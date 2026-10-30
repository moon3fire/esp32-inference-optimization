#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "esp_timer.h"
#include "sdkconfig.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "cnn_model_data.h"
#include "cnn_test_data.h"


namespace {

constexpr std::size_t kArenaBytes = 12 * 1024;
constexpr std::size_t kMaxEvents = 64;
constexpr std::uint32_t kInvalidHandle = UINT32_MAX;
constexpr int kWarmupRuns = 20;
constexpr int kRounds = 3;

alignas(16) std::uint8_t tensor_arena[kArenaBytes];

struct Event {
    const char* tag = nullptr;
    std::int64_t start = 0;
    std::int64_t elapsed = 0;
    unsigned depth = 0;
    bool ended = false;
};

class OperatorProfiler final : public tflite::MicroProfilerInterface {
public:
    void start_capture()
    {
        count_ = 0;
        depth_ = 0;
        failed_ = false;
        enabled_ = true;
    }

    bool stop_capture()
    {
        enabled_ = false;

        if (failed_ || depth_ != 0 || count_ == 0) {
            return false;
        }

        for (std::size_t i = 0; i < count_; ++i) {
            if (!events_[i].ended) {
                return false;
            }
        }

        return true;
    }

    std::uint32_t BeginEvent(const char* tag) override
    {
        if (!enabled_) {
            return kInvalidHandle;
        }

        if (count_ >= kMaxEvents) {
            failed_ = true;
            return kInvalidHandle;
        }

        const auto handle = static_cast<std::uint32_t>(count_++);
        Event& event = events_[handle];

        event.tag = tag != nullptr ? tag : "(unnamed)";
        event.depth = depth_++;
        event.ended = false;
        event.elapsed = 0;
        event.start = esp_timer_get_time();

        return handle;
    }

    void EndEvent(std::uint32_t handle) override
    {
        if (!enabled_ || handle == kInvalidHandle) {
            return;
        }

        const std::int64_t finish = esp_timer_get_time();

        if (handle >= count_) {
            failed_ = true;
            return;
        }

        Event& event = events_[handle];

        if (event.ended ||
            depth_ == 0 ||
            event.depth + 1 != depth_) {
            failed_ = true;
            return;
        }

        event.elapsed = finish - event.start;
        event.ended = true;
        --depth_;
    }

    std::size_t count() const
    {
        return count_;
    }

    const Event& event(std::size_t index) const
    {
        return events_[index];
    }

private:
    Event events_[kMaxEvents]{};
    std::size_t count_ = 0;
    unsigned depth_ = 0;
    bool enabled_ = false;
    bool failed_ = false;
};

struct EventStats {
    const char* tag = nullptr;
    unsigned depth = 0;
    std::int64_t total = 0;
    std::int64_t minimum = 0;
    std::int64_t maximum = 0;
};

OperatorProfiler profiler;
EventStats statistics[kMaxEvents];

int argmax(const std::int8_t* values)
{
    int best = 0;

    for (int i = 1; i < 10; ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }

    return best;
}

void load_image(TfLiteTensor* input, std::size_t sample)
{
    for (std::size_t pixel = 0; pixel < 784; ++pixel) {
        input->data.int8[pixel] = static_cast<std::int8_t>(
            static_cast<int>(cnn_test::images[sample][pixel]) - 128);
    }
}

bool check_tensor_shapes(
    const TfLiteTensor* input,
    const TfLiteTensor* output)
{
    return input != nullptr &&
           output != nullptr &&
           input->type == kTfLiteInt8 &&
           output->type == kTfLiteInt8 &&
           input->dims != nullptr &&
           output->dims != nullptr &&
           input->dims->size == 4 &&
           input->dims->data[0] == 1 &&
           input->dims->data[1] == 28 &&
           input->dims->data[2] == 28 &&
           input->dims->data[3] == 1 &&
           output->dims->size == 2 &&
           output->dims->data[0] == 1 &&
           output->dims->data[1] == 10 &&
           input->bytes == 784 &&
           output->bytes == 10;
}

bool run_profile_round(
    tflite::MicroInterpreter& interpreter,
    TfLiteTensor* input,
    TfLiteTensor* output,
    int round)
{
    std::size_t event_count = 0;
    std::int64_t invoke_total = 0;

    std::size_t correct = 0;
    std::size_t pc_correct = 0;
    std::size_t prediction_matches = 0;
    std::size_t exact_images = 0;
    std::size_t score_mismatches = 0;
    int maximum_score_difference = 0;

    for (std::size_t sample = 0; sample < cnn_test::count; ++sample) {
        load_image(input, sample);

        profiler.start_capture();

        const std::int64_t start = esp_timer_get_time();
        const TfLiteStatus status = interpreter.Invoke();
        const std::int64_t finish = esp_timer_get_time();

        const bool capture_valid = profiler.stop_capture();

        if (status != kTfLiteOk) {
            std::printf("FAIL: Invoke at sample %zu\n", sample);
            return false;
        }

        if (!capture_valid) {
            std::printf(
                "FAIL: profiler capture at sample %zu "
                "(missing events, overflow, or invalid nesting)\n",
                sample);
            return false;
        }

        invoke_total += finish - start;

        // Aggregate after Invoke, outside all measured events.
        if (sample == 0) {
            event_count = profiler.count();

            for (std::size_t i = 0; i < event_count; ++i) {
                const Event& event = profiler.event(i);
                statistics[i] = {
                    event.tag,
                    event.depth,
                    event.elapsed,
                    event.elapsed,
                    event.elapsed
                };
            }
        } else {
            if (profiler.count() != event_count) {
                std::printf("FAIL: event count changed\n");
                return false;
            }

            for (std::size_t i = 0; i < event_count; ++i) {
                const Event& event = profiler.event(i);
                EventStats& stats = statistics[i];

                if (event.depth != stats.depth ||
                    std::strcmp(event.tag, stats.tag) != 0) {
                    std::printf(
                        "FAIL: event sequence changed at event %zu\n", i);
                    return false;
                }

                stats.total += event.elapsed;

                if (event.elapsed < stats.minimum) {
                    stats.minimum = event.elapsed;
                }
                if (event.elapsed > stats.maximum) {
                    stats.maximum = event.elapsed;
                }
            }
        }

        const int predicted = argmax(output->data.int8);
        const int pc_predicted = argmax(cnn_test::expected[sample]);
        const int label = cnn_test::labels[sample];

        correct += (predicted == label);
        pc_correct += (pc_predicted == label);
        prediction_matches += (predicted == pc_predicted);

        bool exact = true;

        for (int digit = 0; digit < 10; ++digit) {
            int difference =
                static_cast<int>(output->data.int8[digit]) -
                static_cast<int>(cnn_test::expected[sample][digit]);

            if (difference < 0) {
                difference = -difference;
            }

            if (difference != 0) {
                exact = false;
                ++score_mismatches;
            }

            if (difference > maximum_score_difference) {
                maximum_score_difference = difference;
            }
        }

        exact_images += exact;

        vTaskDelay(1);
    }

    std::printf("\n--- Operator profile round %d ---\n", round);
    std::printf("Event order follows BeginEvent calls.\n");

    for (std::size_t i = 0; i < event_count; ++i) {
        const EventStats& stats = statistics[i];

        std::printf(
            "[%zu] depth=%u %-24s "
            "mean=%.2f us, min=%lld us, max=%lld us\n",
            i,
            stats.depth,
            stats.tag,
            static_cast<double>(stats.total) / cnn_test::count,
            static_cast<long long>(stats.minimum),
            static_cast<long long>(stats.maximum));
    }

    std::printf(
        "Instrumented Invoke mean: %.2f us\n",
        static_cast<double>(invoke_total) / cnn_test::count);

    std::printf(
        "ESP32 correct: %zu/%zu\n",
        correct, cnn_test::count);
    std::printf(
        "PC correct: %zu/%zu\n",
        pc_correct, cnn_test::count);
    std::printf(
        "Prediction agreement: %zu/%zu\n",
        prediction_matches, cnn_test::count);
    std::printf(
        "Exact output images: %zu/%zu\n",
        exact_images, cnn_test::count);
    std::printf(
        "Output score mismatches: %zu/%zu\n",
        score_mismatches, cnn_test::count * 10);
    std::printf(
        "Maximum score difference: %d integer steps\n",
        maximum_score_difference);

    return true;
}

}  // namespace

extern "C" void app_main()
{
    std::printf("\nESP32 CNN: per-operator profiling\n");
    std::printf(
        "Configured CPU frequency: %d MHz\n",
        static_cast<int>(CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ));

    const tflite::Model* model = tflite::GetModel(cnn_model::data);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        std::printf("FAIL: model schema mismatch\n");
        return;
    }

    static tflite::MicroMutableOpResolver<5> resolver;

    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddRelu() != kTfLiteOk) {
        std::printf("FAIL: operator registration\n");
        return;
    }

    // Fifth argument: resource variables.
    // Sixth argument: profiler.
    static tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        sizeof(tensor_arena),
        nullptr,
        &profiler);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        std::printf("FAIL: AllocateTensors\n");
        return;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    if (!check_tensor_shapes(input, output)) {
        std::printf("FAIL: unexpected tensor type or shape\n");
        return;
    }

    if (input->params.scale != cnn_model::input_scale ||
        input->params.zero_point != cnn_model::input_zero ||
        output->params.scale != cnn_model::output_scale ||
        output->params.zero_point != cnn_model::output_zero) {
        std::printf("FAIL: quantization metadata mismatch\n");
        return;
    }

    std::printf("Model file: %zu bytes\n", cnn_model::size);
    std::printf("Arena reserved: %zu bytes\n", sizeof(tensor_arena));
    std::printf(
        "Arena used: %zu bytes\n",
        interpreter.arena_used_bytes());
    std::printf(
        "Profiler and statistics storage: %zu bytes\n",
        sizeof(profiler) + sizeof(statistics));

    vTaskDelay(pdMS_TO_TICKS(100));

    // Capture remains disabled during warmup.
    for (int i = 0; i < kWarmupRuns; ++i) {
        load_image(
            input,
            static_cast<std::size_t>(i) % cnn_test::count);

        if (interpreter.Invoke() != kTfLiteOk) {
            std::printf("FAIL: warmup Invoke\n");
            return;
        }

        vTaskDelay(1);
    }

    for (int round = 1; round <= kRounds; ++round) {
        if (!run_profile_round(interpreter, input, output, round)) {
            return;
        }

        vTaskDelay(pdMS_TO_TICKS(100));
    }

    std::printf("\nProfiling complete.\n");
}