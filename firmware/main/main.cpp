#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "esp_timer.h"
#include "sdkconfig.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "cnn_model_data.h"
#include "cnn_test_data.h"

namespace
{

    constexpr std::size_t kArenaBytes = 12 * 1024;
    constexpr int kWarmupRuns = 20;
    constexpr int kRounds = 4;

    static_assert(cnn_test::count > 0);

    alignas(16) std::uint8_t tensor_arena[kArenaBytes];
    std::int64_t timings[cnn_test::count];

    int argmax(const std::int8_t *values)
    {
        int best = 0;

        for (int i = 1; i < 10; ++i)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }

    void load_image(TfLiteTensor *input, std::size_t sample)
    {
        for (std::size_t pixel = 0; pixel < 784; ++pixel)
        {
            input->data.int8[pixel] = static_cast<std::int8_t>(
                static_cast<int>(cnn_test::images[sample][pixel]) - 128);
        }
    }

    bool check_tensor_shapes(
        const TfLiteTensor *input,
        const TfLiteTensor *output)
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

    void print_timing_statistics()
    {
        std::int64_t total = 0;

        for (std::size_t i = 0; i < cnn_test::count; ++i)
        {
            total += timings[i];
        }

        std::sort(timings, timings + cnn_test::count);

        const std::size_t count = cnn_test::count;
        const double mean = static_cast<double>(total) / count;

        const double median = (count % 2 == 0)
                                  ? (static_cast<double>(timings[count / 2 - 1]) +
                                     static_cast<double>(timings[count / 2])) /
                                        2.0
                                  : static_cast<double>(timings[count / 2]);

        const std::size_t p95_index = (95 * count + 99) / 100 - 1;

        std::printf(
            "Invoke latency: mean=%.2f us, median=%.2f us, "
            "p95=%lld us, min=%lld us, max=%lld us\n",
            mean,
            median,
            static_cast<long long>(timings[p95_index]),
            static_cast<long long>(timings[0]),
            static_cast<long long>(timings[count - 1]));
    }

    bool run_round(
        tflite::MicroInterpreter &interpreter,
        TfLiteTensor *input,
        TfLiteTensor *output,
        int round)
    {
        std::size_t correct = 0;
        std::size_t pc_correct = 0;
        std::size_t prediction_matches = 0;
        std::size_t exact_images = 0;
        std::size_t score_mismatches = 0;
        int maximum_score_difference = 0;

        for (std::size_t sample = 0; sample < cnn_test::count; ++sample)
        {
            load_image(input, sample);

            // Only Invoke is inside the timed interval.
            const std::int64_t start = esp_timer_get_time();
            const TfLiteStatus status = interpreter.Invoke();
            const std::int64_t finish = esp_timer_get_time();

            timings[sample] = finish - start;

            if (status != kTfLiteOk)
            {
                std::printf(
                    "FAIL: Invoke in round %d, sample %zu\n",
                    round, sample);
                return false;
            }

            const int predicted = argmax(output->data.int8);
            const int pc_predicted = argmax(cnn_test::expected[sample]);
            const int label = cnn_test::labels[sample];

            correct += (predicted == label);
            pc_correct += (pc_predicted == label);
            prediction_matches += (predicted == pc_predicted);

            bool exact = true;

            for (int digit = 0; digit < 10; ++digit)
            {
                int difference =
                    static_cast<int>(output->data.int8[digit]) -
                    static_cast<int>(cnn_test::expected[sample][digit]);

                if (difference < 0)
                {
                    difference = -difference;
                }

                if (difference != 0)
                {
                    exact = false;
                    ++score_mismatches;
                }

                if (difference > maximum_score_difference)
                {
                    maximum_score_difference = difference;
                }
            }

            exact_images += exact;

            // Same scheduling policy as the original benchmark.
            vTaskDelay(1);
        }

        std::printf("\n--- Benchmark round %d ---\n", round);
        print_timing_statistics();

        std::printf(
            "ESP32 correct: %zu/%zu\n",
            correct, cnn_test::count);
        std::printf(
            "PC correct on same subset: %zu/%zu\n",
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

} // namespace

extern "C" void app_main()
{
    std::printf("\nESP32 CNN: whole-network benchmark\n");
    std::printf(
        "Configured CPU frequency: %d MHz\n",
        static_cast<int>(CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ));
    std::printf(
        "Timing scope: Invoke only; %zu images, %d rounds\n",
        cnn_test::count, kRounds);

    const tflite::Model *model = tflite::GetModel(cnn_model::data);

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        std::printf("FAIL: model schema mismatch\n");
        return;
    }

    static tflite::MicroMutableOpResolver<5> resolver;

    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddRelu() != kTfLiteOk)
    {
        std::printf("FAIL: operator registration\n");
        return;
    }

    // No per-operator profiler attached.
    static tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        sizeof(tensor_arena));

    if (interpreter.AllocateTensors() != kTfLiteOk)
    {
        std::printf(
            "FAIL: AllocateTensors; arena budget=%zu bytes\n",
            sizeof(tensor_arena));
        return;
    }

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    if (!check_tensor_shapes(input, output))
    {
        std::printf("FAIL: unexpected tensor type or shape\n");
        return;
    }

    if (input->params.scale != cnn_model::input_scale ||
        input->params.zero_point != cnn_model::input_zero ||
        output->params.scale != cnn_model::output_scale ||
        output->params.zero_point != cnn_model::output_zero)
    {
        std::printf("FAIL: quantization metadata mismatch\n");
        return;
    }

    constexpr std::size_t fixture_bytes =
        sizeof(cnn_test::images) +
        sizeof(cnn_test::labels) +
        sizeof(cnn_test::expected) +
        sizeof(cnn_test::indices);

    std::printf("Model file: %zu bytes\n", cnn_model::size);
    std::printf("Arena reserved: %zu bytes\n", sizeof(tensor_arena));
    std::printf(
        "Arena used: %zu bytes\n",
        interpreter.arena_used_bytes());
    std::printf("Verification data: %zu bytes\n", fixture_bytes);
    std::printf("Timing array: %zu bytes\n", sizeof(timings));

    std::printf(
        "Input: scale=%.10f, zero point=%d\n",
        static_cast<double>(input->params.scale),
        static_cast<int>(input->params.zero_point));
    std::printf(
        "Output: scale=%.10f, zero point=%d\n",
        static_cast<double>(output->params.scale),
        static_cast<int>(output->params.zero_point));

    vTaskDelay(pdMS_TO_TICKS(100));

    for (int i = 0; i < kWarmupRuns; ++i)
    {
        load_image(
            input,
            static_cast<std::size_t>(i) % cnn_test::count);

        if (interpreter.Invoke() != kTfLiteOk)
        {
            std::printf("FAIL: warmup Invoke\n");
            return;
        }

        vTaskDelay(1);
    }

    for (int round = 1; round <= kRounds; ++round)
    {
        if (!run_round(interpreter, input, output, round))
        {
            return;
        }

        vTaskDelay(pdMS_TO_TICKS(100));
    }

    const std::size_t minimum_free_stack =
        static_cast<std::size_t>(uxTaskGetStackHighWaterMark(nullptr));

    const std::size_t configured_stack =
        static_cast<std::size_t>(CONFIG_ESP_MAIN_TASK_STACK_SIZE);

    std::printf("\n--- Main task stack observation ---\n");
    std::printf("Configured stack: %zu bytes\n", configured_stack);
    std::printf("Minimum untouched stack: %zu bytes\n", minimum_free_stack);

    if (minimum_free_stack <= configured_stack)
    {
        std::printf(
            "Observed stack use estimate: %zu bytes\n",
            configured_stack - minimum_free_stack);
    }

    std::printf("\nBenchmark complete.\n");
}