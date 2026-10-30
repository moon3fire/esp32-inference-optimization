# Measurement record

Measured on an original ESP32 rev. 3.0 at 160 MHz, using ESP-IDF 6.1.0,
performance optimization and QIO flash at 80 MHz. Results below are transcribed
from board serial output recorded on October 30, 2026.

## Latest latency and stack comparison

Both builds used the same INT8 model and fixed 100-image subset: 20 warmups,
then four rounds of 100 calls. Profiling and duplicate-kernel verification were
disabled. Only `Invoke()` was timed; interrupts and scheduling during the call
remain included.

| Round | Standard ESP-NN mean (µs) | Custom mean (µs) |
| --- | ---: | ---: |
| 1 | 34284.24 | 21476.52 |
| 2 | 34286.77 | 21479.26 |
| 3 | 34286.31 | 21478.85 |
| 4 | 34286.77 | 21479.30 |
| Average | **34286.0225** | **21478.4825** |

The custom version reduced latency by **37.35%**, approximately **1.60× faster**.
This also reproduced the improvement after the repository layout was cleaned up.

| Memory observation | Standard ESP-NN | Custom |
| --- | ---: | ---: |
| Configured main-task stack | 3584 bytes | 3584 bytes |
| Minimum untouched stack | 1768 bytes | 1640 bytes |
| Observed stack use estimate | 1816 bytes | 1944 bytes |
| Tensor arena reserved | 12288 bytes | 12288 bytes |
| Tensor arena used | 8124 bytes | 8124 bytes |

`uxTaskGetStackHighWaterMark(nullptr)` was sampled before the final completion
message. ESP-IDF reports bytes. Subtracting the watermark from the configured
stack gives a task-lifetime estimate including setup, inference and logging;
it is not an isolated inference measurement or a worst-case stack bound.
The custom build used **128 more observed stack bytes**.

## Compiler and disassembly findings

GCC `-fstack-usage` reports and Xtensa disassembly showed:

| Path | Explicit kernel frame allocation |
| --- | ---: |
| Standard generic convolution | 256 bytes |
| Custom conv1 | 128 bytes |
| Custom dispatcher calling generic conv2 | 128 + 256 = 384 bytes |

The fallback uses a normal call while the dispatcher frame remains active.
Its additional frame is consistent with the 128-byte runtime difference.
These figures exclude upstream callers and any additional callees.

The custom kernel stores nine prepared pixels in 36 bytes within its frame
and reloads them inside the output-channel loop. The dot product uses nine
`mul16s` instructions. The assembly also shows stack-backed loop and
requantization temporaries; their individual cycle costs were not measured.

## Earlier baseline and profiling

[whole_network.csv](whole_network.csv) contains the earlier four-round runs:

| Configuration | Mean Invoke latency (ms) |
| --- | ---: |
| ESP-NN initial | 34.290 |
| TFLite fallback | 190.318 |
| ESP-NN restored | 34.289 |
| Custom conv1 | 21.476 |

The enabled → disabled → enabled sequence reproduced the ESP-NN baseline.
The latest stack-instrumented comparison above is recorded separately from
that CSV. `measured-sdkconfig` and `measured-dependencies.lock` preserve the
earlier captured configuration.

Separate per-operator profiling runs produced these approximate means:

| Implementation | Conv1 ms | Conv2 ms | Instrumented Invoke ms |
| --- | ---: | ---: | ---: |
| Standard ESP-NN | 17.793 | 15.149 | 34.360 |
| Custom conv1, before `noipa` | 5.115 | 22.276 | 28.789 |
| Custom conv1, `noipa` fallback | 4.980 | 15.151 | 21.530 |

Adding `noipa` to the generic fallback restored conv2 performance. The exact
cause of the earlier regression was not established. Profiling uses 4,120
bytes of statistics storage and adds timing overhead; these timings are kept
separate from the whole-network benchmark.

## Numerical validation

- Both board implementations: 99/100 correct and 100/100 predictions matching PC.
- Compared with PC: 98/100 images have exact final scores; two of 1,000 scores
  differ by one INT8 step, without changing predictions.
- Verification mode: custom conv1 matches the original kernel byte-for-byte
  across all 5,408 outputs for each of the 100 images.
- Host kernel test: 692,224 output comparisons across 128 bounded synthetic
  cases passed with UndefinedBehaviorSanitizer.
- Full PC test: FP32 9861/10000; INT8 9858/10000.

The board subset contains the first ten test images per digit, without
filtering for correct predictions. The PC/board score difference remains
unexplained; full-model bit-exact agreement is not claimed.

## Capture another run

Save serial output separately for each mode, plus `sdkconfig`,
`dependencies.lock`, the boot ELF hash and:

```bash
idf.py --version
xtensa-esp32-elf-gcc --version
sha256sum main/cnn_model_data.h
```

Run these commands from `firmware/`. Record whether profiling or verification
was enabled. The reported timings are inference latency, not end-to-end
throughput or power measurements.