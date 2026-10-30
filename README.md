# ESP32 CNN benchmark

An embedded inference project built with C/C++, ESP-IDF,
TensorFlow Lite Micro and ESP-NN. It deploys an INT8 MNIST classifier on an
original ESP32 and optimizes its first convolution using operator profiling
and a guarded, model-specific kernel.

**Result: 37.35% lower inference latency than standard ESP-NN, with 128 bytes
more observed task stack usage and unchanged predictions on the 100-image board test.**

## Results

ESP32 rev. 3.0 · 160 MHz · ESP-IDF 6.1.0 · performance optimization · QIO flash at 80 MHz.
Latency measures `Invoke()` only, averaged over four rounds of 100 images after 20 warmups.

| Configuration | Mean latency | Observed main-task stack use |
| --- | ---: | ---: |
| TFLite fallback, ESP-NN disabled¹ | 190.318 ms | — |
| Standard ESP-NN | 34.286 ms | 1,816 bytes |
| ESP-NN + custom conv1 | **21.478 ms** | 1,944 bytes |

¹ Earlier baseline run with the same model and core settings. ESP-NN provides
the larger baseline improvement; the custom kernel builds on Espressif's implementation.

- **Model:** 8,984-byte INT8 file, down from 23,576 bytes FP32 (61.89% smaller).
- **Accuracy:** 98.58% INT8 versus 98.61% FP32 on the full 10,000-image PC test.
- **Board verification:** 99/100 correct; 100/100 predictions agree with PC.
- **Tensor arena:** 12,288 bytes reserved, 8,124 bytes used in both ESP-NN builds.

Stack figures are task-lifetime watermark estimates, including setup and logging.
The arena is not total RAM. Detailed measurements and numerical checks are in
[results/README.md](results/README.md).

## What changed

Profiling identified conv1 as the largest operator cost. The custom kernel
prepares each 3×3 input patch once and reuses it across eight output channels,
removing generic loop and boundary handling. It retains ESP-NN's requantization
and dispatches all other shapes to the original implementation.

The experiment includes per-operator profiling, byte-level kernel verification,
host tests, and Xtensa disassembly/stack analysis. See
[kernel notes](docs/kernel-notes.md) for the implementation and tradeoffs.

## Build and run

Requires an original ESP32 and ESP-IDF **6.1.0**. Activate your ESP-IDF environment;
for the EIM installation used here:

```bash
source "$HOME/.espressif/tools/activate_idf_v6.1.sh"
```

From the repository root:

```bash
python3 tools/select_mode.py custom
cd firmware
idf.py -DEDGEINFER_PROFILE=OFF reconfigure
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

Use your board's serial port. Exit with **Ctrl+]**. On Ubuntu, your session needs
`dialout` membership; in VMware, attach the USB serial device to the guest.
Keep the included `sdkconfig` for comparable results. Model headers and local
dependency sources are bundled; training is not required to run the firmware.

## Compare implementations

Select a mode from the repository root, then repeat the build/flash commands above:

| Mode | Implementation |
| --- | --- |
| `reference` | TFLite fallback operators, ESP-NN integration disabled |
| `espnn` | Standard ESP-NN |
| `custom` | Specialized conv1 with original generic fallback |
| `verify` | Runs both conv1 kernels and compares all 5,408 outputs per image |

The selector rewrites vendored source from templates in `tools/`; edit the
templates when developing the kernel. Verification executes both kernels and
adds a 5,408-byte buffer, so do not use its timings for performance comparisons.

For per-operator profiling, run `idf.py -DEDGEINFER_PROFILE=ON reconfigure` in
`firmware/`, rebuild and flash. Set it back to `OFF` for whole-network timing.

## Tests and training

Host checks require Python, NumPy and GCC:

```bash
python3 tests/check_artifacts.py
python3 tests/run_kernel_test.py
```

The checks validate saved artifacts and compare 692,224 kernel outputs across
128 synthetic cases with UndefinedBehaviorSanitizer.
Optional training and export commands are in [docs/training.md](docs/training.md).

## License

Project-authored code is released under the [MIT License](LICENSE).
Bundled dependencies and MNIST data retain their own licenses and attribution;
see [THIRD_PARTY.md](THIRD_PARTY.md).
