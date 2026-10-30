# First-convolution specialization

The entry point remains `esp_nn_conv_s8_opt`. It dispatches to the specialized
implementation only when all its guards match:

- Input 28×28×1; filter 3×3×1; output 26×26×8.
- Unit stride, no padding, input offset +128.
- ESP-NN wrapper dilation fields `{0, 0}`. The supplied TFLite wrapper first
  checks actual model dilation is one, then passes these internal fields.

Every other case uses the original generic ESP-NN implementation. This is a
model-shape specialization, not a general replacement for convolution.

For each output position the custom kernel loads nine input pixels, adds the
input offset once, and reuses those values across eight output channels. The
nine multiply-accumulates are explicit. Output layout, bias addition,
per-channel multipliers/shifts, offset and activation clipping follow the
original implementation. No model weights are changed and no runtime scratch
buffer is added by the specialization. Target disassembly shows that the nine
prepared pixels occupy 36 bytes of the 128-byte frame and are reloaded in the
output-channel loop. The generic fallback has a 256-byte frame and is called
with the dispatcher frame still active (384 bytes combined).

The latest board comparison measured 1,816 bytes of observed main-task stack
use with standard ESP-NN and 1,944 bytes with the custom build. This is a
task-lifetime watermark estimate including setup and logging, not an isolated
inference stack bound. See [the measurement record](../results/README.md).

With input values after offset in [0,255], nine signed INT8 products have an
absolute-sum bound of 293,760 before bias. Bias and requantization intermediates
must also fit their respective integer ranges. The kernel inherits the original
implementation's range assumptions; arbitrary models are not covered by the
reported checks.

The original first layer emits 5,408 values versus 1,936 for conv2, so it incurs
more per-output quantization work. Its one input channel also bypasses the
generic four-channel loop. These source observations motivated the experiment;
they are not a cycle-level explanation of every measured cost.

## Verification

`verify` mode runs both kernels on each matching invocation and compares the
5,408 output bytes. Any mismatch prints the first position and aborts. The
reported board test covered the fixed 100 images, not the full test set.
The separate host test broadens coverage with bounded synthetic inputs,
weights, biases and quantization parameters; it does not model Xtensa timing.

## Per-operator profiler

The profiler source is preserved separately from the main benchmark. From
`firmware/`:

```bash
idf.py -DEDGEINFER_PROFILE=ON reconfigure
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

It records events in execution order, so the two convolutions remain separate.
It validates the event sequence across images and prints after measurement.
Nested events must not be summed as if they were disjoint. In the observed model
all six operator events had depth zero. Profiling has a separate 4,120-byte
storage cost in the measured build and adds timing overhead.

Return to the whole-network benchmark explicitly:

```bash
idf.py -DEDGEINFER_PROFILE=OFF reconfigure
idf.py build
```

Kernel selection and profiling are independent. For final performance results,
select `custom`, `espnn`, or `reference` and set profiling OFF. `verify` is only
for numerical testing.