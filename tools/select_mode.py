"""Select a reproducible kernel configuration before rebuilding firmware."""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['reference', 'espnn', 'custom', 'verify'])
args = parser.parse_args()

cmake = ROOT / 'firmware/local_components/esp-tflite-micro/CMakeLists.txt'
text = cmake.read_text()
enabled = 'target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN)'
disabled = '# EDGEINFER: ESP_NN integration disabled'
assert text.count(enabled) + text.count(disabled) == 1
text = text.replace(enabled, disabled)
if args.mode != 'reference':
    text = text.replace(disabled, enabled)

template = 'custom' if args.mode in ('custom', 'verify') else 'stock'
kernel = (ROOT / f'tools/esp_nn_conv_opt.{template}.c').read_text()
if args.mode == 'verify':
    kernel = kernel.replace('#define EDGEINFER_VERIFY_CONV1 0',
                            '#define EDGEINFER_VERIFY_CONV1 1')

cmake.write_text(text)
(ROOT / 'firmware/local_components/esp-nn/src/convolution/esp_nn_conv_opt.c').write_text(kernel)
print(f'Selected {args.mode}; run idf.py reconfigure and idf.py build in firmware/.')
if args.mode == 'verify':
    print('Verification executes both kernels; do not use its latency as a benchmark.')
