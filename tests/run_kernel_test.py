from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
vendor = root / 'firmware/local_components/esp-nn'
with tempfile.TemporaryDirectory() as directory:
    binary = str(Path(directory) / 'kernel_test')
    subprocess.run([
        'gcc', '-std=gnu11', '-O2', '-fsanitize=undefined',
        '-fno-sanitize-recover=all',
        '-I' + str(vendor / 'include'),
        '-I' + str(vendor / 'src/common'),
        str(root / 'tests/kernel_equivalence.c'),
        str(vendor / 'src/convolution/esp_nn_conv_ansi.c'),
        '-o', binary,
    ], check=True)
    subprocess.run([binary], check=True)
