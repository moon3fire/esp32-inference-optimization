# Training and export

The bundled model can run without retraining. Use a separate Python environment
from ESP-IDF for training and conversion. The recorded run used Python 3.12.16,
TensorFlow CPU 2.20.0, Keras 3.15.1 and NumPy 2.5.4; `requirements.txt` pins
these versions, rather than a complete transitive dependency lock.

From the repository root:

```bash
cd training
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Retrains and overwrites saved training artifacts.
python train_cnn.py

# Converts the checkpoint and evaluates FP32/INT8 on the PC.
python export_tflite.py

# Writes and checks headers in ../firmware/main/.
python export_firmware.py
```

To regenerate headers from existing artifacts, run only `export_firmware.py`
after environment setup. The first dataset load downloads MNIST. Conversion
uses an internal TensorFlow freezing helper to avoid a READ_VARIABLE calibration
failure; the exporter is tied to TensorFlow 2.20.0 and emits deprecation warnings.

## Model and data

28×28×1 → Conv3×3/8 + ReLU → MaxPool2×2 → Conv3×3/16 + ReLU → MaxPool2×2
→ Flatten → Dense10 logits. Total: 5,258 parameters.

- Training: 54,001 images; validation: 5,999 images.
- Official test set: 10,000 images.
- Recorded best validation-loss checkpoint: epoch 19.
- Calibration: 500 training images, 50 per digit; no validation/test images.

The saved model is the reference for the reported board results. Fresh training
may produce different weights and measurements.