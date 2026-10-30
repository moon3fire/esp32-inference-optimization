# Third-party code and data

## Dependencies

The repository includes source snapshots of:

- [Espressif ESP-NN 1.4.2](https://github.com/espressif/esp-nn)
- [Espressif esp-tflite-micro 1.4.1](https://github.com/espressif/esp-tflite-micro), including [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro) dependencies

Their license files and copyright notices are retained. Local changes add kernel
mode selection and the guarded first-convolution specialization. The source
templates in `tools/` retain the original Espressif notices.

ESP-IDF, TensorFlow, Keras and NumPy are external build/training dependencies
and remain subject to their respective licenses.

## MNIST

MNIST is credited to Yann LeCun, Corinna Cortes and Christopher J. C. Burges,
with source data from NIST. [Keras documents MNIST](https://keras.io/api/datasets/mnist/)
under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

`firmware/main/cnn_test_data.h` contains 100 MNIST test images converted to C
arrays, with labels, original indices and model-generated reference scores.
Image pixel values are unchanged. The full MNIST dataset is not bundled.
Retain this attribution and the applicable dataset license when redistributing fixtures.

## Project license

The root [MIT License](LICENSE) covers project-authored code and documentation.
It does not replace the licenses of third-party code or datasets.