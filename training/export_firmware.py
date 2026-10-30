from pathlib import Path
import hashlib
import json

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DESTINATION = ROOT.parent / "firmware" / "main"


def format_rows(array):
    return ",\n".join(
        "    {" + ", ".join(str(int(v)) for v in row) + "}"
        for row in array
    )


def main():
    model_bytes = (ARTIFACTS / "cnn_int8.tflite").read_bytes()
    report = json.loads(
        (ARTIFACTS / "conversion_report.json").read_text()
    )

    # Ensure the model is the one evaluated by export_tflite.py.
    assert hashlib.sha256(model_bytes).hexdigest() == report["int8_sha256"]

    (_, _), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )

    with np.load(
        ARTIFACTS / "tflite_test_reference.npz",
        allow_pickle=False,
    ) as reference:
        assert np.array_equal(reference["labels"], test_labels)
        outputs = reference["int8_outputs"].copy()
        predictions = reference["int8_predictions"].copy()

    assert outputs.dtype == np.int8
    assert outputs.shape == (10000, 10)
    assert np.array_equal(
        np.argmax(outputs, axis=1),
        predictions,
    )

    # Fixed selection, independent of whether predictions are correct.
    indices = np.concatenate([
        np.flatnonzero(test_labels == digit)[:10]
        for digit in range(10)
    ])
    indices.sort()

    pixels = test_images[indices].reshape(100, 784)
    labels = test_labels[indices]
    expected = outputs[indices]

    interpreter = tf.lite.Interpreter(
        model_content=model_bytes,
        num_threads=1,
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType
            .BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    interpreter.allocate_tensors()

    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    assert input_info["dtype"] == np.int8
    assert output_info["dtype"] == np.int8
    assert tuple(input_info["shape"]) == (1, 28, 28, 1)
    assert tuple(output_info["shape"]) == (1, 10)

    input_scale, input_zero = input_info["quantization"]
    output_scale, output_zero = output_info["quantization"]

    # Prove pixel - 128 matches the PC quantization for every
    # possible raw pixel, not just the selected images.
    all_pixels = np.arange(256, dtype=np.float32)
    quantized = np.clip(
        np.rint((all_pixels / 255.0) / input_scale) + input_zero,
        -128,
        127,
    ).astype(np.int8)

    direct = (
        np.arange(256, dtype=np.int16) - 128
    ).astype(np.int8)

    assert np.array_equal(quantized, direct)
    print("All 256 pixel mappings: PASS")

    # Recheck saved scores against the exact exported model.
    for sample, raw_pixels in enumerate(pixels):
        input_data = (
            raw_pixels.astype(np.int16) - 128
        ).astype(np.int8).reshape(1, 28, 28, 1)

        interpreter.set_tensor(input_info["index"], input_data)
        interpreter.invoke()
        actual = interpreter.get_tensor(output_info["index"])[0]

        assert np.array_equal(actual, expected[sample]), sample

    print("Selected PC reference scores: PASS")

    DESTINATION.mkdir(parents=True, exist_ok=True)

    byte_lines = [
        "    " + ", ".join(
            f"0x{value:02x}" for value in model_bytes[start:start + 16]
        )
        for start in range(0, len(model_bytes), 16)
    ]

    model_header = (
        "#pragma once\n"
        "#include <cstddef>\n"
        "#include <cstdint>\n\n"
        "namespace cnn_model {\n"
        "alignas(16) static const unsigned char data[] = {\n"
        + ",\n".join(byte_lines)
        + "\n};\n"
        "static constexpr std::size_t size = sizeof(data);\n"
        f"static constexpr float input_scale = {input_scale!r}f;\n"
        f"static constexpr int input_zero = {input_zero};\n"
        f"static constexpr float output_scale = {output_scale!r}f;\n"
        f"static constexpr int output_zero = {output_zero};\n"
        "}\n"
    )

    fixture_header = (
        "#pragma once\n"
        "#include <cstddef>\n"
        "#include <cstdint>\n\n"
        "namespace cnn_test {\n"
        "static constexpr std::size_t count = 100;\n"
        "static const std::uint8_t images[count][784] = {\n"
        + format_rows(pixels)
        + "\n};\n"
        "static const std::uint8_t labels[count] = {\n    "
        + ", ".join(str(int(v)) for v in labels)
        + "\n};\n"
        "static const std::int8_t expected[count][10] = {\n"
        + format_rows(expected)
        + "\n};\n"
        "static const std::uint16_t indices[count] = {\n    "
        + ", ".join(str(int(v)) for v in indices)
        + "\n};\n"
        "}\n"
    )

    (DESTINATION / "cnn_model_data.h").write_text(
        model_header, encoding="utf-8"
    )
    (DESTINATION / "cnn_test_data.h").write_text(
        fixture_header, encoding="utf-8"
    )

    correct = int(np.sum(np.argmax(expected, axis=1) == labels))

    print(f"Model bytes: {len(model_bytes)}")
    print(f"Verification images: {len(indices)}")
    print(f"PC correct on selected images: {correct}/{len(indices)}")
    print(f"Exported headers to: {DESTINATION}")


if __name__ == "__main__":
    main()