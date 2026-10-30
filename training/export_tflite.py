from pathlib import Path
import hashlib
import json

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)


def normalize(image):
    """Convert one uint8 image into a normalized batch of one."""
    return image.astype(np.float32)[None, ..., None] / 255.0


def evaluate(model_bytes, images, integer):
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

    assert tuple(input_info["shape"]) == (1, 28, 28, 1)
    assert tuple(output_info["shape"]) == (1, 10)

    expected_dtype = np.int8 if integer else np.float32
    assert input_info["dtype"] == expected_dtype
    assert output_info["dtype"] == expected_dtype

    if integer:
        float_tensors = [
            detail["name"]
            for detail in interpreter.get_tensor_details()
            if np.issubdtype(detail["dtype"], np.floating)
        ]
        assert not float_tensors, (
            f"Unexpected floating-point tensors: {float_tensors}"
        )

        input_scale, input_zero = input_info["quantization"]
        output_scale, output_zero = output_info["quantization"]

        assert input_scale > 0
        assert output_scale > 0

        print(
            "INT8 input scale / zero point:",
            input_scale,
            input_zero,
        )
        print(
            "INT8 output scale / zero point:",
            output_scale,
            output_zero,
        )

    outputs = np.empty(
        (len(images), 10),
        dtype=expected_dtype,
    )

    for index, image in enumerate(images):
        input_value = normalize(image)

        if integer:
            input_value = np.clip(
                np.rint(input_value / input_scale) + input_zero,
                -128,
                127,
            ).astype(np.int8)

        interpreter.set_tensor(input_info["index"], input_value)
        interpreter.invoke()

        outputs[index] = interpreter.get_tensor(
            output_info["index"]
        )[0]

        if (index + 1) % 2000 == 0:
            print(f"Evaluated {index + 1}/{len(images)} images")

    assert np.isfinite(outputs).all()

    # A shared positive output scale preserves the argmax.
    predictions = np.argmax(outputs, axis=1).astype(np.uint8)

    metadata = {
        "input_shape": input_info["shape"].tolist(),
        "output_shape": output_info["shape"].tolist(),
        "input_scale": float(input_info["quantization"][0]),
        "input_zero_point": int(input_info["quantization"][1]),
        "output_scale": float(output_info["quantization"][0]),
        "output_zero_point": int(output_info["quantization"][1]),
    }

    return outputs, predictions, metadata


def main():
    model = tf.keras.models.load_model(
        ARTIFACTS / "best_model.keras",
        compile=False,
    )
    assert model.count_params() == 5258

    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )

    with np.load(
        ARTIFACTS / "split_indices.npz",
        allow_pickle=False,
    ) as split:
        training_indices = split["training"].copy()
        validation_indices = split["validation"].copy()

    # Use 50 training images per digit for calibration.
    rng = np.random.default_rng(42)
    calibration_parts = []

    for digit in range(10):
        candidates = training_indices[
            train_labels[training_indices] == digit
        ]
        calibration_parts.append(
            rng.choice(candidates, size=50, replace=False)
        )

    calibration_indices = np.concatenate(calibration_parts)
    rng.shuffle(calibration_indices)

    assert len(calibration_indices) == 500
    assert len(np.unique(calibration_indices)) == 500
    assert np.intersect1d(
        calibration_indices, validation_indices
    ).size == 0

    np.save(
        ARTIFACTS / "calibration_indices.npy",
        calibration_indices,
    )

    def representative_dataset():
        for index in calibration_indices:
            yield [normalize(train_images[index])]

    # Fixed batch size avoids dynamic batch-shape operations.
    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=(1, 28, 28, 1),
                dtype=tf.float32,
                name="image",
            )
        ]
    )
    def serving(image):
        return model(image, training=False)

    print("\nFreezing learned weights into constants...")

    concrete = serving.get_concrete_function()
    frozen = convert_variables_to_constants_v2(concrete)

    resource_inputs = [
        tensor.name
        for tensor in frozen.inputs
        if tensor.dtype == tf.resource
    ]
    assert not resource_inputs, resource_inputs

    variable_ops = [
        operation.name
        for operation in frozen.graph.get_operations()
        if operation.type in ("ReadVariableOp", "VarHandleOp")
    ]
    assert not variable_ops, variable_ops

    # Confirm that freezing preserves the model's outputs.
    for index in calibration_indices[:10]:
        sample = normalize(train_images[index])

        expected = model(sample, training=False).numpy()
        actual = frozen(tf.convert_to_tensor(sample))[0].numpy()

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-5,
        )

    print("Frozen graph verification: PASS")

    print("\nConverting FP32 model...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [frozen]
    )
    fp32_bytes = converter.convert()
    (ARTIFACTS / "cnn_fp32.tflite").write_bytes(fp32_bytes)

    print("\nConverting INT8 model...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [frozen]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    int8_bytes = converter.convert()
    (ARTIFACTS / "cnn_int8.tflite").write_bytes(int8_bytes)

    print("\nEvaluating FP32 TFLite...")
    fp32_logits, fp32_predictions, _ = evaluate(
        fp32_bytes,
        test_images,
        integer=False,
    )

    print("\nEvaluating INT8 TFLite...")
    int8_outputs, int8_predictions, quantization = evaluate(
        int8_bytes,
        test_images,
        integer=True,
    )

    with np.load(
        ARTIFACTS / "test_reference.npz",
        allow_pickle=False,
    ) as reference:
        assert np.array_equal(reference["labels"], test_labels)
        keras_predictions = reference["predictions"].copy()
        keras_logits = reference["logits"].copy()

    fp32_correct = int(np.sum(fp32_predictions == test_labels))
    int8_correct = int(np.sum(int8_predictions == test_labels))

    keras_changes = int(np.sum(
        fp32_predictions != keras_predictions
    ))
    quantization_changes = int(np.sum(
        int8_predictions != fp32_predictions
    ))
    fp32_max_error = float(np.max(np.abs(
        fp32_logits - keras_logits
    )))

    # Preserve raw INT8 outputs for later ESP32 comparisons.
    np.savez_compressed(
        ARTIFACTS / "tflite_test_reference.npz",
        labels=test_labels,
        fp32_logits=fp32_logits,
        fp32_predictions=fp32_predictions,
        int8_outputs=int8_outputs,
        int8_predictions=int8_predictions,
    )

    report = {
        "tensorflow": tf.__version__,
        "calibration_images": len(calibration_indices),
        "test_images": len(test_images),
        "fp32_correct": fp32_correct,
        "int8_correct": int8_correct,
        "fp32_vs_keras_changed_predictions": keras_changes,
        "fp32_vs_keras_max_score_error": fp32_max_error,
        "int8_vs_fp32_changed_predictions": quantization_changes,
        "fp32_file_bytes": len(fp32_bytes),
        "int8_file_bytes": len(int8_bytes),
        "fp32_sha256": hashlib.sha256(fp32_bytes).hexdigest(),
        "int8_sha256": hashlib.sha256(int8_bytes).hexdigest(),
        "quantization": quantization,
    }

    (ARTIFACTS / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    test_count = len(test_images)
    accuracy_change = (
        100.0 * (int8_correct - fp32_correct) / test_count
    )
    file_reduction = (
        100.0 * (1.0 - len(int8_bytes) / len(fp32_bytes))
    )

    print("\n--- Conversion results ---")
    print(f"Calibration images: {len(calibration_indices)}")
    print(
        f"FP32 TFLite: {fp32_correct}/{test_count} "
        f"({100.0 * fp32_correct / test_count:.2f}%)"
    )
    print(
        f"INT8 TFLite: {int8_correct}/{test_count} "
        f"({100.0 * int8_correct / test_count:.2f}%)"
    )
    print(
        f"FP32 vs Keras changed predictions: {keras_changes}"
    )
    print(
        f"FP32 vs Keras maximum score error: "
        f"{fp32_max_error:.8f}"
    )
    print(
        f"INT8 vs FP32 changed predictions: "
        f"{quantization_changes}"
    )
    print(
        f"INT8 accuracy change: "
        f"{accuracy_change:+.2f} percentage points"
    )
    print(f"FP32 model file: {len(fp32_bytes)} bytes")
    print(f"INT8 model file: {len(int8_bytes)} bytes")
    print(f"Model file reduction: {file_reduction:.2f}%")
    print(f"Saved artifacts: {ARTIFACTS}")


if __name__ == "__main__":
    main()