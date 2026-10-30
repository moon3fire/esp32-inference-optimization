from pathlib import Path
import json
import platform

import numpy as np
import tensorflow as tf


SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 20

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Limit CPU threads for a predictable workload in the Ubuntu VM.
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()


def make_dataset(images, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if training:
        dataset = dataset.shuffle(
            len(images),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(BATCH_SIZE)

    # Convert only each batch, rather than all images at once.
    dataset = dataset.map(
        lambda image, label: (
            tf.cast(image[..., tf.newaxis], tf.float32) / 255.0,
            label,
        ),
        num_parallel_calls=1,
    )

    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)

    return dataset.prefetch(1)


def build_model():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1), name="image"),
        tf.keras.layers.Conv2D(
            8, 3, padding="valid", activation="relu", name="conv1"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=2, name="pool1"),
        tf.keras.layers.Conv2D(
            16, 3, padding="valid", activation="relu", name="conv2"
        ),
        tf.keras.layers.MaxPooling2D(pool_size=2, name="pool2"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(10, name="scores"),
    ], name="edgeinfer_cnn")


def main():
    (images, labels), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Deterministic stratified split: reserve 10% of each digit
    # for validation.
    rng = np.random.default_rng(SEED)
    training_parts = []
    validation_parts = []

    for digit in range(10):
        indices = np.flatnonzero(labels == digit)
        rng.shuffle(indices)
        validation_count = int(round(0.1 * len(indices)))

        validation_parts.append(indices[:validation_count])
        training_parts.append(indices[validation_count:])

    training_indices = np.concatenate(training_parts)
    validation_indices = np.concatenate(validation_parts)
    rng.shuffle(training_indices)
    rng.shuffle(validation_indices)

    # Every original image must appear exactly once across the two splits.
    all_indices = np.concatenate([
        training_indices,
        validation_indices,
    ])

    assert np.array_equal(
        np.sort(all_indices),
        np.arange(len(images)),
    ), "Training and validation must partition the original dataset"

    # Save the split for later quantization calibration.
    # Calibration must use training images, not test images.
    np.savez(
        OUTPUT / "split_indices.npz",
        training=training_indices,
        validation=validation_indices,
    )

    training_data = make_dataset(
        images[training_indices],
        labels[training_indices],
        training=True,
    )
    validation_data = make_dataset(
        images[validation_indices],
        labels[validation_indices],
    )
    test_data = make_dataset(test_images, test_labels)

    print("Training images:", len(training_indices))
    print("Validation images:", len(validation_indices))
    print("Test images:", len(test_images))

    model = build_model()
    assert model.count_params() == 5258

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        ],
        jit_compile=False,
    )

    model.summary()

    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=MAX_EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(OUTPUT / "best_model.keras"),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=4,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(
                str(OUTPUT / "training_history.csv")
            ),
        ],
        verbose=2,
    )

    # Explicitly reload the saved best checkpoint.
    best_model = tf.keras.models.load_model(
        OUTPUT / "best_model.keras",
        compile=False,
    )

    # One test-set pass, after selecting the model by validation.
    logits = np.concatenate([
        best_model(batch_images, training=False).numpy()
        for batch_images, _ in test_data
    ])

    assert logits.shape == (10000, 10)
    assert np.isfinite(logits).all()

    predictions = np.argmax(logits, axis=1)
    correct = int(np.count_nonzero(predictions == test_labels))

    np.savez_compressed(
        OUTPUT / "test_reference.npz",
        logits=logits,
        predictions=predictions.astype(np.uint8),
        labels=test_labels,
    )

    report = {
        "seed": SEED,
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "numpy": np.__version__,
        "training_images": len(training_indices),
        "validation_images": len(validation_indices),
        "test_images": len(test_images),
        "parameters": best_model.count_params(),
        "epochs_run": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "test_correct": correct,
        "test_accuracy": correct / len(test_images),
    }

    (OUTPUT / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n--- Final FP32 test results ---")
    print(f"Correct: {correct}/{len(test_images)}")
    print(f"Accuracy: {100.0 * correct / len(test_images):.2f}%")
    print(f"Best epoch: {report['best_epoch']}")
    print(f"Parameters: {best_model.count_params()}")
    print(f"Saved artifacts: {OUTPUT}")


if __name__ == "__main__":
    main()