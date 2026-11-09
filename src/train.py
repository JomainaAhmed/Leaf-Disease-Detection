"""
Training Pipeline for Plant Leaf Disease Detection Models.

Features:
- Baseline CNN, MobileNetV2, and EfficientNetB0 architectures.
- 2-phase transfer learning (frozen-base warmup followed by fine-tuning).
- Callbacks: EarlyStopping, ModelCheckpoint (best val_accuracy), ReduceLROnPlateau.
- Fixed random seeds for full reproducibility.
- CLI arguments for hyperparameters and paths.
"""

import os
import json
import argparse
import tensorflow as tf
import pandas as pd

from src.utils import set_random_seeds, NUM_CLASSES, DEFAULT_IMG_SIZE
from src.models import build_baseline_cnn, build_mobilenetv2, build_efficientnetb0


def load_dataset_from_manifest(csv_path: str, img_size=DEFAULT_IMG_SIZE, batch_size=32, is_training=True):
    """
    Build a tf.data.Dataset pipeline from a split manifest CSV file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Manifest CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    file_paths = df["full_path"].values
    labels = df["class_id"].values

    def parse_image(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32)
        label = tf.one_hot(label, NUM_CLASSES)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(df), 2000), seed=42)
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(df)


def load_dataset_from_dir(dir_path: str, img_size=DEFAULT_IMG_SIZE, batch_size=32, is_training=True):
    """
    Load dataset directly from directory using image_dataset_from_directory.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    return tf.keras.utils.image_dataset_from_directory(
        dir_path,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=is_training,
        seed=42 if is_training else None
    )


def train_model(
    model_name: str = "baseline",
    epochs: int = 25,
    fine_tune_epochs: int = 15,
    batch_size: int = 32,
    img_size: int = 128,
    data_dir: str = "data",
    manifest_mode: bool = True,
    learning_rate: float = 1e-4,
    models_dir: str = "models",
    seed: int = 42
):
    """Run end-to-end model training."""
    set_random_seeds(seed)
    os.makedirs(models_dir, exist_ok=True)
    target_size = (img_size, img_size)
    input_shape = (img_size, img_size, 3)

    # 1. Prepare Datasets
    print("=" * 65)
    print(f"Initializing training for model: {model_name.upper()}")
    print("=" * 65)

    if manifest_mode and os.path.exists(os.path.join(data_dir, "train_split.csv")):
        train_csv = os.path.join(data_dir, "train_split.csv")
        val_csv = os.path.join(data_dir, "val_split.csv")
        print(f"Loading datasets via manifests: {train_csv}, {val_csv}")
        train_ds, n_train = load_dataset_from_manifest(train_csv, target_size, batch_size, is_training=True)
        val_ds, n_val = load_dataset_from_manifest(val_csv, target_size, batch_size, is_training=False)
    else:
        train_path = os.path.join(data_dir, "train")
        val_path = os.path.join(data_dir, "valid")
        print(f"Loading datasets from directories: {train_path}, {val_path}")
        train_ds = load_dataset_from_dir(train_path, target_size, batch_size, is_training=True)
        val_ds = load_dataset_from_dir(val_path, target_size, batch_size, is_training=False)

    # 2. Build Model
    if model_name == "baseline":
        model = build_baseline_cnn(input_shape=input_shape, num_classes=NUM_CLASSES)
    elif model_name == "mobilenetv2":
        model = build_mobilenetv2(input_shape=input_shape, num_classes=NUM_CLASSES, fine_tune_at=None)
    elif model_name == "efficientnetb0":
        model = build_efficientnetb0(input_shape=input_shape, num_classes=NUM_CLASSES, fine_tune_at=None)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.summary()

    # 3. Callbacks
    checkpoint_path = os.path.join(models_dir, f"{model_name}_best.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 4. Phase 1 Training (or Full Training for Baseline)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\n--- Phase 1: Training {model_name} for up to {epochs} epochs ---")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    full_history = history1.history

    # 5. Phase 2 Fine-Tuning for Transfer Learning Models
    if model_name in ["mobilenetv2", "efficientnetb0"] and fine_tune_epochs > 0:
        print(f"\n--- Phase 2: Unfreezing base layers for fine-tuning ---")
        base = getattr(model, "base_model", None)
        if base is not None:
            base.trainable = True
            # Freeze bottom 70% of layers, fine-tune the top 30%
            fine_tune_at = int(len(base.layers) * 0.70)
            for layer in base.layers[:fine_tune_at]:
                layer.trainable = False

            fine_tune_lr = learning_rate * 0.1
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            history2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs + fine_tune_epochs,
                initial_epoch=len(history1.epoch),
                callbacks=callbacks
            )

            for k, v in history2.history.items():
                if k in full_history:
                    full_history[k].extend(v)
                else:
                    full_history[k] = v

    # 6. Save Artifacts
    history_path = os.path.join(models_dir, f"{model_name}_history.json")
    # Convert numpy values to serializable Python floats
    serializable_history = {
        k: [float(val) for val in v] for k, v in full_history.items()
    }
    with open(history_path, "w") as f:
        json.dump(serializable_history, f, indent=2)

    final_model_path = os.path.join(models_dir, f"{model_name}_final.keras")
    model.save(final_model_path)
    print(f"\nModel saved to {final_model_path} and best weights to {checkpoint_path}")
    print(f"Training history saved to {history_path}")
    return model, full_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train plant leaf disease classification models.")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "mobilenetv2", "efficientnetb0"])
    parser.add_argument("--epochs", type=int, default=25, help="Number of primary training epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=10, help="Number of fine-tuning epochs for transfer learning")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--img-size", type=int, default=128, help="Image width and height")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data-path", type=str, default="data", help="Directory with CSV manifests or train/val folders")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        data_dir=args.data_path,
        learning_rate=args.lr,
        models_dir=args.models_dir,
        seed=args.seed
    )
