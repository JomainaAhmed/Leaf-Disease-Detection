"""
Model Architectures for Plant Leaf Disease Detection.

Includes:
1. Improved Custom CNN Baseline (with BatchNormalization and Dropout)
2. MobileNetV2 Transfer Learning with fine-tuning support
3. EfficientNetB0 Transfer Learning with fine-tuning support
4. Keras on-the-fly data augmentation pipeline
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_augmentation_layer() -> tf.keras.Sequential:
    """Build an on-the-fly data augmentation Sequential pipeline."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", name="aug_flip"),
        layers.RandomRotation(0.15, fill_mode="reflect", name="aug_rotation"),
        layers.RandomZoom(0.15, fill_mode="reflect", name="aug_zoom"),
        layers.RandomContrast(0.15, name="aug_contrast")
    ], name="data_augmentation")


def build_baseline_cnn(
    input_shape: tuple = (128, 128, 3),
    num_classes: int = 38,
    use_augmentation: bool = True
) -> tf.keras.Model:
    """
    Improved baseline CNN based on the original 5-block convolutional architecture,
    enhanced with BatchNormalization and calibrated Dropout to stabilize training
    and mitigate overfitting.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = inputs

    if use_augmentation:
        x = build_augmentation_layer()(x)

    # Rescale [0, 255] -> [0, 1] if not already normalized
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(x)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(x)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.MaxPooling2D((2, 2), strides=2, name="pool1")(x)
    x = layers.Dropout(0.2, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.MaxPooling2D((2, 2), strides=2, name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.MaxPooling2D((2, 2), strides=2, name="pool3")(x)
    x = layers.Dropout(0.25, name="drop3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", name="conv4_2")(x)
    x = layers.BatchNormalization(name="bn4_2")(x)
    x = layers.MaxPooling2D((2, 2), strides=2, name="pool4")(x)
    x = layers.Dropout(0.3, name="drop4")(x)

    # Block 5 (Final conv block - target for Grad-CAM)
    x = layers.Conv2D(512, (3, 3), padding="same", activation="relu", name="conv5_1")(x)
    x = layers.BatchNormalization(name="bn5_1")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", name="target_conv_layer")(x)
    x = layers.BatchNormalization(name="bn5_2")(x)
    x = layers.MaxPooling2D((2, 2), strides=2, name="pool5")(x)
    x = layers.Dropout(0.35, name="drop5")(x)

    # Dense head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(512, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.4, name="drop_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="improved_baseline_cnn")


def build_mobilenetv2(
    input_shape: tuple = (128, 128, 3),
    num_classes: int = 38,
    use_augmentation: bool = True,
    fine_tune_at: int = None
) -> tf.keras.Model:
    """
    MobileNetV2 transfer learning model with optional fine-tuning.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = inputs

    if use_augmentation:
        x = build_augmentation_layer()(x)

    # MobileNetV2 expects [-1, 1] input range
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    if fine_tune_at is None:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_top")(x)
    x = layers.Dense(256, activation="relu", name="dense_top")(x)
    x = layers.Dropout(0.3, name="drop_top")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_leaf")
    model.base_model = base_model
    return model


def build_efficientnetb0(
    input_shape: tuple = (128, 128, 3),
    num_classes: int = 38,
    use_augmentation: bool = True,
    fine_tune_at: int = None
) -> tf.keras.Model:
    """
    EfficientNetB0 transfer learning model with compound scaling.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = inputs

    if use_augmentation:
        x = build_augmentation_layer()(x)

    x = tf.keras.applications.efficientnet.preprocess_input(x)

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    if fine_tune_at is None:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_top")(x)
    x = layers.Dropout(0.3, name="drop_top")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="efficientnetb0_leaf")
    model.base_model = base_model
    return model
