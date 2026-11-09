"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module.

Provides visual explanations for deep convolutional networks predicting
plant diseases, highlighting the specific leaf regions influencing the diagnosis.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

import matplotlib.cm as cm
from src.utils import CLASS_NAMES, format_class_name, load_image_for_inference


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Search recursively or sequentially for the last convolutional layer name in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # If it's a nested Functional/Sequential model (e.g., MobileNet base)
        if hasattr(layer, "layers"):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer.name
    # Fallback to known common names
    for name in ["target_conv_layer", "conv5_2", "Conv_1", "top_conv"]:
        try:
            model.get_layer(name)
            return name
        except (ValueError, AttributeError):
            pass
    raise ValueError("Could not automatically locate a Conv2D layer in the model.")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = None,
    pred_index: int = None
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a given input image batch and model.

    Args:
        img_array: Batch of shape (1, H, W, 3)
        model: Trained Keras model
        last_conv_layer_name: Name of the last convolutional layer. If None, auto-detected.
        pred_index: Target class index. If None, highest confidence class is used.

    Returns:
        heatmap: 2D numpy array normalized to [0, 1]
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Handle models with nested base models (e.g., MobileNetV2 / EfficientNet)
    target_layer = None
    sub_model = None

    try:
        target_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        # Search nested base models
        for layer in model.layers:
            if hasattr(layer, "layers"):
                try:
                    target_layer = layer.get_layer(last_conv_layer_name)
                    sub_model = layer
                    break
                except ValueError:
                    continue

    if sub_model is not None:
        # Build multi-output for nested architecture
        grad_model = tf.keras.models.Model(
            inputs=sub_model.input,
            outputs=[sub_model.get_layer(last_conv_layer_name).output, sub_model.output]
        )

        with tf.GradientTape() as tape:
            # We assume model preprocesses before sub_model
            # For simplicity, pass through model's earlier layers or base
            conv_outputs, base_output = grad_model(img_array)
            # Pass base_output through remainder of top layers
            x = base_output
            for layer in model.layers[model.layers.index(sub_model) + 1:]:
                x = layer(x)
            preds = x
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
    else:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

    # Compute guided pooling weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU to keep only positive contributions
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(
    original_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet"
) -> Image.Image:
    """
    Superimpose Grad-CAM heatmap over the original PIL image.
    """
    img = np.array(original_img)
    h, w = img.shape[:2]

    # Resize heatmap to match image dimensions
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_resized) / 255.0

    # Apply colormap
    color_mapper = cm.get_cmap(colormap)
    colored_heatmap = color_mapper(heatmap_resized)[:, :, :3]  # drop alpha
    colored_heatmap = np.uint8(255 * colored_heatmap)

    # Blend
    superimposed = np.uint8(img * (1.0 - alpha) + colored_heatmap * alpha)
    return Image.fromarray(superimposed)


def generate_gradcam_samples(
    model_path: str,
    image_paths: list,
    output_dir: str = "assets/gradcam"
):
    """Generate and save Grad-CAM visual explanations for a list of sample images."""
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(model_path):
        print(f"Model path '{model_path}' not found. Skipping Grad-CAM generation.")
        return

    model = tf.keras.models.load_model(model_path)
    last_conv_name = find_last_conv_layer(model)
    print(f"Loaded model from {model_path}. Target conv layer: {last_conv_name}")

    for idx, path in enumerate(image_paths):
        if not os.path.exists(path):
            continue
        arr, pil_img = load_image_for_inference(path)
        preds = model.predict(arr, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        conf = float(preds[top_idx])

        heatmap = make_gradcam_heatmap(arr, model, last_conv_name, pred_index=top_idx)
        cam_img = overlay_gradcam(pil_img, heatmap)

        base_name = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(output_dir, f"gradcam_{base_name}_{CLASS_NAMES[top_idx]}_{conf:.2f}.jpg")
        cam_img.save(save_path)
        print(f"Saved Grad-CAM sample: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for leaf disease diagnosis.")
    parser.add_argument("--model-path", type=str, default="models/baseline_best.keras", help="Path to saved Keras model")
    parser.add_argument("--image", type=str, required=False, help="Path to leaf image")
    parser.add_argument("--output-dir", type=str, default="assets/gradcam", help="Output directory")
    args = parser.parse_args()

    if args.image and os.path.exists(args.image):
        generate_gradcam_samples(args.model_path, [args.image], args.output_dir)
    else:
        print("Provide --image <path> to generate a Grad-CAM overlay.")
