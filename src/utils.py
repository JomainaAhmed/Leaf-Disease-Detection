"""
Utility functions, constants, and helpers for Plant Leaf Disease Detection.
"""

import os
import json
import random
import numpy as np
from PIL import Image

CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

NUM_CLASSES = len(CLASS_NAMES)
DEFAULT_IMG_SIZE = (128, 128)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility across python, numpy, and tensorflow."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def format_class_name(name: str) -> str:
    """Convert raw dataset class name (e.g. Corn_(maize)___Common_rust_) to a clean human-readable string."""
    parts = name.split("___")
    crop = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else ""
    if disease.lower() == "healthy":
        return f"{crop} (Healthy)"
    return f"{crop} - {disease}"


def load_image_for_inference(image_input, target_size=DEFAULT_IMG_SIZE):
    """
    Load and preprocess an image (from file path, file-like object, or PIL Image)
    Returns:
        np.ndarray with shape (1, height, width, 3), dtype float32, normalized to [0, 1] if not already.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        # File-like object (e.g., Streamlit UploadedFile)
        img = Image.open(image_input).convert("RGB")

    img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.asarray(img_resized, dtype=np.float32)
    # Ensure shape is (1, H, W, 3)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    return arr, img_resized


def export_tflite_model(model, output_path: str, quantize_dynamic: bool = False) -> float:
    """
    Convert a trained Keras model to TensorFlow Lite (.tflite) format.
    Returns:
        file size in Megabytes (MB).
    """
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize_dynamic:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    return size_mb
