"""
TensorFlow Lite Model Export and Benchmarking.

Converts a trained Keras model into:
1. Standard Float32 TFLite model (.tflite)
2. Post-training Dynamic Range Quantized TFLite model (.tflite)
Reports file sizes and validates inference output shape (1, 38).
"""

import os
import argparse
import numpy as np
import tensorflow as tf

from src.utils import NUM_CLASSES, DEFAULT_IMG_SIZE


def export_and_benchmark(
    keras_model_path: str,
    output_dir: str = "models",
    quantize: bool = True
):
    """
    Convert model to TFLite, report size reduction, and test inference.
    """
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Model not found: {keras_model_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(keras_model_path))[0]
    keras_size_mb = os.path.getsize(keras_model_path) / (1024 * 1024)

    print("=" * 65)
    print("TENSORFLOW LITE EXPORT & BENCHMARK")
    print("=" * 65)
    print(f"Source Keras Model: {keras_model_path} ({keras_size_mb:.2f} MB)")

    model = tf.keras.models.load_model(keras_model_path)

    # 1. Standard Float32 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model_float = converter.convert()

    float_path = os.path.join(output_dir, f"{base_name}_fp32.tflite")
    with open(float_path, "wb") as f:
        f.write(tflite_model_float)
    float_size_mb = os.path.getsize(float_path) / (1024 * 1024)
    print(f"FP32 TFLite Model : {float_path} ({float_size_mb:.2f} MB)")

    # 2. Dynamic Range Quantization
    quant_size_mb = None
    if quantize:
        converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model_quant = converter_quant.convert()

        quant_path = os.path.join(output_dir, f"{base_name}_dynamic_quant.tflite")
        with open(quant_path, "wb") as f:
            f.write(tflite_model_quant)
        quant_size_mb = os.path.getsize(quant_path) / (1024 * 1024)
        compression = (1.0 - quant_size_mb / keras_size_mb) * 100
        print(f"Quantized TFLite  : {quant_path} ({quant_size_mb:.2f} MB, {compression:.1f}% reduction)")

    # 3. Test TFLite Interpreter
    test_path = quant_path if quantize else float_path
    interpreter = tf.lite.Interpreter(model_path=test_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy_input = np.random.rand(1, DEFAULT_IMG_SIZE[0], DEFAULT_IMG_SIZE[1], 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print(f"Inference Test: Input shape {dummy_input.shape} -> Output shape {output_data.shape}")
    assert output_data.shape == (1, NUM_CLASSES), f"Expected (1, {NUM_CLASSES}), got {output_data.shape}"
    print("Verification PASSED: TFLite model output matches 38 disease classes.")
    print("=" * 65)

    return {
        "keras_size_mb": keras_size_mb,
        "fp32_size_mb": float_size_mb,
        "quant_size_mb": quant_size_mb
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Keras model to TensorFlow Lite.")
    parser.add_argument("--model-path", type=str, default="models/baseline_best.keras", help="Path to input .keras model")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    args = parser.parse_args()

    if os.path.exists(args.model_path):
        export_and_benchmark(args.model_path, args.output_dir)
    else:
        print(f"Model '{args.model_path}' not found. Run training first.")
