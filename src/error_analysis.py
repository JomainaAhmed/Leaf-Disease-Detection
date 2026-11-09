"""
Error Analysis Script for Plant Leaf Disease Classification.

Identifies:
- Top 10 most frequently confused class pairs (True Class vs Predicted Class).
- Underlying error patterns (e.g. early blight vs late blight, bacterial spot across Solanaceae).
- Saves sample misclassified leaf images into assets/error_analysis/ with annotations.
"""

import os
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

from src.utils import CLASS_NAMES, format_class_name, DEFAULT_IMG_SIZE


def run_error_analysis(
    model_path: str,
    test_csv: str = "data/test_split.csv",
    output_dir: str = "assets/error_analysis",
    max_images_per_pair: int = 2
):
    """
    Run diagnostic error analysis on test split.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test split manifest not found: {test_csv}")

    os.makedirs(output_dir, exist_ok=True)
    model = tf.keras.models.load_model(model_path)
    df_test = pd.read_csv(test_csv)

    print("=" * 65)
    print("RUNNING DIAGNOSTIC ERROR ANALYSIS")
    print("=" * 65)

    errors = []
    # Batch predict
    batch_size = 32
    for i in range(0, len(df_test), batch_size):
        batch_slice = df_test.iloc[i:i + batch_size]
        batch_imgs = []
        for p in batch_slice["full_path"].values:
            img = tf.keras.utils.load_img(p, target_size=DEFAULT_IMG_SIZE)
            arr = tf.keras.utils.img_to_array(img)
            batch_imgs.append(arr)
        preds = model.predict(np.array(batch_imgs), verbose=0)
        pred_ids = np.argmax(preds, axis=1)
        confidences = np.max(preds, axis=1)

        for row, pred_id, conf in zip(batch_slice.itertuples(), pred_ids, confidences):
            if row.class_id != pred_id:
                errors.append({
                    "full_path": row.full_path,
                    "filename": row.filename,
                    "true_class_id": row.class_id,
                    "true_class_name": row.class_name,
                    "pred_class_id": pred_id,
                    "pred_class_name": CLASS_NAMES[pred_id],
                    "confidence": float(conf)
                })

    df_errors = pd.DataFrame(errors)
    total_test = len(df_test)
    total_errors = len(df_errors)
    error_rate = (total_errors / total_test) * 100 if total_test > 0 else 0

    print(f"Total Test Samples : {total_test}")
    print(f"Total Errors       : {total_errors} ({error_rate:.2f}%)")
    print("-" * 65)

    if total_errors == 0:
        print("Zero misclassifications detected on test set!")
        return

    # Count pairs
    pair_counts = Counter()
    for e in errors:
        pair_counts[(e["true_class_name"], e["pred_class_name"])] += 1

    top_pairs = pair_counts.most_common(10)

    print("TOP 10 MOST-CONFUSED CLASS PAIRS:")
    report_lines = [
        "# Error Analysis Report\n",
        f"- **Total Test Samples**: {total_test}\n",
        f"- **Total Misclassifications**: {total_errors} ({error_rate:.2f}%)\n",
        "## Top 10 Most-Confused Class Pairs\n",
        "| Rank | True Class | Predicted Class | Error Count | Biological / Visual Root Cause |\n",
        "|------|------------|-----------------|-------------|-------------------------------|\n"
    ]

    for rank, ((true_cls, pred_cls), cnt) in enumerate(top_pairs, 1):
        true_fmt = format_class_name(true_cls)
        pred_fmt = format_class_name(pred_cls)
        # Determine likely confusion cause
        note = "Similar foliar necrotic lesion morphology"
        if "blight" in true_cls.lower() and "blight" in pred_cls.lower():
            note = "Early vs late necrotic symptom stages"
        elif "spot" in true_cls.lower() and "spot" in pred_cls.lower():
            note = "Overlapping circular spot patterns"
        elif "healthy" in pred_cls.lower():
            note = "Faint early-stage symptoms overlooked"

        print(f"{rank:2d}. True: {true_fmt} -> Pred: {pred_fmt} | Count: {cnt}")
        report_lines.append(f"| {rank} | {true_fmt} | {pred_fmt} | {cnt} | {note} |\n")

    # Save sample images for top confused pairs
    for (true_cls, pred_cls), _ in top_pairs:
        pair_matches = [e for e in errors if e["true_class_name"] == true_cls and e["pred_class_name"] == pred_cls]
        for idx, sample in enumerate(pair_matches[:max_images_per_pair]):
            try:
                img = Image.open(sample["full_path"]).convert("RGB")
                draw = ImageDraw.Draw(img)
                # Label on image
                text = f"True: {sample['true_class_name'][:18]}\nPred: {sample['pred_class_name'][:18]} ({sample['confidence']:.2f})"
                draw.rectangle([(0, 0), (img.width, 35)], fill=(0, 0, 0, 180))
                draw.text((4, 2), text, fill=(255, 255, 255))

                save_name = f"err_{sample['true_class_name'][:15]}_vs_{sample['pred_class_name'][:15]}_{idx}.jpg"
                img.save(os.path.join(output_dir, save_name))
            except Exception as ex:
                print(f"Could not save sample error image: {ex}")

    report_path = os.path.join(output_dir, "error_analysis_report.md")
    with open(report_path, "w") as f:
        f.writelines(report_lines)

    print("-" * 65)
    print(f"Error report saved to: {report_path}")
    print(f"Sample misclassified images saved to: {output_dir}/")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run error analysis on test set.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test-manifest", type=str, default="data/test_split.csv", help="Test split CSV")
    parser.add_argument("--output-dir", type=str, default="assets/error_analysis", help="Output directory")
    args = parser.parse_args()

    run_error_analysis(args.model_path, args.test_manifest, args.output_dir)
