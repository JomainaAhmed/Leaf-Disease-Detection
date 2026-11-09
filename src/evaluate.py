"""
Comprehensive Model Evaluation on Truly Held-out Test Set.

Computes:
- Overall Accuracy
- Macro and Weighted Precision, Recall, and F1-score
- Detailed per-class classification report
- High-resolution normalized Confusion Matrix saved to assets/
- Updates or generates results.md comparison table
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import tensorflow as tf

from src.utils import CLASS_NAMES, format_class_name, DEFAULT_IMG_SIZE


def evaluate_test_set(
    model_path: str,
    test_csv: str = "data/test_split.csv",
    output_dir: str = "assets",
    results_path: str = "results.md",
    batch_size: int = 32
):
    """
    Run evaluation on the held-out test manifest and generate reports + plots.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test split CSV not found: {test_csv}")

    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    print("=" * 65)
    print(f"Evaluating Model: {model_name} on Held-Out Test Set")
    print(f"Manifest: {test_csv}")
    print("=" * 65)

    model = tf.keras.models.load_model(model_path)
    df_test = pd.read_csv(test_csv)
    total_samples = len(df_test)

    y_true = df_test["class_id"].values
    y_pred = []
    y_scores = []

    # Process in batches
    for i in range(0, total_samples, batch_size):
        batch_paths = df_test["full_path"].iloc[i:i + batch_size].values
        batch_imgs = []
        for p in batch_paths:
            img = tf.keras.utils.load_img(p, target_size=DEFAULT_IMG_SIZE)
            arr = tf.keras.utils.img_to_array(img)
            batch_imgs.append(arr)
        batch_arr = np.array(batch_imgs)
        preds = model.predict(batch_arr, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_scores.extend(np.max(preds, axis=1))

    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    print("\n--- HELD-OUT TEST PERFORMANCE ---")
    print(f"Test Accuracy        : {acc * 100:.2f}%")
    print(f"Macro Precision      : {prec_macro:.4f}")
    print(f"Macro Recall         : {rec_macro:.4f}")
    print(f"Macro F1-Score       : {f1_macro:.4f}")
    print(f"Weighted F1-Score    : {f1_weighted:.4f}")
    print("-" * 65)

    # Classification report
    readable_labels = [format_class_name(c) for c in CLASS_NAMES]
    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=readable_labels,
        digits=4,
        zero_division=0
    )

    report_txt_path = os.path.join(output_dir, f"classification_report_{model_name}.txt")
    with open(report_txt_path, "w") as f:
        f.write(clf_report)
    print(f"Classification report saved to {report_txt_path}")

    # Normalized Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(24, 20))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=readable_labels,
        yticklabels=readable_labels
    )
    plt.title(f"Normalized Confusion Matrix - {model_name} (Test Accuracy: {acc*100:.2f}%)", fontsize=16, pad=15)
    plt.xlabel("Predicted Class", fontsize=14, labelpad=10)
    plt.ylabel("True Class", fontsize=14, labelpad=10)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_path}")

    metrics = {
        "model": model_name,
        "test_samples": total_samples,
        "accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted)
    }

    metrics_json = os.path.join(output_dir, f"metrics_{model_name}.json")
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate plant disease model on held-out test split.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model (.keras)")
    parser.add_argument("--test-manifest", type=str, default="data/test_split.csv", help="CSV manifest of test set")
    parser.add_argument("--output-dir", type=str, default="assets", help="Directory for plots and reports")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()

    evaluate_test_set(
        model_path=args.model_path,
        test_csv=args.test_manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
