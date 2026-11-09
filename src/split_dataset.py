"""
Leakage-Safe Dataset Splitting Module.

Groups images by source leaf (to prevent offline-augmented variants of the same leaf
from leaking across splits) and performs a stratified GroupKFold / GroupSplit
into Train (70%), Validation (15%), and Test (15%) with a fixed random seed.
Saves reproducible manifest CSV files.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

try:
    from src.utils import CLASS_NAMES, set_random_seeds
    from src.leakage_check import extract_base_image_id
except ImportError:
    from utils import CLASS_NAMES, set_random_seeds
    from leakage_check import extract_base_image_id


def build_manifest(raw_dataset_dir: str) -> pd.DataFrame:
    """
    Scan the dataset directory and extract metadata:
    filepath, filename, class_name, class_id, and group_id (source leaf id).
    """
    records = []
    class_to_id = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}

    # Directory may contain direct class folders or subfolders
    for root, dirs, files in os.walk(raw_dataset_dir):
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            continue

        dir_name = os.path.basename(root)
        if dir_name not in class_to_id:
            # Check if parent directory or folder matches known classes
            for c in CLASS_NAMES:
                if c.lower() in dir_name.lower():
                    dir_name = c
                    break

        if dir_name in class_to_id:
            cls_name = dir_name
            cls_id = class_to_id[cls_name]
            for fname in image_files:
                fpath = os.path.join(root, fname)
                group_id = f"{cls_name}___{extract_base_image_id(fname)}"
                records.append({
                    "filepath": os.path.relpath(fpath, raw_dataset_dir).replace("\\", "/"),
                    "full_path": os.path.abspath(fpath),
                    "filename": fname,
                    "class_name": cls_name,
                    "class_id": cls_id,
                    "group_id": group_id
                })

    return pd.DataFrame(records)


def create_leakage_safe_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> tuple:
    """
    Perform a leakage-safe split by splitting unique source leaf groups (group_id),
    preserving class proportions.
    """
    set_random_seeds(seed)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # Aggregate by group_id
    group_df = df.groupby("group_id").agg({
        "class_name": "first",
        "class_id": "first",
        "filepath": "count"
    }).rename(columns={"filepath": "image_count"}).reset_index()

    train_groups = []
    val_groups = []
    test_groups = []

    # Stratify by class
    for cls_name, cls_group_df in group_df.groupby("class_name"):
        groups = cls_group_df["group_id"].values
        np.random.seed(seed)
        np.random.shuffle(groups)

        n_total = len(groups)
        n_train = int(np.floor(n_total * train_ratio))
        n_val = int(np.floor(n_total * val_ratio))

        g_train = groups[:n_train]
        g_val = groups[n_train:n_train + n_val]
        g_test = groups[n_train + n_val:]

        train_groups.extend(g_train)
        val_groups.extend(g_val)
        test_groups.extend(g_test)

    train_set = set(train_groups)
    val_set = set(val_groups)
    test_set = set(test_groups)

    # Verify no group overlaps
    assert len(train_set.intersection(val_set)) == 0, "Leakage detected: train & val overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Leakage detected: train & test overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Leakage detected: val & test overlap!"

    train_df = df[df["group_id"].isin(train_set)].copy().reset_index(drop=True)
    val_df = df[df["group_id"].isin(val_set)].copy().reset_index(drop=True)
    test_df = df[df["group_id"].isin(test_set)].copy().reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = "data"):
    """Save split manifests to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_split.csv")
    val_path = os.path.join(output_dir, "val_split.csv")
    test_path = os.path.join(output_dir, "test_split.csv")
    full_path = os.path.join(output_dir, "full_manifest.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df.to_csv(full_path, index=False)

    print("=" * 65)
    print("           LEAKAGE-SAFE SPLIT GENERATION SUMMARY")
    print("=" * 65)
    print(f"Train set: {len(train_df):6d} images ({len(train_df['group_id'].unique())} unique leaf groups)")
    print(f"Val set  : {len(val_df):6d} images ({len(val_df['group_id'].unique())} unique leaf groups)")
    print(f"Test set : {len(test_df):6d} images ({len(test_df['group_id'].unique())} unique leaf groups)")
    print(f"Total    : {len(full_df):6d} images ({len(full_df['group_id'].unique())} unique leaf groups)")
    print("-" * 65)
    print(f"Manifests saved to: {output_dir}/")
    print(f" - {train_path}")
    print(f" - {val_path}")
    print(f" - {test_path}")
    print(f" - {full_path}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create leakage-safe group split for plant leaf dataset.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw or combined image directory")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for CSV manifests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if os.path.exists(args.data_dir):
        df_all = build_manifest(args.data_dir)
        if len(df_all) > 0:
            tr, va, te = create_leakage_safe_split(df_all, seed=args.seed)
            save_splits(tr, va, te, args.output_dir)
        else:
            print(f"No matching leaf images found in {args.data_dir}.")
    else:
        print(f"Data directory '{args.data_dir}' not found.")
        print("Provide --data-dir <path> after dataset is downloaded or extracted.")
