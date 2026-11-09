"""
Data Leakage Check & Grouping Module.

Detects whether augmented copies or perceptual duplicates of the same source
leaf image exist across 'train' and 'valid' splits in the offline-augmented dataset.
"""

import os
import re
import argparse
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from PIL import Image


def compute_dhash(image_path: str, hash_size: int = 8) -> int:
    """
    Compute difference hash (dhash) for an image to detect visually identical
    or slightly augmented variants across dataset directories.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
            diff = []
            for row in range(hash_size):
                row_start = row * (hash_size + 1)
                for col in range(hash_size):
                    diff.append(pixels[row_start + col] > pixels[row_start + col + 1])
            decimal_val = 0
            for idx, bit in enumerate(diff):
                if bit:
                    decimal_val |= 1 << idx
            return decimal_val
    except Exception:
        return 0


def extract_base_image_id(filename: str) -> str:
    """
    Extract source image identifier from filename if filenames contain augmentation suffixes
    (e.g., 'image (1)_flip.JPG' -> 'image (1)', '0a1b2c_rot.jpg' -> '0a1b2c').
    """
    base = os.path.splitext(filename)[0]
    # Remove common augmentation markers
    cleaned = re.sub(r'(_zoom|_flip|_rot\d+|_aug\d*|_flip\w*|_brightness|copy|\(\d+\))', '', base, flags=re.IGNORECASE)
    # Remove trailing dashes or underscores
    cleaned = cleaned.rstrip('_ -')
    return cleaned if len(cleaned) > 2 else base


def scan_split_for_hashes(split_dir: str, max_samples_per_class: int = None) -> Dict[str, Dict[str, List[str]]]:
    """
    Scan a directory (e.g., train/ or valid/) and compute hashes grouped by class.
    Returns:
        dict: { class_name: { hash: [filepaths] } }
    """
    results = defaultdict(lambda: defaultdict(list))
    if not os.path.exists(split_dir):
        return results

    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    for cls in classes:
        cls_path = os.path.join(split_dir, cls)
        filenames = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if max_samples_per_class:
            filenames = filenames[:max_samples_per_class]

        for fname in filenames:
            fpath = os.path.join(cls_path, fname)
            h = compute_dhash(fpath)
            results[cls][h].append(fpath)
    return results


def check_leakage(train_dir: str, valid_dir: str, max_samples_per_class: int = None) -> Dict:
    """
    Check data leakage between train_dir and valid_dir using perceptual hash overlap.
    """
    print(f"Scanning train directory: {train_dir} ...")
    train_hashes = scan_split_for_hashes(train_dir, max_samples_per_class)
    print(f"Scanning valid directory: {valid_dir} ...")
    valid_hashes = scan_split_for_hashes(valid_dir, max_samples_per_class)

    leakage_by_class = {}
    total_leaked_valid_images = 0
    total_valid_images = 0

    all_classes = set(train_hashes.keys()).union(set(valid_hashes.keys()))

    for cls in sorted(all_classes):
        train_cls_hashes = train_hashes.get(cls, {})
        valid_cls_hashes = valid_hashes.get(cls, {})

        cls_valid_total = sum(len(fps) for fps in valid_cls_hashes.values())
        total_valid_images += cls_valid_total

        leaked_in_class = 0
        sample_pairs = []

        for h, v_fps in valid_cls_hashes.items():
            if h in train_cls_hashes:
                leaked_in_class += len(v_fps)
                if len(sample_pairs) < 3:
                    sample_pairs.append({
                        "valid_sample": os.path.basename(v_fps[0]),
                        "train_match": os.path.basename(train_cls_hashes[h][0])
                    })

        total_leaked_valid_images += leaked_in_class
        if leaked_in_class > 0:
            leakage_by_class[cls] = {
                "leaked_images_in_valid": leaked_in_class,
                "total_valid_images": cls_valid_total,
                "leakage_percentage": (leaked_in_class / cls_valid_total * 100) if cls_valid_total > 0 else 0,
                "samples": sample_pairs
            }

    overall_leakage_pct = (total_leaked_valid_images / total_valid_images * 100) if total_valid_images > 0 else 0.0

    report = {
        "train_dir": train_dir,
        "valid_dir": valid_dir,
        "total_classes_checked": len(all_classes),
        "total_valid_images_checked": total_valid_images,
        "total_leaked_valid_images": total_leaked_valid_images,
        "overall_leakage_percentage": overall_leakage_pct,
        "leakage_by_class": leakage_by_class
    }
    return report


def print_leakage_report(report: Dict) -> None:
    """Format and display leakage findings."""
    print("=" * 70)
    print("              DATA LEAKAGE AUDIT REPORT")
    print("=" * 70)
    print(f"Total Valid Images Checked: {report['total_valid_images_checked']}")
    print(f"Total Leaked Valid Images : {report['total_leaked_valid_images']}")
    print(f"Leakage Rate              : {report['overall_leakage_percentage']:.2f}%")
    print("-" * 70)

    if report["leakage_by_class"]:
        print("Top 10 Most Severely Leaked Classes:")
        sorted_classes = sorted(
            report["leakage_by_class"].items(),
            key=lambda x: x[1]["leaked_images_in_valid"],
            reverse=True
        )[:10]
        for cls, info in sorted_classes:
            print(f" - {cls[:40]:40s}: {info['leaked_images_in_valid']}/{info['total_valid_images']} "
                  f"({info['leakage_percentage']:.1f}%)")
            for s in info["samples"][:1]:
                print(f"    Sample: valid '{s['valid_sample']}' matches train '{s['train_match']}'")
    else:
        print("No perceptual hash leakage detected between provided directories.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit dataset for train/val data leakage.")
    parser.add_argument("--train-dir", type=str, default="data/train", help="Path to training directory")
    parser.add_argument("--valid-dir", type=str, default="data/valid", help="Path to validation directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per class for quick testing")
    args = parser.parse_args()

    if os.path.exists(args.train_dir) and os.path.exists(args.valid_dir):
        rep = check_leakage(args.train_dir, args.valid_dir, args.max_samples)
        print_leakage_report(rep)
    else:
        print(f"Dataset directories '{args.train_dir}' and/or '{args.valid_dir}' not found.")
        print("Run with --train-dir and --valid-dir when dataset is unpacked.")
