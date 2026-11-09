"""
Unit and Integration Tests for Plant Leaf Disease Detection Pipeline.

Tests:
1. Label mapping and 38-class integrity
2. Image preprocessing shape and range
3. Model architecture construction and output tensor shape (None, 38)
4. Agronomic disease database completeness
5. Leakage-safe group splitting logic
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.utils import CLASS_NAMES, NUM_CLASSES, format_class_name, load_image_for_inference
from src.split_dataset import create_leakage_safe_split, extract_base_image_id


def test_label_mapping_count():
    """Verify exactly 38 distinct plant disease classes are registered."""
    assert len(CLASS_NAMES) == 38
    assert NUM_CLASSES == 38
    assert len(set(CLASS_NAMES)) == 38, "Duplicate class names found!"


def test_format_class_name():
    """Verify human-readable name formatting converts internal identifiers properly."""
    raw = "Apple___Apple_scab"
    formatted = format_class_name(raw)
    assert "Apple" in formatted
    assert "scab" in formatted.lower()

    healthy_raw = "Tomato___healthy"
    assert format_class_name(healthy_raw) == "Tomato (Healthy)"

    corn_rust = "Corn_(maize)___Common_rust_"
    assert "Corn" in format_class_name(corn_rust)


def test_preprocessing_shape():
    """Verify input image preprocessing returns (1, 128, 128, 3) float32 array."""
    # Create synthetic test image
    dummy_img = Image.new("RGB", (256, 256), color=(120, 180, 75))
    arr, resized = load_image_for_inference(dummy_img, target_size=(128, 128))

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 128, 128, 3)
    assert arr.dtype == np.float32
    assert resized.size == (128, 128)


def test_model_output_shape():
    """Verify baseline CNN builds with output shape (None, 38)."""
    try:
        import tensorflow as tf
        from src.models import build_baseline_cnn

        model = build_baseline_cnn(input_shape=(128, 128, 3), num_classes=38, use_augmentation=False)
        assert model.output_shape == (None, 38)
        assert model.input_shape == (None, 128, 128, 3)

        dummy_batch = np.random.rand(2, 128, 128, 3).astype(np.float32)
        preds = model(dummy_batch, training=False)
        assert preds.shape == (2, 38)
        # Sum of softmax probabilities should be close to 1
        np.testing.assert_allclose(np.sum(preds.numpy(), axis=1), [1.0, 1.0], atol=1e-5)
    except ImportError:
        pytest.skip("TensorFlow not installed in test environment")


def test_disease_database_completeness():
    """Verify disease_info.json covers all 38 classes with treatment and prevention tips."""
    db_path = os.path.join(os.path.dirname(__file__), "..", "app", "disease_info.json")
    assert os.path.exists(db_path), "disease_info.json does not exist!"

    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cls in CLASS_NAMES:
        assert cls in data, f"Missing class in disease_info.json: {cls}"
        entry = data[cls]
        assert "crop" in entry
        assert "condition" in entry
        assert "treatment" in entry and len(entry["treatment"]) > 0
        assert "prevention" in entry and len(entry["prevention"]) > 0


def test_leakage_group_split():
    """Verify group splitting ensures zero group overlap across train, val, and test splits."""
    # Synthesize small dataset with offline augmented duplicates
    mock_records = []
    classes = ["Tomato___healthy", "Apple___Apple_scab"]
    for cls in classes:
        for leaf_id in range(10):
            # 3 augmented copies per leaf
            for aug in ["orig", "rot90", "flip"]:
                fname = f"leaf_{leaf_id}_{aug}.jpg"
                mock_records.append({
                    "filepath": f"{cls}/{fname}",
                    "full_path": f"/mock/{cls}/{fname}",
                    "filename": fname,
                    "class_name": cls,
                    "class_id": 0 if "Tomato" in cls else 1,
                    "group_id": f"{cls}___leaf_{leaf_id}"
                })

    df_mock = pd.DataFrame(mock_records)
    train_df, val_df, test_df = create_leakage_safe_split(
        df_mock,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    train_groups = set(train_df["group_id"].unique())
    val_groups = set(val_df["group_id"].unique())
    test_groups = set(test_df["group_id"].unique())

    # Check zero overlap
    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0

    assert len(train_df) + len(val_df) + len(test_df) == len(df_mock)
