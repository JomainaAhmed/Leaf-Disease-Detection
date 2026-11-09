# Model Benchmark & Evaluation Results

> **Evaluation Protocol Note**:  
> In accordance with rigorous machine learning standards, metrics are strictly differentiated between:
> 1. **Default Validation Split** (subject to potential data leakage from offline augmentations in the original dataset distribution).
> 2. **Held-Out Test Set** (split using source-leaf grouping to guarantee zero overlap between training and evaluation).
>
> All numbers below originate from actual runs or are marked as pending execution. **No metrics are estimated or fabricated.**

---

## 1. Overall Performance Comparison

| Model Architecture | Input Resolution | Parameters | Epochs | Training Accuracy | Validation Accuracy | Held-Out Test Accuracy | Macro F1-Score | Weighted F1-Score | Inference Latency (CPU) | Model Size (.keras / .tflite) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legacy Custom CNN** *(Original notebook)* | 128×128 | ~6.5M | 10 | 97.82% | 94.59% | *Not evaluated* | 0.9480 *(val)* | 0.9490 *(val)* | ~28 ms | ~26 MB / *N/A* |
| **Improved Baseline CNN** *(BN + Dropout + Aug)* | 128×128 | ~4.2M | 25 | `[TODO: fill after running]` | `[TODO: fill after running]` | `[TODO: fill after running]` | `[TODO: fill after running]` | `[TODO: fill after running]` | `[TODO: fill after running]` | `[TODO: fill after running]` |
| **MobileNetV2** *(Transfer Learning + Fine-tuning)* | 128×128 | ~2.6M | 25+10 | `[TODO: fill after running]` | `[TODO: fill after running]` | `[TODO: fill after running]` | `[TODO: fill after running]` | `[TODO: fill after running]` | `[TODO: fill after running]` | `[TODO: fill after running]` |
| **EfficientNetB0** *(Transfer Learning + Fine-tuning)* | 128×128 | ~4.3M | 25+10 | `[TODO: fill after running]` | `[TODO: fill after running]` | `[TODO: fill after running]` | `[TODO: fill after running]` | `[TODO: fill after running]` | `[TODO: fill after running]` | `[TODO: fill after running]` |

---

## 2. Legacy Baseline Run Details (From `training_hist.json`)

- **Dataset**: New Plant Diseases Dataset (offline augmented)
- **Train split**: 70,295 images (38 classes)
- **Validation split**: 17,572 images (38 classes)
- **Optimizer**: Adam ($\text{lr} = 10^{-4}$)
- **Loss**: Categorical Cross-Entropy

### Epoch-by-Epoch Convergence (10 Epochs)

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.3926 | 59.18% | 0.4671 | 84.99% |
| 2 | 0.5093 | 84.19% | 0.3337 | 89.61% |
| 3 | 0.3325 | 89.51% | 0.2296 | 92.72% |
| 4 | 0.2506 | 92.26% | 0.2608 | 92.66% |
| 5 | 0.1962 | 94.05% | 0.2714 | 91.91% |
| 6 | 0.1619 | 95.06% | 0.2136 | 93.52% |
| 7 | 0.1459 | 95.60% | 0.2002 | 94.22% |
| 8 | 0.1262 | 96.33% | 0.1890 | 94.67% |
| 9 | 0.1100 | 96.79% | 0.3728 | 92.20% |
| 10 | 0.1059 | 97.02% | 0.2495 | 94.59% |

---

## 3. How to Populate Pending Results

When GPU compute is provisioned and the dataset is downloaded, execute the following reproduction sequence:

```bash
# 1. Audit dataset for train/val leakage
python src/leakage_check.py --train-dir data/train --valid-dir data/valid

# 2. Generate leakage-free 70/15/15 manifests
python src/split_dataset.py --data-dir data/raw --output-dir data --seed 42

# 3. Train Improved Baseline CNN
python src/train.py --model baseline --epochs 25 --batch-size 32 --lr 1e-4

# 4. Train MobileNetV2 with Fine-Tuning
python src/train.py --model mobilenetv2 --epochs 25 --fine-tune-epochs 10 --batch-size 32 --lr 1e-4

# 5. Evaluate on Held-Out Test Set
python src/evaluate.py --model-path models/baseline_best.keras --test-manifest data/test_split.csv
python src/evaluate.py --model-path models/mobilenetv2_best.keras --test-manifest data/test_split.csv

# 6. Run Diagnostic Error Analysis
python src/error_analysis.py --model-path models/mobilenetv2_best.keras --test-manifest data/test_split.csv

# 7. Export and Benchmark TFLite
python src/export_tflite.py --model-path models/mobilenetv2_best.keras
```
