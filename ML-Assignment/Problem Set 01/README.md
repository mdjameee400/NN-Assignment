# Problem Set 01 — Pneumonia Classification from Chest X-Rays (CNN)

Classifies paediatric anterior-posterior chest X-ray radiographs as **NORMAL** or **PNEUMONIA**. Dataset: 5,863 JPEG images (Kermany et al.), organized into `train` (5,216), `val` (16), and `test` (624) splits, each containing `NORMAL` and `PNEUMONIA` subfolders.

---

## How to Run

### Option 1: Google Colab (Recommended)
1. Open [`problem_set_1.ipynb`](problem_set_1.ipynb) in Google Colab.
2. Mount Google Drive containing `xray_dataset.zip` (at `/content/drive/MyDrive/ML_Assignment/xray_dataset.zip` or update path as needed).
3. Run the notebook sequentially. All dependencies (`tensorflow`, `matplotlib`, `seaborn`, `scikit-learn`) are pre-installed in Colab.

### Option 2: Local Python / Jupyter Environment
```bash
pip install tensorflow matplotlib seaborn scikit-learn
jupyter notebook "problem_set_1.ipynb"
```

The dataset is expected unzipped at `/content/xray_data` (or set the paths to your local directory e.g., `./Archive/{train,val,test}`).

> **Note:** The notebook is already executed and committed with real outputs (training logs, loss/accuracy curves, test metrics, and confusion matrix heatmap) saved directly in the notebook file.

---

## Approach & Methodology

### 1. Transfer Learning with EfficientNetB0
Rather than training a shallow CNN from scratch with limited training examples (~5.2k images), which struggled with feature generalization (our previous baseline CNN achieved only ~75.48% test accuracy), this solution adopts **Transfer Learning using EfficientNetB0** pretrained on ImageNet:
- Pretrained weights provide robust low-level edge, texture, and anatomical feature detectors.
- The backbone is frozen during initial training to preserve pretrained representations while optimizing a custom classification head.

### 2. Medical-Safe Data Augmentation
Standard computer vision augmentations like random horizontal flipping or arbitrary rotations are clinically hazardous for chest radiographs:
- Chest X-rays have a strictly defined, clinically meaningful orientation (e.g., cardiac silhouette located on the left hemithorax, liver on the right, aortic arch placement). Mirroring or arbitrarily tilting an X-ray creates anatomically unrealistic images.
- Therefore, augmentation is strictly constrained to acquisition-safe variations:
  - `RandomZoom(0.15)`
  - `RandomTranslation(0.08, 0.08)`
  - `RandomContrast(0.12)`

### 3. Handling Class Imbalance
The training dataset is notably imbalanced:
- `PNEUMONIA`: 3,875 images (~74.3%)
- `NORMAL`: 1,341 images (~25.7%)

To prevent the loss function from trivially collapsing towards predicting only the majority class, inverse frequency class weights are dynamically calculated:
- Class 0 (NORMAL): **1.9448**
- Class 1 (PNEUMONIA): **0.6730**

These weights are passed directly to `.fit(..., class_weight=class_weight)` during training.

### 4. Classification Head & Two-Phase Training
- **Custom Head Architecture**:
  - `GlobalAveragePooling2D()`
  - `BatchNormalization()`
  - `Dropout(0.4)` (regularization to mitigate overfitting)
  - `Dense(64, activation='relu')`
  - `Dense(1, activation='sigmoid')` (binary classification output)
- **Phase 1 (Feature Extraction)**:
  - Backbone frozen (`base_effnet.trainable = False`).
  - Optimized with `AdamW(learning_rate=1e-3, weight_decay=1e-4)`.
  - Monitored with `EarlyStopping` and `ReduceLROnPlateau` targeting `val_auc`.
- **Phase 2 (Fine-Tuning)**:
  - Top 30 layers of the EfficientNetB0 backbone are unfrozen (`base_effnet.layers[:-30]` remain frozen).
  - Recompiled with a very low learning rate (`1e-5`) using `AdamW` to gently tune high-level feature representations without destroying low-level pretrained weights.

### 5. Clinically Meaningful Metrics
Along with standard accuracy, the training and evaluation monitor **Precision**, **Recall**, and **ROC-AUC**. In medical diagnosis, high recall (sensitivity) on disease cases is critical to minimize false negatives (avoiding missing active pneumonia infections), while maintaining high precision to prevent unnecessary clinical alarm.

---

## Findings & Evaluation

All numbers below are from the actual execution of [`problem_set_1.ipynb`](problem_set_1.ipynb) on the 624 held-out test images:

### Test Set Performance Comparison

| Metric | Previous Baseline (From-Scratch CNN) | Updated Model (EfficientNetB0 Transfer Learning) |
|---|:---:|:---:|
| **Test Accuracy** | 75.48% | **85.42%** |
| **Test Loss** | 0.6161 | **0.3459** |
| **Test ROC-AUC** | — | **0.9369** |
| **NORMAL Precision** | 0.99 | **0.89** |
| **NORMAL Recall** | 0.35 | **0.70** *(doubled)* |
| **NORMAL F1-Score** | 0.52 | **0.78** |
| **PNEUMONIA Precision** | 0.72 | **0.84** |
| **PNEUMONIA Recall** | 1.00 | **0.95** |
| **PNEUMONIA F1-Score** | 0.84 | **0.89** |
| **Macro Average F1** | 0.68 | **0.84** |
| **Weighted Average F1** | 0.72 | **0.85** |

### Detailed Analysis
1. **Resolution of Recall Collapse**: In the earlier scratch CNN, the model suffered from severe majority-class bias; healthy subjects had an unacceptable recall of only 35% (misclassifying 65% of healthy patients as diseased). The updated model doubles NORMAL recall to **70%** with **89% precision**.
2. **Clinical Sensitivity**: For PNEUMONIA, the model achieves **95% recall** (identifying 369 out of 390 pneumonia cases) and **84% precision**, ensuring minimal false negatives for triage.
3. **Discriminative Capacity**: A Test ROC-AUC of **0.9369** confirms strong feature separability across the entire decision boundary.

---

## Validation Set Limitations
- The provided `val/` split contains only 16 images (8 NORMAL, 8 PNEUMONIA).
- With such a small sample size, validation accuracy and AUC are heavily quantized (moving only in increments of 1/16 = 6.25%). In practice, `val_auc` quickly plateaued at 0.9844 – 1.0000.
- `EarlyStopping` and `ReduceLROnPlateau` receive a coarse feedback signal due to this limited sample. While the 624-image test set remains a reliable benchmark, future iterations should carve a larger stratified validation split (e.g., 90/10) directly from the 5,216 training images.

---

## Summary of Key Improvements
- **Model Migration**: Replaced from-scratch CNN with ImageNet-pretrained `EfficientNetB0` with top-30 layer fine-tuning.
- **Medical Augmentation**: Restricted jitter to contrast, translation, and zoom, eliminating invalid flips/rotations.
- **Optimization**: Switched from standard SGD/Adam to decoupled weight decay `AdamW` with dynamic learning rate scheduling.
- **Performance Gain**: Accuracy increased by **+9.94%** (75.48% $\rightarrow$ 85.42%) alongside balanced diagnostic recall across both classes.
