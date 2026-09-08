# Machine Learning & Neural Networks — Assignment Repository

This repository contains the complete solutions, documented code, and analytical reports for the **Neural Networks / Machine Learning** course problem sets.

- **Institution**: East Delta University (EDU)
- **Program / Term**: 9th Semester
- **Course Instructor**: Ashraf Sir

---

## Assignment Instructions & Submission Guidelines

The repository strictly adheres to the following submission parameters:
1. **Problem Statements & Datasets**: Solutions implemented for each announced problem statement using the provided real-world datasets.
2. **Dedicated Directories**: Maintained two separate, isolated directories for the two problem sets.
3. **Comprehensive Documentation**: Each directory contains the self-contained executable code (`.ipynb`) and a dedicated `README.md` file covering the approach, methodology, empirical findings, and critical evaluations.
4. **Academic Integrity**: Original formulations and rigorous experimental analyses implemented without copy-pasting.

---

## Repository Structure

```
NN-Assignment/
├── README.md                             # Central course overview & assignment summary
└── ML-Assignment/
    ├── Problem Set 01/
    │   ├── README.md                     # Methodology, architecture & findings for PS 01
    │   └── problem_set_1.ipynb           # Executed CNN / Transfer Learning notebook
    └── Problem Set 02/
        ├── README.md                     # Methodology, evaluation & imbalance study for PS 02
        └── problem_set_2.ipynb           # Executed Logistic Regression notebook
```

---

## Summary of Problem Sets

### [Problem Set 01: Pneumonia Classification from Chest X-Rays](ML-Assignment/Problem%20Set%2001/)

- **Objective**: Accurately classify paediatric anterior-posterior chest radiographs as **NORMAL** or **PNEUMONIA** to support automated clinical triage and diagnostic workflows.
- **Dataset Provided**: Kermany et al. Chest X-Ray image dataset, consisting of **5,863 JPEG images** partitioned into `train` (5,216), `val` (16), and `test` (624) sets across `NORMAL` and `PNEUMONIA` classes.
- **Methodology & Key Techniques**:
  - **Transfer Learning Backbone**: Replaced basic shallow CNNs with ImageNet-pretrained **EfficientNetB0** for superior visual feature extraction.
  - **Clinically Aware Augmentation**: Restricted image transformations strictly to zoom, translation, and contrast jitter—avoiding invalid horizontal flips or arbitrary tilts that disrupt anatomical asymmetry.
  - **Class Imbalance Mitigation**: Applied dynamic inverse frequency loss weighting (`NORMAL: 1.9448`, `PNEUMONIA: 0.6730`).
  - **Two-Phase Fine-Tuning**: Initial feature extraction of the custom classification head followed by unfreezing and fine-tuning the top 30 layers with AdamW (`lr=1e-5`).
- **Empirical Findings**:
  - **Test Accuracy**: **85.42%** (+9.94% improvement over from-scratch baseline)
  - **Test ROC-AUC**: **0.9369**
  - **NORMAL Recall**: Doubled from 0.35 to **0.70** (89% Precision)
  - **PNEUMONIA Recall**: **0.95** (84% Precision), ensuring high sensitivity for clinical safety.
- **Directory**: [`ML-Assignment/Problem Set 01/`](ML-Assignment/Problem%20Set%2001/)

---

### [Problem Set 02: Bank Term Deposit Prediction](ML-Assignment/Problem%20Set%2002/)

- **Objective**: Build a predictive binary classification model to determine whether a prospective client will subscribe to a bank term deposit (`y`: yes/no) based on direct telemarketing campaign contacts.
- **Dataset Provided**: UCI Machine Learning **Bank Marketing Dataset** (`bank-full.csv`), comprising **45,211 instances** and **17 features** (economic indicators, campaign contacts, past outcomes, and client demographics).
- **Methodology & Key Techniques**:
  - **Preprocessing & Encoding**: Handled nominal categorical attributes via one-hot dummy encoding (`drop_first=True`) to avoid multicollinearity across 42 predictor dimensions.
  - **Stratified Partitioning**: 80/20 train-test split (`stratify=y`, `random_state=42`) preserving the intrinsic 88:12 class distribution.
  - **Leakage-Free Scaling**: Normalized features with `StandardScaler` fitted exclusively on training data.
  - **Logistic Regression Model**: Trained with `max_iter=1000` to ensure L-BFGS solver convergence; serialized with `joblib`.
  - **Class Imbalance Study**: Documented an auxiliary experiment comparing default weighting against `class_weight='balanced'`.
- **Empirical Findings**:
  - **Overall Test Accuracy**: **90.1%** (0.90) on 9,043 held-out test records.
  - **ROC-AUC Score**: **0.91** (0.9054), proving high continuous discriminative capacity.
  - **Imbalance Insights**: Demonstrated how `class_weight='balanced'` shifts the decision threshold to elevate subscriber recall from **35%** to **81%** (capturing 861 out of 1,058 depositors) while maintaining an identical ROC-AUC of 0.91.
- **Directory**: [`ML-Assignment/Problem Set 02/`](ML-Assignment/Problem%20Set%2002/)

---

## Environment & Dependencies

All notebooks are pre-executed with outputs embedded, and can be run either locally or on Google Colab.

```bash
# Core Machine Learning & Data Processing Stack
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow joblib
```
