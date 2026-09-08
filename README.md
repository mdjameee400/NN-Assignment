# Machine Learning & Neural Networks — Assignment Repository

This repository contains the complete solutions, documented code, and analytical reports for the **Neural Networks / Machine Learning** assignment problem sets.

- **Institution**: East Delta University (EDU)
- **Program / Term**: 9th Semester
- **Course**: Neural Networks (NN)
- **Course Instructor**: Ashrafur Rahman Chowdhury
- **Total Marks**: 15

---

## Assignment Instructions & Submission Guidelines

The assignment was announced with the following core directives:

- **Total Marks**: 15
- **Dataset Utilization**: Implement solutions for the given problem statements using the official datasets provided.
- **Git Repository Submission**: Upload the complete solutions to a Git repository and submit the repository link.
- **Dedicated Directories**: Maintain **two separate, isolated directories** for the two problem sets.
- **Required Deliverables per Directory**:
  - **Code**: Clean, executable implementation file (`.ipynb`).
  - **`README.md`**: Rigorous technical documentation explaining the approach, methodology, empirical findings, and critical evaluations.
- **Academic Integrity**: Strict anti-plagiarism policy (*"Please do not COPY PASTE. I won't be responsible for your mark 0.0"*). All solutions, model designs, and analyses are implemented and written originally from scratch.

---

## Repository Structure

```
NN-Assignment/
├── README.md                             # Central assignment documentation & course metadata
└── ML-Assignment/
    ├── Problem Set 01/
    │   ├── README.md                     # Methodology, architecture & findings for PS 01
    │   └── problem_set_1.ipynb           # Executed CNN / Transfer Learning notebook
    └── Problem Set 02/
        ├── README.md                     # Methodology, evaluation & imbalance study for PS 02
        └── problem_set_2.ipynb           # Executed Logistic Regression notebook
```

---

## Problem Set Overviews

### [Problem Set 01: Pneumonia Classification from Chest X-Rays](ML-Assignment/Problem%20Set%2001/)

- **Task**: Develop a **Convolutional Neural Network (CNN)** to classify paediatric chest radiographs as either **NORMAL** or **PNEUMONIA** to assist in clinical diagnosis.
- **Dataset Provided**: Kermany et al. Chest X-Ray image dataset, comprising **5,863 JPEG images** partitioned into `train` (5,216), `val` (16), and `test` (624) directories.
- **Methodology**:
  - **Transfer Learning Backbone**: Utilized ImageNet-pretrained **EfficientNetB0** with custom classification head (`GlobalAveragePooling2D` $\rightarrow$ `BatchNorm` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Dense(64, relu)` $\rightarrow$ `Dense(1, sigmoid)`).
  - **Clinically Aware Augmentation**: Restricted variations to zoom, translation, and contrast jitter, strictly avoiding arbitrary horizontal flips or tilting that would disrupt bilateral anatomical asymmetry.
  - **Class Imbalance Mitigation**: Dynamic inverse frequency class weighting (`NORMAL: 1.9448`, `PNEUMONIA: 0.6730`).
  - **Two-Phase Optimization**: Head training followed by fine-tuning the top 30 layers using `AdamW` (`lr=1e-5`).
- **Core Results**:
  - **Test Accuracy**: **85.42%** (+9.94% over from-scratch baseline)
  - **Test ROC-AUC**: **0.9369**
  - **Balanced Sensitivity**: Doubled NORMAL recall to **0.70** (89% Precision) while preserving **0.95** PNEUMONIA recall (84% Precision).
- **Directory**: [`ML-Assignment/Problem Set 01/`](ML-Assignment/Problem%20Set%2001/)

---

### [Problem Set 02: Bank Term Deposit Prediction](ML-Assignment/Problem%20Set%2002/)

- **Task**: Construct a **Logistic Regression** model to predict whether a prospective banking customer will subscribe to a term deposit (`y`: yes/no) based on marketing campaign features.
- **Dataset Provided**: UCI Machine Learning **Bank Marketing Dataset** (`bank-full.csv`), consisting of **45,211 records** and **17 attributes** (client demographics, credit defaults, account balance, contact campaign history, and macro indicators).
- **Methodology**:
  - **Feature Preprocessing**: One-hot dummy encoding with reference levels dropped (`drop_first=True`) across nominal categories to avoid multicollinearity across 42 predictors.
  - **Stratified Partitioning**: 80/20 train-test split (`stratify=y`, `random_state=42`) preserving the intrinsic 88:12 class distribution.
  - **Leakage-Free Normalization**: Applied `StandardScaler` fitted exclusively on training data.
  - **Model Fitting & Serialization**: Fitted `LogisticRegression(max_iter=1000)` and serialized model using `joblib`.
  - **Class Imbalance Analysis**: Conducted an auxiliary investigation comparing default weighting against `class_weight='balanced'`.
- **Core Results**:
  - **Overall Test Accuracy**: **~90.1%** (0.90) on 9,043 held-out test records.
  - **Test ROC-AUC Score**: **0.91** (0.9054).
  - **Imbalance Mechanics**: Documented how class-weighted learning shifts the operating threshold to increase minority subscriber recall from **35%** to **81%** (capturing 861 out of 1,058 depositors) while preserving identical ROC-AUC rank-ordering.
- **Directory**: [`ML-Assignment/Problem Set 02/`](ML-Assignment/Problem%20Set%2002/)

---

## Environment & Execution

All notebooks contain executed outputs, metrics, and visualization figures saved in the file. They are ready to run both in local environments and Google Colab.

```bash
# Clone the repository
git clone https://github.com/mdjameee400/NN-Assignment.git
cd NN-Assignment

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow joblib
```
