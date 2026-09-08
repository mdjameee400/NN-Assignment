# Problem Set 02 — Bank Term Deposit Prediction (Logistic Regression)

Binary classification model to predict whether a prospective banking customer will subscribe to a term deposit (`y`: yes/no) based on direct telemarketing campaigns. The solution is developed using the **Bank Marketing Dataset** (UCI Machine Learning Repository), containing 45,211 instances and 17 attributes from a Portuguese banking institution.

---

## How to Run

### Option 1: Google Colab (Recommended)
1. Open [`problem_set_2.ipynb`](problem_set_2.ipynb) in Google Colab.
2. Mount Google Drive containing `bank-full.csv` at `/content/drive/MyDrive/ML_Assignment/bank-full.csv` (or modify `data_path` to match your Drive storage location).
3. Execute all cells sequentially. Required libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`) are standard across Colab runtimes.

### Option 2: Local Python / Jupyter Environment
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
jupyter notebook "problem_set_2.ipynb"
```

Configure `data_path` to point to the local CSV file path (e.g., `./bank-data/bank-full.csv`).

> **Execution Note:** The notebook is committed with executed outputs and visualizations (confusion matrix heatmap and ROC curves) saved directly in the notebook file.

---

## Approach & Methodology

### 1. Exploratory Data Analysis & Verification
- **Dataset Dimensions**: 45,211 rows × 17 features (7 numerical, 10 categorical).
- **Data Integrity**: Verified 0 null or missing values across all columns.
- **Target Distribution**:
  - Class `no` (non-subscribers): **39,922** (~88.30%)
  - Class `yes` (subscribers): **5,289** (~11.70%)
  - Confirmed an intrinsic class imbalance ratio of approximately **88:12**, which dictates evaluation beyond raw accuracy.

### 2. Feature Preprocessing & Categorical Encoding
- **Binary Target Mapping**: Mapped `df['y']` to binary values (`'yes': 1, 'no': 0`).
- **One-Hot Encoding**: Applied `pd.get_dummies(df, drop_first=True)` to encode nominal categorical features (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`).
  - Dropping the reference level (`drop_first=True`) prevents multicollinearity (dummy variable trap) in linear estimators.
  - Expands the feature space to **42 predictor features**.

### 3. Stratified Train-Test Partitioning
- Partitioned dataset into **80% training** (36,168 samples) and **20% testing** (9,043 samples) using `train_test_split`.
- Parameter `stratify=y` ensures both train and test splits retain the identical 88.3% / 11.7% class proportion, preventing sampling distortion.
- Seeded with `random_state=42` for exact reproducibility.

### 4. Feature Standardization (`StandardScaler`)
- Fitted `StandardScaler` strictly on `X_train` (`fit_transform`) and subsequently transformed `X_test` (`transform`), eliminating data leakage.
- Standardizing to zero mean and unit variance ensures gradient descent stability during optimization and prevents features with wide numerical ranges (e.g., `balance`, `duration`) from artificially skewing coefficient weights.

### 5. Model Training & Persistence
- Trained a binary classifier using `LogisticRegression(max_iter=1000)`.
- Set `max_iter=1000` to guarantee solver convergence without early termination warnings.
- Serialized the trained estimator using `joblib.dump(model, 'bank_logistic_model.pkl')` for deployment readiness.

---

## Model Evaluation (Primary Submission)

The baseline submission employs the default classification threshold (0.50) without synthetic sampling or weight inflation, evaluated on the 9,043 held-out test samples:

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **No (0)** | 0.92 | 0.97 | 0.95 | 7,985 |
| **Yes (1)** | 0.64 | 0.35 | 0.45 | 1,058 |
| **Macro Average** | 0.78 | 0.66 | 0.70 | 9,043 |
| **Weighted Average** | 0.89 | 0.90 | 0.89 | 9,043 |

**Overall Metrics:**
- **Overall Accuracy**: **~90.1%** (0.90)
- **ROC-AUC Score**: **0.91** (0.9054)

### Confusion Matrix

| | Predicted: No | Predicted: Yes | Total Actual |
|---|:---:|:---:|:---:|
| **Actual: No** | **7,781** (TN) | **204** (FP) | 7,985 |
| **Actual: Yes** | **689** (FN) | **369** (TP) | 1,058 |
| **Total Predicted** | 8,470 | 573 | 9,043 |

### Interpretation & Critical Analysis
- **Strong Discriminative Capacity**: The model achieves an ROC-AUC of **0.91**, proving that the logistic sigmoid function cleanly differentiates prospective subscribers from non-subscribers across continuous probability estimates.
- **The Imbalance Caveat**: While an overall accuracy of 90.1% is high, accuracy alone is misleading in imbalanced settings (a naive dummy classifier predicting all `no` would naturally achieve 88.3%).
- **Conservative Positive Predictions**: The model is selective when predicting "yes" — when it alerts that a client will subscribe, it is accurate **64%** of the time (precision). However, at the default 0.5 decision threshold, it only captures **35%** of actual subscribers (recall = 0.35), letting 689 potential depositors slip past undetected as false negatives.

---

## Limitations & Class Imbalance Experimentation

A key limitation of the primary model is its low sensitivity (35% recall) for the positive class (`yes`), resulting directly from the natural ~88:12 distribution of marketing outcomes. Without imbalance compensation, standard maximum likelihood estimation prioritizes minimizing total errors across the dominant majority class.

### Comparative Experiment: Class Weighting (`class_weight='balanced'`)
To analyze the mechanics of decision boundary shift without altering the submitted primary code, an auxiliary experiment was conducted by fitting `LogisticRegression(max_iter=1000, class_weight='balanced')`. 

Under `balanced` mode, Scikit-Learn inversely scales class weights according to class frequencies:
$$\text{Weight}(k) = \frac{N_{\text{samples}}}{N_{\text{classes}} \times N_k}$$

This penalizes misclassifying a rare positive subscriber ~7.5 times more heavily than misclassifying a non-subscriber.

#### Performance Comparison Table

| Metric | Default Model (Submitted) | With `class_weight='balanced'` (Experiment) |
|---|:---:|:---:|
| **Overall Accuracy** | **90.1%** | 84.6% |
| **Class 1 ("Yes") Recall** | 35% | **81%** *(+46% gain)* |
| **Class 1 ("Yes") Precision** | **64%** | 42% |
| **Class 1 ("Yes") F1-Score** | 45% | **55%** |
| **ROC-AUC Score** | **0.91** | **0.91** *(Identical)* |

#### Confusion Matrix Comparison

| Actual \ Predicted | Default Model (`class_weight=None`) | Balanced Model (`class_weight='balanced'`) |
|---|:---:|:---:|
| **True Negatives (TN)** | **7,781** | 6,789 |
| **False Positives (FP)** | **204** | 1,196 |
| **False Negatives (FN)** | 689 | **197** *(drastically reduced)* |
| **True Positives (TP)** | 369 | **861** *(more than doubled)* |

#### Technical Insights from the Experiment
1. **Identical ROC-AUC (0.91)**: 
   - Despite drastic changes in precision and recall, the ROC-AUC remained identical at 0.91. 
   - This occurs because ROC-AUC measures rank ordering across all operating thresholds, independent of the decision cutoff.
2. **Boundary Shift Mechanics**:
   - Applying `class_weight='balanced'` shifts the effective operating point on the ROC curve. The model trades off specificity to capture **861 out of 1,058 subscribers** (81.4% recall), cutting missed subscribers from 689 down to 197, at the expense of increasing outreach to 1,196 uninterested leads (false positives).

---

## Strategic Justification & Business Trade-offs

The decision of which model to prioritize is a classic machine learning trade-off dictated by bank operational constraints:

| Objective / Scenario | Recommended Model | Rationale |
|---|---|---|
| **High Outreach Cost / Call Center Capacity Limit** | **Default Model (Submitted)** | Minimizes wasted agent time on unqualified leads by ensuring 64% precision and minimal false alarms (only 204 FP). |
| **High Customer Lifetime Value / Growth Priority** | **Balanced Model** | Prioritizes customer acquisition by capturing 81% of all depositors, accepting higher telemarketing costs as acceptable overhead. |

The **default model** is submitted as the canonical solution because it represents the pure logistic formulation with high overall fidelity (~90.1%), with this comparative study documenting the exact practical ramifications of cost-sensitive learning.

---

## Future Improvements & Recommendations
- **Dynamic Threshold Calibration**: Rather than rigid weighting, tune the probability threshold (e.g., test cutoffs between 0.25 and 0.35) using Precision-Recall Curves (PR-AUC) to meet specific cost-per-lead requirements.
- **Resampling Methods**: Benchmark synthetic oversampling (SMOTE, ADASYN) and intelligent undersampling (Tomek Links).
- **Leakage Prevention (`duration` feature)**: The call duration attribute (`duration`) is only known *after* a phone call concludes. In real production deployment where leads are pre-screened *before* dialing, models should be evaluated both with and without `duration` to prevent lookahead bias.
- **Non-Linear Ensembles**: Compare against tree-based architectures (Random Forests, LightGBM, XGBoost) capable of modeling complex nonlinear cross-feature interactions.
