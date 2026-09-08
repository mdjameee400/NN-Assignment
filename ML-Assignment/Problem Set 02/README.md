# Problem Set 02 — Bank Term Deposit Prediction (Logistic Regression)

Binary classification model to predict whether a client will subscribe to a bank term deposit (`y`: yes/no) based on direct marketing telemarketing campaigns. Dataset: **Bank Marketing Dataset** (UCI Machine Learning Repository) consisting of 45,211 records and 17 features from a Portuguese banking institution.

---

## File Structure

```
ML-Assignment/
├── Problem Set 01/
│   ├── README.md
│   └── problem_set_1.ipynb
└── Problem Set 02/
    ├── README.md
    └── problem_set_2.ipynb
```

---

## How to Run

### Option 1: Google Colab (Recommended)
1. Open [`problem_set_2.ipynb`](problem_set_2.ipynb) in Google Colab.
2. Mount Google Drive containing `bank-full.csv` at `/content/drive/MyDrive/ML_Assignment/bank-full.csv` (or adjust `data_path` to your storage location).
3. Run the notebook sequentially. All dependencies (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`) are pre-installed in Google Colab.

### Option 2: Local Python / Jupyter Environment
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
jupyter notebook "problem_set_2.ipynb"
```

The dataset is expected at `./bank-data/bank-full.csv` (or configure `data_path` to match your local CSV path).

> **Note:** The notebook is already executed and committed with actual outputs (data summaries, encoded shapes, classification report, ROC-AUC score, confusion matrix heatmap, and ROC curve plot) saved directly in the notebook file.

---

## Approach & Methodology

### 1. Exploratory Data Analysis & Verification
- **Dataset Dimensions**: 45,211 records across 17 attributes (7 numerical, 10 categorical).
- **Missing Value Check**: 0 null/missing values across all columns.
- **Class Imbalance**:
  - Class `no` (did not subscribe): **39,922** (~88.3%)
  - Class `yes` (subscribed): **5,289** (~11.7%)
  - The dataset exhibits an approximate 88:12 negative-to-positive imbalance ratio.

### 2. Data Preprocessing & Categorical Encoding
- **Target Encoding**: Mapped binary target variable `y` (`{'yes': 1, 'no': 0}`).
- **One-Hot Encoding**: Applied `pd.get_dummies(df, drop_first=True)` to encode all categorical features (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`).
  - Dropping the first level prevents multicollinearity (dummy variable trap).
  - Expanded the feature space to **42 predictor features**.

### 3. Stratified Train-Test Splitting
- Split into **80% training** (36,168 samples) and **20% test** (9,043 samples) using `train_test_split`.
- Parameter `stratify=y` ensures both train and test splits retain the identical 88.3% / 11.7% class proportion, preventing sampling bias.
- Seeded with `random_state=42` for strict reproducibility.

### 4. Feature Standardization
- Applied `StandardScaler` to normalize features:
  - Fitted scaler exclusively on training data (`scaler.fit_transform(X_train)`).
  - Applied fitted parameters to transform test data (`scaler.transform(X_test)`), eliminating data leakage.
  - Necessary for Logistic Regression to ensure gradient descent stability and prevent features with larger scales (e.g., `balance`, `duration`) from dominating weights.

### 5. Model Architecture & Training
- **Model**: `LogisticRegression(max_iter=1000)`
- Set `max_iter=1000` to guarantee solver convergence on the 42-dimensional feature space.
- Fitted on standardized features (`X_train_scaled`, `y_train`).

### 6. Model Serialization
- Model persisted via `joblib.dump(model, 'bank_logistic_model.pkl')` for deployment and downstream inference pipelines.

---

## Findings & Evaluation

Evaluation performed on the held-out test set (**9,043 samples**):

### Test Set Performance Metrics

| Metric | Class 0 (`no`) | Class 1 (`yes`) | Macro Avg | Weighted Avg | Overall |
|---|:---:|:---:|:---:|:---:|:---:|
| **Precision** | **0.92** | **0.64** | 0.78 | 0.89 | — |
| **Recall** | **0.97** | **0.35** | 0.66 | 0.90 | — |
| **F1-Score** | **0.95** | **0.45** | 0.70 | 0.89 | — |
| **Support** | 7,985 | 1,058 | 9,043 | 9,043 | 9,043 |
| **Accuracy** | — | — | — | — | **90%** (0.90) |
| **ROC-AUC Score** | — | — | — | — | **0.9054** |

---

### Detailed Analysis
1. **Strong Overall Discriminative Ability (ROC-AUC = 0.9054)**:
   - An ROC-AUC score of **~90.54%** demonstrates that the Logistic Regression model effectively ranks positive subscription probabilities higher than negative instances across various decision thresholds.
2. **High Precision on Majority Class (0.92 Precision, 0.97 Recall)**:
   - The model reliably filters clients who are not interested in subscribing, minimizing wasted marketing efforts on non-converters.
3. **Class Imbalance & Minority Recall (0.35 Recall)**:
   - Due to the ~88:12 imbalance, the default 0.5 decision threshold causes lower sensitivity (35% recall) for actual term deposit subscribers, while maintaining reasonable precision (64%).
4. **Recommended Future Enhancements**:
   - Apply `class_weight='balanced'` in `LogisticRegression` to penalize minority misclassifications.
   - Tune probability classification thresholds (lowering from 0.5 to ~0.3) to maximize subscriber identification.
   - Explore nonlinear algorithms such as Random Forest or Gradient Boosting (XGBoost/LightGBM).

---

## Key Highlights
- **Clean Code**: Zero extraneous comments, cleanly structured pipeline from ingestion to serialization.
- **Leak-Free Scaling**: Scaling parameters computed only on training split.
- **Stratified Validation**: Representative class distribution across train/test splits.
- **Persistent Model**: Serialized `.pkl` model ready for deployment.
