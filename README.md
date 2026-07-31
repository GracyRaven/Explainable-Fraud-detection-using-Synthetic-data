# Explainable Fraud Detection for M-Pesa Using Machine Learning and SHAP

**Author:** Gracy Kisia — BSc Data Science and Analytics

An explainable fraud detection system for Kenya's M-Pesa mobile money platform. Using a synthetic dataset of 120,000 transactions (2.92% fraud rate), the project compares standard vs. cost-sensitive Logistic Regression, Random Forest, and XGBoost models, and uses SHAP to identify which transaction features drive fraud predictions. A key contribution is formally testing through Spearman rank correlation,whether cost-sensitive weighting distorts SHAP feature importance rankings.

## Dataset

| Detail | Value |
|---|---|
| Source | [Kaggle: M-Pesa Transactions Fraud] (https://www.kaggle.com/datasets/calebboen/mpesa-transactions-fraud/data) |
| Size | 120,000 transactions × 13 columns, Jan–Dec 2026 |
| Fraud rate | ~2.92% (3,504 fraudulent transactions) |
| Missing values | None |

## Objectives

1. Compare standard vs. cost-sensitive models using F2-score and AUC-PR as evaluation metrics
2. Use SHAP to identify the transaction features that drive fraud predictions
3. Test whether cost-sensitive weighting distorts SHAP rankings (Spearman ρ ≥ 0.80 = stable)

## Methodology

Leakage columns (`sender_balance_after`, `receiver_balance_after`, `transaction_id`) were removed, categoricals one-hot encoded, and three features engineered (`amount_to_balance_ratio`, `is_late_night`, `low_sender_balance`). All 20 features passed VIF screening (max VIF = 3.73, threshold = 10). A temporal split (months 1–9 train / 10–12 test) was used to avoid data leakage, and six models: standard and cost-sensitive versions of Logistic Regression, Random Forest, and XGBoost were trained and explained using SHAP TreeExplainer on matched sample rows, with a Spearman rank-correlation stability check and F2-optimal threshold tuning.

## Results

| Model | F2 Score | AUC-PR | FP | Fraud Caught |
|---|---|---|---|---|
| LR Standard | 0.6755 | 0.6857 | 5 | 62.6% |
| LR Cost-Sensitive | 0.5902 | 0.6849 | 852 | 66.7% |
| **RF Standard** | **0.7043** | 0.6796 | **0** | 65.6% |
| RF Cost-Sensitive | 0.7043 | 0.6818 | 0 | 65.6% |
| XGB Standard | 0.7022 | 0.6866 | 0 | 65.4% |
| XGB Cost-Sensitive | 0.6578 | 0.6879 | 324 | 66.3% |

**Best model:** Random Forest Standard (F2 = 0.7043, zero false positives).

**Top SHAP predictors** (both models agree): `amount_to_balance_ratio` > `receiver_balance_before`/`sender_balance_before` > `amount` > `hour`. The engineered `amount_to_balance_ratio` feature outperformed every raw feature in both models.

**SHAP stability:** Spearman ρ = 0.9188 (XGBoost) and 0.9293 (Random Forest), both well above the 0.80 threshold (p < 0.0001). This means that cost-sensitive weighting does not distort SHAP rankings.

**Threshold tuning:** After tuning, cost-sensitive Logistic Regression improved from F2 = 0.5902 to 0.6925, exceeding its standard counterpart; Random Forest and XGBoost cost-sensitive matched their standard versions.

## Key Findings

- Random Forest Standard is the best-performing, most precise model (highest F2, zero false positives).
- `amount_to_balance_ratio` is the single strongest fraud predictor in both tree-based models, confirming the value of feature engineering.
- Cost-sensitive learning and interpretability are compatible: SHAP rankings remain stable (ρ > 0.80) after class weighting.
- Threshold tuning — not class weighting alone — is what lets cost-sensitive models match or beat standard ones.


## How to Run

```bash
git clone https://github.com/GracyRaven/Explainable-Fraud-detection-using-Synthetic-data.git
cd Explainable-Fraud-detection-using-Synthetic-data
pip install -r requirements.txt
jupyter notebook PROJECT_IMPLEMENTATION.ipynb
```
Update the CSV path in Cell 06 to point to `mpesa_synthetic.csv` on your machine.
