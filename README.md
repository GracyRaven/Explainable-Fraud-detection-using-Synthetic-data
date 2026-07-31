# Explainable Fraud Detection for M-Pesa Using Machine Learning and SHAP

**Author:** Gracy Kisia - BSc Data Science and Analytics

An explainable fraud detection system for Kenya's M-Pesa mobile money platform. Using a synthetic dataset of 120,000 transactions (2.92% fraud rate), the project compares standard vs. cost-sensitive Logistic Regression, Random Forest, and XGBoost models, and uses SHAP to identify which transaction features drive fraud predictions. A key contribution is formally testing through Spearman rank correlation,whether cost-sensitive weighting distorts SHAP feature importance rankings.

## Dataset

| Detail | Value |
|---|---|
| Source | [Kaggle: M-Pesa Transactions Fraud] (https://www.kaggle.com/datasets/calebboen/mpesa-transactions-fraud/data) |
| Size | 120,000 transactions × 13 columns, Jan–Dec 2026 |
| Fraud rate | ~2.92% (3,504 fraudulent transactions) |
| Missing values | None |

## Objectives

1. Compare standard and cost-sensitive machine learning models using F2-score and AUC-PR.
2. Use SHAP to explain the transaction features driving fraud predictions.
3. Evaluate whether cost-sensitive learning alters SHAP feature importance rankings using Spearman rank correlation.

## Methodology

Leakage columns (`sender_balance_after`, `receiver_balance_after`, `transaction_id`) were removed, categoricals one-hot encoded, and three features engineered (`amount_to_balance_ratio`, `is_late_night`, `low_sender_balance`). 
All 20 features passed VIF screening (max VIF = 3.73, threshold = 10). 
A temporal split (months 1–9 train / 10–12 test) was used to avoid data leakage, and six models: standard and cost-sensitive versions of Logistic Regression, Random Forest, and XGBoost were trained and explained using SHAP TreeExplainer on matched sample rows, with a Spearman rank-correlation stability check and F2-optimal threshold tuning.

## How to Run

```bash
git clone https://github.com/GracyRaven/Explainable-Fraud-detection-using-Synthetic-data.git
cd Explainable-Fraud-detection-using-Synthetic-data
pip install -r requirements.txt
jupyter notebook PROJECT_IMPLEMENTATION.ipynb
```
Update the CSV path in Cell 06 to point to `mpesa_synthetic.csv` on your machine.
