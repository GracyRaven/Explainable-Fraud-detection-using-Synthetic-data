# Explainable Fraud Detection for M-Pesa Using Machine Learning and SHAP

**Author:** Gracy Kisia - BSc Data Science and Analytics

This project explores explainable machine learning for fraud detection in Kenya's M-Pesa mobile money platform. Using a synthetic dataset of 120,000 transactions with a fraud rate of 2.92%, it compares standard and cost-sensitive versions of Logistic Regression, Random Forest, and XGBoost to evaluate their effectiveness in detecting fraudulent transactions. Beyond predictive performance, the project focuses on model interpretability by using SHAP to explain which transaction features have the greatest influence on fraud predictions. It also investigates whether applying cost-sensitive learning changes these explanations by comparing SHAP feature importance rankings using Spearman rank correlation. The goal is to develop a fraud detection approach that is not only accurate but also transparent and trustworthy, making it easier to understand the reasoning behind model predictions.

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


## Project Structure

```text
Explainable-Fraud-detection-using-Synthetic-data/
├── Data/
├── docs/
├── outputs/
├── PROJECT IMPLEMENTATION.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

```bash
git clone https://github.com/GracyRaven/Explainable-Fraud-detection-using-Synthetic-data.git
cd Explainable-Fraud-detection-using-Synthetic-data
pip install -r requirements.txt
jupyter notebook PROJECT_IMPLEMENTATION.ipynb
```
N.B: Update the CSV path in Cell 06 to point to `mpesa_synthetic.csv` on your machine.
