# Explainable Fraud Detection for M-Pesa Using Machine Learning and SHAP

**Author:** Gracy Kisia - BSc Data Science and Analytics

This project explored explainable machine learning for fraud detection in Kenya's M-Pesa mobile money platform. Using a synthetic dataset of 120,000 transactions with a fraud rate of 2.92%, it compared standard and cost-sensitive versions of Logistic Regression, Random Forest, and XGBoost models to evaluate their ability to detect fraudulent transactions. The project focused not only on predictive performance but also on model transparency by using SHAP to identify the key transaction features influencing fraud predictions. In addition, Spearman rank correlation was used to compare SHAP feature importance rankings between standard and cost-sensitive models, helping determine whether cost-sensitive learning changed the ordering of important features. The overall aim was to develop a fraud detection approach that balanced accuracy, interpretability, and reliability for mobile money fraud analysis.

## Dataset

| Detail | Value |
|---|---|
| Source | [Kaggle: M-Pesa Transactions Fraud](https://www.kaggle.com/datasets/calebboen/mpesa-transactions-fraud/data) |
| Size | 120,000 transactions × 13 columns, Jan–Dec 2026 |
| Fraud rate | ~2.92% (3,504 fraudulent transactions) |
| Missing values | None |

## Dataset License

The dataset is dedicated to the public domain under the CC0 1.0 Universal (CC0 1.0) license, permitting unrestricted use, modification, and distribution without requiring permission or attribution.

## Objectives

1. Compared standard and cost-sensitive machine learning models using F2-score and AUC-PR.
2. Used SHAP to explain the transaction features driving fraud predictions.
3. Evaluated whether cost-sensitive learning altered SHAP feature importance rankings using Spearman rank correlation.

## Methodology

Leakage-related columns (sender_balance_after, receiver_balance_after, and transaction_id) were removed to prevent the models from learning information that would not be available during real-world fraud detection.

Categorical variables were transformed using one-hot encoding, and three new features were engineered: amount_to_balance_ratio, is_late_night, and low_sender_balance. After preprocessing and feature engineering, the resulting 20 features were evaluated for multicollinearity using Variance Inflation Factor (VIF), with the highest VIF recorded at 3.73, below the threshold of 10.

A temporal data split was applied, using months 1–9 for training and months 10–12 for testing, to simulate real-world deployment and reduce data leakage risks. Six models, consisting of standard and cost-sensitive versions of Logistic Regression, Random Forest, and XGBoost, were trained and interpreted using SHAP TreeExplainer.

SHAP explanations were compared using matched sample rows, with Spearman rank correlation used to assess the stability of feature importance rankings. Finally, F2-score threshold tuning was applied to optimize model performance, with greater emphasis placed on correctly identifying fraudulent transactions.

## Project Summary

This project demonstrated an explainable machine learning approach to fraud detection in M-Pesa transactions. It combined cost-sensitive learning, SHAP explanations, and Spearman rank correlation to evaluate model performance and interpretability while addressing the imbalanced transaction data. A fraud_risk_console.html was also developed to provide a simple interface for viewing fraud-risk predictions and explanations.

## Project Structure

```text
Explainable-Fraud-detection-using-Synthetic-data/
├── Data/
├── docs/
├── outputs/
├── PROJECT IMPLEMENTATION.ipynb
├── requirements.txt
├── .gitignore
├── README.md
├── fraud_risk_console.html
```

## How to Run

```bash
git clone https://github.com/GracyRaven/Explainable-Fraud-detection-using-Synthetic-data.git
cd Explainable-Fraud-detection-using-Synthetic-data
pip install -r requirements.txt
jupyter notebook PROJECT_IMPLEMENTATION.ipynb
```
N.B: Update the CSV path in Cell 06 to point to `mpesa_synthetic.csv` on your machine.
