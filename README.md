# Explainable Fraud Detection for M-Pesa Using Machine Learning and SHAP

This project investigates whether cost-sensitive machine learning can detect 
M-Pesa mobile money fraud more effectively than standard models, while keeping 
SHAP-based explanations stable and trustworthy enough for real-world regulatory use.

## Author
Gracy Betty Kisia - BSc Data Science and Analytics

## Dataset
- **Name:** Synthetic M-Pesa Transaction Dataset
- **Source:** Kaggle
- **Link:** (https://www.kaggle.com/datasets/calebboen/mpesa-transactions-fraud/data)
- **Size:** 120,000 transactions across 12 months
- **Fraud rate:** ~2.9%
- **Location:** mpesa_synthetic.csv

## Overview
Mobile money platforms like M-Pesa process millions of transactions daily, 
with fraud typically under 3% of volume. This creates two challenges: severe 
class imbalance, and the need for models that can explain why a transaction 
was flagged, not just flag it.

This project compares standard and cost-sensitive versions of three algorithms: Logistic Regression, Random Forest, and XGBoost, then uses SHAP to test 
whether cost-sensitive weighting changes which features the model relies on.

## Research Objectives
1. Compare standard vs. cost-sensitive machine learning models for M-Pesa 
   fraud detection using the F2-score and AUC-PR evaluation metrics
2. Use SHAP to identify which transaction features drive fraud predictions
3. Test whether cost-sensitive weighting distorts SHAP feature importance 
   rankings, using Spearman rank correlation

## Methodology

| Step | Approach |
|---|---|
| Data cleaning | Removed leakage columns and transaction ID |
| Feature engineering | One-hot encoding of categoricals; engineered ratio and time-based features |
| Multicollinearity check | Variance Inflation Factor (VIF) |
| Train/test split | Temporal split.months 1–9 will be used for training, months 10–12 will be used for testing |
| Class imbalance | Cost-sensitive learning via class weighting |
| Models | The standard and cost-sensitive variants of Logistic Regression, Random Forest, XGBoost |
| Explainability | SHAP TreeExplainer for tree-based models |
| Stability test | Spearman rank correlation between standard and cost-sensitive SHAP feature rankings |

## Current Status
All three research objectives have been completed:
- All six models were trained and evaluated
- SHAP analysis completed. The amount_to_balance_ratio feature was identified as top fraud predictor as shown in the python code given.
- Spearman stability test confirmed: XGBoost ρ=0.9188, Random Forest ρ=0.9293
- Threshold tuning completed.The Logistic Regression cost-sensitive exceeded standard after tuning
- Currently finalising the final written report

## Key Results

| Model | F2 Score | AUC-PR |
|---|---|---|
| LR Standard | 0.6755 | 0.6854 |
| LR Cost-Sensitive | 0.5902 | 0.6849 |
| RF Standard | 0.7043 | 0.6796 |
| RF Cost-Sensitive | 0.7043 | 0.6818 |
| XGB Standard | 0.7022 | 0.6866 |
| XGB Cost-Sensitive | 0.6578 | 0.6879 |

**Best model:** Random Forest Standard (F2 = 0.7043, zero false positives). This will be shown through the python code given.

**SHAP Stability (Spearman ρ):**
- XGBoost: 0.9188 ✅ Stable
- Random Forest: 0.9293 ✅ Stable
