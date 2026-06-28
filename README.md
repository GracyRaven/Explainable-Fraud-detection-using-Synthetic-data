# Explainable Fraud Detection for M-Pesa Using Machine Learning and SHAP

## Project Description
This project develops an explainable fraud detection system for Kenya's M-Pesa mobile money platform. Using a synthetic dataset of 120,000 transactions with a 2.92% fraud rate, the study compares standard and cost-sensitive machine learning models of Logistic Regression, Random Forest, and XGBoost and uses SHAP (SHapley Additive exPlanations) to identify which transaction features drive fraud predictions. A key contribution is formally testing whether cost-sensitive weighting distorts SHAP feature importance rankings using Spearman rank correlation.

## Author
Gracy B. Kisia - BSc Data Science and Analytics

## Dataset
| Detail | Value |
|---|---|
| Name | Synthetic M-Pesa Transaction Dataset |
| Source | Kaggle |
| Size | 120,000 transactions × 13 columns |
| Months covered | 12 months (January–December 2026) |
| Fraud rate | 2.92% (3,504 fraudulent transactions) |
| Missing values | None |
| Location | mpesa_synthetic.csv (https://www.kaggle.com/datasets/calebboen/mpesa-transactions-fraud/data) |

## Research Objectives
1. Comparing standard vs. cost-sensitive machine learning models for M-Pesa fraud detection using F2-score and AUC-PR evaluation metrics
2. Using SHAP to identify which transaction features drive fraud predictions
3. Formally test whether cost-sensitive weighting distorts SHAP feature importance rankings using Spearman rank correlation (stability threshold: ρ ≥ 0.80)

## Methodology

| Step | Detail |
|---|---|
| Data cleaning | Removed sender_balance_after and receiver_balance_after (data leakage) and transaction_id (non-predictive) |
| Feature engineering | One-hot encoded categoricals; engineered amount_to_balance_ratio, is_late_night, low_sender_balance |
| Multicollinearity check | Variance Inflation Factor (VIF) — all 20 features below threshold of 10, none dropped |
| Train/test split | Temporal — months 1–9 train (90,165 rows), months 10–12 test (29,835 rows) |
| Class imbalance | Cost-sensitive learning via class weighting — not SMOTE |
| Models | Logistic Regression, Random Forest, XGBoost — standard and cost-sensitive variants (6 models total) |
| Explainability | SHAP TreeExplainer on same sampled rows for fair comparison |
| Stability test | Spearman rank correlation between standard and cost-sensitive SHAP feature rankings |
| Threshold tuning | Precision-recall curve used to find optimal F2 threshold for each cost-sensitive model |

## Results

### Model Performance (Default 0.5 Threshold)

| Model | F2 Score | AUC-PR | TP | FP | FN | TN | Fraud Caught |
|---|---|---|---|---|---|---|---|
| LR Standard | 0.6755 | 0.6857 | 540 | 5 | 323 | 28967 | 62.6% |
| LR Cost-Sensitive | 0.5902 | 0.6849 | 576 | 852 | 287 | 28120 | 66.7% |
| RF Standard | **0.7043** | 0.6796 | 566 | **0** | 297 | 28972 | 65.6% |
| RF Cost-Sensitive | **0.7043** | 0.6818 | 566 | **0** | 297 | 28972 | 65.6% |
| XGB Standard | 0.7022 | 0.6866 | 564 | **0** | 299 | 28972 | 65.4% |
| XGB Cost-Sensitive | 0.6578 | 0.6879 | 572 | 324 | 291 | 28648 | 66.3% |

**Best model: Random Forest Standard — F2 = 0.7043, zero false positives**

### VIF Screening Results
All 20 features passed — highest VIF was 3.73 (hour), well below the threshold of 10. No features were dropped.

### SHAP Feature Importance — Top 10

**XGBoost Cost-Sensitive:**

| Rank | Feature | Mean |SHAP| | Interpretation |
|---|---|---|---|
| 1 | amount_to_balance_ratio | 1.4328 | Accounts being drained — strongest fraud signal |
| 2 | receiver_balance_before | 0.4387 | Receiver account pattern |
| 3 | amount | 0.3738 | Raw transaction size |
| 4 | sender_balance_before | 0.3650 | Sender financial position |
| 5 | hour | 0.2681 | Time of day |
| 6 | device_type_smartphone | 0.0891 | Device used |
| 7 | region_Kisumu | 0.0592 | Geographic signal |
| 8 | day_of_week_Sat | 0.0585 | Weekend pattern |
| 9 | day_of_week_Sun | 0.0564 | Weekend pattern |
| 10 | transaction_type_peer | 0.0561 | Peer-to-peer transfers |

Both models independently agreed on the same top 5 features. The engineered feature amount_to_balance_ratio outperformed all raw dataset features in both models.

### Spearman Stability Test

| Model | Spearman ρ | p-value | Decision |
|---|---|---|---|
| XGBoost | 0.9188 | < 0.0001 | Stable ✅ |
| Random Forest | 0.9293 | < 0.0001 | Stable ✅ |

Both values exceed the 0.80 stability threshold. Cost-sensitive weighting does NOT distort SHAP feature importance rankings.

### Threshold Tuning Results

| Model | Default F2 | Tuned Threshold | Tuned F2 | Improvement |
|---|---|---|---|---|
| Logistic Regression Cost-Sensitive | 0.5902 | 0.893 | 0.6925 | +0.1023 |
| Random Forest Cost-Sensitive | 0.7043 | 0.660 | 0.7043 | +0.0000 |
| XGB Cost-Sensitive | 0.6578 | 0.790 | 0.7028 | +0.0450 |

After tuning, LR cost-sensitive (0.6925) exceeded LR standard (0.6755). RF and XGB cost-sensitive matched standard model performance.

## Key Findings

1. **Best model:** Random Forest Standard achieved the highest F2 score (0.7043) with zero false positives
2. **Top fraud predictor:** amount_to_balance_ratio — an engineered feature — was the single most important predictor in both models (SHAP value 1.43 for XGBoost)
3. **SHAP stability confirmed:** Spearman ρ of 0.9188 and 0.9293 — both well above 0.80 — proving cost-sensitive learning and model interpretability are compatible
4. **Threshold matters:** Cost-sensitive models require threshold calibration beyond the default 0.5 cutoff to demonstrate their full benefit

## Hypotheses Summary

| Hypothesis | Status | Evidence |
|---|---|---|
| H1: Cost-sensitive models outperform standard | ⚠️ Partially confirmed | Standard won at default threshold; cost-sensitive matched/exceeded after tuning |
| H2: SHAP identifies meaningful fraud features | ✅ Confirmed | amount_to_balance_ratio dominant in both models |
| H3: SHAP rankings stable under cost-sensitive weighting | ✅ Confirmed | ρ = 0.9188 and 0.9293, both above 0.80 |

## Current Status
- ✅ Data pipeline complete
- ✅ All six models trained and evaluated
- ✅ SHAP analysis completed
- ✅ Spearman stability test confirmed
- ✅ Threshold tuning completed
- ✅ All outputs saved
- 📝 Currently finalising written report

## Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── PROJECT_IMPLEMENTATION.ipynb
├── mpesa_synthetic.csv
├── mpesa_engineered.csv
├── outputs/
│   ├── full_results.csv
│   ├── spearman_results.csv
│   ├── shap_importance_comparison.csv
│   ├── threshold_tuning_results.csv
│   ├── shap_xgb_summary.png
│   ├── shap_rf_summary.png
│   ├── shap_xgb_bar.png
│   ├── spearman_comparison.png
│   ├── confusion_matrices_all.png
│   ├── f2_comparison.png
│   ├── f2_tuned_comparison.png
│   └── threshold_tradeoff.png
└── docs/
    └── progress.md
```


