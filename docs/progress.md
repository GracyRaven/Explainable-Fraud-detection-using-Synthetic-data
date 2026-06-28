# Progress Log

## What Has Been Done
- Loaded and cleaned synthetic M-Pesa dataset (120,000 transactions, 2.92% fraud rate)
- Removed data leakage columns and performed feature engineering (22 columns, 3 new features)
- Applied VIF screening: all 20 features clean, none dropped
- Temporal train/test split: months 1–9 train, months 10–12 test
- Trained 6 models: standard and cost-sensitive versions of Logistic Regression, Random Forest and XGBoost
- Evaluated all models using F2-score and AUC-PR
- Completed SHAP analysis:amount_to_balance_ratio identified as top fraud predictor
- Spearman stability test confirmed: XGBoost ρ=0.9188, Random Forest ρ=0.9293. This means they are both stable
- Threshold tuning completed: LR cost-sensitive exceeded standard after tuning (F2: 0.6925 vs 0.6755)
- All outputs saved (plots and CSV files)

## What Is Planned Next
- Writing Chapter 4 (Results)
- Writing Chapter 5 (Evaluation, Validation AND Deployment)
- Writing Chapter 6 (Discussion, Conclusions AND Recommendations)
- Finalising and submitting the final report
