# ML/RUL Upgrade — Judge Requirements

## Dataset used
`backend/data/training.csv` and `public/BURN_AI_INSPECTOR_FRONTEND_25000.csv` are the supplied 25,000-component dataset. It contains the 0h/24h/96h/168h values, drift features, `Failure_Time_h`, `RUL_at_168h`, and `Risk`.

The supplied `Failure_Flag_168h` is 0 for all 25,000 rows, so the classifier does not fabricate a failure label. Instead, the imbalanced supervised target is `Risk != Normal` (Normal vs Warning).

## Implemented pipeline
1. Train/test split.
2. Fisher discriminant fitted on training data.
3. SMOTE applied only to training data.
4. CatBoost classifier predicts failure-risk probability.
5. Random Forest Regressor predicts `RUL_at_168h`.
6. Isolation Forest provides unsupervised anomaly detection.
7. SHAP explains CatBoost predictions for the selected component.
8. Model A remains the hard 168h safety guard.
9. Existing OLS and robust statistical anomaly calculations remain as transparent baselines.

## Held-out metrics from the supplied dataset
- Normal: 20,777; Warning: 4,223 (16.9% Warning before SMOTE).
- After SMOTE: 16,622 / 16,622.
- CatBoost Accuracy: 96.76%
- CatBoost Precision: 85.03%
- CatBoost Recall: 98.11%
- CatBoost F1: 91.10%
- CatBoost ROC-AUC: 99.57%
- Random Forest RUL MAE: 13.75 h
- Random Forest RUL RMSE: 17.81 h
- Random Forest R²: 0.954

These are computed metrics from the 20% held-out test split; they are not hard-coded into the model.
