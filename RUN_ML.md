# BURN AI INSPECTOR — ML/RUL upgrade

## 1. Start backend

```bash
cd backend
python -m pip install -r requirements.txt
python train.py
python -m uvicorn main:app --reload --port 8000
```

## 2. Start frontend

Open another terminal:

```bash
npm install
npm run dev
```

Set `VITE_ML_API_URL=http://localhost:8000` in `.env.local` if needed.

## What the upgrade now demonstrates

- **SMOTE** balances the imbalanced Warning/Normal training classes.
- **Fisher discriminant** provides a supervised separation score.
- **CatBoost** predicts failure-risk probability.
- **Random Forest Regressor** predicts RUL at 168h.
- **Isolation Forest** flags unsupervised anomalies.
- **SHAP** explains the selected component's CatBoost prediction.
- **Model A** remains the hard safety-limit guard.
- Existing OLS and robust statistical anomaly logic remain as transparent baselines.

## Supplied dataset

`backend/data/training.csv` is the supplied 25,000-component dataset with `RUL_at_168h`, `Failure_Time_h`, and `Risk`. The frontend also loads the same dataset automatically from `public/BURN_AI_INSPECTOR_FRONTEND_25000.csv`.

The classifier target is `Risk != Normal`, because `Failure_Flag_168h` is 0 for all 25,000 rows. This creates the imbalanced classification problem the judge asked you to address without inventing positive 168h failure labels.
