# BURN AI INSPECTOR ML Backend

Python FastAPI backend for the RUL upgrade requested by the SIH judge.

## Models

- SMOTE: balances the Warning vs Normal classification training set.
- Fisher discriminant: one-dimensional supervised separation feature.
- CatBoost: predicts Warning/Critical failure-risk probability.
- Random Forest Regressor: predicts RUL at the 168h inspection point.
- Isolation Forest: unsupervised anomaly detector.
- SHAP: explains CatBoost risk predictions.

## Dataset used

`data/training.csv` is the supplied 25,000-component frontend validation dataset. It contains:

- `Value_0h`, `Value_24h`, `Value_96h`, `Value_168h`
- trajectory drift features
- `Failure_Time_h`
- `RUL_at_168h`
- `Risk`

`RUL_at_168h` and other outcome columns are targets only and are never model inputs.

The classification target is `Risk != Normal` because `Failure_Flag_168h` is zero for all 25,000 rows in the supplied inference dataset. This gives the judge-requested imbalanced classification task (20,777 Normal vs 4,223 Warning).

## Run

```bash
cd backend
python -m pip install -r requirements.txt
python train.py
uvicorn main:app --reload --port 8000
```

The frontend should use `VITE_ML_API_URL=http://localhost:8000` during local development.
