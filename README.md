# BURN AI INSPECTOR — SIH 2026 PS 26170

Early-warning reliability intelligence for component burn-in screening, upgraded with a real Python ML backend and Remaining Useful Life (RUL) prediction.

## Frontend

React + Vite dashboard. It loads the supplied 25,000-component dataset from `public/BURN_AI_INSPECTOR_FRONTEND_25000.csv` and sends component batches to the FastAPI ML backend.

## ML backend

`backend/` contains the training and inference service:

- SMOTE — handles Normal/Warning class imbalance on training data only.
- Fisher discriminant — supervised feature separation.
- CatBoost — failure-risk classification.
- Random Forest Regressor — RUL prediction at the 168h inspection point.
- Isolation Forest — unsupervised anomaly detection.
- SHAP — explanation for the selected component's CatBoost prediction.
- Model A — deterministic safety-limit guard remains independent of ML.
- Existing OLS and robust statistical anomaly engines remain as transparent baselines.

## Dataset

The supplied 25,000-component frontend dataset contains `Value_0h`, `Value_24h`, `Value_96h`, `Value_168h`, drift features, `Failure_Time_h`, `RUL_at_168h`, and `Risk`. The classifier target is `Risk != Normal` because `Failure_Flag_168h` is 0 for all rows. The RUL target is `RUL_at_168h`.

## Run

Terminal 1:

```bash
cd backend
python -m pip install -r requirements.txt
python train.py
python -m uvicorn main:app --reload --port 8000
```

Terminal 2:

```bash
npm install
npm run dev
```

For deployment, set:

```text
VITE_ML_API_URL=https://YOUR-BACKEND-URL
```

## Current CSV upload format
The uploader accepts the time-series BURN AI INSPECTOR dataset with columns:
`Component_ID, Lot_ID, Time_h, Temperature_C, Applied_Voltage_V, Leakage_Current_mA, Supply_Current_mA, Functional_Frequency_MHz, Propagation_Delay_ns, Health_Score_pct, Anomaly_Score, Failure_Flag, Failure_Time_h, RUL_h, Risk_Class`.
It aggregates each component's 0h/24h/96h/168h leakage measurements for the existing frontend and sends the resulting component features to the ML engine. RUL, failure time, and risk labels are retained as validation targets and are not used as inference inputs.

### Run
Terminal 1 (ML backend):
`cd backend && python3 -m pip install -r requirements.txt && uvicorn main:app --reload --port 8000`

Terminal 2 (frontend):
`npm install && npm run dev`

MongoDB is optional for the UI/ML flow. If MongoDB is running locally on `127.0.0.1:27017`, uploaded normalized components are also persisted in database `burn_ai_inspector`, collection `components`. Set `MONGODB_URI` if using Atlas.
