from pathlib import Path
import os
from typing import Any
import json
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient

from features import build_features, FEATURES

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

app = FastAPI(title="BURN AI INSPECTOR ML Engine", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

cat_model = None
rul_model = None
iso_model = None
fisher_scaler = None
fisher_lda = None
shap_explainer = None
metrics = {}

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017/")

def mongo_collection():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return client, client["burn_ai_inspector"]["components"]


def load_artifacts():
    global cat_model, rul_model, iso_model, fisher_scaler, fisher_lda, shap_explainer, metrics
    required = [
        ART / "catboost_classifier.joblib",
        ART / "random_forest_rul.joblib",
        ART / "isolation_forest.joblib",
        ART / "fisher_scaler.joblib",
        ART / "fisher_lda.joblib",
        ART / "metrics.json",
    ]
    if not all(p.exists() for p in required):
        raise RuntimeError("ML artifacts are missing. Run: python train.py")
    cat_model = joblib.load(ART / "catboost_classifier.joblib")
    rul_model = joblib.load(ART / "random_forest_rul.joblib")
    iso_model = joblib.load(ART / "isolation_forest.joblib")
    fisher_scaler = joblib.load(ART / "fisher_scaler.joblib")
    fisher_lda = joblib.load(ART / "fisher_lda.joblib")
    shap_explainer = shap.TreeExplainer(cat_model)
    metrics = json.loads((ART / "metrics.json").read_text())


@app.on_event("startup")
def startup():
    load_artifacts()


@app.get("/health")
def health():
    return {
        "status": "online",
        "service": "BURN AI INSPECTOR ML Engine",
        "models": [
            "CatBoost Classifier",
            "Random Forest RUL Regressor",
            "Isolation Forest",
            "Fisher Discriminant",
            "SHAP",
        ],
    }


def _predict(frame: pd.DataFrame) -> list[dict[str, Any]]:
    X = build_features(frame)
    X_aug = X.copy()
    fisher_score = fisher_lda.transform(fisher_scaler.transform(X)).ravel()
    X_aug["fisher_score"] = fisher_score

    probability = cat_model.predict_proba(X_aug)[:, 1]
    classification = (probability >= 0.5).astype(int)
    rul = np.maximum(0.0, rul_model.predict(X))

    # Random-forest tree spread is a useful uncertainty band, not a formal CI.
    tree_preds = np.vstack([tree.predict(X) for tree in rul_model.estimators_])
    rul_low = np.maximum(0.0, np.percentile(tree_preds, 10, axis=0))
    rul_high = np.maximum(0.0, np.percentile(tree_preds, 90, axis=0))

    anomaly = iso_model.predict(X)
    anomaly_score = -iso_model.score_samples(X)

    out = []
    values = None
    # SHAP is deliberately calculated only for small/single-component calls.
    # Batch prediction stays fast enough for 25,000-row uploads.
    if len(frame) <= 5:
        shap_values = shap_explainer(X_aug)
        values = np.asarray(shap_values.values)
        if values.ndim == 3:
            values = values[:, :, 1]

    for i, row in frame.reset_index(drop=True).iterrows():
        feature_impacts = []
        if values is not None:
            feature_impacts = sorted(
                [
                    {"feature": FEATURES[j], "impact": float(values[i, j])}
                    for j in range(len(FEATURES))
                ],
                key=lambda z: abs(z["impact"]),
                reverse=True,
            )[:5]
        current_time = float(row.get("Current_Time_h", 168) or 168)
        pred_life = current_time + float(rul[i])
        risk = "Warning" if classification[i] else "Normal"
        if probability[i] >= 0.80:
            risk = "Critical"
        elif probability[i] >= 0.50:
            risk = "Warning"

        out.append({
            "component_id": str(row.get("Component_ID", row.get("component_id", f"row-{i}"))),
            "lot_id": str(row.get("Lot_ID", row.get("lot_id", "—"))),
            "failure_probability": float(probability[i]),
            "risk": risk,
            "rul_h": float(rul[i]),
            "predicted_lifetime_h": pred_life,
            "rul_low_h": float(rul_low[i]),
            "rul_high_h": float(rul_high[i]),
            "current_time_h": current_time,
            "anomaly": bool(anomaly[i] == -1),
            "anomaly_score": float(anomaly_score[i]),
            "fisher_score": float(fisher_score[i]),
            "shap": feature_impacts,
        })
    return out


@app.post("/predict")
def predict(payload: dict[str, Any]):
    try:
        frame = pd.DataFrame([payload])
        return _predict(frame)[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict-batch")
def predict_batch(payload: dict[str, Any]):
    try:
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("payload.rows must be a non-empty list")
        frame = pd.DataFrame(rows)
        return {"results": _predict(frame)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/components")
def get_components():
    try:
        client, collection = mongo_collection()
        docs = list(collection.find({}, {"_id": 0}).limit(25000))
        client.close()
        return docs
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/components/bulk")
def save_components(components: list[dict[str, Any]]):
    try:
        if not components:
            raise ValueError("No component data received")
        client, collection = mongo_collection()
        collection.delete_many({})
        collection.insert_many(components, ordered=False)
        client.close()
        return {"success": True, "inserted": len(components)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics")
def get_metrics():
    return metrics
