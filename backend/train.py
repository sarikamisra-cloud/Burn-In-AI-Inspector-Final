from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

from features import build_features, FEATURES

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "training.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATA)
    required = ["Component_ID", "Risk", "RUL_at_168h"] + [
        "Value_0h", "Value_24h", "Value_96h", "Value_168h"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required training columns: {missing}")

    X = build_features(df)
    # Judge-requested imbalanced classification target: Normal vs Warning.
    # The supplied 25k frontend dataset contains 20,777 Normal and 4,223 Warning.
    y_cls = (df["Risk"].astype(str).str.lower() != "normal").astype(int)
    y_rul = pd.to_numeric(df["RUL_at_168h"], errors="coerce")

    train_ids, test_ids = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_cls,
    )

    X_train = X.iloc[train_ids].copy()
    X_test = X.iloc[test_ids].copy()
    y_train = y_cls.iloc[train_ids].copy()
    y_test = y_cls.iloc[test_ids].copy()

    # Fisher/LDA is fitted only on the training partition.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    fisher = LinearDiscriminantAnalysis(n_components=1)
    fisher_train = fisher.fit_transform(X_train_scaled, y_train).ravel()
    fisher_test = fisher.transform(X_test_scaled).ravel()

    # SMOTE is applied ONLY to training data; test data remains untouched.
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    X_smote = X_smote.copy()
    X_smote["fisher_score"] = fisher.fit_transform(
        scaler.fit_transform(X_smote[FEATURES]), y_smote
    ).ravel()
    # Refit the persisted Fisher/scaler on the original training data so the
    # deployed transformation matches evaluation/inference.
    scaler = StandardScaler().fit(X_train)
    fisher = LinearDiscriminantAnalysis(n_components=1).fit(
        scaler.transform(X_train), y_train
    )
    X_train_aug = X_train.copy()
    X_test_aug = X_test.copy()
    X_train_aug["fisher_score"] = fisher.transform(scaler.transform(X_train)).ravel()
    X_test_aug["fisher_score"] = fisher.transform(scaler.transform(X_test)).ravel()
    X_smote = X_smote[FEATURES].copy()
    X_smote["fisher_score"] = fisher.transform(scaler.transform(X_smote[FEATURES])).ravel()

    cat_model = CatBoostClassifier(
        iterations=350,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="F1",
        random_seed=RANDOM_STATE,
        verbose=False,
    )
    cat_model.fit(X_smote, y_smote)

    cls_pred = cat_model.predict(X_test_aug).astype(int).ravel()
    cls_prob = cat_model.predict_proba(X_test_aug)[:, 1]

    # RUL model is a separate regression task. It is not SMOTE'd and does not
    # use RUL/Future labels as features.
    rul_model = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rul_model.fit(X_train, y_rul.iloc[train_ids])
    rul_pred = rul_model.predict(X_test)

    # Unsupervised anomaly detector trained without labels.
    iso_model = IsolationForest(
        n_estimators=120,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso_model.fit(X_train)
    iso_pred = iso_model.predict(X_test)
    iso_score = -iso_model.score_samples(X_test)

    # Distribution statistics for dashboard display.
    dist = {}
    for col in FEATURES:
        s = X[col]
        dist[col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "median": float(s.median()),
            "max": float(s.max()),
        }

    metrics = {
        "dataset_rows": int(len(df)),
        "components": int(df["Component_ID"].nunique()),
        "classification_target": "Warning vs Normal",
        "before_smote": {
            "normal": int((y_cls == 0).sum()),
            "warning": int((y_cls == 1).sum()),
            "warning_pct": float(y_cls.mean() * 100),
        },
        "after_smote": {
            "normal": int((y_smote == 0).sum()),
            "warning": int((y_smote == 1).sum()),
        },
        "catboost": {
            "accuracy": float(accuracy_score(y_test, cls_pred)),
            "precision": float(precision_score(y_test, cls_pred, zero_division=0)),
            "recall": float(recall_score(y_test, cls_pred, zero_division=0)),
            "f1": float(f1_score(y_test, cls_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, cls_prob)),
        },
        "rul_random_forest": {
            "mae_h": float(mean_absolute_error(y_rul.iloc[test_ids], rul_pred)),
            "rmse_h": float(mean_squared_error(y_rul.iloc[test_ids], rul_pred) ** 0.5),
            "r2": float(r2_score(y_rul.iloc[test_ids], rul_pred)),
        },
        "isolation_forest": {
            "test_anomaly_rate_pct": float((iso_pred == -1).mean() * 100),
            "mean_anomaly_score": float(np.mean(iso_score)),
        },
        "fisher": {
            "method": "Linear Discriminant Analysis (Fisher discriminant)",
            "n_components": 1,
        },
        "data_leakage": {
            "excluded_targets": ["Risk", "Failure_Time_h", "RUL_at_168h", "Failure_Flag_168h"],
            "split_unit": "component row (one row per component in this dataset)",
        },
        "distribution": dist,
        "rul_range_h": {
            "min": int(y_rul.min()),
            "max": int(y_rul.max()),
            "mean": float(y_rul.mean()),
        },
    }

    joblib.dump(cat_model, ART / "catboost_classifier.joblib")
    joblib.dump(rul_model, ART / "random_forest_rul.joblib")
    joblib.dump(iso_model, ART / "isolation_forest.joblib")
    joblib.dump(scaler, ART / "fisher_scaler.joblib")
    joblib.dump(fisher, ART / "fisher_lda.joblib")
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ART / "feature_names.json").write_text(json.dumps(FEATURES, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
