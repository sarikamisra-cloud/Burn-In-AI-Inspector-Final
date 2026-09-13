import numpy as np
import pandas as pd

FEATURES = [
    "value_0h",
    "value_24h",
    "value_96h",
    "value_168h",
    "drift_0_24_per_h",
    "drift_24_96_per_h",
    "drift_96_168_per_h",
    "delta_0_168",
    "curvature",
    "relative_growth",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the exact feature matrix used by both training and inference.

    The target columns (Risk, Failure_Time_h, RUL_at_168h) are never used as
    model inputs. This keeps the RUL and classification predictions leakage-free.
    """
    x = pd.DataFrame(index=df.index)
    x["value_0h"] = pd.to_numeric(df["Value_0h"], errors="coerce")
    x["value_24h"] = pd.to_numeric(df["Value_24h"], errors="coerce")
    x["value_96h"] = pd.to_numeric(df["Value_96h"], errors="coerce")
    x["value_168h"] = pd.to_numeric(df["Value_168h"], errors="coerce")

    # Prefer supplied trajectory rates from the dataset; recompute if absent.
    x["drift_0_24_per_h"] = pd.to_numeric(
        df.get("Drift_0_24_per_h", (x.value_24h - x.value_0h) / 24),
        errors="coerce",
    )
    x["drift_24_96_per_h"] = pd.to_numeric(
        df.get("Drift_24_96_per_h", (x.value_96h - x.value_24h) / 72),
        errors="coerce",
    )
    x["drift_96_168_per_h"] = pd.to_numeric(
        df.get("Drift_96_168_per_h", (x.value_168h - x.value_96h) / 72),
        errors="coerce",
    )

    x["delta_0_168"] = x.value_168h - x.value_0h
    x["curvature"] = x.drift_96_168_per_h - x.drift_0_24_per_h
    x["relative_growth"] = x.delta_0_168 / np.maximum(np.abs(x.value_0h), 1e-6)

    return x[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
