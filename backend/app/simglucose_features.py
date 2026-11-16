"""Causal CGM, insulin, and carbohydrate features for SimGlucose models."""

from __future__ import annotations

import numpy as np
import pandas as pd

SIM_FEATURE_COLUMNS = [
    "cgm_lag_1", "cgm_lag_2", "cgm_lag_3", "cgm_lag_6", "cgm_lag_12",
    "cgm_mean_30", "cgm_std_30", "cgm_mean_60", "cgm_std_60",
    "cgm_mean_90", "cgm_std_90", "cgm_mean_120", "cgm_std_120",
    "cgm_delta_5", "cgm_delta_30", "hour", "hour_sin", "hour_cos",
    "insulin_5", "insulin_15", "insulin_30", "insulin_60", "insulin_120",
    "cho_5", "cho_15", "cho_30", "cho_60", "cho_120",
]


def build_sim_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create past-only predictors and future-BG targets from 5-minute rows."""
    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    g = out["cgm"].astype(float)
    out["cgm_lag_1"] = g.shift(1)
    out["cgm_lag_2"] = g.shift(2)
    out["cgm_lag_3"] = g.shift(3)
    out["cgm_lag_6"] = g.shift(6)
    out["cgm_lag_12"] = g.shift(12)
    for minutes in (30, 60, 90, 120):
        window = minutes // 5
        out[f"cgm_mean_{minutes}"] = g.rolling(window).mean()
        out[f"cgm_std_{minutes}"] = g.rolling(window).std()
    out["cgm_delta_5"] = g - out["cgm_lag_1"]
    out["cgm_delta_30"] = g - out["cgm_lag_6"]
    out["hour"] = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    for col in ("insulin", "cho"):
        signal = out[col].fillna(0).astype(float)
        for minutes in (5, 15, 30, 60, 120):
            out[f"{col}_{minutes}"] = signal.rolling(minutes // 5, min_periods=1).sum()
    for horizon in (30, 60, 90, 120):
        out[f"target_{horizon}"] = out["bg"].shift(-(horizon // 5))
    return out
