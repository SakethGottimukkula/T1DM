"""Feature engineering — mirrors the notebook exactly.

Feature columns (18 total, same for all horizons):
  lag_1, lag_2, lag_3, lag_6, lag_12,
  roll_mean_30, roll_std_30, roll_mean_60, roll_std_60,
  roll_mean_90, roll_std_90, roll_mean_120, roll_std_120,
  delta_5, delta_30,
  hour, hour_sin, hour_cos
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical feature column order — must match the notebook
FEATURE_COLUMNS: list[str] = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "roll_mean_30",
    "roll_std_30",
    "roll_mean_60",
    "roll_std_60",
    "roll_mean_90",
    "roll_std_90",
    "roll_mean_120",
    "roll_std_120",
    "delta_5",
    "delta_30",
    "hour",
    "hour_sin",
    "hour_cos",
]


def build_features(df: pd.DataFrame, *, group_col: str = "patient_id") -> pd.DataFrame:
    """Add all 18 feature columns + 4 targets, grouped by *group_col*.

    This is meant for the **training / backtest** path where multiple
    patients may be present in the same DataFrame.
    The DataFrame must already contain ``timestamp`` (datetime) and
    ``glucose`` (float) columns.

    Returns a copy — the original is not mutated.
    """

    df = df.copy()
    g = df.groupby(group_col)["glucose"]

    # -- Lags --
    df["lag_1"] = g.shift(1)
    df["lag_2"] = g.shift(2)
    df["lag_3"] = g.shift(3)
    df["lag_6"] = g.shift(6)
    df["lag_12"] = g.shift(12)

    # -- Rolling statistics --
    df["roll_mean_30"] = g.transform(lambda x: x.rolling(6).mean())
    df["roll_std_30"] = g.transform(lambda x: x.rolling(6).std())
    df["roll_mean_60"] = g.transform(lambda x: x.rolling(12).mean())
    df["roll_std_60"] = g.transform(lambda x: x.rolling(12).std())
    df["roll_mean_90"] = g.transform(lambda x: x.rolling(18).mean())
    df["roll_std_90"] = g.transform(lambda x: x.rolling(18).std())
    df["roll_mean_120"] = g.transform(lambda x: x.rolling(24).mean())
    df["roll_std_120"] = g.transform(lambda x: x.rolling(24).std())

    # -- Deltas --
    df["delta_5"] = df["glucose"] - df["lag_1"]
    df["delta_30"] = df["glucose"] - df["lag_6"]

    # -- Time-of-day --
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # -- Targets --
    df["target_30"] = g.shift(-6)
    df["target_60"] = g.shift(-12)
    df["target_90"] = g.shift(-18)
    df["target_120"] = g.shift(-24)

    return df


def build_features_single(df: pd.DataFrame) -> pd.DataFrame:
    """Build features for a *single-patient* series (no groupby).

    Used by the API inference path where readings come from one patient.
    Targets are NOT computed — only the 18 feature columns are added.
    """

    df = df.copy()
    g = df["glucose"]

    # -- Lags --
    df["lag_1"] = g.shift(1)
    df["lag_2"] = g.shift(2)
    df["lag_3"] = g.shift(3)
    df["lag_6"] = g.shift(6)
    df["lag_12"] = g.shift(12)

    # -- Rolling statistics --
    df["roll_mean_30"] = g.rolling(6).mean()
    df["roll_std_30"] = g.rolling(6).std()
    df["roll_mean_60"] = g.rolling(12).mean()
    df["roll_std_60"] = g.rolling(12).std()
    df["roll_mean_90"] = g.rolling(18).mean()
    df["roll_std_90"] = g.rolling(18).std()
    df["roll_mean_120"] = g.rolling(24).mean()
    df["roll_std_120"] = g.rolling(24).std()

    # -- Deltas --
    df["delta_5"] = df["glucose"] - df["lag_1"]
    df["delta_30"] = df["glucose"] - df["lag_6"]

    # -- Time-of-day --
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df
