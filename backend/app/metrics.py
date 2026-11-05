"""Evaluation metrics calculation for glucose predictions."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.ega import classify_array

ZONES = ["A", "B", "C", "D", "E"]


def summarize(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Compute RMSE, MAE, and Clarke EGA zone metrics for predictions vs actuals."""
    n_rows = len(actual)
    if n_rows == 0 or len(predicted) == 0:
        return {
            "n_rows": 0,
            "rmse": 0.0,
            "mae": 0.0,
            "zone_counts": {z: 0 for z in ZONES},
            "zone_percentages": {z: 0.0 for z in ZONES},
            "a_plus_b_pct": 0.0,
        }

    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)

    rmse = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2)
    mae = round(float(mean_absolute_error(y_true, y_pred)), 2)

    _, counts = classify_array(y_true.tolist(), y_pred.tolist())
    zone_counts = {z: int(counts.get(z, 0)) for z in ZONES}

    total = sum(zone_counts.values())
    zone_percentages = {
        z: round(zone_counts[z] / total * 100.0, 2) if total > 0 else 0.0
        for z in ZONES
    }

    a_plus_b = zone_counts["A"] + zone_counts["B"]
    a_plus_b_pct = round(a_plus_b / total * 100.0, 2) if total > 0 else 0.0

    return {
        "n_rows": n_rows,
        "rmse": rmse,
        "mae": mae,
        "zone_counts": zone_counts,
        "zone_percentages": zone_percentages,
        "a_plus_b_pct": a_plus_b_pct,
    }
