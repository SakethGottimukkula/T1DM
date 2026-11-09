"""Tests for render_readme_tables script."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.render_readme_tables import (
    build_table_1,
    build_table_2,
    build_table_3,
    build_table_4,
    render_all_tables,
)


@pytest.fixture
def sample_metrics_dict() -> dict:
    """Provide a synthetic metrics dictionary matching metrics.json schema."""
    return {
        "schema_version": "1.0",
        "n_train_rows": 10000,
        "n_eval_rows": 2500,
        "patients": ["559", "540", "544"],
        "horizons": {
            "30": {
                "overall": {
                    "rmse": 14.523,
                    "mae": 10.128,
                    "zone_percentages": {"A": 85.123, "B": 13.234, "C": 1.0, "D": 0.5, "E": 0.143},
                    "a_plus_b_pct": 98.357,
                },
                "per_patient": {
                    "540": {"rmse": 13.50, "a_plus_b_pct": 99.00},
                    "544": {"rmse": 15.20, "a_plus_b_pct": 98.00},
                    "559": {"rmse": 14.80, "a_plus_b_pct": 98.10},
                },
            },
            "60": {
                "overall": {
                    "rmse": 22.351,
                    "mae": 16.789,
                    "zone_percentages": {"A": 75.0, "B": 21.0, "C": 2.0, "D": 1.5, "E": 0.5},
                    "a_plus_b_pct": 96.00,
                },
                "per_patient": {
                    "540": {"rmse": 21.00, "a_plus_b_pct": 97.00},
                    "544": {"rmse": 23.00, "a_plus_b_pct": 95.50},
                    "559": {"rmse": 22.50, "a_plus_b_pct": 96.00},
                },
            },
        },
        "baselines": {
            "persistence": {
                "30": {"rmse": 16.12, "mae": 11.23, "a_plus_b_pct": 97.50},
                "60": {"rmse": 25.40, "mae": 18.90, "a_plus_b_pct": 94.00},
            },
            "ridge": {
                "30": {"rmse": 14.80, "mae": 10.40, "a_plus_b_pct": 98.10},
                "60": {"rmse": 22.80, "mae": 17.10, "a_plus_b_pct": 95.80},
            },
            "lasso": {
                "30": {"rmse": 14.90, "mae": 10.50, "a_plus_b_pct": 98.00},
                "60": {"rmse": 22.95, "mae": 17.20, "a_plus_b_pct": 95.70},
            },
        },
        "feature_importance": {
            "30": {
                "glucose_lag_0": 105.432,
                "glucose_roll_mean_6": 42.129,
                "glucose_roll_std_6": 25.781,
                "iob": 15.654,
                "hour_sin": 10.231,
                "cob": 5.123,
            },
            "60": {
                "glucose_lag_0": 85.123,
                "glucose_roll_mean_12": 52.341,
                "iob": 30.123,
                "glucose_roll_std_12": 22.456,
                "hour_sin": 12.345,
                "cob": 8.765,
            },
        },
    }


def test_build_table_1_format(sample_metrics_dict: dict) -> None:
    t1 = build_table_1(sample_metrics_dict)
    assert "### T1: XGBoost performance (test set, 2,500 rows)" in t1
    assert "| Horizon | RMSE (mg/dL) | MAE (mg/dL) | Clarke A (%) | Clarke B (%) | Clarke C (%) | Clarke D (%) | Clarke E (%) | A+B (%) |" in t1
    assert "| 30 min | 14.52 | 10.13 | 85.12 | 13.23 | 1.00 | 0.50 | 0.14 | 98.36 |" in t1
    assert "| 60 min | 22.35 | 16.79 | 75.00 | 21.00 | 2.00 | 1.50 | 0.50 | 96.00 |" in t1


def test_build_table_2_baselines_no_arima(sample_metrics_dict: dict) -> None:
    t2 = build_table_2(sample_metrics_dict)
    assert "### T2: Baseline comparison" in t2
    assert "| Horizon | Model | RMSE (mg/dL) | MAE (mg/dL) | A+B (%) |" in t2
    assert "| 30 min | XGBoost | 14.52 | 10.13 | 98.36 |" in t2
    assert "| 30 min | Persistence | 16.12 | 11.23 | 97.50 |" in t2
    assert "| 30 min | Ridge | 14.80 | 10.40 | 98.10 |" in t2
    assert "| 30 min | Lasso | 14.90 | 10.50 | 98.00 |" in t2
    # Ensure ARIMA is not present in the table
    assert "ARIMA" not in t2
    assert "arima" not in t2.lower()


def test_build_table_3_per_patient_sorting(sample_metrics_dict: dict) -> None:
    t3 = build_table_3(sample_metrics_dict)
    assert "### T3: Per-patient RMSE and A+B (%)" in t3
    assert "| Patient | 30 min | 60 min |" in t3
    lines = [line.strip() for line in t3.split("\n") if line.strip().startswith("| 5")]
    # Check numerical ordering: 540, 544, 559
    assert len(lines) == 3
    assert lines[0].startswith("| 540 | 13.50 / 99.00 | 21.00 / 97.00 |")
    assert lines[1].startswith("| 544 | 15.20 / 98.00 | 23.00 / 95.50 |")
    assert lines[2].startswith("| 559 | 14.80 / 98.10 | 22.50 / 96.00 |")


def test_build_table_4_feature_importance_top5(sample_metrics_dict: dict) -> None:
    t4 = build_table_4(sample_metrics_dict)
    assert "### T4: Feature importance (gain), top 5 per horizon" in t4
    assert "| Rank | 30 min | 60 min |" in t4
    # Rank 1 must have glucose_lag_0 for both
    assert "| 1 | `glucose_lag_0` (105.43) | `glucose_lag_0` (85.12) |" in t4
    # cob had rank 6 in 30 min, so it shouldn't appear
    assert "`cob`" not in t4


def test_render_all_tables_complete(sample_metrics_dict: dict) -> None:
    all_tables = render_all_tables(sample_metrics_dict)
    assert "### T1:" in all_tables
    assert "### T2:" in all_tables
    assert "### T3:" in all_tables
    assert "### T4:" in all_tables
