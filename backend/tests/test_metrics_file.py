"""Tests for backend/models/metrics.json structure and validation."""

import json
from pathlib import Path
import pytest

_METRICS_PATH = Path(__file__).resolve().parent.parent / "models" / "metrics.json"


def test_metrics_json():
    if not _METRICS_PATH.exists():
        pytest.skip("backend/models/metrics.json does not exist")

    with open(_METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("schema_version") == 1
    horizons = data.get("horizons", {})
    assert set(horizons.keys()) == {"30", "60", "90", "120"}

    n_eval_rows = data.get("n_eval_rows")
    assert isinstance(n_eval_rows, int)

    feature_importance = data.get("feature_importance", {})
    feature_columns = data.get("feature_columns", [])
    assert len(feature_columns) == 18

    for hz in ["30", "60", "90", "120"]:
        hz_data = horizons[hz]
        assert "overall" in hz_data
        assert "per_patient" in hz_data

        overall_rows = hz_data["overall"].get("n_rows")
        assert overall_rows == n_eval_rows

        per_patient = hz_data["per_patient"]
        sum_patient_rows = sum(p["n_rows"] for p in per_patient.values())
        assert sum_patient_rows == overall_rows

        # Check feature importance for this horizon
        hz_fi = feature_importance.get(hz, {})
        for col in feature_columns:
            assert col in hz_fi
