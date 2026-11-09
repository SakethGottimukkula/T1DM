"""Tests for prediction context and model/metrics endpoints."""

from datetime import datetime, timedelta
import math
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.features import FEATURE_COLUMNS, build_features_single
from app.main import app, get_store
import pandas as pd


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_48_readings(start_dt: datetime | None = None) -> list[dict]:
    base = start_dt or datetime(2020, 1, 15, 8, 0, 0)
    readings = []
    for i in range(48):
        ts = base + timedelta(minutes=5 * i)
        glucose = round(120.0 + 15.0 * math.sin(i / 5.0) + (i % 7), 1)
        readings.append({"timestamp": ts.isoformat(), "glucose": glucose})
    return readings


def test_predict_context_fields_and_statistics(client):
    health = client.get("/health").json()
    if health["models_loaded"] == 0:
        pytest.skip("No models loaded — skipping context test")

    readings = _make_48_readings()
    r = client.post("/v1/predict", json={"readings": readings, "horizon": "all"})
    assert r.status_code == 200
    data = r.json()

    assert "context" in data
    ctx = data["context"]

    # Basic counts and window span
    assert ctx["readings_submitted"] == 48
    assert ctx["window_readings"] == 24
    assert ctx["window_end"] == data["timestamp"]
    assert ctx["window_start"] == readings[24]["timestamp"]
    assert ctx["window_span_minutes"] == 115.0

    # Window glucose statistics over the last 24 readings
    last_24_gluc = [r["glucose"] for r in readings[24:]]
    assert abs(ctx["current_glucose"] - round(last_24_gluc[-1], 2)) < 0.01
    assert abs(ctx["window_min"] - round(min(last_24_gluc), 2)) < 0.01
    assert abs(ctx["window_max"] - round(max(last_24_gluc), 2)) < 0.01
    assert abs(ctx["window_mean"] - round(sum(last_24_gluc) / 24.0, 2)) < 0.01

    # Deltas
    assert abs(ctx["change_5min"] - round(last_24_gluc[-1] - last_24_gluc[-2], 2)) < 0.01
    assert abs(ctx["change_30min"] - round(last_24_gluc[-1] - last_24_gluc[-7], 2)) < 0.01

    # Features order and roll_mean_120 consistency
    assert list(ctx["features"].keys()) == FEATURE_COLUMNS
    assert abs(ctx["window_mean"] - ctx["features"]["roll_mean_120"]) < 0.01


def test_predictions_unchanged_matches_store_predict(client):
    health = client.get("/health").json()
    if health["models_loaded"] == 0:
        pytest.skip("No models loaded — skipping predictions equality test")

    readings = _make_48_readings()
    r = client.post("/v1/predict", json={"readings": readings, "horizon": "all"})
    assert r.status_code == 200
    data = r.json()

    # Compute features directly via build_features_single
    df = pd.DataFrame(readings)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_feat = build_features_single(df)
    last_row = {col: float(df_feat[col].iloc[-1]) for col in FEATURE_COLUMNS}

    store = get_store()
    for pred in data["predictions"]:
        hz = pred["horizon"]
        expected_raw = store.predict(last_row, hz)
        assert abs(pred["predicted_glucose"] - round(expected_raw, 2)) < 1e-4


def test_predict_csv_patient_id_in_context(client):
    health = client.get("/health").json()
    if health["models_loaded"] == 0:
        pytest.skip("No models loaded — skipping csv patient_id context test")

    readings = _make_48_readings()

    # 1. With patient_id column
    csv_lines_with_pid = ["timestamp,glucose,patient_id"]
    for r_item in readings:
        csv_lines_with_pid.append(f"{r_item['timestamp']},{r_item['glucose']},559")
    csv_with_pid = "\n".join(csv_lines_with_pid)

    resp_with_pid = client.post(
        "/v1/predict/csv",
        data={"horizon": "30"},
        files={"file": ("test.csv", csv_with_pid, "text/csv")},
    )
    assert resp_with_pid.status_code == 200
    assert resp_with_pid.json()["context"]["patient_id"] == "559"

    # 2. Without patient_id column
    csv_lines_no_pid = ["timestamp,glucose"]
    for r_item in readings:
        csv_lines_no_pid.append(f"{r_item['timestamp']},{r_item['glucose']}")
    csv_no_pid = "\n".join(csv_lines_no_pid)

    resp_no_pid = client.post(
        "/v1/predict/csv",
        data={"horizon": "30"},
        files={"file": ("test.csv", csv_no_pid, "text/csv")},
    )
    assert resp_no_pid.status_code == 200
    assert resp_no_pid.json()["context"]["patient_id"] is None


def test_models_endpoint_hashes_and_metrics_info(client):
    health = client.get("/health").json()
    if health["models_loaded"] == 0:
        pytest.skip("No models loaded — skipping models info test")

    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()

    for m in data["models"]:
        if m["loaded"]:
            assert m["sha256"] is not None
            assert len(m["sha256"]) == 64
            assert m["size_bytes"] is not None
            assert m["size_bytes"] > 0

    assert isinstance(data["metrics_status"], str)
    assert isinstance(data["metrics_warnings"], list)
    assert data["feature_importance"] is not None


def test_metrics_endpoint_real_and_missing(client, monkeypatch, tmp_path: Path):
    # 1. Real metrics.json exists
    r = client.get("/v1/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert isinstance(data["warnings"], list)
    assert "horizons" in data["metrics"]
    assert list(data["metrics"]["horizons"].keys()) == ["30", "60", "90", "120"]

    # 2. Monkeypatch METRICS_PATH to a missing file -> 503
    missing_file = tmp_path / "nonexistent_metrics.json"
    monkeypatch.setattr("app.main.METRICS_PATH", missing_file)

    r_missing = client.get("/v1/metrics")
    assert r_missing.status_code == 503
    assert "scripts/evaluate.py" in r_missing.text
