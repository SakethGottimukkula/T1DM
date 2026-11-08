"""Tests for timestamp policy, gap policy, validation order, and patient selection."""

import io
from datetime import datetime, timedelta
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.predictor import ModelStore


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan so ModelStore is initialised."""
    with TestClient(app) as c:
        yield c


def _make_readings(n: int = 48, start_glucose: float = 120.0) -> list[dict]:
    """Generate *n* synthetic 5-minute-spaced readings as dicts."""
    base = datetime(2020, 1, 15, 8, 0, 0)
    readings = []
    for i in range(n):
        ts = base + timedelta(minutes=5 * i)
        glucose = start_glucose + (i % 10) * 2 - 10
        readings.append(
            {
                "timestamp": ts.isoformat(),
                "glucose": round(max(50.0, min(glucose, 350.0)), 1),
            }
        )
    return readings


def _make_csv(rows: list[dict]) -> str:
    """Helper to convert list of dicts to CSV string."""
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in headers))
    return "\n".join(lines)


# ===========================================================================
# 1. Tests that need no models
# ===========================================================================


def test_duplicate_timestamps_early_part_rejected(client):
    """Duplicate timestamps anywhere in the input must raise 422."""
    readings = _make_readings(n=48)
    readings[5]["timestamp"] = readings[4]["timestamp"]
    r = client.post("/v1/predict", json={"readings": readings, "horizon": 30})
    assert r.status_code == 422
    assert "Timestamps must be unique." in r.text


def test_gap_inside_last_24_readings_rejected(client):
    """A gap inside the last 24 readings must be rejected with 422 and descriptive message."""
    base = datetime(2020, 1, 15, 8, 0, 0)
    readings = []
    for i in range(48):
        # Introduce a 20-minute gap at index 35 (inside last 24 rows)
        offset = 5 * i if i < 35 else 5 * i + 15
        ts = base + timedelta(minutes=offset)
        readings.append({"timestamp": ts.isoformat(), "glucose": 120.0})

    r = client.post("/v1/predict", json={"readings": readings, "horizon": 30})
    assert r.status_code == 422
    data = r.json()
    assert "Found 1 gap(s) outside 3–7 min" in data["detail"]
    assert "Only the most recent 24 readings (2 hours) are checked" in data["detail"]


def test_fewer_than_24_readings_rejected(client):
    """Fewer than 24 readings must be rejected with 422."""
    readings = _make_readings(n=15)
    r = client.post("/v1/predict", json={"readings": readings, "horizon": 30})
    assert r.status_code == 422
    assert "At least 24 readings required" in r.text


def test_backtest_missing_columns_rejected(client):
    """Backtest CSV missing required columns must return 422."""
    csv_data = "timestamp,glucose\n2020-01-15T08:00:00,120\n"
    r = client.post(
        "/v1/backtest",
        data={"horizon": 30},
        files={"file": ("test.csv", csv_data, "text/csv")},
    )
    assert r.status_code == 422
    assert "CSV must contain 'timestamp', 'glucose', and 'patient_id' columns" in r.text


def test_backtest_unparseable_timestamp_rejected(client):
    """Backtest CSV with unparseable timestamps must return 422."""
    csv_data = "timestamp,glucose,patient_id\ninvalid_ts,120,559\n"
    r = client.post(
        "/v1/backtest",
        data={"horizon": 30},
        files={"file": ("test.csv", csv_data, "text/csv")},
    )
    assert r.status_code == 422
    assert "Invalid timestamp 'invalid_ts'" in r.text


def test_predict_csv_multi_patient_without_patient_id_rejected(client):
    """Predict CSV with multiple patients and no patient_id specified must return 422."""
    rows = []
    base = datetime(2020, 1, 15, 8, 0, 0)
    for i in range(30):
        ts = (base + timedelta(minutes=5 * i)).isoformat()
        rows.append({"timestamp": ts, "glucose": 120.0, "patient_id": "559"})
        rows.append({"timestamp": ts, "glucose": 130.0, "patient_id": "563"})
    csv_content = _make_csv(rows)

    r = client.post(
        "/v1/predict/csv",
        data={"horizon": 30},
        files={"file": ("multi.csv", csv_content, "text/csv")},
    )
    assert r.status_code == 422
    assert "Multiple patients found in CSV" in r.text


def test_predict_csv_unknown_patient_id_rejected(client):
    """Predict CSV with an unknown patient_id must return 422 listing available IDs."""
    rows = []
    base = datetime(2020, 1, 15, 8, 0, 0)
    for i in range(30):
        ts = (base + timedelta(minutes=5 * i)).isoformat()
        rows.append({"timestamp": ts, "glucose": 120.0, "patient_id": "559"})
    csv_content = _make_csv(rows)

    r = client.post(
        "/v1/predict/csv",
        data={"horizon": 30, "patient_id": "999"},
        files={"file": ("single.csv", csv_content, "text/csv")},
    )
    assert r.status_code == 422
    assert "Patient '999' not found" in r.text


# ===========================================================================
# 2. Validation-before-503 test
# ===========================================================================


def test_backtest_validation_before_503(client, monkeypatch):
    """Invalid input must return 422 even when model is not loaded; valid input returns 503."""
    # Force store to report model not loaded
    monkeypatch.setattr(ModelStore, "is_loaded", lambda self, hz: False)

    # 1. Invalid CSV (missing columns) -> must return 422, NOT 503
    invalid_csv = "timestamp,glucose\n2020-01-15T08:00:00,120\n"
    r_inv = client.post(
        "/v1/backtest",
        data={"horizon": 30},
        files={"file": ("invalid.csv", invalid_csv, "text/csv")},
    )
    assert r_inv.status_code == 422

    # 2. Valid CSV -> all validation passes, then missing model triggers 503
    base = datetime(2020, 1, 15, 8, 0, 0)
    valid_rows = [
        {"timestamp": (base + timedelta(minutes=5 * i)).isoformat(), "glucose": 120.0, "patient_id": 559}
        for i in range(35)
    ]
    valid_csv = _make_csv(valid_rows)
    r_val = client.post(
        "/v1/backtest",
        data={"horizon": 30},
        files={"file": ("valid.csv", valid_csv, "text/csv")},
    )
    assert r_val.status_code == 503
    assert "Model for 30-min horizon is not loaded" in r_val.text


# ===========================================================================
# 3. Tests that need models
# ===========================================================================


def test_predict_accepts_day_first_and_returns_clean_iso(client):
    """Predict accepts DD-MM-YYYY format and returns clean ISO format with 'T' and no '+'/'Z'."""
    health = client.get("/health").json()
    if health["models_loaded"] == 0:
        pytest.skip("No models loaded — skipping model-dependent test")

    base = datetime(2020, 1, 15, 8, 0, 0)
    readings = [
        {
            "timestamp": (base + timedelta(minutes=5 * i)).strftime("%d-%m-%Y %H:%M:%S"),
            "glucose": 120.0 + (i % 5) * 2,
        }
        for i in range(48)
    ]

    r = client.post("/v1/predict", json={"readings": readings, "horizon": 30})
    assert r.status_code == 200
    ts = r.json()["timestamp"]
    assert "T" in ts
    assert "+" not in ts
    assert "Z" not in ts
    assert " " not in ts


def test_predict_gap_near_start_allowed_and_matches_last_24(client):
    """A 20-min gap near start (outside last 24) returns 200, matching predictions for last 24 rows."""
    health = client.get("/health").json()
    if health["models_loaded"] == 0:
        pytest.skip("No models loaded — skipping model-dependent test")

    base = datetime(2020, 1, 15, 8, 0, 0)
    readings_60 = []
    for i in range(60):
        # Gap at index 10 (early in the series, before the last 24 rows)
        offset = 5 * i if i < 10 else 5 * i + 15
        ts = base + timedelta(minutes=offset)
        glucose = 120.0 + (i % 8) * 3
        readings_60.append({"timestamp": ts.isoformat(), "glucose": glucose})

    r60 = client.post("/v1/predict", json={"readings": readings_60, "horizon": 30})
    assert r60.status_code == 200

    # Predictions for just the last 24 readings
    readings_24 = readings_60[-24:]
    r24 = client.post("/v1/predict", json={"readings": readings_24, "horizon": 30})
    assert r24.status_code == 200

    # Predictions must be equal
    assert r60.json()["predictions"] == r24.json()["predictions"]


def test_predict_csv_multi_patient_and_single_patient(client):
    """Predict CSV works with patient_id on multi-patient CSV, and without patient_id on single-patient."""
    health = client.get("/health").json()
    if health["models_loaded"] == 0:
        pytest.skip("No models loaded — skipping model-dependent test")

    base = datetime(2020, 1, 15, 8, 0, 0)
    # Multi-patient CSV
    multi_rows = []
    for i in range(30):
        ts = (base + timedelta(minutes=5 * i)).isoformat()
        multi_rows.append({"timestamp": ts, "glucose": 120.0 + i, "patient_id": "559"})
        multi_rows.append({"timestamp": ts, "glucose": 140.0 + i, "patient_id": "563"})
    multi_csv = _make_csv(multi_rows)

    r_multi = client.post(
        "/v1/predict/csv",
        data={"horizon": 30, "patient_id": "559"},
        files={"file": ("multi.csv", multi_csv, "text/csv")},
    )
    assert r_multi.status_code == 200

    # Single-patient CSV with patient_id column
    single_rows = [
        {"timestamp": (base + timedelta(minutes=5 * i)).isoformat(), "glucose": 120.0 + i, "patient_id": "559"}
        for i in range(30)
    ]
    single_csv = _make_csv(single_rows)
    r_single = client.post(
        "/v1/predict/csv",
        data={"horizon": 30},
        files={"file": ("single.csv", single_csv, "text/csv")},
    )
    assert r_single.status_code == 200

    # Single-patient CSV without patient_id column
    no_pid_rows = [
        {"timestamp": (base + timedelta(minutes=5 * i)).isoformat(), "glucose": 120.0 + i}
        for i in range(30)
    ]
    no_pid_csv = _make_csv(no_pid_rows)
    r_no_pid = client.post(
        "/v1/predict/csv",
        data={"horizon": 30},
        files={"file": ("no_pid.csv", no_pid_csv, "text/csv")},
    )
    assert r_no_pid.status_code == 200
