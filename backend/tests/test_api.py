"""Tests for the /v1/predict endpoint (happy path + 422 cases)."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app


def _make_readings(n: int = 48, start_glucose: float = 120.0) -> list[dict]:
    """Generate *n* synthetic 5-minute-spaced readings as dicts."""
    from datetime import datetime, timedelta

    base = datetime(2020, 1, 15, 8, 0, 0)
    readings = []
    for i in range(n):
        ts = base + timedelta(minutes=5 * i)
        glucose = start_glucose + (i % 10) * 2 - 10  # oscillates 110–128
        readings.append(
            {
                "timestamp": ts.isoformat(),
                "glucose": round(max(50.0, min(glucose, 350.0)), 1),
            }
        )
    return readings


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan so the ModelStore is initialised."""
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("ok", "degraded")


class TestModels:
    def test_models_endpoint(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["horizons"] == [30, 60, 90, 120]
        assert len(data["feature_columns"]) == 18


class TestPredict:
    def test_predict_requires_min_history(self, client):
        """Sending fewer than 24 readings should fail with 422."""
        readings = _make_readings(n=10)
        r = client.post("/v1/predict", json={"readings": readings, "horizon": "all"})
        assert r.status_code == 422

    def test_predict_rejects_out_of_range_glucose(self, client):
        """Glucose outside 40–400 should fail validation."""
        readings = _make_readings(n=48)
        readings[5]["glucose"] = 10.0  # below 40
        r = client.post("/v1/predict", json={"readings": readings, "horizon": 30})
        assert r.status_code == 422

    def test_predict_rejects_non_increasing_timestamps(self, client):
        """Duplicate/reversed timestamps should fail."""
        readings = _make_readings(n=48)
        readings[10]["timestamp"] = readings[5]["timestamp"]
        r = client.post("/v1/predict", json={"readings": readings, "horizon": 30})
        assert r.status_code == 422

    def test_predict_rejects_bad_horizon(self, client):
        """Invalid horizon should fail."""
        readings = _make_readings(n=48)
        r = client.post("/v1/predict", json={"readings": readings, "horizon": 45})
        assert r.status_code == 422

    def test_predict_happy_path_if_models_loaded(self, client):
        """If models are loaded, a valid request should return predictions.

        This test is conditional: it skips if models aren't loaded (e.g. in CI
        without trained models).
        """
        health = client.get("/health").json()
        if health["models_loaded"] == 0:
            pytest.skip("No models loaded — skipping happy-path test")

        readings = _make_readings(n=48)
        r = client.post("/v1/predict", json={"readings": readings, "horizon": "all"})
        assert r.status_code == 200
        data = r.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 4
        for p in data["predictions"]:
            assert p["horizon"] in [30, 60, 90, 120]
            assert 30 <= p["predicted_glucose"] <= 500  # plausible range
            assert p["zone_label"] in ("low", "in-range", "high")

    def test_predict_single_horizon(self, client):
        """If models are loaded, requesting a single horizon should return 1 result."""
        health = client.get("/health").json()
        if health["models_loaded"] == 0:
            pytest.skip("No models loaded — skipping happy-path test")

        readings = _make_readings(n=48)
        r = client.post("/v1/predict", json={"readings": readings, "horizon": 60})
        assert r.status_code == 200
        data = r.json()
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["horizon"] == 60


class TestBacktest:
    def test_backtest_invalid_horizon(self, client):
        csv_data = "timestamp,glucose,patient_id\n01-01-2020 08:00:00,120,559\n"
        r = client.post(
            "/v1/backtest",
            data={"horizon": 45},
            files={"file": ("test.csv", csv_data, "text/csv")},
        )
        assert r.status_code == 422

    def test_backtest_missing_columns(self, client):
        csv_data = "timestamp,glucose\n01-01-2020 08:00:00,120\n"
        r = client.post(
            "/v1/backtest",
            data={"horizon": 30},
            files={"file": ("test.csv", csv_data, "text/csv")},
        )
        assert r.status_code == 422

    def test_backtest_happy_path(self, client):
        health = client.get("/health").json()
        if health["models_loaded"] == 0:
            pytest.skip("No models loaded — skipping happy-path backtest")

        # Generate 60 rows for patient 559 with 5-minute spacing and one gap
        from datetime import datetime, timedelta

        base = datetime(2020, 1, 15, 8, 0, 0)
        lines = ["timestamp,glucose,patient_id"]
        for i in range(60):
            # Introduce a 15-minute gap at index 30
            offset_mins = 5 * i if i < 30 else 5 * i + 10
            ts = base + timedelta(minutes=offset_mins)
            ts_str = ts.strftime("%d-%m-%Y %H:%M:%S")
            glucose = 120 + (i % 10) * 3
            lines.append(f"{ts_str},{glucose},559")

        csv_content = "\n".join(lines)
        r = client.post(
            "/v1/backtest",
            data={"horizon": 30},
            files={"file": ("test.csv", csv_content, "text/csv")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "n_rows" in data
        assert data["n_rows"] > 0
        assert "rmse" in data
        assert "mae" in data
        assert "zone_counts" in data
        assert "zone_percentages" in data
        assert "a_plus_b_pct" in data
        assert "warnings" in data
        assert data["warnings"].get("559") == 1  # 1 non-5-minute gap
        assert len(data["timestamps"]) <= 2000
        assert len(data["actual"]) == len(data["timestamps"])
        assert len(data["predicted"]) == len(data["timestamps"])

