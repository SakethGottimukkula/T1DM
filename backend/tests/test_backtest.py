"""Tests for /v1/backtest canonical evaluation, per-patient metrics, and sampling."""

from datetime import datetime, timedelta
import io
import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from app.config import HORIZONS
from app.features import FEATURE_COLUMNS, build_features
from app.main import app, get_store
from app.metrics import summarize
from app.predictor import ModelStore


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan so ModelStore is initialised."""
    with TestClient(app) as c:
        yield c


def _make_synthetic_backtest_csv(patient_lengths: dict[str, int]) -> str:
    """Generate a multi-patient CSV with given row counts per patient."""
    base = datetime(2020, 1, 15, 8, 0, 0)
    lines = ["timestamp,glucose,patient_id"]
    for pid, count in patient_lengths.items():
        for i in range(count):
            ts = base + timedelta(minutes=5 * i)
            ts_str = ts.strftime("%d-%m-%Y %H:%M:%S")
            # Varied glucose pattern spanning diverse ranges and EGA zones
            glucose = round(80.0 + (i % 25) * 5.0 + (int(pid) % 7) * 2.0, 1)
            lines.append(f"{ts_str},{glucose},{pid}")
    return "\n".join(lines)


# ===========================================================================
# ModelStore.predict_batch Unit Test
# ===========================================================================


def test_model_store_predict_batch():
    """Verify ModelStore.predict_batch matches single-row predict within tolerance."""
    store = ModelStore()
    store.load_all()
    if 30 not in store.loaded_horizons:
        pytest.skip("Model 30 not loaded — skipping predict_batch test")

    np.random.seed(42)
    sample_data = {
        col: np.random.uniform(70.0, 200.0, size=10)
        for col in FEATURE_COLUMNS
    }
    df_features = pd.DataFrame(sample_data)

    # Batch prediction
    batch_preds = store.predict_batch(df_features, 30)
    assert len(batch_preds) == 10
    assert isinstance(batch_preds, np.ndarray)

    # Single-row prediction for comparison
    single_pred = store.predict(df_features.iloc[0].to_dict(), 30)
    assert abs(batch_preds[0] - single_pred) < 1e-4


# ===========================================================================
# Backtest Endpoint Tests
# ===========================================================================


def test_backtest_canonical_row_count_across_all_horizons(client):
    """Verify n_rows == sum(N_patient - 47) and is identical across all four horizons."""
    health = client.get("/health").json()
    if health["models_loaded"] < len(HORIZONS):
        pytest.skip("Not all models loaded — skipping backtest horizon test")

    patient_lengths = {"559": 100, "563": 160, "570": 80}
    # Expected valid rows: (100 - 47) + (160 - 47) + (80 - 47) = 53 + 113 + 33 = 199
    expected_rows = sum(n - 47 for n in patient_lengths.values())
    csv_content = _make_synthetic_backtest_csv(patient_lengths)

    horizon_n_rows = []
    for hz in HORIZONS:
        r = client.post(
            "/v1/backtest",
            data={"horizon": hz},
            files={"file": ("synthetic.csv", csv_content, "text/csv")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["n_rows"] == expected_rows
        assert data["n_rows_raw"] == sum(patient_lengths.values())
        horizon_n_rows.append(data["n_rows"])

    assert len(set(horizon_n_rows)) == 1, "n_rows must be identical across all horizons"


def test_backtest_per_patient_metrics_breakdown(client):
    """Verify per_patient metrics structure, row counts, and sum matching n_rows."""
    health = client.get("/health").json()
    if health["models_loaded"] == 0:
        pytest.skip("No models loaded — skipping backtest per-patient test")

    patient_lengths = {"559": 100, "563": 160, "570": 80}
    csv_content = _make_synthetic_backtest_csv(patient_lengths)

    r = client.post(
        "/v1/backtest",
        data={"horizon": 30},
        files={"file": ("synthetic.csv", csv_content, "text/csv")},
    )
    assert r.status_code == 200
    data = r.json()

    per_patient = data["per_patient"]
    assert set(per_patient.keys()) == {"559", "563", "570"}

    for pid, count in patient_lengths.items():
        expected_patient_rows = count - 47
        p_metrics = per_patient[pid]
        assert p_metrics["n_rows"] == expected_patient_rows
        assert isinstance(p_metrics["rmse"], float)
        assert isinstance(p_metrics["mae"], float)
        assert "zone_counts" in p_metrics
        assert "zone_percentages" in p_metrics
        assert "a_plus_b_pct" in p_metrics

    sum_patient_rows = sum(p["n_rows"] for p in per_patient.values())
    assert sum_patient_rows == data["n_rows"]


def test_backtest_plot_sampling_and_determinism(client):
    """Verify plot sampling respects max_points, includes all patients, and is deterministic."""
    health = client.get("/health").json()
    if health["models_loaded"] == 0:
        pytest.skip("No models loaded — skipping backtest sampling test")

    patient_lengths = {"559": 100, "563": 160, "570": 80}
    csv_content = _make_synthetic_backtest_csv(patient_lengths)

    r1 = client.post(
        "/v1/backtest",
        data={"horizon": 30, "max_points": 50},
        files={"file": ("synthetic.csv", csv_content, "text/csv")},
    )
    assert r1.status_code == 200
    data1 = r1.json()

    ts1 = data1["timestamps"]
    act1 = data1["actual"]
    pred1 = data1["predicted"]
    pids1 = data1["patient_ids"]
    zones1 = data1["zones"]

    # All parallel arrays must have equal length <= 50
    assert len(ts1) == len(act1) == len(pred1) == len(pids1) == len(zones1)
    assert len(ts1) <= 50

    # Every patient appears at least once in the sampled points
    unique_pids_sampled = {str(p) for p in pids1}
    assert unique_pids_sampled == {"559", "563", "570"}

    # Every zone letter is in A-E
    for z in zones1:
        assert z in {"A", "B", "C", "D", "E"}

    # Second identical call must return identical arrays (determinism)
    r2 = client.post(
        "/v1/backtest",
        data={"horizon": 30, "max_points": 50},
        files={"file": ("synthetic.csv", csv_content, "text/csv")},
    )
    assert r2.status_code == 200
    data2 = r2.json()

    assert data1["timestamps"] == data2["timestamps"]
    assert data1["actual"] == data2["actual"]
    assert data1["predicted"] == data2["predicted"]
    assert data1["patient_ids"] == data2["patient_ids"]
    assert data1["zones"] == data2["zones"]


def test_backtest_max_points_out_of_bounds_rejected(client):
    """max_points < 50 or > 5000 must be rejected with HTTP 422."""
    csv_content = "timestamp,glucose,patient_id\n01-01-2020 08:00:00,120,559\n"

    r_low = client.post(
        "/v1/backtest",
        data={"horizon": 30, "max_points": 10},
        files={"file": ("test.csv", csv_content, "text/csv")},
    )
    assert r_low.status_code == 422
    assert "max_points must be between 50 and 5000" in r_low.text

    r_high = client.post(
        "/v1/backtest",
        data={"horizon": 30, "max_points": 6000},
        files={"file": ("test.csv", csv_content, "text/csv")},
    )
    assert r_high.status_code == 422
    assert "max_points must be between 50 and 5000" in r_high.text


def test_backtest_all_legacy_and_new_fields_present(client):
    """Ensure all legacy fields and newly added fields are present in response."""
    health = client.get("/health").json()
    if health["models_loaded"] == 0:
        pytest.skip("No models loaded — skipping fields check")

    patient_lengths = {"559": 60}
    csv_content = _make_synthetic_backtest_csv(patient_lengths)

    r = client.post(
        "/v1/backtest",
        data={"horizon": 30},
        files={"file": ("test.csv", csv_content, "text/csv")},
    )
    assert r.status_code == 200
    data = r.json()

    legacy_fields = [
        "n_rows", "rmse", "mae", "zone_counts", "zone_percentages",
        "clarke_zone_counts", "clarke_zone_percentages", "clarke_zones",
        "a_plus_b_pct", "clarke_ab_pct", "warnings", "timestamps",
        "actual", "predicted", "horizon"
    ]
    for field in legacy_fields:
        assert field in data, f"Missing legacy field: {field}"

    new_fields = ["n_rows_raw", "evaluation_rule", "per_patient", "patient_ids", "zones"]
    for field in new_fields:
        assert field in data, f"Missing new field: {field}"

    assert data["evaluation_rule"] == "rows where all 18 features and all 4 targets are non-null (identical to Model/XGBoost.ipynb)"


def test_backtest_overall_metrics_match_summarize(client):
    """Verify response rmse, mae, and zone_counts equal app.metrics.summarize() applied to same data."""
    health = client.get("/health").json()
    if health["models_loaded"] == 0:
        pytest.skip("No models loaded — skipping summarize parity test")

    patient_lengths = {"559": 100, "563": 120}
    csv_content = _make_synthetic_backtest_csv(patient_lengths)

    r = client.post(
        "/v1/backtest",
        data={"horizon": 30},
        files={"file": ("test.csv", csv_content, "text/csv")},
    )
    assert r.status_code == 200
    resp_data = r.json()

    # Recompute ground truth using the app's own feature extraction
    df = pd.read_csv(io.StringIO(csv_content))
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S")
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    df_feat = build_features(df, group_col="patient_id")
    target_cols = [f"target_{h}" for h in HORIZONS]
    valid_cols = FEATURE_COLUMNS + target_cols
    df_valid = df_feat.dropna(subset=valid_cols).reset_index(drop=True)

    actuals = df_valid["target_30"].to_numpy(dtype=float)
    store = get_store()
    preds = store.predict_batch(df_valid[FEATURE_COLUMNS], 30)

    expected_metrics = summarize(actuals, preds)
    assert resp_data["n_rows"] == expected_metrics["n_rows"]
    assert resp_data["rmse"] == expected_metrics["rmse"]
    assert resp_data["mae"] == expected_metrics["mae"]
    assert resp_data["zone_counts"] == expected_metrics["zone_counts"]
    assert resp_data["zone_percentages"] == expected_metrics["zone_percentages"]
    assert resp_data["a_plus_b_pct"] == expected_metrics["a_plus_b_pct"]
