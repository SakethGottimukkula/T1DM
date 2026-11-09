"""Unit and integration tests for server-side test data, /v1/cases, and backtest without upload."""

from datetime import datetime, timedelta
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

import app.config as config
from app.casedata import CaseData, clear_cache, get_case_data
from app.main import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan so ModelStore is initialised."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_case_cache():
    """Ensure casedata cache is clean before and after each test."""
    clear_cache()
    yield
    clear_cache()


def _make_synthetic_test_csv(tmp_path: Path) -> Path:
    """Generate a multi-patient synthetic CSV with controlled gaps and counts.

    - Patient 101: 200 continuous 5-minute readings (many eligible cases).
    - Patient 102: 150 readings with a 20-minute gap at index 80 (windows spanning it are ineligible).
    - Patient 103: 40 readings (< 61 readings, 0 eligible cases).
    """
    rows = []
    base_dt = datetime(2021, 1, 1, 8, 0, 0)

    # Patient 101: 200 readings
    for i in range(200):
        dt = base_dt + timedelta(minutes=5 * i)
        gl = round(100.0 + (i % 20) * 4.0, 1)
        rows.append({"timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"), "glucose": gl, "patient_id": "101"})

    # Patient 102: 150 readings with a 20-min gap after index 80
    current_dt = base_dt
    for i in range(150):
        if i == 81:
            current_dt += timedelta(minutes=20)
        elif i > 0:
            current_dt += timedelta(minutes=5)
        gl = round(110.0 + (i % 15) * 3.5, 1)
        rows.append({"timestamp": current_dt.strftime("%Y-%m-%d %H:%M:%S"), "glucose": gl, "patient_id": "102"})

    # Patient 103: 40 readings
    for i in range(40):
        dt = base_dt + timedelta(minutes=5 * i)
        gl = round(95.0 + (i % 10) * 5.0, 1)
        rows.append({"timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"), "glucose": gl, "patient_id": "103"})

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "synthetic_test.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ===========================================================================
# CaseData Unit Tests
# ===========================================================================


def test_vectorised_eligibility_matches_reference_loop(tmp_path: Path):
    """Verify vectorised prefix-sum eligibility matches an independent loop."""
    csv_path = _make_synthetic_test_csv(tmp_path)
    cdata = CaseData(csv_path)

    assert cdata.total_patients == 3
    assert cdata.total_readings == 390

    for pid, pdata in cdata._patients.items():
        p_df = pdata["df"]
        ts = p_df["timestamp"]
        n = len(p_df)
        expected_indices = []

        if n >= 61:
            for i in range(36, n - 24):
                # 60 intervals from i - 36 to i + 24
                diffs = [
                    (ts.iloc[j + 1] - ts.iloc[j]).total_seconds()
                    for j in range(i - 36, i + 24)
                ]
                if all(180 <= d <= 420 for d in diffs):
                    expected_indices.append(i)

        assert np.array_equal(pdata["eligible_indices"], np.array(expected_indices))

    # Patient 103 should have 0 eligible cases
    assert len(cdata._patients["103"]["eligible_indices"]) == 0
    # Patient 101 should have 200 - 60 = 140 eligible cases
    assert len(cdata._patients["101"]["eligible_indices"]) == 140
    # Patient 102 should have fewer eligible cases due to the 20-min gap
    assert len(cdata._patients["102"]["eligible_indices"]) < (150 - 60)


def test_deterministic_case_selection(tmp_path: Path):
    """Verify get_cases returns segment midpoints and determinism across calls."""
    csv_path = _make_synthetic_test_csv(tmp_path)
    cdata = CaseData(csv_path)

    cases1 = cdata.get_cases(per_patient=3)
    cases2 = cdata.get_cases(per_patient=3)
    assert cases1 == cases2

    pids = [c["patient_id"] for c in cases1]
    assert "101" in pids
    assert "102" in pids
    assert "103" not in pids  # No eligible cases for 103

    # Check case counts per patient
    assert pids.count("101") == 3
    assert pids.count("102") == 3

    # Assert new indices against independent midpoint formula
    for pid in ("101", "102"):
        el = cdata._patients[pid]["eligible_indices"]
        m = len(el)
        k = 3
        assert m > k
        # Independent formula: floor((j + 0.5) * m / k)
        expected_step = [int(np.floor((j + 0.5) * m / k)) for j in range(k)]
        expected_indices = el[expected_step]

        # First and last eligible positions are NOT automatically chosen
        assert 0 not in expected_step
        assert (m - 1) not in expected_step

        actual_cases = [c for c in cases1 if c["patient_id"] == pid]
        p_df = cdata._patients[pid]["df"]
        actual_timestamps = [c["timestamp"] for c in actual_cases]
        expected_timestamps = [p_df.iloc[idx]["timestamp"].isoformat() for idx in expected_indices]
        assert actual_timestamps == expected_timestamps


def test_case_detail_structure(tmp_path: Path):
    """Verify get_case_detail builds 37 history, 24 future, and actual_at."""
    csv_path = _make_synthetic_test_csv(tmp_path)
    cdata = CaseData(csv_path)

    cases = cdata.get_cases(per_patient=1)
    case_id = cases[0]["case_id"]

    detail = cdata.get_case_detail(case_id)
    assert detail is not None
    assert detail["case_id"] == case_id
    assert detail["timestamp"] == cases[0]["timestamp"]
    assert detail["glucose"] == cases[0]["glucose"]

    assert len(detail["history"]) == 37
    assert len(detail["future"]) == 24
    assert detail["history"][-1]["timestamp"] == detail["timestamp"]

    # Ground truth actual_at
    actual_at = detail["actual_at"]
    assert set(actual_at.keys()) == {"30", "60", "90", "120"}
    assert actual_at["30"] == detail["future"][5]["glucose"]
    assert actual_at["60"] == detail["future"][11]["glucose"]
    assert actual_at["90"] == detail["future"][17]["glucose"]
    assert actual_at["120"] == detail["future"][23]["glucose"]


# ===========================================================================
# API Endpoint Integration Tests
# ===========================================================================


def test_cases_list_endpoint_success(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """GET /v1/cases returns 200 with data_source, patients, and cases."""
    csv_path = _make_synthetic_test_csv(tmp_path)
    monkeypatch.setattr(config, "TEST_CSV_PATH", csv_path)

    res = client.get("/v1/cases?per_patient=2")
    assert res.status_code == 200
    data = res.json()

    assert data["data_source"]["configured"] is True
    assert data["data_source"]["status"] == "ok"
    assert data["data_source"]["file_name"] == "synthetic_test.csv"
    # Never leak directory path
    assert str(csv_path.parent) not in str(data)

    assert len(data["patients"]) == 3
    # 2 cases for 101, 2 for 102, 0 for 103 -> 4 total
    assert data["total_cases"] == 4
    assert len(data["cases"]) == 4


def test_cases_list_validation_errors(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """GET /v1/cases returns 422 if per_patient is outside 1..10."""
    csv_path = _make_synthetic_test_csv(tmp_path)
    monkeypatch.setattr(config, "TEST_CSV_PATH", csv_path)

    res_zero = client.get("/v1/cases?per_patient=0")
    assert res_zero.status_code == 422

    res_twelve = client.get("/v1/cases?per_patient=11")
    assert res_twelve.status_code == 422


def test_cases_not_configured_returns_503(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """GET /v1/cases and /v1/cases/{id} return 503 when TEST_CSV_PATH is unset."""
    monkeypatch.setattr(config, "TEST_CSV_PATH", None)

    res_list = client.get("/v1/cases")
    assert res_list.status_code == 503
    assert "TEST_CSV_PATH environment variable is unset" in res_list.json()["detail"]

    res_detail = client.get("/v1/cases/101-20210101T120000")
    assert res_detail.status_code == 503
    assert "TEST_CSV_PATH environment variable is unset" in res_detail.json()["detail"]


def test_cases_file_missing_returns_503(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """GET /v1/cases returns 503 with base filename when file does not exist."""
    missing_path = tmp_path / "missing_data.csv"
    monkeypatch.setattr(config, "TEST_CSV_PATH", missing_path)

    res = client.get("/v1/cases")
    assert res.status_code == 503
    detail = res.json()["detail"]
    assert "'missing_data.csv' not found" in detail
    assert str(tmp_path) not in detail


def test_case_detail_endpoint(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """GET /v1/cases/{id} returns 200 for valid ID and 404 for unknown/malformed."""
    csv_path = _make_synthetic_test_csv(tmp_path)
    monkeypatch.setattr(config, "TEST_CSV_PATH", csv_path)

    # 1. Fetch case list to obtain valid case ID
    list_res = client.get("/v1/cases?per_patient=1")
    assert list_res.status_code == 200
    case_id = list_res.json()["cases"][0]["case_id"]

    # 2. Fetch case detail
    detail_res = client.get(f"/v1/cases/{case_id}")
    assert detail_res.status_code == 200
    d = detail_res.json()
    assert d["case_id"] == case_id
    assert len(d["history"]) == 37
    assert len(d["future"]) == 24
    assert set(d["actual_at"].keys()) == {"30", "60", "90", "120"}

    # 3. 404 on nonexistent case
    res_404 = client.get("/v1/cases/101-19990101T000000")
    assert res_404.status_code == 404

    # 4. 404 on malformed ID
    res_malformed = client.get("/v1/cases/not_a_valid_id")
    assert res_malformed.status_code == 404


def test_round_trip_case_history_to_predict(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that history from /v1/cases/{id} can be posted directly to /v1/predict."""
    csv_path = _make_synthetic_test_csv(tmp_path)
    monkeypatch.setattr(config, "TEST_CSV_PATH", csv_path)

    list_res = client.get("/v1/cases?per_patient=1")
    case_id = list_res.json()["cases"][0]["case_id"]

    detail_res = client.get(f"/v1/cases/{case_id}")
    history = detail_res.json()["history"]

    pred_res = client.post("/v1/predict", json={"readings": history, "horizon": "all"})
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert len(pred_data["predictions"]) == 4
    assert pred_data["latest_glucose"] == history[-1]["glucose"]


def test_backtest_without_upload(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """POST /v1/backtest without file uses server test data and returns data_source='server'."""
    csv_path = _make_synthetic_test_csv(tmp_path)
    monkeypatch.setattr(config, "TEST_CSV_PATH", csv_path)

    res = client.post("/v1/backtest", data={"horizon": 30, "max_points": 500})
    assert res.status_code == 200
    data = res.json()

    assert data["data_source"] == "server"
    assert data["horizon"] == 30
    assert data["n_rows"] > 0
    assert len(data["actual"]) == len(data["predicted"])


def test_backtest_without_upload_error_messages(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify specific 422 messages for not_configured, file_missing, and invalid data."""
    # 1. not_configured
    monkeypatch.setattr(config, "TEST_CSV_PATH", None)
    clear_cache()
    res1 = client.post("/v1/backtest", data={"horizon": 30, "max_points": 500})
    assert res1.status_code == 422
    assert res1.json()["detail"] == "No file uploaded and TEST_CSV_PATH is not set. Upload a CSV or configure TEST_CSV_PATH."

    # 2. file_missing
    missing_file = tmp_path / "absent_file.csv"
    monkeypatch.setattr(config, "TEST_CSV_PATH", missing_file)
    clear_cache()
    res2 = client.post("/v1/backtest", data={"horizon": 30, "max_points": 500})
    assert res2.status_code == 422
    assert res2.json()["detail"] == "No file uploaded and the server test data file 'absent_file.csv' was not found."

    # 3. invalid
    bad_file = tmp_path / "corrupt_data.csv"
    bad_file.write_text("invalid,columns\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(config, "TEST_CSV_PATH", bad_file)
    clear_cache()
    res3 = client.post("/v1/backtest", data={"horizon": 30, "max_points": 500})
    assert res3.status_code == 422
    assert res3.json()["detail"] == "No file uploaded and the server test data file 'corrupt_data.csv' is invalid."


def test_leak_hygiene_and_permission_error_log(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    """Verify directory paths are not leaked in error responses or logs on load failure."""
    secret_dir = tmp_path / "secret_dir_xyz"
    secret_dir.mkdir()
    secret_file = secret_dir / "secret_cases.csv"
    secret_file.write_text("timestamp,glucose,patient_id\n", encoding="utf-8")
    monkeypatch.setattr(config, "TEST_CSV_PATH", secret_file)

    def mock_read_csv(*args, **kwargs):
        raise PermissionError(f"[Errno 13] Permission denied: '{secret_file}'")

    import app.casedata as casedata_mod
    monkeypatch.setattr(casedata_mod.pd, "read_csv", mock_read_csv)

    clear_cache()
    with caplog.at_level("WARNING"):
        res_cases = client.get("/v1/cases")
        assert res_cases.status_code == 503
        assert "secret_dir_xyz" not in res_cases.text

        res_bt = client.post("/v1/backtest", data={"horizon": 30})
        assert res_bt.status_code == 422
        assert "secret_dir_xyz" not in res_bt.text

        assert "secret_dir_xyz" not in caplog.text
        assert "Failed to load test CSV secret_cases.csv: PermissionError" in caplog.text
