"""Tests for metricsfile module (loading and consistency checks)."""

import json
from pathlib import Path
import pytest

from app.metricsfile import check_consistency, load_metrics, overall_status


class FakeStore:
    """Minimal fake ModelStore for unit testing consistency checks."""

    def __init__(
        self,
        loaded_horizons=(30, 60, 90, 120),
        hashes=None,
        feature_columns=None,
    ):
        self._loaded = set(loaded_horizons)
        self._hashes = hashes or {h: f"sha256_for_model_{h}" for h in loaded_horizons}
        self.feature_columns = (
            feature_columns
            if feature_columns is not None
            else ["lag_1", "lag_2", "lag_3", "roll_mean_30"]
        )

    def is_loaded(self, horizon: int) -> bool:
        return horizon in self._loaded

    def model_sha256(self, horizon: int) -> str | None:
        return self._hashes.get(horizon)


def _make_valid_metrics_dict(store: FakeStore) -> dict:
    return {
        "schema_version": 1,
        "feature_columns": list(store.feature_columns),
        "models": {
            str(h): {"sha256": store.model_sha256(h), "size_bytes": 1234}
            for h in [30, 60, 90, 120]
        },
        "horizons": {"30": {}, "60": {}, "90": {}, "120": {}},
        "feature_importance": {"importance_type": "weight", "30": {}},
    }


def test_missing_file(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.json"
    metrics, status = load_metrics(missing_path)
    assert metrics is None
    assert status == "missing"
    assert overall_status(status, [], None) == "missing"


def test_invalid_json(tmp_path: Path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{ broken json", encoding="utf-8")
    metrics, status = load_metrics(bad_file)
    assert metrics is None
    assert status == "unreadable"
    assert overall_status(status, [], None) == "unreadable"


def test_non_dict_json(tmp_path: Path):
    list_file = tmp_path / "list.json"
    list_file.write_text("[1, 2, 3]", encoding="utf-8")
    metrics, status = load_metrics(list_file)
    assert metrics is None
    assert status == "unreadable"
    assert overall_status(status, [], None) == "unreadable"


def test_unsupported_schema_version(tmp_path: Path):
    schema2_file = tmp_path / "v2.json"
    schema2_file.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    metrics, status = load_metrics(schema2_file)
    assert metrics is None
    assert status == "unsupported_schema"
    assert overall_status(status, [], None) == "unsupported_schema"


def test_ok_when_hashes_and_features_match(tmp_path: Path):
    store = FakeStore()
    valid_data = _make_valid_metrics_dict(store)
    file_path = tmp_path / "metrics.json"
    file_path.write_text(json.dumps(valid_data), encoding="utf-8")

    metrics, load_status = load_metrics(file_path)
    assert load_status == "ok"
    assert metrics is not None

    warnings = check_consistency(metrics, store)
    assert warnings == []
    assert overall_status(load_status, warnings, metrics) == "ok"


def test_model_mismatch_when_one_hash_differs(tmp_path: Path):
    store = FakeStore()
    data = _make_valid_metrics_dict(store)
    # Tamper with horizon 30 hash
    data["models"]["30"]["sha256"] = "different_hash_value"
    file_path = tmp_path / "metrics.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    metrics, load_status = load_metrics(file_path)
    assert load_status == "ok"

    warnings = check_consistency(metrics, store)
    assert len(warnings) == 1
    assert "30" in warnings[0]
    assert "scripts/evaluate.py" in warnings[0]
    assert overall_status(load_status, warnings, metrics) == "model_mismatch"


def test_feature_mismatch_when_only_feature_list_differs(tmp_path: Path):
    store = FakeStore()
    data = _make_valid_metrics_dict(store)
    data["feature_columns"] = ["different_feature_col"]
    file_path = tmp_path / "metrics.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    metrics, load_status = load_metrics(file_path)
    assert load_status == "ok"

    warnings = check_consistency(metrics, store)
    assert len(warnings) == 1
    assert "feature_columns" in warnings[0]
    assert overall_status(load_status, warnings, metrics) == "feature_mismatch"


def test_model_not_loaded_warning(tmp_path: Path):
    # Horizon 120 is not loaded in store
    store = FakeStore(loaded_horizons=(30, 60, 90))
    data = _make_valid_metrics_dict(store)
    file_path = tmp_path / "metrics.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    metrics, load_status = load_metrics(file_path)
    assert load_status == "ok"

    warnings = check_consistency(metrics, store)
    assert any("120" in w and "not loaded" in w for w in warnings)
