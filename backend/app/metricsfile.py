"""Metrics file loading and consistency check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import HORIZONS

_cache: dict[str, Any] = {
    "path": None,
    "mtime": None,
    "result": None,
}


def load_metrics(path: Path | str) -> tuple[dict[str, Any] | None, str]:
    """Load and parse metrics.json with mtime-based caching.

    Returns
    -------
    tuple[dict | None, str]
        (metrics_dict, status_str) where status_str is one of:
        - "missing": file does not exist
        - "unreadable": not valid JSON or not a JSON object
        - "unsupported_schema": schema_version != 1
        - "ok": successfully loaded
    """
    p = Path(path)
    if not p.exists():
        return None, "missing"

    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None, "missing"

    resolved = str(p.resolve())
    if _cache["path"] == resolved and _cache["mtime"] == mtime and _cache["result"] is not None:
        return _cache["result"]

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        res = (None, "unreadable")
        _cache["path"] = resolved
        _cache["mtime"] = mtime
        _cache["result"] = res
        return res

    if not isinstance(data, dict):
        res = (None, "unreadable")
        _cache["path"] = resolved
        _cache["mtime"] = mtime
        _cache["result"] = res
        return res

    if data.get("schema_version") != 1:
        res = (None, "unsupported_schema")
        _cache["path"] = resolved
        _cache["mtime"] = mtime
        _cache["result"] = res
        return res

    res = (data, "ok")
    _cache["path"] = resolved
    _cache["mtime"] = mtime
    _cache["result"] = res
    return res


def check_consistency(metrics: dict[str, Any] | None, store: Any) -> list[str]:
    """Compare loaded models in store with metrics.json.

    Returns a list of plain English warnings describing any discrepancies.
    """
    if metrics is None:
        return []

    warnings: list[str] = []
    models_entry = metrics.get("models", {}) if isinstance(metrics.get("models"), dict) else {}

    for hz in HORIZONS:
        if store.is_loaded(hz):
            hz_entry = models_entry.get(str(hz), {}) if isinstance(models_entry, dict) else {}
            expected_sha = hz_entry.get("sha256") if isinstance(hz_entry, dict) else None
            actual_sha = store.model_sha256(hz)
            if actual_sha is None or expected_sha is None or actual_sha.lower() != expected_sha.lower():
                warnings.append(
                    f"Model for horizon {hz} differs from metrics.json (metrics.json was generated for different model files and scripts/evaluate.py should be re-run)."
                )
        else:
            warnings.append(f"model for horizon {hz} is not loaded")

    if metrics.get("feature_columns") != store.feature_columns:
        warnings.append("feature_columns in metrics.json does not match model feature columns")

    return warnings


def overall_status(
    load_status: str,
    warnings: list[str],
    metrics: dict[str, Any] | None,
) -> str:
    """Determine the overall consistency status string."""
    if load_status != "ok":
        return load_status

    if any("scripts/evaluate.py" in w for w in warnings):
        return "model_mismatch"

    if any("feature_columns" in w for w in warnings):
        return "feature_mismatch"

    return "ok"
