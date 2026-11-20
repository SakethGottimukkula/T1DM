"""Ephemeral storage and sanitization for uploaded patient measurements."""

from __future__ import annotations

import re
import secrets
import threading
import time
from typing import Any

import numpy as np
import pandas as pd


_TTL_SECONDS = 6 * 60 * 60
_MAX_CONTEXTS = 64
_DENIED_COLUMN_PARTS = re.compile(
    r"(?:^|_)(?:row|file|path|source|code|script|model|train|test|target|label|"
    r"feature|fold|split|secret|credential|api|token|password|email|address|"
    r"index|prompt|system|config|version)(?:_|$)|"
    r"(?:^|_)(?:patient|subject|record)_?(?:id|identifier|name|number|uuid)$|"
    r"^(?:id|identifier|uuid|guid|hash|key|patient|subject|record)$|"
    r"(?:^|_)(?:filename|filepath)$",
    re.IGNORECASE,
)
_CLINICAL_TEXT_COLUMN = re.compile(
    r"(?:meal|food|activity|exercise|sleep|symptom|event|condition|treatment|diagnos|"
    r"medication|therapy|status|sex|gender|stage|category|device|sensor|site)", re.IGNORECASE
)
_SAFE_CATEGORY_VALUE = re.compile(r"[\w .,+/()%-]{1,80}", re.UNICODE)
_COLUMN_SEPARATORS = re.compile(r"[^a-z0-9]+")
_TIME_NAMES = {"timestamp", "datetime", "date_time", "time", "date"}


def normalize_column_name(value: Any) -> str:
    """Normalize a dataset header to a predictable, user-readable key."""
    name = _COLUMN_SEPARATORS.sub("_", str(value).strip().lower()).strip("_")
    return name or "measurement"


def sanitize_patient_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep timestamp and patient measurements, dropping identifiers and project metadata."""
    if frame.empty:
        raise ValueError("The uploaded patient dataset has no rows.")

    result = pd.DataFrame(index=frame.index)
    used_names: set[str] = set()
    timestamp_series: pd.Series | None = None

    for index, original_name in enumerate(frame.columns):
        name = normalize_column_name(original_name)
        series = frame.iloc[:, index]
        if name in _TIME_NAMES and timestamp_series is None:
            timestamp_series = series
            continue
        if _DENIED_COLUMN_PARTS.search(name) or name.startswith("unnamed_"):
            continue

        nonempty = int(series.notna().sum())
        if not nonempty:
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if int(numeric.notna().sum()) >= max(1, int(nonempty * 0.5)):
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            if numeric.notna().any():
                clean_name = name
                suffix = 2
                while clean_name in used_names:
                    clean_name = f"{name}_{suffix}"
                    suffix += 1
                result[clean_name] = numeric
                used_names.add(clean_name)
            continue

        # Keep only concise categorical health labels; free-form notes are excluded.
        if _CLINICAL_TEXT_COLUMN.search(name):
            values = series.astype("string").str.strip()
            if values.dropna().map(lambda value: bool(_SAFE_CATEGORY_VALUE.fullmatch(value))).all():
                clean_name = name
                suffix = 2
                while clean_name in used_names:
                    clean_name = f"{name}_{suffix}"
                    suffix += 1
                result[clean_name] = values
                used_names.add(clean_name)

    if timestamp_series is None:
        raise ValueError("The uploaded patient dataset needs a timestamp or time column.")
    # Normalize mixed timezone/format input so range comparisons remain safe.
    try:
        timestamps = pd.to_datetime(timestamp_series, errors="coerce", format="mixed", utc=True)
    except TypeError:  # pandas < 2.0 does not accept format="mixed"
        timestamps = pd.to_datetime(timestamp_series, errors="coerce", utc=True)
    if timestamps.isna().all():
        raise ValueError("The uploaded patient dataset has no readable timestamps.")
    result.insert(0, "timestamp", timestamps)
    result = result.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if len(result.columns) < 2:
        raise ValueError("The uploaded dataset has no supported patient measurements.")
    return result


class PatientDatasetStore:
    """Process-local store for sanitized uploaded data with expiring opaque IDs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, tuple[float, pd.DataFrame]] = {}

    def _purge_expired(self, now: float) -> None:
        expired = [key for key, (expires, _) in self._entries.items() if expires <= now]
        for key in expired:
            del self._entries[key]

    def register(self, frame: pd.DataFrame) -> tuple[str, int, list[str]]:
        clean = sanitize_patient_dataframe(frame)
        now = time.monotonic()
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._purge_expired(now)
            while len(self._entries) >= _MAX_CONTEXTS:
                oldest = min(self._entries, key=lambda key: self._entries[key][0])
                del self._entries[oldest]
            self._entries[token] = (now + _TTL_SECONDS, clean)
        return token, len(clean), list(clean.columns[1:])

    def get(self, token: str) -> pd.DataFrame | None:
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            entry = self._entries.get(token)
            return entry[1].copy(deep=True) if entry else None

    def clear(self, token: str) -> None:
        with self._lock:
            self._entries.pop(token, None)


patient_dataset_store = PatientDatasetStore()
