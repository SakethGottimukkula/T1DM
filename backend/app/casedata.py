"""Server-side test data loader and case extractor for NutraT1D."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .timeutil import parse_timestamps

logger = logging.getLogger(__name__)

CASE_ID_REGEX = re.compile(r"^(.+)-(\d{8}T\d{6})$")
HISTORY_ROWS = 36
FUTURE_ROWS = 24
TOTAL_WINDOW_ROWS = HISTORY_ROWS + FUTURE_ROWS + 1  # 61


class CaseData:
    """Manages loaded test dataset, patient grouping, and eligible case extraction."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Test CSV file not found: {self.path.name}")

        df = pd.read_csv(self.path)
        required_cols = {"timestamp", "glucose", "patient_id"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(
                f"Test CSV '{self.path.name}' missing required columns: {sorted(missing)}"
            )

        df["timestamp"] = parse_timestamps(df["timestamp"])
        df["patient_id"] = df["patient_id"].astype(str)
        df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
        if df["glucose"].isna().any():
            raise ValueError(f"Test CSV '{self.path.name}' contains NaN glucose values")

        # Sort by patient_id, then timestamp
        df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
        self._raw_frame = df

        # Group by patient_id and compute eligibility
        self._patients: dict[str, dict[str, Any]] = {}
        self._eligible_cases: dict[str, tuple[str, int]] = {}
        self._total_eligible_cases = 0

        for pid, group in df.groupby("patient_id", sort=False):
            p_df = group.reset_index(drop=True)
            n_rows = len(p_df)
            eligible_indices = self._compute_eligible_indices(p_df)
            self._patients[pid] = {
                "df": p_df,
                "n_readings": n_rows,
                "eligible_indices": eligible_indices,
                "first_timestamp": p_df["timestamp"].iloc[0].isoformat() if n_rows > 0 else None,
                "last_timestamp": p_df["timestamp"].iloc[-1].isoformat() if n_rows > 0 else None,
            }
            self._total_eligible_cases += len(eligible_indices)

            for idx in eligible_indices:
                ts = p_df["timestamp"].iloc[idx]
                case_id = f"{pid}-{ts.strftime('%Y%m%dT%H%M%S')}"
                self._eligible_cases[case_id] = (pid, int(idx))

    @staticmethod
    def _compute_eligible_indices(p_df: pd.DataFrame) -> np.ndarray:
        n = len(p_df)
        if n < TOTAL_WINDOW_ROWS:
            return np.array([], dtype=int)

        ts_sec = p_df["timestamp"].values.astype("datetime64[s]").astype(np.int64)
        diffs = np.diff(ts_sec)
        good = (diffs >= 180) & (diffs <= 420)

        cumsum = np.zeros(n, dtype=np.int64)
        cumsum[1:] = np.cumsum(good)

        i_vals = np.arange(HISTORY_ROWS, n - FUTURE_ROWS)
        sums = cumsum[i_vals + FUTURE_ROWS] - cumsum[i_vals - HISTORY_ROWS]
        eligible_mask = (sums == (HISTORY_ROWS + FUTURE_ROWS))
        return i_vals[eligible_mask]

    @property
    def raw_frame(self) -> pd.DataFrame:
        return self._raw_frame

    @property
    def file_name(self) -> str:
        return self.path.name

    @property
    def total_readings(self) -> int:
        return len(self._raw_frame)

    @property
    def total_patients(self) -> int:
        return len(self._patients)

    @property
    def total_eligible_cases(self) -> int:
        return self._total_eligible_cases

    def get_patients_info(self) -> list[dict[str, Any]]:
        info = []
        for pid, pdata in self._patients.items():
            info.append({
                "patient_id": pid,
                "n_readings": pdata["n_readings"],
                "n_eligible_cases": len(pdata["eligible_indices"]),
                "first_timestamp": pdata["first_timestamp"],
                "last_timestamp": pdata["last_timestamp"],
            })
        return info

    def get_cases(self, per_patient: int = 3) -> list[dict[str, Any]]:
        cases = []
        for pid, pdata in self._patients.items():
            eligible = pdata["eligible_indices"]
            n_el = len(eligible)
            if n_el == 0:
                continue
            if n_el <= per_patient:
                selected_indices = eligible
            else:
                step_indices = np.round(np.linspace(0, n_el - 1, per_patient)).astype(int)
                selected_indices = eligible[step_indices]

            p_df = pdata["df"]
            for idx in selected_indices:
                row = p_df.iloc[idx]
                ts = row["timestamp"]
                case_id = f"{pid}-{ts.strftime('%Y%m%dT%H%M%S')}"
                cases.append({
                    "case_id": case_id,
                    "patient_id": pid,
                    "timestamp": ts.isoformat(),
                    "glucose": float(row["glucose"]),
                })
        return cases

    def get_case_detail(self, case_id: str) -> dict[str, Any] | None:
        match = CASE_ID_REGEX.match(case_id)
        if not match:
            return None
        if case_id not in self._eligible_cases:
            return None

        pid, idx = self._eligible_cases[case_id]
        pdata = self._patients[pid]
        p_df = pdata["df"]

        # Build history: 37 readings from idx - 36 to idx inclusive
        history_rows = p_df.iloc[idx - HISTORY_ROWS : idx + 1]
        history = [
            {"timestamp": r.timestamp.isoformat(), "glucose": float(r.glucose)}
            for r in history_rows.itertuples()
        ]

        # Build future: 24 readings from idx + 1 to idx + 24 inclusive
        future_rows = p_df.iloc[idx + 1 : idx + FUTURE_ROWS + 1]
        future = [
            {"timestamp": r.timestamp.isoformat(), "glucose": float(r.glucose)}
            for r in future_rows.itertuples()
        ]

        # actual_at for horizons 30, 60, 90, 120 (steps 6, 12, 18, 24)
        actual_at: dict[str, float] = {}
        for hz in (30, 60, 90, 120):
            step = hz // 5
            actual_val = float(p_df.iloc[idx + step]["glucose"])
            actual_at[str(hz)] = actual_val

        current_row = p_df.iloc[idx]
        return {
            "case_id": case_id,
            "patient_id": pid,
            "timestamp": current_row["timestamp"].isoformat(),
            "glucose": float(current_row["glucose"]),
            "history": history,
            "future": future,
            "actual_at": actual_at,
        }


_cache: dict[str, Any] = {
    "path": None,
    "mtime": None,
    "instance": None,
    "status": "not_configured",
}


def clear_cache() -> None:
    """Clear internal cache (useful for tests)."""
    _cache["path"] = None
    _cache["mtime"] = None
    _cache["instance"] = None
    _cache["status"] = "not_configured"


def get_case_data() -> tuple[CaseData | None, str]:
    """Retrieve or load the cached CaseData singleton based on config.TEST_CSV_PATH."""
    csv_path = config.TEST_CSV_PATH
    if csv_path is None or str(csv_path).strip() == "":
        return None, "not_configured"

    p = Path(csv_path)
    if not p.exists() or not p.is_file():
        return None, "file_missing"

    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None, "file_missing"

    if _cache["path"] == p and _cache["mtime"] == mtime and _cache["instance"] is not None:
        return _cache["instance"], _cache["status"]

    try:
        instance = CaseData(p)
        _cache["path"] = p
        _cache["mtime"] = mtime
        _cache["instance"] = instance
        _cache["status"] = "ok"
        return instance, "ok"
    except Exception as exc:
        logger.warning("Failed to load test CSV %s: %s", p.name, exc)
        _cache["path"] = p
        _cache["mtime"] = mtime
        _cache["instance"] = None
        _cache["status"] = "invalid"
        return None, "invalid"
