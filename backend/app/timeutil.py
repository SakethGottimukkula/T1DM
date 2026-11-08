"""Timestamp parsing utilities with strict pattern-based format matching."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Sequence

import pandas as pd

# ISO 8601 regex: starts with YYYY-MM-DD, then 'T' or a space, then HH:MM[:SS[.fraction]],
# optional suffix 'Z' or offset like +05:30, -0500, +05:30
ISO_REGEX = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?(?:Z|[+-]\d{2}(?::?\d{2})?)?$"
)

# Day-first regex: exactly "DD-MM-YYYY HH:MM:SS"
DAY_FIRST_REGEX = re.compile(
    r"^(\d{2})-(\d{2})-(\d{4}) (\d{2}):(\d{2}):(\d{2})$"
)


def parse_timestamps(values: Sequence[str] | pd.Series) -> pd.Series:
    """Parse timestamps with strict pattern matching and wall-clock preservation.

    Accepted formats:
      - ISO 8601 (e.g. 'YYYY-MM-DDTHH:MM:SS', 'YYYY-MM-DD HH:MM:SS', optional fractional seconds,
        optional 'Z' or UTC offset which is discarded).
      - Day-first (exactly 'DD-MM-YYYY HH:MM:SS').

    Raises:
      ValueError: if an invalid pattern, invalid date, or mixed formats are found.
    """
    if len(values) == 0:
        return pd.Series([], dtype="datetime64[ns]")

    parsed_dates = []
    detected_formats: set[str] = set()

    for val in values:
        val_str = str(val) if not isinstance(val, str) else val

        # Check Day-first
        m_day = DAY_FIRST_REGEX.match(val_str)
        if m_day:
            d, m, y, H, M, S = m_day.groups()
            try:
                dt = datetime(int(y), int(m), int(d), int(H), int(M), int(S))
            except ValueError:
                quoted = val_str[:40]
                raise ValueError(
                    f"Invalid timestamp '{quoted}'. Accepted formats are ISO 8601 (e.g. 'YYYY-MM-DDTHH:MM:SS') "
                    f"and Day-first ('DD-MM-YYYY HH:MM:SS')."
                )
            detected_formats.add("DAY_FIRST")
            parsed_dates.append(dt)
            continue

        # Check ISO
        m_iso = ISO_REGEX.match(val_str)
        if m_iso:
            y, m, d, H, M, S, frac = m_iso.groups()
            sec = int(S) if S else 0
            micro = int(frac[:6].ljust(6, "0")) if frac else 0
            try:
                dt = datetime(int(y), int(m), int(d), int(H), int(M), sec, micro)
            except ValueError:
                quoted = val_str[:40]
                raise ValueError(
                    f"Invalid timestamp '{quoted}'. Accepted formats are ISO 8601 (e.g. 'YYYY-MM-DDTHH:MM:SS') "
                    f"and Day-first ('DD-MM-YYYY HH:MM:SS')."
                )
            detected_formats.add("ISO")
            parsed_dates.append(dt)
            continue

        quoted = val_str[:40]
        raise ValueError(
            f"Invalid timestamp '{quoted}'. Accepted formats are ISO 8601 (e.g. 'YYYY-MM-DDTHH:MM:SS') "
            f"and Day-first ('DD-MM-YYYY HH:MM:SS')."
        )

    if len(detected_formats) > 1:
        raise ValueError("Mixed timestamp formats")

    index = values.index if isinstance(values, pd.Series) else None
    return pd.Series(pd.to_datetime(parsed_dates), index=index, dtype="datetime64[ns]")
