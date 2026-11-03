"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Default MODEL_DIR is backend/models relative to the repo root.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", str(_BACKEND_DIR / "models")))

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

_DEFAULT_ORIGINS = "http://127.0.0.1:5500,http://localhost:5500"
CORS_ORIGINS: list[str] = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if o.strip()
]

# ---------------------------------------------------------------------------
# Feature / validation constants  (derived from the notebook)
# ---------------------------------------------------------------------------

HORIZONS: list[int] = [30, 60, 90, 120]

# Shifts for each horizon (rows at 5-min spacing)
HORIZON_SHIFTS: dict[int, int] = {30: 6, 60: 12, 90: 18, 120: 24}

# Largest lookback needed to compute all features.
# rolling(24) is the widest window  →  24 prior values required.
# So we need the current row + 23 prior = 24 total readings minimum.
MIN_HISTORY: int = 24

# Glucose validation bounds (mg/dL)
GLUCOSE_MIN: float = 40.0
GLUCOSE_MAX: float = 400.0
