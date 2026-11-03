"""Pydantic v2 request / response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .config import GLUCOSE_MAX, GLUCOSE_MIN


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class GlucoseReading(BaseModel):
    """A single CGM reading."""

    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp, e.g. '2020-01-15T07:35:00'",
    )
    glucose: float = Field(
        ...,
        ge=GLUCOSE_MIN,
        le=GLUCOSE_MAX,
        description="Blood glucose in mg/dL (40–400)",
    )


class PredictRequest(BaseModel):
    """POST /v1/predict request body."""

    readings: list[GlucoseReading] = Field(
        ...,
        min_length=1,
        description="Time-ordered CGM readings",
    )
    horizon: int | Literal["all"] = Field(
        default="all",
        description="Forecast horizon in minutes: 30, 60, 90, 120, or 'all'",
    )

    @field_validator("horizon")
    @classmethod
    def _validate_horizon(cls, v: int | str) -> int | str:
        if isinstance(v, str) and v != "all":
            raise ValueError("horizon must be 30, 60, 90, 120, or 'all'")
        if isinstance(v, int) and v not in (30, 60, 90, 120):
            raise ValueError("horizon must be 30, 60, 90, 120, or 'all'")
        return v


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HorizonResult(BaseModel):
    """Prediction result for a single horizon."""

    horizon: int
    predicted_glucose: float
    change_from_latest: float
    zone_label: str = Field(
        ...,
        description="'low' (<70), 'in-range' (70–180), or 'high' (>180)",
    )


class PredictResponse(BaseModel):
    """POST /v1/predict response body."""

    predictions: list[HorizonResult]
    latest_glucose: float
    timestamp: str


class BacktestRow(BaseModel):
    """Single row in the backtest result arrays."""

    timestamp: str
    actual: float
    predicted: float


class ClarkeZoneCounts(BaseModel):
    """Clarke Error Grid zone counts."""

    A: int = 0
    B: int = 0
    C: int = 0
    D: int = 0
    E: int = 0


class BacktestResponse(BaseModel):
    """POST /v1/backtest response body."""

    n_rows: int
    rmse: float
    mae: float
    zone_counts: dict[str, int]
    zone_percentages: dict[str, float]
    clarke_zone_counts: dict[str, int]
    clarke_zone_percentages: dict[str, float]
    clarke_zones: dict[str, Any]
    a_plus_b_pct: float
    warnings: dict[str, int]
    timestamps: list[str]
    actual: list[float]
    predicted: list[float]
    horizon: int | None = None
    clarke_ab_pct: float | None = None


class ModelInfo(BaseModel):
    """Information about a single loaded model."""

    horizon: int
    loaded: bool
    model_file: str


class ModelsResponse(BaseModel):
    """GET /v1/models response body."""

    horizons: list[int]
    feature_columns: list[str]
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str
    models_loaded: int
    models_expected: int
    detail: str | None = None
