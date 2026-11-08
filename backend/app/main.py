"""Glucose Horizon — FastAPI application."""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import CORS_ORIGINS, GLUCOSE_MAX, GLUCOSE_MIN, HORIZONS, MIN_HISTORY
from .ega import classify_array
from .features import FEATURE_COLUMNS, build_features, build_features_single
from .timeutil import parse_timestamps
from .predictor import ModelStore
from .schemas import (
    BacktestResponse,
    HealthResponse,
    HorizonResult,
    ModelInfo,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

_store: ModelStore | None = None


def get_store() -> ModelStore:
    assert _store is not None, "ModelStore not initialised"
    return _store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _store
    _store = ModelStore()
    _store.load_all()
    loaded = len(_store.loaded_horizons)
    logger.info(
        "Startup complete — %d / %d models loaded from %s",
        loaded,
        len(HORIZONS),
        _store.model_dir,
    )
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Glucose Horizon API",
    description="CGM-only blood glucose forecasting for Type 1 Diabetes",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zone_label(glucose: float) -> str:
    if glucose < 70:
        return "low"
    if glucose > 180:
        return "high"
    return "in-range"


def _validate_readings_df(df: pd.DataFrame) -> None:
    """Common validation for a DataFrame of readings."""

    # Glucose bounds
    oob = df[(df["glucose"] < GLUCOSE_MIN) | (df["glucose"] > GLUCOSE_MAX)]
    if not oob.empty:
        raise HTTPException(
            422,
            f"Glucose values must be between {GLUCOSE_MIN} and {GLUCOSE_MAX} mg/dL. "
            f"Found {len(oob)} out-of-range value(s).",
        )

    # Duplicate timestamps anywhere in the input
    if df["timestamp"].duplicated().any():
        raise HTTPException(422, "Timestamps must be unique.")

    # Timestamps must be strictly increasing
    if not df["timestamp"].is_monotonic_increasing:
        raise HTTPException(422, "Timestamps must be strictly increasing.")

    # Minimum history
    if len(df) < MIN_HISTORY:
        raise HTTPException(
            422,
            f"At least {MIN_HISTORY} readings required (largest rolling window = 24 rows). "
            f"Got {len(df)}.",
        )

    # Only the LAST MIN_HISTORY readings must be continuous (~5-minute spacing: allow 3–7 min gaps)
    last_window = df.iloc[-MIN_HISTORY:]
    diffs = last_window["timestamp"].diff().dropna().dt.total_seconds()
    bad = diffs[(diffs < 180) | (diffs > 420)]
    if not bad.empty:
        raise HTTPException(
            422,
            f"Timestamps should be ~5 minutes apart. Found {len(bad)} gap(s) outside 3–7 min. "
            f"Only the most recent {MIN_HISTORY} readings (2 hours) are checked.",
        )


def _predict_horizons(
    store: ModelStore,
    features_row: dict[str, float],
    latest_glucose: float,
    horizons: list[int],
) -> list[HorizonResult]:
    results: list[HorizonResult] = []
    for hz in horizons:
        if not store.is_loaded(hz):
            raise HTTPException(
                503, f"Model for {hz}-min horizon is not loaded."
            )
        pred = store.predict(features_row, hz)
        results.append(
            HorizonResult(
                horizon=hz,
                predicted_glucose=round(pred, 2),
                change_from_latest=round(pred - latest_glucose, 2),
                zone_label=_zone_label(pred),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    store = get_store()
    loaded = len(store.loaded_horizons)
    expected = len(HORIZONS)
    missing = set(HORIZONS) - set(store.loaded_horizons)
    if loaded == expected:
        return HealthResponse(
            status="ok",
            models_loaded=loaded,
            models_expected=expected,
        )
    return HealthResponse(
        status="degraded",
        models_loaded=loaded,
        models_expected=expected,
        detail=f"Missing models for horizons: {sorted(missing)}",
    )


@app.get("/v1/models", response_model=ModelsResponse)
def models_info() -> ModelsResponse:
    store = get_store()
    model_infos = [
        ModelInfo(
            horizon=hz,
            loaded=store.is_loaded(hz),
            model_file=f"model_{hz}.json",
        )
        for hz in HORIZONS
    ]
    return ModelsResponse(
        horizons=HORIZONS,
        feature_columns=store.feature_columns,
        models=model_infos,
    )


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    store = get_store()

    # Build DataFrame from readings
    rows = [{"timestamp": r.timestamp, "glucose": r.glucose} for r in req.readings]
    df = pd.DataFrame(rows)
    try:
        df["timestamp"] = parse_timestamps(df["timestamp"])
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    df = df.sort_values("timestamp").reset_index(drop=True)

    _validate_readings_df(df)

    # Build features (single patient)
    df = build_features_single(df)

    # Take the last row
    last = df.iloc[-1]
    if last[FEATURE_COLUMNS].isna().any():
        raise HTTPException(
            422,
            f"Not enough history to compute all features. Need at least {MIN_HISTORY} readings.",
        )

    features_row = {col: float(last[col]) for col in FEATURE_COLUMNS}
    latest_glucose = float(last["glucose"])
    ts = last["timestamp"].isoformat()

    # Determine horizons
    if req.horizon == "all":
        horizons = HORIZONS
    else:
        horizons = [int(req.horizon)]

    preds = _predict_horizons(store, features_row, latest_glucose, horizons)

    return PredictResponse(
        predictions=preds,
        latest_glucose=latest_glucose,
        timestamp=ts,
    )


@app.post("/v1/predict/csv", response_model=PredictResponse)
async def predict_csv(
    file: UploadFile = File(...),
    horizon: str = Form(default="all"),
    patient_id: str | None = Form(default=None),
) -> PredictResponse:
    store = get_store()

    # Parse horizon parameter
    if horizon == "all":
        horizons = HORIZONS
    else:
        try:
            hz_int = int(horizon)
        except ValueError:
            raise HTTPException(422, "horizon must be 30, 60, 90, 120, or 'all'")
        if hz_int not in HORIZONS:
            raise HTTPException(422, "horizon must be 30, 60, 90, 120, or 'all'")
        horizons = [hz_int]

    # Read CSV
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(422, f"Could not parse CSV: {exc}")

    if "timestamp" not in df.columns or "glucose" not in df.columns:
        raise HTTPException(422, "CSV must contain 'timestamp' and 'glucose' columns.")

    try:
        df["timestamp"] = parse_timestamps(df["timestamp"])
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    # Patient selection
    if "patient_id" in df.columns:
        unique_pids = [str(p) for p in df["patient_id"].unique()]
        pid_str = str(patient_id) if patient_id is not None and str(patient_id).strip() != "" else None

        if len(unique_pids) == 1:
            single_pid = unique_pids[0]
            if pid_str is not None and pid_str != single_pid:
                raise HTTPException(
                    422,
                    f"Patient '{patient_id}' not found. Available patient IDs: {unique_pids[:20]}.",
                )
            df = df[df["patient_id"].astype(str) == single_pid]
        elif len(unique_pids) > 1:
            if pid_str is None:
                raise HTTPException(
                    422,
                    f"Multiple patients found in CSV: {unique_pids[:20]}. Please specify 'patient_id'.",
                )
            if pid_str not in unique_pids:
                raise HTTPException(
                    422,
                    f"Patient '{patient_id}' not found. Available patient IDs: {unique_pids[:20]}.",
                )
            df = df[df["patient_id"].astype(str) == pid_str]

    df = df.sort_values("timestamp").reset_index(drop=True)

    _validate_readings_df(df)

    df = build_features_single(df)
    last = df.iloc[-1]

    if last[FEATURE_COLUMNS].isna().any():
        raise HTTPException(
            422,
            f"Not enough history to compute all features. Need at least {MIN_HISTORY} readings.",
        )

    features_row = {col: float(last[col]) for col in FEATURE_COLUMNS}
    latest_glucose = float(last["glucose"])
    ts = last["timestamp"].isoformat()

    preds = _predict_horizons(store, features_row, latest_glucose, horizons)

    return PredictResponse(
        predictions=preds,
        latest_glucose=latest_glucose,
        timestamp=ts,
    )


@app.post("/v1/backtest", response_model=BacktestResponse)
async def backtest(
    file: UploadFile = File(...),
    horizon: int = Form(...),
) -> BacktestResponse:
    store = get_store()

    # 1. Horizon valid (422)
    if horizon not in HORIZONS:
        raise HTTPException(422, "horizon must be 30, 60, 90, or 120")

    # 2. Read and parse CSV (422)
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(422, f"Could not parse CSV: {exc}")

    # 3. Required columns (422)
    required_cols = {"timestamp", "glucose", "patient_id"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(
            422,
            f"CSV must contain 'timestamp', 'glucose', and 'patient_id' columns. Got: {list(df.columns)}",
        )

    # 4. Timestamps parse (422)
    try:
        df["timestamp"] = parse_timestamps(df["timestamp"])
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    # 5. Model loaded check (503)
    if not store.is_loaded(horizon):
        raise HTTPException(503, f"Model for {horizon}-min horizon is not loaded.")

    # 1. Sort by patient_id, then timestamp
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    # Compute count of non-5-minute gaps per patient
    warnings: dict[str, int] = {}
    for pid, group in df.groupby("patient_id"):
        diffs = group["timestamp"].diff().dropna()
        non_5m = int((diffs != pd.Timedelta(minutes=5)).sum())
        warnings[str(pid)] = non_5m

    # 2. Build features with groupby('patient_id') exactly as in Model/XGBoost.ipynb.
    # No gap segmentation, and do not reject files that contain gaps.
    df_feat = build_features(df, group_col="patient_id")

    # 3. Drop rows with NaN features or target
    target_col = f"target_{horizon}"
    valid_cols = FEATURE_COLUMNS + [target_col]
    df_valid = df_feat.dropna(subset=valid_cols).reset_index(drop=True)

    if len(df_valid) == 0:
        raise HTTPException(422, "No rows with enough history and valid targets.")

    # 4. Predict every remaining row with the model for that horizon and compare against the true glucose at t+horizon
    X = df_valid[FEATURE_COLUMNS].values
    model = store._models[horizon]  # type: ignore[attr-defined]
    preds = model.predict(X)
    actuals = df_valid[target_col].values

    # Metrics
    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae = float(mean_absolute_error(actuals, preds))

    # Clarke EGA
    _, zone_counts = classify_array(actuals.tolist(), preds.tolist())
    n_rows = len(df_valid)
    zone_percentages = {
        z: round((zone_counts[z] / n_rows) * 100, 2) if n_rows else 0.0
        for z in ["A", "B", "C", "D", "E"]
    }
    a_plus_b_pct = (
        round(((zone_counts["A"] + zone_counts["B"]) / n_rows) * 100, 2)
        if n_rows
        else 0.0
    )

    # Arrays timestamps/actual/predicted capped at 2000 points for plotting
    max_plot_points = 2000
    plot_df = df_valid.iloc[:max_plot_points]
    timestamps_arr = [ts.isoformat() for ts in plot_df["timestamp"]]
    actual_arr = [round(float(v), 2) for v in actuals[:max_plot_points]]
    predicted_arr = [round(float(v), 2) for v in preds[:max_plot_points]]

    return BacktestResponse(
        n_rows=n_rows,
        rmse=round(rmse, 2),
        mae=round(mae, 2),
        zone_counts=zone_counts,
        zone_percentages=zone_percentages,
        clarke_zone_counts=zone_counts,
        clarke_zone_percentages=zone_percentages,
        clarke_zones={"counts": zone_counts, "percentages": zone_percentages},
        a_plus_b_pct=a_plus_b_pct,
        clarke_ab_pct=a_plus_b_pct,
        warnings=warnings,
        timestamps=timestamps_arr,
        actual=actual_arr,
        predicted=predicted_arr,
        horizon=horizon,
    )
