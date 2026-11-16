"""Glucose Horizon — FastAPI application."""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import CORS_ORIGINS, GLUCOSE_MAX, GLUCOSE_MIN, HORIZONS, MIN_HISTORY, OHIO_TEST_CSV
from .ega import classify_array
from .features import FEATURE_COLUMNS, build_features, build_features_single
from .predictor import ModelStore
from .simglucose_features import SIM_FEATURE_COLUMNS, build_sim_features
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
_sim_models: dict[int, object] = {}


def get_store() -> ModelStore:
    assert _store is not None, "ModelStore not initialised"
    return _store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _store
    _store = ModelStore()
    _store.load_all()
    _load_simglucose_models()
    loaded = len(_store.loaded_horizons)
    logger.info(
        "Startup complete — %d / %d models loaded from %s",
        loaded,
        len(HORIZONS),
        _store.model_dir,
    )
    yield


def _load_simglucose_models() -> None:
    import xgboost as xgb

    model_dir = Path(__file__).resolve().parent.parent / "models" / "simglucose"
    _sim_models.clear()
    for horizon in HORIZONS:
        path = model_dir / f"model_{horizon}.json"
        if path.is_file():
            model = xgb.XGBRegressor()
            model.load_model(str(path))
            _sim_models[horizon] = model


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

    # Timestamps must be strictly increasing
    if not df["timestamp"].is_monotonic_increasing:
        raise HTTPException(422, "Timestamps must be strictly increasing.")

    # Roughly 5-minute spacing: allow 3–7 min gaps
    if len(df) >= 2:
        diffs = df["timestamp"].diff().dropna().dt.total_seconds()
        bad = diffs[(diffs < 180) | (diffs > 420)]
        if not bad.empty:
            raise HTTPException(
                422,
                f"Timestamps should be ~5 minutes apart. Found {len(bad)} gap(s) outside 3–7 min.",
            )

    # Minimum history
    if len(df) < MIN_HISTORY:
        raise HTTPException(
            422,
            f"At least {MIN_HISTORY} readings required (largest rolling window = 24 rows). "
            f"Got {len(df)}.",
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


@app.get("/v1/models/simglucose")
def simglucose_models_info() -> dict:
    return {"model_family": "SimGlucose XGBoost", "models_loaded": len(_sim_models) == len(HORIZONS),
            "horizons_loaded": sorted(_sim_models), "feature_columns": SIM_FEATURE_COLUMNS}


@app.post("/v1/predict/simglucose/csv")
async def predict_simglucose_csv(file: UploadFile = File(...), horizon: str = Form(default="all")) -> dict:
    if len(_sim_models) != len(HORIZONS):
        raise HTTPException(503, "SimGlucose models are not trained yet; generate the training data and run the training script first.")
    if horizon == "all":
        horizons = HORIZONS
    else:
        try:
            horizons = [int(horizon)]
        except ValueError:
            raise HTTPException(422, "horizon must be 30, 60, 90, 120, or 'all'")
        if horizons[0] not in HORIZONS:
            raise HTTPException(422, "horizon must be 30, 60, 90, 120, or 'all'")
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(422, f"Could not parse CSV: {exc}")
    df.columns = [str(col).strip().lower() for col in df.columns]
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    if "timestamp" not in df.columns and "unnamed: 0" in df.columns:
        df = df.rename(columns={"unnamed: 0": "timestamp"})
    for alias in ("glucose", "bg"):
        if "cgm" not in df.columns and alias in df.columns:
            df = df.rename(columns={alias: "cgm"})
    meal_defaulted = "cho" not in df.columns
    if "cho" not in df.columns:
        df["cho"] = 0.0
    if not {"timestamp", "cgm", "insulin"}.issubset(df.columns):
        raise HTTPException(422, "SimGlucose CSV needs timestamp/Time, CGM, and insulin columns. CHO is optional and defaults to zero.")
    if "bg" not in df.columns:
        df["bg"] = df["cgm"]
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    except (ValueError, TypeError):
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    for col in ("cgm", "bg", "insulin", "cho"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "cgm", "bg", "insulin"]).sort_values("timestamp")
    df = df.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    df = df.resample("5min").agg({"cgm": "mean", "bg": "mean", "insulin": "sum", "cho": "sum"}).dropna(subset=["cgm", "bg"])
    if len(df) < MIN_HISTORY:
        raise HTTPException(422, f"At least {MIN_HISTORY} valid 5-minute readings are required. Got {len(df)}.")
    feature_df = build_sim_features(df.reset_index())
    last = feature_df.iloc[-1]
    if last[SIM_FEATURE_COLUMNS].isna().any():
        raise HTTPException(422, "Not enough history to compute model features.")
    x = last[SIM_FEATURE_COLUMNS].to_numpy(dtype=float).reshape(1, -1)
    latest = float(last["cgm"])
    predictions = []
    for hz in horizons:
        predicted = float(_sim_models[hz].predict(x)[0])
        predictions.append({"horizon": hz, "predicted_glucose": round(predicted, 2),
                            "change_from_latest": round(predicted - latest, 2), "zone_label": _zone_label(predicted)})
    return {"predictions": predictions, "latest_glucose": latest, "timestamp": str(last["timestamp"]),
            "meal_defaulted_to_zero": meal_defaulted}


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    store = get_store()

    # Build DataFrame from readings
    rows = [{"timestamp": r.timestamp, "glucose": r.glucose} for r in req.readings]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
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
    ts = str(last["timestamp"])

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


@app.post("/v1/predict/csv")
async def predict_csv(
    file: UploadFile = File(...),
    horizon: str = Form(default="all"),
    patient_id: str | None = Form(default=None),
) -> dict:
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

    df.columns = [str(col).strip().lower() for col in df.columns]
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    if "timestamp" not in df.columns and "unnamed: 0" in df.columns:
        df = df.rename(columns={"unnamed: 0": "timestamp"})
    if "glucose" not in df.columns and "cgm" in df.columns:
        df = df.rename(columns={"cgm": "glucose"})
    if "glucose" not in df.columns and "bg" in df.columns:
        df = df.rename(columns={"bg": "glucose"})

    if "timestamp" not in df.columns or "glucose" not in df.columns:
        raise HTTPException(422, "CSV must contain 'timestamp' and 'glucose' columns.")

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S")
    except (ValueError, TypeError):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, f"Could not parse timestamps: {exc}")

    has_patient = "patient_id" in df.columns
    if not has_patient:
        df["patient_id"] = "all"
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    patients_available = df["patient_id"].drop_duplicates().tolist()
    if patient_id is not None and patient_id not in {str(value) for value in patients_available}:
        raise HTTPException(422, f"patient_id '{patient_id}' not found. Available: {patients_available}")
    patient_used = patient_id if patient_id is not None else str(patients_available[0])
    df = df[df["patient_id"].astype(str) == patient_used].sort_values("timestamp").reset_index(drop=True)
    warnings = {"gaps_outside_3_7_minutes": 0}
    tail_count = 25
    if len(df) < tail_count:
        raise HTTPException(422, f"At least {tail_count} readings required to validate feature history. Got {len(df)}.")
    if len(df) >= 2:
        all_diffs = df["timestamp"].diff().dropna().dt.total_seconds()
        warnings["gaps_outside_3_7_minutes"] = int(((all_diffs < 180) | (all_diffs > 420)).sum())
    tail = df.tail(tail_count).reset_index(drop=True)
    diffs = tail["timestamp"].diff().dropna().dt.total_seconds()
    bad = diffs[(diffs < 180) | (diffs > 420)]
    if not bad.empty:
        index = int(bad.index[0])
        raise HTTPException(422, f"Timestamps should be ~5 minutes apart in feature window. Gap at {tail.loc[index, 'timestamp']}.")
    df = tail.drop(columns=["patient_id"]).reset_index(drop=True)

    df = build_features_single(df)
    last = df.iloc[-1]

    if last[FEATURE_COLUMNS].isna().any():
        raise HTTPException(
            422,
            f"Not enough history to compute all features. Need at least {MIN_HISTORY} readings.",
        )

    features_row = {col: float(last[col]) for col in FEATURE_COLUMNS}
    latest_glucose = float(last["glucose"])
    ts = str(last["timestamp"])

    preds = _predict_horizons(store, features_row, latest_glucose, horizons)

    return {
        "predictions": [pred.model_dump() for pred in preds],
        "latest_glucose": latest_glucose,
        "timestamp": ts,
        "patient_used": patient_used,
        "patients_available": [str(value) for value in patients_available],
        "warnings": warnings,
    }


@app.post("/v1/backtest", response_model=BacktestResponse)
async def backtest(
    file: UploadFile = File(...),
    horizon: int = Form(...),
) -> BacktestResponse:
    store = get_store()

    if horizon not in HORIZONS:
        raise HTTPException(422, "horizon must be 30, 60, 90, or 120")
    if not store.is_loaded(horizon):
        raise HTTPException(503, f"Model for {horizon}-min horizon is not loaded.")

    # Read CSV
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(422, f"Could not parse CSV: {exc}")

    required_cols = {"timestamp", "glucose", "patient_id"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(
            422,
            f"CSV must contain 'timestamp', 'glucose', and 'patient_id' columns. Got: {list(df.columns)}",
        )

    try:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%d-%m-%Y %H:%M:%S"
        )
    except Exception as exc:
        raise HTTPException(
            422, f"Error parsing timestamp with format '%d-%m-%Y %H:%M:%S': {exc}"
        )

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
    timestamps_arr = [str(ts) for ts in plot_df["timestamp"]]
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


@app.get("/v1/backtest/overall")
def backtest_overall_metrics() -> dict:
    """Return pooled Ohio held-out RMSE/MAE at each forecast horizon."""
    store = get_store()
    if not OHIO_TEST_CSV.is_file():
        raise HTTPException(404, f"Ohio test CSV not found: {OHIO_TEST_CSV}")
    if any(not store.is_loaded(horizon) for horizon in HORIZONS):
        raise HTTPException(503, "Models for all forecast horizons must be loaded to calculate overall RMSE.")

    try:
        df = pd.read_csv(OHIO_TEST_CSV)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S")
    except Exception as exc:
        raise HTTPException(422, f"Could not read or parse Ohio test CSV: {exc}")
    required = {"timestamp", "glucose", "patient_id"}
    if not required.issubset(df.columns):
        raise HTTPException(422, "Ohio test CSV must contain timestamp, glucose, and patient_id columns.")

    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    df_feat = build_features(df, group_col="patient_id")
    metrics: dict[str, dict[str, float | int]] = {}
    for horizon in HORIZONS:
        target_col = f"target_{horizon}"
        valid = df_feat.dropna(subset=FEATURE_COLUMNS + [target_col])
        if valid.empty:
            raise HTTPException(422, f"No valid Ohio test rows for {horizon}-minute horizon.")
        actual = valid[target_col].to_numpy()
        predicted = store._models[horizon].predict(valid[FEATURE_COLUMNS].values)  # type: ignore[attr-defined]
        metrics[str(horizon)] = {
            "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
            "mae": round(float(mean_absolute_error(actual, predicted)), 2),
            "n_rows": int(len(valid)),
        }
    return {"dataset": "OhioT1DM held-out test set", "metrics": metrics}


@app.get("/v1/backtest/patient/{patient_id}")
def backtest_patient_metrics(patient_id: str) -> dict:
    """Return Ohio held-out RMSE/MAE for one patient at every horizon."""
    store = get_store()
    if not OHIO_TEST_CSV.is_file():
        raise HTTPException(404, f"Ohio test CSV not found: {OHIO_TEST_CSV}")

    try:
        df = pd.read_csv(OHIO_TEST_CSV)
    except Exception as exc:
        raise HTTPException(422, f"Could not read Ohio test CSV: {exc}")
    required_cols = {"timestamp", "glucose", "patient_id"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(422, "Ohio test CSV must contain timestamp, glucose, and patient_id columns.")
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S")
    except (ValueError, TypeError):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, f"Could not parse Ohio test timestamps: {exc}")

    patient_rows = df[df["patient_id"].astype(str) == patient_id]
    if patient_rows.empty:
        available = sorted(df["patient_id"].dropna().astype(str).unique().tolist())
        raise HTTPException(404, f"Patient '{patient_id}' not found in Ohio test CSV. Available: {available}")
    if any(not store.is_loaded(horizon) for horizon in HORIZONS):
        raise HTTPException(503, "Models for all forecast horizons must be loaded to calculate patient RMSE.")

    patient_rows = patient_rows.sort_values("timestamp").reset_index(drop=True)
    df_feat = build_features(patient_rows, group_col="patient_id")
    metrics: dict[str, dict[str, float | int]] = {}
    for horizon in HORIZONS:
        target_col = f"target_{horizon}"
        valid = df_feat.dropna(subset=FEATURE_COLUMNS + [target_col]).reset_index(drop=True)
        if valid.empty:
            raise HTTPException(422, f"Not enough valid test rows for patient '{patient_id}'.")
        actual = valid[target_col].to_numpy()
        predicted = store._models[horizon].predict(valid[FEATURE_COLUMNS].values)  # type: ignore[attr-defined]
        metrics[str(horizon)] = {
            "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
            "mae": round(float(mean_absolute_error(actual, predicted)), 2),
            "n_rows": int(len(valid)),
        }

    return {"patient_id": patient_id, "metrics": metrics}
