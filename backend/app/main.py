"""Glucose Horizon — FastAPI application."""

from __future__ import annotations

import io
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import (
    CORS_ORIGINS,
    GLUCOSE_MAX,
    GLUCOSE_MIN,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    HORIZONS,
    MIN_HISTORY,
    OHIO_TEST_CSV,
)
from .ega import classify_array
from .features import FEATURE_COLUMNS, build_features, build_features_single
from .predictor import ModelStore
from .wellness import get_wellness_suggestion
from agent.router import router as agent_router
from agent.patient_data import patient_dataset_store
from .simglucose_features import SIM_FEATURE_COLUMNS, build_sim_features
from .schemas import (
    BacktestResponse,
    HealthResponse,
    HorizonResult,
    ModelInfo,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
    WellnessRequest,
    WellnessResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

_store: ModelStore | None = None
_sim_models: dict[int, object] = {}
_sim_eval_models: dict[int, object] = {}
_sim_eval_patient_ids: list[str] = []


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
    _sim_eval_models.clear()
    _sim_eval_patient_ids.clear()
    for horizon in HORIZONS:
        path = model_dir / f"model_{horizon}.json"
        if path.is_file():
            model = xgb.XGBRegressor()
            model.load_model(str(path))
            _sim_models[horizon] = model
    eval_dir = model_dir / "evaluation"
    ids_path = eval_dir / "patient_ids.json"
    if ids_path.is_file():
        _sim_eval_patient_ids.extend(json.loads(ids_path.read_text()))
    for horizon in HORIZONS:
        path = eval_dir / f"model_{horizon}.json"
        if path.is_file():
            model = xgb.XGBRegressor()
            model.load_model(str(path))
            _sim_eval_models[horizon] = model


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


app.include_router(agent_router)


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
    raw_row_count = len(df)
    df.columns = [str(col).strip().lower() for col in df.columns]
    csv_headers = list(df.columns)
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
    invalid_rows = int(df[["timestamp", "cgm", "bg", "insulin"]].isna().any(axis=1).sum())
    df = df.dropna(subset=["timestamp", "cgm", "bg", "insulin"]).sort_values("timestamp")
    duplicate_timestamps = int(df["timestamp"].duplicated(keep="last").sum())
    df = df.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    patient_dataset = df.reset_index().copy()
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
    evaluation: dict[str, dict[str, object]] = {}
    for eval_horizon in HORIZONS:
        target_col = f"target_{eval_horizon}"
        valid = feature_df.dropna(subset=SIM_FEATURE_COLUMNS + [target_col]).reset_index(drop=True)
        if valid.empty:
            continue
        actual = valid[target_col].to_numpy()
        predicted = _sim_models[eval_horizon].predict(valid[SIM_FEATURE_COLUMNS].to_numpy(dtype=float))
        labels, zone_counts = classify_array(actual.tolist(), predicted.tolist())
        count = len(valid)
        stride = max(1, int(np.ceil(count / 1500)))
        evaluation[str(eval_horizon)] = {
            "n_rows": count,
            "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
            "mae": round(float(mean_absolute_error(actual, predicted)), 2),
            "zone_percentages": {zone: round(zone_counts[zone] * 100 / count, 2) for zone in ["A", "B", "C", "D", "E"]},
            "points": [
                {"x": round(float(a), 2), "y": round(float(p), 2), "z": zone}
                for a, p, zone in zip(actual[::stride], predicted[::stride], labels[::stride])
            ],
            "timestamps": [str(ts) for ts in valid["timestamp"].iloc[::stride]],
            "actual": [round(float(value), 2) for value in actual[::stride]],
            "predicted": [round(float(value), 2) for value in predicted[::stride]],
        }
    history = feature_df.tail(48)[["timestamp", "cgm"]].rename(columns={"cgm": "glucose"})
    agent_dataset_id, agent_dataset_rows, agent_dataset_columns = patient_dataset_store.register(patient_dataset)
    return {"predictions": predictions, "latest_glucose": latest, "timestamp": str(last["timestamp"]),
            "meal_defaulted_to_zero": meal_defaulted, "headers": csv_headers, "row_count": raw_row_count,
            "agent_dataset_id": agent_dataset_id, "agent_dataset_rows": agent_dataset_rows,
            "agent_dataset_columns": agent_dataset_columns,
            "readings": [{"timestamp": str(row.timestamp), "glucose": round(float(row.glucose), 2)} for row in history.itertuples(index=False)],
            "evaluation": evaluation,
            "warnings": {"invalid_rows_dropped": invalid_rows, "duplicate_timestamps_removed": duplicate_timestamps,
                         "rows_resampled_or_missing": max(0, raw_row_count - len(df))}}


@app.get("/v1/simglucose/test/patients")
def simglucose_test_patients() -> dict:
    if not _sim_eval_patient_ids:
        raise HTTPException(503, "SimGlucose evaluation models are unavailable; rerun the SimGlucose training script.")
    return {"patients": _sim_eval_patient_ids, "horizons": sorted(_sim_eval_models)}


@app.get("/v1/simglucose/test/patient/{patient_id}")
def simglucose_test_patient(patient_id: str, horizon: int = 30, case_index: int = 2) -> dict:
    if patient_id not in _sim_eval_patient_ids:
        raise HTTPException(404, f"Held-out SimGlucose patient '{patient_id}' not found.")
    if horizon not in HORIZONS or horizon not in _sim_eval_models:
        raise HTTPException(422, "horizon must be 30, 60, 90, or 120")
    csv_path = Path(__file__).resolve().parent.parent / "data" / "simglucose_training.csv"
    if not csv_path.is_file():
        raise HTTPException(404, "SimGlucose training dataset is unavailable.")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df[df["patient_id"].astype(str) == patient_id].sort_values("timestamp")
    if df.empty:
        raise HTTPException(404, f"No SimGlucose records found for patient '{patient_id}'.")
    df = df.set_index("timestamp").resample("5min").agg({"bg": "mean", "cgm": "mean", "insulin": "sum", "cho": "sum"}).dropna().reset_index()
    targets = [f"target_{hz}" for hz in HORIZONS]
    features = build_sim_features(df).dropna(subset=SIM_FEATURE_COLUMNS + targets).reset_index(drop=True)
    if features.empty:
        raise HTTPException(422, "No valid evaluation rows for this patient.")
    case_index = max(0, min(4, case_index))
    row_index = round(case_index * (len(features) - 1) / 4)
    selected = features.iloc[row_index]
    horizon_predictions = {}
    patient_metrics = {}
    scatter = {}
    for hz in HORIZONS:
        actual_values = features[f"target_{hz}"].to_numpy()
        predicted_values = _sim_eval_models[hz].predict(features[SIM_FEATURE_COLUMNS].to_numpy())
        labels, zone_counts = classify_array(actual_values.tolist(), predicted_values.tolist())
        row_actual = float(actual_values[row_index])
        row_predicted = float(predicted_values[row_index])
        horizon_predictions[str(hz)] = {
            "actual": round(row_actual, 2), "predicted": round(row_predicted, 2),
            "error": round(abs(row_actual - row_predicted), 2),
            "clarke_zone": labels[row_index],
        }
        n = len(actual_values)
        patient_metrics[str(hz)] = {
            "rmse": round(float(np.sqrt(mean_squared_error(actual_values, predicted_values))), 2),
            "mae": round(float(mean_absolute_error(actual_values, predicted_values)), 2),
            "zone_percentages": {z: round(zone_counts[z] * 100 / n, 2) for z in ["A", "B", "C", "D", "E"]},
            "n_rows": n,
        }
        stride = max(1, int(np.ceil(n / 500)))
        scatter[str(hz)] = [{"x": round(float(a), 2), "y": round(float(p), 2), "z": zone}
                           for a, p, zone in zip(actual_values[::stride], predicted_values[::stride], labels[::stride])]
    selected_prediction = horizon_predictions[str(horizon)]
    selected_actual = selected_prediction["actual"]
    selected_predicted = selected_prediction["predicted"]
    context_rows = df[df["timestamp"] <= pd.Timestamp(selected["timestamp"])].tail(36)
    cgm = context_rows["cgm"].astype(float)
    insulin_total = float(context_rows["insulin"].tail(12).sum())
    cho_total = float(context_rows["cho"].tail(12).sum())
    history = [{"x": (i - len(context_rows) + 1) * 5, "y": round(float(value), 2)}
               for i, value in enumerate(cgm.tolist())]
    future_rows = df[df["timestamp"] > pd.Timestamp(selected["timestamp"])].head(24)
    future = [{"x": (i + 1) * 5, "y": round(float(value), 2)}
              for i, value in enumerate(future_rows["bg"].tolist())]
    return {"patient_id": patient_id, "horizon": horizon, "n_rows": len(features),
            "rmse": patient_metrics[str(horizon)]["rmse"],
            "mae": patient_metrics[str(horizon)]["mae"],
            "timestamp": str(selected["timestamp"]), "actual": round(selected_actual, 2),
            "predicted": round(selected_predicted, 2),
            "error": round(abs(selected_actual - selected_predicted), 2),
            "clarke_zone": selected_prediction["clarke_zone"],
            "predictions": horizon_predictions, "metrics": patient_metrics, "scatter": scatter,
            "series": {"history": history, "actual_future": future},
            "context": {"current": round(float(cgm.iloc[-1]), 2),
                        "roc": round(float(cgm.iloc[-1] - cgm.iloc[-4]), 2) if len(cgm) >= 4 else 0.0,
                        "mean": round(float(cgm.mean()), 2),
                        "sd": round(float(cgm.std(ddof=0)), 2),
                        "tir": round(float(((cgm >= 70) & (cgm <= 180)).mean() * 100), 1),
                        "insulin_60": round(insulin_total, 3), "cho_60": round(cho_total, 2)}}


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
    raw_row_count = len(df)

    df.columns = [str(col).strip().lower() for col in df.columns]
    csv_headers = list(df.columns)
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
    patient_dataset = df.drop(columns=["patient_id"], errors="ignore").copy()
    eval_rows = df.copy()
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

    # Score historical horizons in this upload wherever a future glucose value exists.
    eval_features = build_features(eval_rows, group_col="patient_id")
    evaluation: dict[str, dict[str, object]] = {}
    for eval_horizon in HORIZONS:
        target_col = f"target_{eval_horizon}"
        valid = eval_features.dropna(subset=FEATURE_COLUMNS + [target_col]).reset_index(drop=True)
        if valid.empty:
            continue
        actual = valid[target_col].to_numpy()
        predicted = store._models[eval_horizon].predict(valid[FEATURE_COLUMNS].values)
        labels, zone_counts = classify_array(actual.tolist(), predicted.tolist())
        count = len(valid)
        stride = max(1, int(np.ceil(count / 1500)))
        evaluation[str(eval_horizon)] = {
            "n_rows": count,
            "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
            "mae": round(float(mean_absolute_error(actual, predicted)), 2),
            "zone_percentages": {zone: round(zone_counts[zone] * 100 / count, 2) for zone in ["A", "B", "C", "D", "E"]},
            "points": [
                {"x": round(float(a), 2), "y": round(float(p), 2), "z": zone}
                for a, p, zone in zip(actual[::stride], predicted[::stride], labels[::stride])
            ],
            "timestamps": [str(ts) for ts in valid["timestamp"].iloc[::stride]],
            "actual": [round(float(value), 2) for value in actual[::stride]],
            "predicted": [round(float(value), 2) for value in predicted[::stride]],
        }

    agent_dataset_id, agent_dataset_rows, agent_dataset_columns = patient_dataset_store.register(patient_dataset)
    return {
        "predictions": [pred.model_dump() for pred in preds],
        "latest_glucose": latest_glucose,
        "timestamp": ts,
        "patient_used": patient_used,
        "patients_available": [str(value) for value in patients_available],
        "warnings": warnings,
        "headers": csv_headers,
        "row_count": raw_row_count,
        "agent_dataset_id": agent_dataset_id,
        "agent_dataset_rows": agent_dataset_rows,
        "agent_dataset_columns": agent_dataset_columns,
        "readings": [
            {"timestamp": str(row.timestamp), "glucose": round(float(row.glucose), 2)}
            for row in tail[["timestamp", "glucose"]].itertuples(index=False)
        ],
        "evaluation": evaluation,
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
    clarke: dict[str, dict[str, object]] = {}
    for horizon in HORIZONS:
        target_col = f"target_{horizon}"
        valid = df_feat.dropna(subset=FEATURE_COLUMNS + [target_col]).reset_index(drop=True)
        if valid.empty:
            raise HTTPException(422, f"Not enough valid test rows for patient '{patient_id}'.")
        actual = valid[target_col].to_numpy()
        predicted = store._models[horizon].predict(valid[FEATURE_COLUMNS].values)  # type: ignore[attr-defined]
        labels, zone_counts = classify_array(actual.tolist(), predicted.tolist())
        n_rows = len(actual)
        zone_percentages = {
            zone: round(zone_counts[zone] * 100 / n_rows, 2)
            for zone in ["A", "B", "C", "D", "E"]
        }
        metrics[str(horizon)] = {
            "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
            "mae": round(float(mean_absolute_error(actual, predicted)), 2),
            "n_rows": int(n_rows),
        }
        stride = max(1, int(np.ceil(n_rows / 500)))
        clarke[str(horizon)] = {
            "percentages": zone_percentages,
            "points": [
                {"x": round(float(a), 2), "y": round(float(p), 2), "z": zone}
                for a, p, zone in zip(actual[::stride], predicted[::stride], labels[::stride])
            ],
        }

    return {"patient_id": patient_id, "metrics": metrics, "clarke": clarke}


# ---------------------------------------------------------------------------
# Wellness agent (Groq)
# ---------------------------------------------------------------------------


@app.post("/v1/wellness", response_model=WellnessResponse)
async def wellness(req: WellnessRequest) -> WellnessResponse:
    """Return a Groq-powered wellness suggestion (meal ideas, workout plans, etc.)."""
    return await get_wellness_suggestion(req.message)
