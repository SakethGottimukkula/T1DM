"""Glucose Horizon — FastAPI application."""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from . import config
from .casedata import get_case_data
from .config import (
    CORS_ORIGINS,
    GLUCOSE_MAX,
    GLUCOSE_MIN,
    HORIZONS,
    METRICS_PATH,
    MIN_HISTORY,
)
from .ega import classify_point
from .features import FEATURE_COLUMNS, build_features, build_features_single
from .metrics import summarize
from .metricsfile import check_consistency, load_metrics, overall_status
from .timeutil import parse_timestamps
from .predictor import ModelStore
from .schemas import (
    BacktestResponse,
    CaseDetail,
    CaseListResponse,
    CaseReading,
    CaseSummary,
    DataSourceInfo,
    HealthResponse,
    HorizonResult,
    MetricsResponse,
    ModelInfo,
    ModelsResponse,
    PatientInfo,
    PredictContext,
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


def _build_context(
    df_with_features: pd.DataFrame,
    features_row: dict[str, float],
    n_submitted: int,
    patient_id: str | None = None,
) -> PredictContext:
    window_df = df_with_features.iloc[-MIN_HISTORY:]
    window_start_dt = window_df["timestamp"].iloc[0]
    window_end_dt = window_df["timestamp"].iloc[-1]
    span_min = round((window_end_dt - window_start_dt).total_seconds() / 60.0, 2)
    gluc_series = window_df["glucose"]

    return PredictContext(
        readings_submitted=int(n_submitted),
        window_readings=MIN_HISTORY,
        window_span_minutes=span_min,
        window_start=window_start_dt.isoformat(),
        window_end=window_end_dt.isoformat(),
        current_glucose=round(float(gluc_series.iloc[-1]), 2),
        window_min=round(float(gluc_series.min()), 2),
        window_max=round(float(gluc_series.max()), 2),
        window_mean=round(float(gluc_series.mean()), 2),
        change_5min=round(float(features_row["delta_5"]), 2),
        change_30min=round(float(features_row["delta_30"]), 2),
        patient_id=str(patient_id) if patient_id is not None else None,
        features={col: round(float(features_row[col]), 4) for col in FEATURE_COLUMNS},
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
            sha256=store.model_sha256(hz),
            size_bytes=store.model_size(hz),
        )
        for hz in HORIZONS
    ]
    metrics, load_status = load_metrics(METRICS_PATH)
    warnings = check_consistency(metrics, store) if load_status == "ok" else []
    status = overall_status(load_status, warnings, metrics)
    feat_imp = (
        metrics.get("feature_importance")
        if (metrics is not None and load_status == "ok")
        else None
    )
    return ModelsResponse(
        horizons=HORIZONS,
        feature_columns=store.feature_columns,
        models=model_infos,
        metrics_status=status,
        metrics_warnings=warnings,
        feature_importance=feat_imp,
    )


@app.get("/v1/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    store = get_store()
    metrics, load_status = load_metrics(METRICS_PATH)
    if load_status != "ok" or metrics is None:
        raise HTTPException(
            503,
            f"Metrics file could not be loaded ({load_status}). "
            "Please run 'python scripts/evaluate.py' to generate backend/models/metrics.json.",
        )
    warnings = check_consistency(metrics, store)
    status = overall_status(load_status, warnings, metrics)
    return MetricsResponse(
        status=status,
        warnings=warnings,
        metrics=metrics,
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
    context = _build_context(df, features_row, len(df), patient_id=None)

    return PredictResponse(
        predictions=preds,
        latest_glucose=latest_glucose,
        timestamp=ts,
        context=context,
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
    selected_pid: str | None = None
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
            selected_pid = single_pid
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
            selected_pid = pid_str

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
    context = _build_context(df, features_row, len(df), patient_id=selected_pid)

    return PredictResponse(
        predictions=preds,
        latest_glucose=latest_glucose,
        timestamp=ts,
        context=context,
    )


@app.post("/v1/backtest", response_model=BacktestResponse)
async def backtest(
    file: UploadFile | None = File(default=None),
    horizon: int = Form(...),
    max_points: int = Form(default=2000),
) -> BacktestResponse:
    store = get_store()

    # 1. Horizon & max_points valid (422)
    if horizon not in HORIZONS:
        raise HTTPException(422, "horizon must be 30, 60, 90, or 120")
    if max_points < 50 or max_points > 5000:
        raise HTTPException(422, "max_points must be between 50 and 5000")

    # 2. Source data (upload vs server)
    if file is not None:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(422, f"Could not parse CSV: {exc}")

        n_rows_raw = len(df)
        data_source = "upload"

        required_cols = {"timestamp", "glucose", "patient_id"}
        if not required_cols.issubset(df.columns):
            raise HTTPException(
                422,
                f"CSV must contain 'timestamp', 'glucose', and 'patient_id' columns. Got: {list(df.columns)}",
            )

        try:
            df["timestamp"] = parse_timestamps(df["timestamp"])
        except ValueError as exc:
            raise HTTPException(422, str(exc))
    else:
        case_data, data_status = get_case_data()
        if data_status != "ok" or case_data is None:
            if data_status == "not_configured":
                raise HTTPException(
                    422,
                    "No file uploaded and TEST_CSV_PATH is not set. Upload a CSV or configure TEST_CSV_PATH.",
                )
            name = config.TEST_CSV_PATH.name if config.TEST_CSV_PATH else "unknown"
            if data_status == "file_missing":
                raise HTTPException(
                    422,
                    f"No file uploaded and the server test data file '{name}' was not found.",
                )
            raise HTTPException(
                422,
                f"No file uploaded and the server test data file '{name}' is invalid.",
            )
        df = case_data.raw_frame.copy()
        n_rows_raw = len(df)
        data_source = "server"

    # 3. Model loaded check (503)
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
    df_feat = build_features(df, group_col="patient_id")

    # 3. Canonical row selection: dropna on FEATURE_COLUMNS + all four targets
    target_cols = [f"target_{h}" for h in HORIZONS]
    valid_cols = FEATURE_COLUMNS + target_cols
    df_valid = df_feat.dropna(subset=valid_cols).reset_index(drop=True)

    if len(df_valid) == 0:
        raise HTTPException(422, "No rows with enough history and valid targets.")

    # 4. Predict using store.predict_batch
    target_col = f"target_{horizon}"
    actuals = df_valid[target_col].to_numpy(dtype=float)
    preds = store.predict_batch(df_valid[FEATURE_COLUMNS], horizon)

    # 5. Metrics via summarize()
    overall = summarize(actuals, preds)

    per_patient: dict[str, dict] = {}
    for pid in df_valid["patient_id"].unique():
        p_mask = (df_valid["patient_id"] == pid).to_numpy()
        p_actuals = actuals[p_mask]
        p_preds = preds[p_mask]
        per_patient[str(pid)] = summarize(p_actuals, p_preds)

    # 6. Plot sampling: at most max_points rows
    n = len(df_valid)
    if n <= max_points:
        indices = np.arange(n)
    else:
        indices = np.unique(np.round(np.linspace(0, n - 1, max_points)).astype(int))

    timestamps_arr = [df_valid["timestamp"].iloc[i].isoformat() for i in indices]
    actual_arr = [round(float(actuals[i]), 2) for i in indices]
    predicted_arr = [round(float(preds[i]), 2) for i in indices]
    patient_ids_arr = [
        int(p) if isinstance(p, (int, np.integer)) or (isinstance(p, str) and p.isdigit()) else str(p)
        for p in df_valid["patient_id"].iloc[indices]
    ]
    zones_arr = [classify_point(float(actuals[i]), float(preds[i])) for i in indices]

    return BacktestResponse(
        n_rows=overall["n_rows"],
        rmse=overall["rmse"],
        mae=overall["mae"],
        zone_counts=overall["zone_counts"],
        zone_percentages=overall["zone_percentages"],
        clarke_zone_counts=overall["zone_counts"],
        clarke_zone_percentages=overall["zone_percentages"],
        clarke_zones={"counts": overall["zone_counts"], "percentages": overall["zone_percentages"]},
        a_plus_b_pct=overall["a_plus_b_pct"],
        clarke_ab_pct=overall["a_plus_b_pct"],
        warnings=warnings,
        timestamps=timestamps_arr,
        actual=actual_arr,
        predicted=predicted_arr,
        horizon=horizon,
        n_rows_raw=n_rows_raw,
        evaluation_rule="rows where all 18 features and all 4 targets are non-null (identical to Model/XGBoost.ipynb)",
        per_patient=per_patient,
        patient_ids=patient_ids_arr,
        zones=zones_arr,
        data_source=data_source,
    )


# ---------------------------------------------------------------------------
# Case inspection endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/cases", response_model=CaseListResponse)
async def list_cases(
    per_patient: int = Query(default=3, ge=1, le=10, description="Cases per patient (1–10)"),
) -> CaseListResponse:
    case_data, data_status = get_case_data()

    if data_status != "ok" or case_data is None:
        if data_status == "not_configured":
            raise HTTPException(
                status_code=503,
                detail="Server test data is not configured (TEST_CSV_PATH environment variable is unset).",
            )
        file_name = config.TEST_CSV_PATH.name if config.TEST_CSV_PATH else "unknown"
        if data_status == "file_missing":
            raise HTTPException(
                status_code=503,
                detail=f"Server test data file '{file_name}' not found.",
            )
        raise HTTPException(
            status_code=503,
            detail=f"Server test data file '{file_name}' is invalid.",
        )

    ds_info = DataSourceInfo(
        configured=True,
        file_name=case_data.file_name,
        status="ok",
        total_readings=case_data.total_readings,
        total_patients=case_data.total_patients,
        total_eligible_cases=case_data.total_eligible_cases,
    )
    patients = [PatientInfo(**p) for p in case_data.get_patients_info()]
    cases = [CaseSummary(**c) for c in case_data.get_cases(per_patient=per_patient)]

    return CaseListResponse(
        data_source=ds_info,
        patients=patients,
        cases=cases,
        total_cases=len(cases),
    )


@app.get("/v1/cases/{case_id}", response_model=CaseDetail)
async def get_case(case_id: str) -> CaseDetail:
    case_data, data_status = get_case_data()

    if data_status != "ok" or case_data is None:
        if data_status == "not_configured":
            raise HTTPException(
                status_code=503,
                detail="Server test data is not configured (TEST_CSV_PATH environment variable is unset).",
            )
        file_name = config.TEST_CSV_PATH.name if config.TEST_CSV_PATH else "unknown"
        if data_status == "file_missing":
            raise HTTPException(
                status_code=503,
                detail=f"Server test data file '{file_name}' not found.",
            )
        raise HTTPException(
            status_code=503,
            detail=f"Server test data file '{file_name}' is invalid.",
        )

    detail = case_data.get_case_detail(case_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")

    return CaseDetail(**detail)
