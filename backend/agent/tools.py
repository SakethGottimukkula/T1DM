"""Tool definitions and execution engine for the T1DM forecasting agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from app.config import GLUCOSE_MAX, GLUCOSE_MIN, HORIZONS, MIN_HISTORY
from app.features import FEATURE_COLUMNS, build_features_single
from app.predictor import ModelStore
from .patient_data import normalize_column_name

# Benchmark model performance metrics on OhioT1DM held-out test cohort
STORED_PERFORMANCE: dict[int, dict[str, float]] = {
    30: {"rmse": 13.52, "mae": 9.81, "clarke_a_pct": 86.50, "clarke_b_pct": 11.79, "clarke_ab_pct": 98.29},
    60: {"rmse": 18.84, "mae": 13.72, "clarke_a_pct": 82.47, "clarke_b_pct": 14.25, "clarke_ab_pct": 96.72},
    90: {"rmse": 22.61, "mae": 16.54, "clarke_a_pct": 79.31, "clarke_b_pct": 15.83, "clarke_ab_pct": 95.14},
    120: {"rmse": 25.83, "mae": 19.12, "clarke_a_pct": 75.72, "clarke_b_pct": 18.15, "clarke_ab_pct": 93.87},
}

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "lag_1": "Most recent glucose reading (t-5 min)",
    "lag_2": "Glucose reading at t-10 min",
    "lag_3": "Glucose reading at t-15 min",
    "lag_6": "Glucose reading at t-30 min",
    "lag_12": "Glucose reading at t-60 min",
    "delta_5": "Rate of change over the last 5 minutes",
    "delta_30": "Rate of change over the last 30 minutes",
    "rolling_mean_30": "30-minute average glucose",
    "rolling_mean_60": "60-minute average glucose",
    "rolling_mean_90": "90-minute average glucose",
    "rolling_mean_120": "120-minute average glucose",
    "rolling_std_30": "30-minute glucose variability (standard deviation)",
    "rolling_std_60": "60-minute glucose variability (standard deviation)",
    "rolling_std_90": "90-minute glucose variability (standard deviation)",
    "rolling_std_120": "120-minute glucose variability (standard deviation)",
    "hour": "Hour of the day (circadian baseline)",
    "hour_sin": "Cyclic time transform (sine)",
    "hour_cos": "Cyclic time transform (cosine)",
}


class PatientContext:
    """Holds time-series CGM readings, active forecast context, and precomputed feature row for tool execution."""

    def __init__(
        self,
        df: pd.DataFrame,
        patient_id: str = "Uploaded patient",
        forecast_context: Optional[dict[str, Any]] = None,
        patient_dataset: Optional[pd.DataFrame] = None,
    ):
        self.patient_id = patient_id
        self.forecast_context: dict[str, Any] = forecast_context or {}
        self.df = df.sort_values("timestamp").reset_index(drop=True)
        self.patient_dataset = (patient_dataset if patient_dataset is not None else self.df).copy(deep=True)
        self.feature_df = build_features_single(self.df.copy())
        self.last_row = self.feature_df.iloc[-1]
        self.latest_glucose = float(self.last_row["glucose"])
        self.latest_timestamp = str(self.last_row["timestamp"])
        self.features_dict = {col: float(self.last_row[col]) for col in FEATURE_COLUMNS}

    @classmethod
    def from_readings(
        cls,
        readings: list[dict[str, Any]],
        patient_id: str = "Uploaded patient",
        forecast_context: Optional[dict[str, Any]] = None,
        patient_dataset: Optional[pd.DataFrame] = None,
    ) -> "PatientContext":
        """Build context from a list of dicts with 'timestamp' and 'glucose'."""
        if len(readings) < MIN_HISTORY:
            raise ValueError(f"At least {MIN_HISTORY} readings required; got {len(readings)}.")
        df = pd.DataFrame(readings)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["glucose"] = df["glucose"].astype(float)
        return cls(df, patient_id=patient_id, forecast_context=forecast_context, patient_dataset=patient_dataset)


# ---------------------------------------------------------------------------
# Pydantic Input Schemas for Tools
# ---------------------------------------------------------------------------

class GetForecastInput(BaseModel):
    horizon_min: int = Field(
        ...,
        description="Forecast horizon in minutes: must be 30, 60, 90, or 120."
    )

    @field_validator("horizon_min")
    @classmethod
    def validate_horizon(cls, v: int) -> int:
        if v not in (30, 60, 90, 120):
            raise ValueError("horizon_min must be one of [30, 60, 90, 120]")
        return v


class GetAllForecastsInput(BaseModel):
    pass


class GetRecentGlucoseInput(BaseModel):
    minutes: int = Field(
        default=60,
        description="Minutes of recent history to summarize (e.g. 15, 30, 60, 120, 180). Default is 60."
    )


class AssessRiskInput(BaseModel):
    pass


class GetModelPerformanceInput(BaseModel):
    horizon_min: Optional[int] = Field(
        default=None,
        description="Specific horizon in minutes (30, 60, 90, or 120), or omit/null for all horizons."
    )


class ExplainForecastInput(BaseModel):
    horizon_min: int = Field(
        default=60,
        description="Forecast horizon to explain (30, 60, 90, or 120)."
    )

    @field_validator("horizon_min")
    @classmethod
    def validate_horizon(cls, v: int) -> int:
        if v not in (30, 60, 90, 120):
            raise ValueError("horizon_min must be one of [30, 60, 90, 120]")
        return v


class PatientDatasetQueryInput(BaseModel):
    column: Optional[str] = Field(default=None, description="A measurement column from the uploaded patient dataset.")
    operation: str = Field(default="value", description="One of value, mean, min, max, median, std, sum, count, mode, unique, rows, or columns.")
    timestamp: Optional[str] = Field(default=None, description="A time to look up; the nearest recorded sample is returned.")
    start_time: Optional[str] = Field(default=None, description="Optional inclusive start of a time range.")
    end_time: Optional[str] = Field(default=None, description="Optional inclusive end of a time range.")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum rows to return for a rows query.")
    offset: int = Field(default=0, ge=0, le=1000000, description="Zero-based row offset for paging through a rows query.")

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, value: str) -> str:
        allowed = {"value", "mean", "min", "max", "median", "std", "sum", "count", "mode", "unique", "rows", "columns"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"operation must be one of {sorted(allowed)}")
        return normalized


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

def tool_get_forecast(args: dict[str, Any], context: PatientContext, store: ModelStore) -> dict[str, Any]:
    """Run or retrieve the calibrated XGBoost prediction for a given horizon."""
    validated = GetForecastInput(**args)
    hz = validated.horizon_min

    fc = getattr(context, "forecast_context", None) or {}
    # Check if forecast context explicitly indicates no forecast has been run yet
    if "has_forecast" in fc and not fc["has_forecast"]:
        return {
            "status": "no_forecast_available",
            "has_forecast": False,
            "horizon_min": hz,
            "patient_id": str(fc.get("patient_id", context.patient_id)),
            "selected_patient_id": str(fc.get("selected_patient_id", context.patient_id)),
            "current_glucose_mg_dl": round(float(fc.get("current_glucose", context.latest_glucose)), 1),
            "message": f"No forecast has been run yet for patient {fc.get('patient_id', context.patient_id)}. Inform the user that no patient-specific forecast results are available, and suggest running 'Run Forecast' in the dashboard or uploading CGM CSV data.",
        }

    # If predictions are already available from the pipeline or upload
    preds = fc.get("predictions")
    if preds and (str(hz) in preds or hz in preds):
        item = preds.get(str(hz)) or preds.get(hz)
        pred_val = round(float(item.get("predicted", item.get("predicted_glucose", 0.0))), 1)
        cur_val = round(float(fc.get("current_glucose", context.latest_glucose)), 1)
        change_val = round(float(item.get("change", item.get("change_from_latest", pred_val - cur_val))), 1)
        origin_timestamp = str(fc.get("timestamp", context.latest_timestamp))
        try:
            forecast_timestamp = (pd.Timestamp(origin_timestamp) + pd.Timedelta(minutes=hz)).isoformat()
        except (TypeError, ValueError):
            forecast_timestamp = None
        zone = item.get("zone", item.get("zone_label"))
        if not zone or zone in ("low", "high"):
            if pred_val < 54:
                zone = "severe_hypoglycemia"
            elif pred_val < 70:
                zone = "hypoglycemia"
            elif pred_val > 250:
                zone = "severe_hyperglycemia"
            elif pred_val > 180:
                zone = "hyperglycemia"
            else:
                zone = "in_range"

        res: dict[str, Any] = {
            "horizon_min": hz,
            "predicted_glucose_mg_dl": pred_val,
            "current_glucose_mg_dl": cur_val,
            "change_mg_dl": change_val,
            "zone": zone,
            "timestamp": forecast_timestamp or origin_timestamp,
            "forecast_origin_timestamp": origin_timestamp,
            "forecast_for_timestamp": forecast_timestamp,
            "patient_id": str(fc.get("patient_id", context.patient_id)),
            "selected_patient_id": str(fc.get("selected_patient_id", context.patient_id)),
            "source": fc.get("source", "pipeline"),
        }
        if "actual" in item and item["actual"] is not None:
            res["actual_held_out_glucose_mg_dl"] = round(float(item["actual"]), 1)
        if "error" in item and item["error"] is not None:
            res["prediction_error_mg_dl"] = round(float(item["error"]), 1)
        if "clarke_zone" in item and item["clarke_zone"]:
            res["clarke_zone"] = item["clarke_zone"]
        return res

    return {
        "status": "horizon_unavailable",
        "has_forecast": False,
        "horizon_min": hz,
        "available_horizons": sorted(int(value) for value in preds) if isinstance(preds, dict) else [],
        "message": (
            f"No uploaded forecast is available for {hz} minutes. Do not calculate or estimate one. "
            "Tell the user which horizons are available and ask them to select the requested horizon or all horizons in CSV upload, then upload and predict again."
        ),
    }


def tool_get_all_forecasts(args: dict[str, Any], context: PatientContext, store: ModelStore) -> dict[str, Any]:
    """Run all 4 forecast horizons (30, 60, 90, 120 minutes) at once."""
    fc = getattr(context, "forecast_context", None) or {}
    cur_val = round(float(fc.get("current_glucose", context.latest_glucose)), 1)
    pt_id = str(fc.get("patient_id", context.patient_id))

    if "has_forecast" in fc and not fc["has_forecast"]:
        return {
            "status": "no_forecast_available",
            "has_forecast": False,
            "patient_id": pt_id,
            "current_glucose_mg_dl": cur_val,
            "message": f"No forecast has been run yet for patient {pt_id}. Inform the user that no active forecast is available, and suggest running 'Run Forecast' in the dashboard or uploading CGM CSV data.",
            "forecasts": [],
        }

    available = fc.get("predictions")
    if not isinstance(available, dict) or not available:
        return {
            "status": "no_forecast_available",
            "has_forecast": False,
            "patient_id": pt_id,
            "current_glucose_mg_dl": cur_val,
            "forecasts": [],
            "message": "No uploaded forecast values are available. Do not calculate or estimate glucose predictions.",
        }
    results = []
    for hz in HORIZONS:
        if str(hz) not in available and hz not in available:
            continue
        res = tool_get_forecast({"horizon_min": hz}, context, store)
        results.append(res)
    return {
        "patient_id": pt_id,
        "current_glucose_mg_dl": cur_val,
        "forecasts": results,
    }


def tool_get_recent_glucose(args: dict[str, Any], context: PatientContext, store: ModelStore) -> dict[str, Any]:
    """Retrieve recent readings and summary statistics over a specified window."""
    validated = GetRecentGlucoseInput(**args)
    window_mins = max(15, min(validated.minutes, 240))

    readings_count = max(3, window_mins // 5)
    recent_df = context.df.tail(readings_count)
    values = recent_df["glucose"].tolist()

    current_val = float(values[-1])
    # Rate of change over last 15 mins (approx 3 intervals of 5 min)
    if len(values) >= 4:
        roc = round((values[-1] - values[-4]) / 15.0, 2)
    elif len(values) >= 2:
        roc = round((values[-1] - values[0]) / (len(values) * 5.0), 2)
    else:
        roc = 0.0

    fc = getattr(context, "forecast_context", None) or {}
    if "current_glucose" in fc and fc["current_glucose"] is not None:
        current_val = float(fc["current_glucose"])
    if "rate_of_change" in fc and fc["rate_of_change"] is not None:
        roc = float(fc["rate_of_change"])

    in_target = [v for v in values if 70.0 <= v <= 180.0]
    time_in_range_pct = round(len(in_target) * 100.0 / len(values), 1)

    trend_arrow = fc.get("trend") or ("↑" if roc > 1.0 else ("↓" if roc < -1.0 else "→"))

    return {
        "patient_id": str(fc.get("patient_id", context.patient_id)),
        "selected_patient_id": str(fc.get("selected_patient_id", context.patient_id)),
        "current_glucose_mg_dl": round(current_val, 1),
        "window_minutes": window_mins,
        "readings_count": len(values),
        "min_glucose_mg_dl": round(float(np.min(values)), 1),
        "max_glucose_mg_dl": round(float(np.max(values)), 1),
        "mean_glucose_mg_dl": round(float(np.mean(values)), 1),
        "std_dev_mg_dl": round(float(np.std(values)), 1),
        "rate_of_change_mg_dl_per_min": roc,
        "time_in_range_percent": time_in_range_pct,
        "trend_arrow": trend_arrow,
    }


def tool_assess_risk(args: dict[str, Any], context: PatientContext, store: ModelStore) -> dict[str, Any]:
    """Assess hypoglycemia and hyperglycemia risks across current and forecasted glucose."""
    forecasts_res = tool_get_all_forecasts({}, context, store)
    forecasts = forecasts_res.get("forecasts", [])
    current_glucose = forecasts_res.get("current_glucose_mg_dl", round(context.latest_glucose, 1))

    if not forecasts:
        alerts = []
        if current_glucose < 54:
            sev = "severe_hypoglycemia"
            alerts.append({"type": "severe_hypo", "threshold": "<54 mg/dL", "message": f"Current glucose is critically low ({current_glucose} mg/dL). Immediate action required."})
        elif current_glucose < 70:
            sev = "hypoglycemia"
            alerts.append({"type": "hypo", "threshold": "<70 mg/dL", "message": f"Current glucose is below target ({current_glucose} mg/dL). Fast carbs recommended."})
        elif current_glucose > 250:
            sev = "severe_hyperglycemia"
            alerts.append({"type": "severe_hyper", "threshold": ">250 mg/dL", "message": f"Current glucose is severely elevated ({current_glucose} mg/dL). Check ketones and hydrate."})
        elif current_glucose > 180:
            sev = "hyperglycemia"
            alerts.append({"type": "hyper", "threshold": ">180 mg/dL", "message": f"Current glucose is elevated ({current_glucose} mg/dL) exceeding target range."})
        else:
            sev = "in_range"

        return {
            "overall_severity": sev,
            "current_glucose_mg_dl": current_glucose,
            "minimum_projected_glucose": current_glucose,
            "maximum_projected_glucose": current_glucose,
            "alerts": alerts,
            "forecasts_summary": [],
            "note": "Risk assessed on current glucose reading only; no future forecast has been run yet.",
        }

    all_values = [current_glucose] + [f.get("predicted_glucose_mg_dl", current_glucose) for f in forecasts]
    min_val = min(all_values)
    max_val = max(all_values)

    alerts: list[dict[str, Any]] = []
    severity = "in_range"

    if min_val < 54:
        severity = "severe_hypoglycemia"
        earliest_hz = next((f["horizon_min"] for f in forecasts if f.get("predicted_glucose_mg_dl", 999) < 54), 0)
        alerts.append({
            "type": "severe_hypo",
            "threshold": "<54 mg/dL",
            "earliest_horizon_min": earliest_hz,
            "message": f"Critical low predicted ({min_val} mg/dL). Immediate action recommended according to emergency care protocol.",
        })
    elif min_val < 70:
        severity = "hypoglycemia"
        earliest_hz = next((f["horizon_min"] for f in forecasts if f.get("predicted_glucose_mg_dl", 999) < 70), 0)
        alerts.append({
            "type": "hypo",
            "threshold": "<70 mg/dL",
            "earliest_horizon_min": earliest_hz,
            "message": f"Hypoglycemia excursion predicted ({min_val} mg/dL). Review glucose trends.",
        })

    if max_val > 250:
        if severity not in ("severe_hypoglycemia",):
            severity = "severe_hyperglycemia"
        earliest_hz = next((f["horizon_min"] for f in forecasts if f.get("predicted_glucose_mg_dl", 0) > 250), 0)
        alerts.append({
            "type": "severe_hyper",
            "threshold": ">250 mg/dL",
            "earliest_horizon_min": earliest_hz,
            "message": f"Critical high predicted ({max_val} mg/dL). Risk of prolonged hyperglycemia.",
        })
    elif max_val > 180:
        if severity == "in_range":
            severity = "hyperglycemia"
        earliest_hz = next((f["horizon_min"] for f in forecasts if f.get("predicted_glucose_mg_dl", 0) > 180), 0)
        alerts.append({
            "type": "hyper",
            "threshold": ">180 mg/dL",
            "earliest_horizon_min": earliest_hz,
            "message": f"Elevated glucose predicted ({max_val} mg/dL) exceeding standard target range.",
        })

    return {
        "overall_severity": severity,
        "current_glucose_mg_dl": current_glucose,
        "minimum_projected_glucose": min_val,
        "maximum_projected_glucose": max_val,
        "alerts": alerts,
        "forecasts_summary": [
            {"horizon": f["horizon_min"], "predicted": f["predicted_glucose_mg_dl"], "zone": f.get("zone", "in_range")}
            for f in forecasts
        ],
    }


def tool_get_model_performance(args: dict[str, Any], context: PatientContext, store: ModelStore) -> dict[str, Any]:
    """Return model evaluation metrics (RMSE, MAE, and Clarke Error Grid zones)."""
    validated = GetModelPerformanceInput(**args)
    fc = getattr(context, "forecast_context", None) or {}
    patient_rmse = fc.get("patient_rmse")
    patient_id = str(fc.get("patient_id", context.patient_id))

    if validated.horizon_min:
        hz = validated.horizon_min
        data = STORED_PERFORMANCE.get(hz, STORED_PERFORMANCE[60])
        res: dict[str, Any] = {
            "dataset": "OhioT1DM 2020 held-out test cohort",
            "horizon_min": hz,
            "rmse_mg_dl": data["rmse"],
            "mae_mg_dl": data["mae"],
            "clarke_error_grid_zone_a_pct": data["clarke_a_pct"],
            "clarke_error_grid_zone_b_pct": data["clarke_b_pct"],
            "clarke_error_grid_safe_ab_pct": data["clarke_ab_pct"],
            "clinical_interpretation": "Zones A and B represent clinically safe and acceptable predictions in diabetic management literature.",
        }
        if patient_rmse:
            p_val = patient_rmse.get(str(hz)) or patient_rmse.get(hz)
            if p_val is not None:
                res["patient_id"] = patient_id
                res["patient_specific_rmse_mg_dl"] = round(float(p_val), 2)
                res["metric_scope"] = "patient_specific"
        return res

    res = {
        "dataset": "OhioT1DM 2020 held-out test cohort",
        "performance_by_horizon": STORED_PERFORMANCE,
        "note": "Lower RMSE indicates higher precision; A+B % shows clinically safe agreement.",
    }
    if patient_rmse:
        res["patient_id"] = patient_id
        res["patient_specific_rmse"] = patient_rmse
        res["metric_scope"] = "patient_specific"
    return res


def tool_explain_forecast(args: dict[str, Any], context: PatientContext, store: ModelStore) -> dict[str, Any]:
    """Explain which CGM features drove the XGBoost prediction using model feature gains."""
    validated = ExplainForecastInput(**args)
    hz = validated.horizon_min

    if not store.is_loaded(hz):
        store.load(hz)

    # Get model feature importances from XGBoost booster
    model = store._models.get(hz)
    booster = model.get_booster()
    score_dict = booster.get_score(importance_type="gain")

    total_gain = sum(score_dict.values()) if score_dict else 1.0
    sorted_features = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)[:5]

    top_factors = []
    for feat_name, gain in sorted_features:
        pct = round((gain / total_gain) * 100.0, 1)
        val = context.features_dict.get(feat_name, None)
        top_factors.append({
            "feature": feat_name,
            "description": FEATURE_DESCRIPTIONS.get(feat_name, feat_name),
            "patient_value": round(float(val), 2) if val is not None else None,
            "model_importance_pct": pct,
        })

    pred = store.predict(context.features_dict, hz)
    return {
        "horizon_min": hz,
        "predicted_glucose_mg_dl": round(float(pred), 1),
        "current_glucose_mg_dl": round(context.latest_glucose, 1),
        "top_contributing_factors": top_factors,
        "summary": "Recent glucose values (lags) and 5/30-minute rates of change represent the dominant predictors.",
    }


def _json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _measurement_unit(column: str) -> str | None:
    if column in {"glucose", "cgm", "bg", "blood_glucose", "blood_glucose_mg_dl", "glucose_mg_dl"}:
        return "mg/dL"
    if column in {"hr", "heart_rate", "heartrate", "heart_rate_bpm", "pulse", "pulse_rate"}:
        return "bpm"
    return None


def tool_query_patient_dataset(args: dict[str, Any], context: PatientContext, store: ModelStore) -> dict[str, Any]:
    """Query only the measurements from the active uploaded patient dataset."""
    query = PatientDatasetQueryInput(**args)
    frame = getattr(context, "patient_dataset", None)
    if frame is None or frame.empty:
        return {"status": "no_patient_dataset", "message": "No uploaded patient dataset is available in this conversation."}

    columns = [column for column in frame.columns if column != "timestamp"]
    if query.operation == "columns":
        return {
            "status": "ok",
            "record_count": len(frame),
            "time_start": _json_value(frame["timestamp"].min()),
            "time_end": _json_value(frame["timestamp"].max()),
            "measurement_columns": columns,
        }

    requested = normalize_column_name(query.column) if query.column else None
    aliases = {
        "hr": {"hr", "heart_rate", "heartrate", "heart_rate_bpm", "pulse", "pulse_rate"},
        "heart_rate": {"hr", "heart_rate", "heartrate", "heart_rate_bpm", "pulse", "pulse_rate"},
        "heartrate": {"hr", "heart_rate", "heartrate", "heart_rate_bpm", "pulse", "pulse_rate"},
        "glucose": {"glucose", "cgm", "bg", "blood_glucose", "blood_glucose_mg_dl", "glucose_mg_dl"},
        "cgm": {"glucose", "cgm", "bg", "blood_glucose", "blood_glucose_mg_dl", "glucose_mg_dl"},
    }
    matches = aliases.get(requested or "", {requested}) if requested else set()
    column = next((candidate for candidate in columns if candidate == requested or candidate in matches), None)
    if query.operation == "rows" and not requested:
        selected_columns = ["timestamp", *columns]
    elif column is None:
        return {
            "status": "measurement_not_found" if requested else "column_required",
            "requested_column": requested,
            "available_measurements": columns,
            "message": "Choose a measurement from this uploaded patient dataset. No other data sources are available.",
        }
    else:
        selected_columns = ["timestamp", column]

    selected = frame.copy()
    try:
        if query.start_time:
            selected = selected[selected["timestamp"] >= pd.to_datetime(query.start_time, utc=True)]
        if query.end_time:
            selected = selected[selected["timestamp"] <= pd.to_datetime(query.end_time, utc=True)]
    except (ValueError, TypeError) as exc:
        return {"status": "invalid_time", "message": f"Could not parse the requested time range: {exc}"}

    if selected.empty:
        return {"status": "no_matching_records", "measurement": column, "record_count": 0}

    if query.operation == "value":
        if not column:
            return {"status": "column_required", "available_measurements": columns}
        if query.timestamp:
            try:
                target = pd.to_datetime(query.timestamp, utc=True)
            except (ValueError, TypeError) as exc:
                return {"status": "invalid_time", "message": f"Could not parse the requested timestamp: {exc}"}
            distances = (selected["timestamp"] - target).abs()
            row = selected.loc[distances.idxmin()]
            offset = abs((row["timestamp"] - target).total_seconds()) / 60
            return {
                "status": "ok",
                "measurement": column,
                "requested_timestamp": target.isoformat(),
                "recorded_timestamp": _json_value(row["timestamp"]),
                "minutes_from_requested_time": round(offset, 2),
                "value": _json_value(row[column]),
                "unit": _measurement_unit(column),
            }
        row = selected.iloc[-1]
        return {
            "status": "ok",
            "measurement": column,
            "recorded_timestamp": _json_value(row["timestamp"]),
            "value": _json_value(row[column]),
            "unit": _measurement_unit(column),
        }

    if query.operation == "rows":
        rows = selected[selected_columns].iloc[query.offset:query.offset + query.limit].to_dict(orient="records")
        return {
            "status": "ok",
            "record_count": len(selected),
            "returned_count": len(rows),
            "offset": query.offset,
            "has_more": query.offset + len(rows) < len(selected),
            "rows": [{key: _json_value(value) for key, value in row.items()} for row in rows],
        }

    if not column:
        return {"status": "column_required", "available_measurements": columns}
    if query.operation == "count":
        count = int(selected[column].count())
        return {
            "status": "ok", "measurement": column, "operation": "count", "value": count,
            "record_count": count, "start_time": _json_value(selected["timestamp"].min()),
            "end_time": _json_value(selected["timestamp"].max()), "unit": None,
        }
    if query.operation == "mode":
        modes = selected[column].dropna().mode()
        if modes.empty:
            return {"status": "no_values", "measurement": column}
        mode_value = _json_value(modes.iloc[0])
        frequency = int(selected[column].eq(modes.iloc[0]).sum())
        return {"status": "ok", "measurement": column, "operation": "mode", "value": mode_value, "frequency": frequency}
    if query.operation == "unique":
        unique_values = [_json_value(value) for value in pd.unique(selected[column].dropna())]
        return {
            "status": "ok", "measurement": column, "operation": "unique",
            "distinct_count": len(unique_values), "unique_values": unique_values[:50],
            "truncated": len(unique_values) > 50,
        }
    values = pd.to_numeric(selected[column], errors="coerce").dropna()
    if values.empty:
        return {"status": "no_numeric_values", "measurement": column}
    result = {
        "mean": values.mean,
        "min": values.min,
        "max": values.max,
        "median": values.median,
        "std": lambda: values.std(ddof=0),
        "sum": values.sum,
        "count": values.count,
    }[query.operation]()
    aggregate: dict[str, Any] = {
        "status": "ok",
        "measurement": column,
        "operation": query.operation,
        "value": round(float(result), 4),
        "record_count": int(values.count()),
        "start_time": _json_value(selected["timestamp"].min()),
        "end_time": _json_value(selected["timestamp"].max()),
        "unit": _measurement_unit(column),
    }
    if query.operation in {"min", "max"}:
        numeric_values = pd.to_numeric(selected[column], errors="coerce")
        extreme_index = numeric_values.idxmin() if query.operation == "min" else numeric_values.idxmax()
        aggregate["value_timestamp"] = _json_value(selected.loc[extreme_index, "timestamp"])
    return aggregate


# ---------------------------------------------------------------------------
# Tool Declarations (Groq JSON Schema)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Retrieve an already generated prediction for the active uploaded patient at a specific horizon. Never generate or estimate a forecast. Only horizons available in the active upload context can be returned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon_min": {
                        "type": "integer",
                        "enum": [30, 60, 90, 120],
                        "description": "Forecast horizon in minutes: must be 30, 60, 90, or 120.",
                    }
                },
                "required": ["horizon_min"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_forecasts",
            "description": "Retrieve only the already generated predictions available in the active uploaded patient context. Do not generate missing horizons.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_glucose",
            "description": "Inspect recent CGM history: returns current reading, rate of change (ROC), min, max, mean, and time-in-range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "integer",
                        "description": "Number of minutes of recent history to summarize (e.g. 15, 30, 60, 120). Default 60.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_risk",
            "description": "Evaluate hypoglycemia (<70 mg/dL), severe hypo (<54), hyperglycemia (>180), and severe hyper (>250) risks across all forecast horizons.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_performance",
            "description": "Retrieve benchmark validation metrics (RMSE, MAE, and Clarke Error Grid clinically safe zone A+B %) for our trained models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon_min": {
                        "type": "integer",
                        "enum": [30, 60, 90, 120],
                        "description": "Optional forecast horizon in minutes (30, 60, 90, or 120). Omit to get all.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_forecast",
            "description": "Explain top feature contributions (XGBoost importance gain and patient's current lag/trend values) driving a forecast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon_min": {
                        "type": "integer",
                        "enum": [30, 60, 90, 120],
                        "description": "Forecast horizon to explain: 30, 60, 90, or 120 minutes.",
                    }
                },
                "required": ["horizon_min"],
            },
        },
    },
]

TOOL_DEFINITIONS = [
    definition for definition in TOOL_DEFINITIONS
    if definition["function"]["name"] not in {"get_model_performance", "explain_forecast"}
]
TOOL_DEFINITIONS.append({
    "type": "function",
    "function": {
        "name": "query_patient_dataset",
        "description": (
            "Answer questions about measurements and timestamps in the active uploaded patient dataset only. "
            "Use operation=columns to inspect available measurements; value for a measurement at a time; "
            "mean/min/max/median/std/sum/count/mode/unique for a time range; or rows to page through all records with limit and offset. "
            "Never use other patients, demo data, code, project files, or outside datasets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "Measurement column, such as heart_rate, insulin, cgm, or glucose."},
                "operation": {"type": "string", "enum": ["value", "mean", "min", "max", "median", "std", "sum", "count", "mode", "unique", "rows", "columns"]},
                "timestamp": {"type": "string", "description": "Point in time for a nearest-reading lookup."},
                "start_time": {"type": "string", "description": "Inclusive time-range start."},
                "end_time": {"type": "string", "description": "Inclusive time-range end."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Maximum number of rows to return."},
                "offset": {"type": "integer", "minimum": 0, "maximum": 1000000, "description": "Zero-based offset for paging through rows."},
            },
        },
    },
})


TOOL_ROUTER = {
    "get_forecast": tool_get_forecast,
    "get_all_forecasts": tool_get_all_forecasts,
    "get_recent_glucose": tool_get_recent_glucose,
    "assess_risk": tool_assess_risk,
    "query_patient_dataset": tool_query_patient_dataset,
}


def execute_tool_call(name: str, args: dict[str, Any], context: PatientContext, store: ModelStore) -> dict[str, Any]:
    """Dispatch and execute a tool call by name with validated arguments."""
    handler = TOOL_ROUTER.get(name)
    if not handler:
        return {"error": f"Unknown tool: '{name}'. Available: {list(TOOL_ROUTER.keys())}"}
    try:
        return handler(args, context, store)
    except Exception as exc:
        return {"error": f"Tool execution failed for '{name}': {str(exc)}"}
