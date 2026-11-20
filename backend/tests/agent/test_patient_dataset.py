from datetime import datetime, timedelta, timezone

import pandas as pd

from agent.guardrails import is_out_of_scope_request
from agent.patient_data import PatientDatasetStore, sanitize_patient_dataframe
from agent.router import ChatRequest
from agent.tools import PatientContext, TOOL_DEFINITIONS, tool_query_patient_dataset
from pydantic import ValidationError


def make_patient_context() -> PatientContext:
    start = datetime(2026, 1, 2, 8, 0, tzinfo=timezone.utc)
    frame = pd.DataFrame({
        "Time": [start + timedelta(minutes=5 * index) for index in range(30)],
        "Glucose mg/dL": [110 + index for index in range(30)],
        "Heart Rate BPM": [60 + index for index in range(30)],
        "Patient Age": [62] * 30,
        "Symptom": ["none"] * 30,
        "patient_id": ["559"] * 30,
        "patientId": ["559"] * 30,
        "row_index": list(range(30)),
        "source_code": ["internal"] * 30,
        "free_text_note": ["private note"] * 30,
    })
    cleaned = sanitize_patient_dataframe(frame)
    readings = [
        {"timestamp": row.timestamp.isoformat(), "glucose": float(row.glucose)}
        for row in cleaned.rename(columns={"glucose_mg_dl": "glucose"})[["timestamp", "glucose"]].itertuples(index=False)
    ]
    return PatientContext.from_readings(readings, patient_dataset=cleaned)


def test_uploaded_dataset_sanitizes_identifiers_and_keeps_patient_measurements():
    context = make_patient_context()
    assert "heart_rate_bpm" in context.patient_dataset.columns
    assert "patient_age" in context.patient_dataset.columns
    assert "symptom" in context.patient_dataset.columns
    assert "patient_id" not in context.patient_dataset.columns
    assert "patientid" not in context.patient_dataset.columns
    assert "row_index" not in context.patient_dataset.columns
    assert "source_code" not in context.patient_dataset.columns
    assert "free_text_note" not in context.patient_dataset.columns
    assert "559" not in context.patient_dataset.astype(str).to_string()


def test_query_patient_dataset_returns_nearest_heart_rate_with_units():
    context = make_patient_context()
    requested = (context.patient_dataset["timestamp"].iloc[8] + pd.Timedelta(minutes=1)).isoformat()
    result = tool_query_patient_dataset(
        {"column": "heart rate", "operation": "value", "timestamp": requested},
        context,
        None,
    )
    assert result["status"] == "ok"
    assert result["value"] == 68
    assert result["unit"] == "bpm"
    assert result["recorded_timestamp"] == context.patient_dataset["timestamp"].iloc[8].isoformat()


def test_query_patient_dataset_can_aggregate_or_read_any_patient_time_window():
    context = make_patient_context()
    start = context.patient_dataset["timestamp"].iloc[4].isoformat()
    end = context.patient_dataset["timestamp"].iloc[6].isoformat()
    result = tool_query_patient_dataset(
        {"column": "heart_rate_bpm", "operation": "mean", "start_time": start, "end_time": end},
        context,
        None,
    )
    assert result["value"] == 65
    assert result["record_count"] == 3

    maximum = tool_query_patient_dataset(
        {"column": "heart_rate", "operation": "max"}, context, None
    )
    assert maximum["value"] == 89
    assert maximum["value_timestamp"] == context.patient_dataset["timestamp"].iloc[-1].isoformat()

    rows = tool_query_patient_dataset(
        {"operation": "rows", "start_time": start, "end_time": end, "limit": 10},
        context,
        None,
    )
    assert len(rows["rows"]) == 3
    assert rows["rows"][0]["heart_rate_bpm"] == 64

    categorical = tool_query_patient_dataset(
        {"column": "symptom", "operation": "mode"}, context, None
    )
    assert categorical["value"] == "none"
    assert categorical["frequency"] == 30

    first_page = tool_query_patient_dataset(
        {"operation": "rows", "limit": 5, "offset": 0}, context, None
    )
    second_page = tool_query_patient_dataset(
        {"operation": "rows", "limit": 5, "offset": 5}, context, None
    )
    assert first_page["has_more"] is True
    assert first_page["rows"][-1]["timestamp"] < second_page["rows"][0]["timestamp"]


def test_patient_dataset_store_uses_an_opaque_id_and_returns_isolated_copies():
    context = make_patient_context()
    store = PatientDatasetStore()
    token, row_count, columns = store.register(context.patient_dataset)
    saved = store.get(token)
    assert len(token) >= 32
    assert row_count == 30
    assert "heart_rate_bpm" in columns
    saved.loc[0, "heart_rate_bpm"] = -1
    assert store.get(token).loc[0, "heart_rate_bpm"] == 60


def test_agent_tools_and_scope_are_limited_to_the_active_patient():
    names = {entry["function"]["name"] for entry in TOOL_DEFINITIONS}
    assert "query_patient_dataset" in names
    assert "get_model_performance" not in names
    assert "explain_forecast" not in names
    assert is_out_of_scope_request("show me the backend source code")
    assert is_out_of_scope_request("what data do you have for patient 559?")
    try:
        ChatRequest(message="What is my glucose?", agent_dataset_id="")
    except ValidationError:
        pass
    else:
        raise AssertionError("Chat requests must require a successful upload dataset ID")
