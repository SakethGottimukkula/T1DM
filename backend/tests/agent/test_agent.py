"""Comprehensive tests for the T1DM Groq Agent layer.

Tests cover:
- Tool functions and outputs
- Risk assessment thresholds
- Safety guardrails and dosing refusal
- LLM wrapper with openai/gpt-oss-120b parameters and reasoning suppression
- Agent loop with mocked Groq responses
- Router endpoints (/agent/health, /agent/chat, /agent/reset)
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agent.agent import T1DMAgent, append_session_message, get_session_history, reset_session, select_reasoning_effort
from agent.config import GROQ_MODEL
from agent.guardrails import (
    EMERGENCY_ADVISORY,
    RESEARCH_DISCLAIMER,
    SAFE_REFUSAL_MESSAGE,
    check_dosing_violation,
    post_process_response,
    sanitize_answer,
)
from agent.llm import (
    GroqClientWrapper,
    GroqLLMError,
    ToolArgumentError,
    deduplicate_tool_calls,
    parse_tool_args,
    sanitize_response_text,
)
from agent.tools import (
    PatientContext,
    STORED_PERFORMANCE,
    execute_tool_call,
    tool_assess_risk,
    tool_explain_forecast,
    tool_get_all_forecasts,
    tool_get_forecast,
    tool_get_model_performance,
    tool_get_recent_glucose,
)
from agent.patient_data import patient_dataset_store
from app.main import app
from app.predictor import ModelStore


@pytest.fixture(scope="module")
def model_store():
    store = ModelStore()
    store.load_all()
    return store


@pytest.fixture
def default_context():
    return uploaded_test_context()


def uploaded_test_context(forecast_context=None):
    base = datetime(2026, 1, 1, 8, 0)
    readings = [
        {"timestamp": (base + timedelta(minutes=5 * index)).isoformat(), "glucose": 120.0 + index / 2}
        for index in range(30)
    ]
    return PatientContext.from_readings(readings, forecast_context=forecast_context)


def register_uploaded_test_dataset():
    import pandas as pd

    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 08:00", periods=30, freq="5min"),
        "glucose": [120.0 + index / 2 for index in range(30)],
    })
    token, _, _ = patient_dataset_store.register(frame)
    return token


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ===========================================================================
# 1. Tool Function Tests
# ===========================================================================

class TestAgentTools:
    def test_get_forecast_valid_horizons(self, default_context, model_store):
        for hz in (30, 60, 90, 120):
            res = tool_get_forecast({"horizon_min": hz}, default_context, model_store)
            assert res["horizon_min"] == hz
            assert isinstance(res["predicted_glucose_mg_dl"], float)
            assert isinstance(res["current_glucose_mg_dl"], float)
            assert res["zone"] in ("in_range", "hypoglycemia", "severe_hypoglycemia", "hyperglycemia", "severe_hyperglycemia")

    def test_get_forecast_invalid_horizon_fails(self, default_context, model_store):
        with pytest.raises(Exception):
            tool_get_forecast({"horizon_min": 45}, default_context, model_store)

    def test_get_all_forecasts(self, default_context, model_store):
        res = tool_get_all_forecasts({}, default_context, model_store)
        assert len(res["forecasts"]) == 4
        horizons = [f["horizon_min"] for f in res["forecasts"]]
        assert horizons == [30, 60, 90, 120]

    def test_get_recent_glucose_stats(self, default_context, model_store):
        res = tool_get_recent_glucose({"minutes": 60}, default_context, model_store)
        assert res["window_minutes"] == 60
        assert res["readings_count"] >= 3
        assert res["min_glucose_mg_dl"] <= res["max_glucose_mg_dl"]
        assert isinstance(res["rate_of_change_mg_dl_per_min"], float)
        assert res["trend_arrow"] in ("↑", "↓", "→")

    def test_get_model_performance(self, default_context, model_store):
        # Specific horizon
        p60 = tool_get_model_performance({"horizon_min": 60}, default_context, model_store)
        assert p60["horizon_min"] == 60
        assert p60["rmse_mg_dl"] == STORED_PERFORMANCE[60]["rmse"]
        assert p60["clarke_error_grid_safe_ab_pct"] == STORED_PERFORMANCE[60]["clarke_ab_pct"]

        # All horizons
        p_all = tool_get_model_performance({}, default_context, model_store)
        assert "performance_by_horizon" in p_all

    def test_explain_forecast_returns_top_factors(self, default_context, model_store):
        res = tool_explain_forecast({"horizon_min": 60}, default_context, model_store)
        assert res["horizon_min"] == 60
        assert len(res["top_contributing_factors"]) <= 5
        top_1 = res["top_contributing_factors"][0]
        assert "feature" in top_1
        assert "model_importance_pct" in top_1
        assert "patient_value" in top_1


# ===========================================================================
# 2. Risk Assessment Thresholds
# ===========================================================================

class TestRiskAssessment:
    def test_normal_risk(self, default_context, model_store):
        res = tool_assess_risk({}, default_context, model_store)
        assert "overall_severity" in res
        assert "minimum_projected_glucose" in res
        assert "maximum_projected_glucose" in res

    def test_severe_hypo_threshold_under_54(self, model_store):
        # Create readings with extreme low
        import pandas as pd
        from datetime import datetime, timedelta
        base = datetime(2020, 1, 1, 8, 0)
        readings = [{"timestamp": (base + timedelta(minutes=5 * i)).isoformat(), "glucose": 45.0} for i in range(30)]
        ctx = PatientContext.from_readings(readings)
        res = tool_assess_risk({}, ctx, model_store)
        assert res["overall_severity"] == "severe_hypoglycemia"
        assert any(a["type"] == "severe_hypo" for a in res["alerts"])

    def test_severe_hyper_threshold_over_250(self, model_store):
        # Create readings with extreme high
        from datetime import datetime, timedelta
        base = datetime(2020, 1, 1, 8, 0)
        readings = [{"timestamp": (base + timedelta(minutes=5 * i)).isoformat(), "glucose": 320.0} for i in range(30)]
        ctx = PatientContext.from_readings(readings)
        res = tool_assess_risk({}, ctx, model_store)
        assert res["overall_severity"] == "severe_hyperglycemia"
        assert any(a["type"] == "severe_hyper" for a in res["alerts"])


# ===========================================================================
# 3. Guardrails & Refusal
# ===========================================================================

class TestGuardrails:
    def test_detects_insulin_dosing_violation(self):
        assert check_dosing_violation("You should take 4 units of insulin before lunch.") is True
        assert check_dosing_violation("Inject 2 u of Humalog now.") is True
        assert check_dosing_violation("Increase your basal rate by 1 unit.") is True
        assert check_dosing_violation("Your 60-minute forecast is 140 mg/dL.") is False

    def test_refusal_sentence_is_not_removed(self):
        text = "I can't advise you to adjust your insulin. Instead, please ask your care team."
        clean, was_modified = sanitize_answer(text)
        assert was_modified is False
        assert "can't advise you to adjust your insulin" in clean

    def test_dosing_directive_removed_while_meal_workout_kept(self):
        text = (
            "For dinner, grilled salmon with quinoa is a great choice. "
            "Take 5 units of insulin before eating. "
            "A light 20-minute walk afterwards will help keep your glucose steady."
        )
        clean, was_modified = sanitize_answer(text)
        assert was_modified is True
        assert "take 5 units of insulin" not in clean.lower()
        assert "grilled salmon with quinoa" in clean
        assert "20-minute walk" in clean
        assert "care team" in clean

    def test_meal_workout_no_dosing_passes_unchanged(self):
        text = (
            "Here are some great options: steel-cut oats with walnuts for breakfast, "
            "and a 30-minute moderate cycling routine in the afternoon. "
            "Be sure to stay well hydrated and check your glucose before starting."
        )
        clean, was_modified = sanitize_answer(text)
        assert was_modified is False
        assert clean == text

    def test_post_process_replaces_dosing_with_safe_refusal(self):
        dosing_output = "Take 3 units of rapid-acting insulin."
        cleaned = post_process_response(dosing_output, [])
        assert cleaned == SAFE_REFUSAL_MESSAGE

    def test_post_process_injects_emergency_for_critical_excursions(self):
        text = "Your projected glucose is 48 mg/dL in 30 minutes."
        tool_results = [{"result": {"predicted_glucose_mg_dl": 48.0, "overall_severity": "severe_hypoglycemia"}}]
        cleaned = post_process_response(text, tool_results)
        assert EMERGENCY_ADVISORY in cleaned


# ===========================================================================
# 4. LLM Wrapper & openai/gpt-oss-120b Tuning
# ===========================================================================

class TestLLMWrapper:
    def test_sanitize_response_text_strips_think_tags(self):
        raw = "<think>Analyzing patient trends</think>The 60-minute prediction is 125 mg/dL."
        assert sanitize_response_text(raw) == "The 60-minute prediction is 125 mg/dL."

    def test_parse_tool_args_handles_clean_and_markdown(self):
        assert parse_tool_args('{"horizon_min": 60}') == {"horizon_min": 60}
        assert parse_tool_args('```json\n{"horizon_min": 90}\n```') == {"horizon_min": 90}

    def test_parse_tool_args_raises_on_garbage(self):
        with pytest.raises(ToolArgumentError):
            parse_tool_args("Not valid JSON at all")

    def test_deduplicate_tool_calls(self):
        tc1 = MagicMock()
        tc1.function.name = "get_forecast"
        tc1.function.arguments = '{"horizon_min": 60}'

        tc2 = MagicMock()
        tc2.function.name = "get_forecast"
        tc2.function.arguments = '{"horizon_min": 60}'

        tc3 = MagicMock()
        tc3.function.name = "get_forecast"
        tc3.function.arguments = '{"horizon_min": 90}'

        unique = deduplicate_tool_calls([tc1, tc2, tc3])
        assert len(unique) == 2

    def test_select_reasoning_effort(self):
        assert select_reasoning_effort("Why is my forecast higher than current?") == "high"
        assert select_reasoning_effort("Explain the risk of a low") == "high"
        assert select_reasoning_effort("What will my glucose be in 60 minutes?") == "low"
        assert select_reasoning_effort("Hello there") == "medium"

    def test_groq_params_no_base_url_and_include_reasoning_false(self):
        wrapper = GroqClientWrapper(api_key="gsk_fake_key")
        assert not hasattr(wrapper, "base_url")

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test answer"
        mock_choice.message.tool_calls = []
        mock_choice.finish_reason = "stop"
        mock_resp.choices = [mock_choice]
        mock_resp.usage = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        wrapper._get_client = MagicMock(return_value=mock_client)
        asyncio.run(wrapper.chat([{"role": "user", "content": "hi"}]))

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("include_reasoning") is False
        assert "reasoning_format" not in call_kwargs
        assert "base_url" not in call_kwargs


# ===========================================================================
# 5. Agent Loop & Router Endpoints
# ===========================================================================

class TestAgentRouter:
    def test_agent_health(self, client):
        r = client.get("/agent/health")
        assert r.status_code == 200
        data = r.json()
        assert "model" in data
        assert "openai/gpt-oss-120b" in data["model"]
        assert "reasoning_effort" in data

    def test_agent_reset(self, client):
        append_session_message("test_sess", "user", "Hello")
        assert len(get_session_history("test_sess")) == 1
        r = client.post("/agent/reset", json={"session_id": "test_sess"})
        assert r.status_code == 200
        assert len(get_session_history("test_sess")) == 0

    def test_agent_chat_missing_key_returns_503(self, client, monkeypatch):
        monkeypatch.setattr("agent.router.GROQ_API_KEY", "")
        r = client.post("/agent/chat", json={
            "session_id": "s1", "message": "What is my forecast?",
            "agent_dataset_id": register_uploaded_test_dataset(),
        })
        assert r.status_code == 503

    def test_agent_chat_mocked_end_to_end(self, client, monkeypatch, model_store):
        monkeypatch.setattr("agent.router.GROQ_API_KEY", "gsk_fake_key_12345")

        # Mock LLM calls: Turn 1 requests tool 'get_forecast', Turn 2 outputs final answer
        mock_tc = MagicMock()
        mock_tc.id = "call_abc123"
        mock_tc.function.name = "get_forecast"
        mock_tc.function.arguments = '{"horizon_min": 60}'

        response_turn_1 = {
            "text": "",
            "tool_calls": [mock_tc],
            "usage": {"prompt_tokens": 120, "completion_tokens": 25},
        }

        response_turn_2 = {
            "text": "Your forecasted blood glucose in 60 minutes is projected to be 138.5 mg/dL, which is within the target range.",
            "tool_calls": [],
            "usage": {"prompt_tokens": 210, "completion_tokens": 40},
        }

        wrapper_mock = MagicMock(spec=GroqClientWrapper)
        wrapper_mock.chat = AsyncMock(side_effect=[response_turn_1, response_turn_2])

        with patch("agent.router.T1DMAgent") as MockAgentClass:
            # Create real agent with mocked LLM wrapper
            real_agent = T1DMAgent(store=model_store, client_wrapper=wrapper_mock)
            MockAgentClass.return_value = real_agent

            r = client.post(
                "/agent/chat",
                json={
                    "session_id": "test_e2e_session", "message": "What will my glucose be in 60 minutes?",
                    "agent_dataset_id": register_uploaded_test_dataset(),
                },
            )
            assert r.status_code == 200
            data = r.json()
            assert "138.5 mg/dL" in data["answer"]
            assert len(data["tool_calls"]) == 1
            assert data["tool_calls"][0]["name"] == "get_forecast"
            assert data["tool_calls"][0]["args"]["horizon_min"] == 60
            assert "predicted_glucose_mg_dl" in data["tool_calls"][0]["result"]
            assert data["disclaimer"] == RESEARCH_DISCLAIMER
            assert data["session_id"] == "test_e2e_session"

    def test_agent_chat_with_forecast_context_payload(self, client, model_store):
        """Verify chat endpoint accepts forecast_context and grounds predictions in it."""
        mock_tc = MagicMock()
        mock_tc.id = "call_fc_1"
        mock_tc.function.name = "get_forecast"
        mock_tc.function.arguments = '{"horizon_min": 60}'

        response_turn_1 = {
            "text": "",
            "tool_calls": [mock_tc],
            "usage": {"prompt_tokens": 140, "completion_tokens": 20},
        }
        response_turn_2 = {
            "text": "Your actual 60-minute prediction from the pipeline is 155.0 mg/dL.",
            "tool_calls": [],
            "usage": {"prompt_tokens": 220, "completion_tokens": 30},
        }

        wrapper_mock = MagicMock(spec=GroqClientWrapper)
        wrapper_mock.chat = AsyncMock(side_effect=[response_turn_1, response_turn_2])

        with patch("agent.router.T1DMAgent") as MockAgentClass:
            real_agent = T1DMAgent(store=model_store, client_wrapper=wrapper_mock)
            MockAgentClass.return_value = real_agent

            forecast_context = {
                "has_forecast": True,
                "patient_id": "559",
                "current_glucose": 140.0,
                "predictions": {
                    "60": {
                        "predicted": 155.0,
                        "change": 15.0,
                        "actual": 152.0,
                        "error": 3.0,
                        "clarke_zone": "A",
                    }
                },
                "patient_rmse": {"60": 16.2},
            }

            r = client.post(
                "/agent/chat",
                json={
                    "session_id": "test_fc_session",
                    "message": "What is my 60-minute forecast?",
                    "agent_dataset_id": register_uploaded_test_dataset(),
                    "forecast_context": forecast_context,
                },
            )
            assert r.status_code == 200
            data = r.json()
            assert len(data["tool_calls"]) == 1
            tc_result = data["tool_calls"][0]["result"]
            assert tc_result["predicted_glucose_mg_dl"] == 155.0
            assert tc_result["current_glucose_mg_dl"] == 140.0
            assert tc_result["change_mg_dl"] == 15.0
            assert "actual_held_out_glucose_mg_dl" not in tc_result
            assert "clarke_zone" not in tc_result


# ===========================================================================
# 6. Pipeline Forecast Context & Grounding Tests
# ===========================================================================

class TestForecastContextTools:
    def test_tool_get_forecast_no_forecast_run(self, model_store):
        """When has_forecast is False, tool reports no forecast is available."""
        context = uploaded_test_context(forecast_context={"has_forecast": False, "patient_id": "559"})
        res = tool_get_forecast({"horizon_min": 60}, context, model_store)
        assert res["status"] == "no_forecast_available"
        assert res["has_forecast"] is False
        assert "No forecast has been run yet" in res["message"]

    def test_tool_get_forecast_with_pipeline_predictions(self, model_store):
        """When pipeline predictions are provided, tool returns exact pipeline numbers."""
        fc = {
            "has_forecast": True,
            "patient_id": "559",
            "current_glucose": 135.0,
            "predictions": {
                "30": {"predicted": 142.5, "change": 7.5, "actual": 140.0, "error": 2.5, "clarke_zone": "A"},
                "60": {"predicted": 150.0, "change": 15.0, "actual": 148.0, "error": 2.0, "clarke_zone": "A"},
            },
        }
        context = uploaded_test_context(forecast_context=fc)
        res30 = tool_get_forecast({"horizon_min": 30}, context, model_store)
        assert res30["predicted_glucose_mg_dl"] == 142.5
        assert res30["current_glucose_mg_dl"] == 135.0
        assert res30["actual_held_out_glucose_mg_dl"] == 140.0
        assert res30["prediction_error_mg_dl"] == 2.5
        assert res30["clarke_zone"] == "A"

    def test_tool_get_all_forecasts_no_forecast_run(self, model_store):
        """When has_forecast is False, get_all_forecasts returns empty forecasts list and status."""
        context = uploaded_test_context(forecast_context={"has_forecast": False, "patient_id": "559"})
        res = tool_get_all_forecasts({}, context, model_store)
        assert res["status"] == "no_forecast_available"
        assert res["has_forecast"] is False
        assert res["forecasts"] == []

    def test_tool_get_model_performance_patient_specific(self, model_store):
        """Prioritizes patient-specific RMSE over cohort benchmark when available."""
        fc = {
            "has_forecast": True,
            "patient_id": "559",
            "patient_rmse": {"30": 11.2, "60": 15.4, "90": 19.8, "120": 22.1},
        }
        context = uploaded_test_context(forecast_context=fc)
        res60 = tool_get_model_performance({"horizon_min": 60}, context, model_store)
        assert res60["patient_id"] == "559"
        assert res60["patient_specific_rmse_mg_dl"] == 15.4
        assert res60["metric_scope"] == "patient_specific"

        res_all = tool_get_model_performance({}, context, model_store)
        assert res_all["patient_id"] == "559"
        assert res_all["patient_specific_rmse"]["60"] == 15.4
        assert res_all["metric_scope"] == "patient_specific"
