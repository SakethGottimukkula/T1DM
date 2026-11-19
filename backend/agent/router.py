"""FastAPI router for the Groq-powered T1DM forecasting agent."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.predictor import ModelStore
from .agent import T1DMAgent, reset_session
from .config import GROQ_API_KEY, GROQ_MODEL, GROQ_REASONING_EFFORT
from .guardrails import RESEARCH_DISCLAIMER
from .llm import GroqClientWrapper
from .patient_data import patient_dataset_store
from .tools import PatientContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])
_SESSION_DATASET_IDS: dict[str, str] = {}


def get_agent_store() -> ModelStore:
    """Retrieve the singleton ModelStore from main or initialize fallback."""
    try:
        from app.main import _store
        if _store is not None:
            return _store
    except Exception:
        pass
    store = ModelStore()
    store.load_all()
    return store


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ToolCallLog(BaseModel):
    id: str
    name: str
    args: dict[str, Any]
    result: dict[str, Any]


class ChatRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=120, description="Conversation session ID")
    message: str = Field(..., min_length=1, max_length=2000, description="User question")
    agent_dataset_id: str = Field(..., min_length=32, max_length=128, description="Opaque ID returned by a successful patient CSV upload")
    forecast_context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Active uploaded-patient forecast values; text metadata is ignored.",
    )


class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallLog]
    disclaimer: str
    session_id: str
    latency_ms: Optional[float] = None
    tokens: Optional[dict[str, int]] = None


class ResetSessionRequest(BaseModel):
    session_id: str = Field(default="default")


class HealthAgentResponse(BaseModel):
    status: str
    model: str
    reasoning_effort: str
    groq_configured: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthAgentResponse)
async def agent_health() -> HealthAgentResponse:
    """Check Groq agent connectivity and configuration."""
    configured = bool(GROQ_API_KEY.strip())
    status = "ok" if configured else "unconfigured"
    return HealthAgentResponse(
        status=status,
        model=GROQ_MODEL,
        reasoning_effort=GROQ_REASONING_EFFORT,
        groq_configured=configured,
    )


def _safe_number(value: Any, minimum: float, maximum: float) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and minimum <= result <= maximum else None


def _safe_forecast_context(raw: Optional[dict[str, Any]], frame) -> dict[str, Any]:
    """Retain only numeric forecast facts and fixed labels from the browser payload."""
    source = raw if isinstance(raw, dict) else {}
    safe: dict[str, Any] = {
        "source": "upload",
        "patient_id": "Uploaded patient",
        "selected_patient_id": "Uploaded patient",
        "has_forecast": source.get("has_forecast") is True,
    }
    current = _safe_number(source.get("current_glucose"), 40, 400)
    if current is not None:
        safe["current_glucose"] = current
    if not frame.empty:
        safe["timestamp"] = frame["timestamp"].iloc[-1].isoformat()

    predictions: dict[str, dict[str, Any]] = {}
    raw_predictions = source.get("predictions")
    if isinstance(raw_predictions, dict):
        for horizon in (30, 60, 90, 120):
            item = raw_predictions.get(str(horizon), raw_predictions.get(horizon))
            if not isinstance(item, dict):
                continue
            predicted = _safe_number(item.get("predicted"), 40, 400)
            if predicted is None:
                continue
            prediction: dict[str, Any] = {"predicted": predicted}
            change = _safe_number(item.get("change"), -400, 400)
            if change is not None:
                prediction["change"] = change
            zone = item.get("zone")
            if isinstance(zone, str) and zone in {"in_range", "hypoglycemia", "severe_hypoglycemia", "hyperglycemia", "severe_hyperglycemia"}:
                prediction["zone"] = zone
            predictions[str(horizon)] = prediction
    if predictions:
        safe["predictions"] = predictions
    return safe


def _context_for_dataset(dataset_id: str, forecast_context: Optional[dict[str, Any]] = None) -> PatientContext:
    frame = patient_dataset_store.get(dataset_id)
    if frame is None:
        raise HTTPException(410, "This uploaded patient context has expired. Upload the patient CSV again to continue.")
    glucose_aliases = {"glucose", "cgm", "bg", "blood_glucose", "blood_glucose_mg_dl", "glucose_mg_dl"}
    glucose_column = next((name for name in frame.columns if name in glucose_aliases), None)
    if glucose_column is None:
        raise HTTPException(422, "The uploaded patient dataset does not contain a glucose measurement for this agent context.")
    readings_frame = frame[["timestamp", glucose_column]].rename(columns={glucose_column: "glucose"}).dropna()
    if len(readings_frame) < 24:
        raise HTTPException(422, "The uploaded patient dataset has too few glucose readings to start the agent. Upload at least 24 readings.")
    readings = [
        {"timestamp": row.timestamp.isoformat(), "glucose": float(row.glucose)}
        for row in readings_frame.itertuples(index=False)
    ]
    safe_context = _safe_forecast_context(forecast_context, frame)
    return PatientContext.from_readings(
        readings,
        patient_id="Uploaded patient",
        forecast_context=safe_context,
        patient_dataset=frame,
    )


@router.post("/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest) -> ChatResponse:
    """Send a query to the T1DM forecasting agent."""
    context = _context_for_dataset(req.agent_dataset_id, req.forecast_context)
    previous_dataset = _SESSION_DATASET_IDS.get(req.session_id)
    if previous_dataset != req.agent_dataset_id:
        reset_session(req.session_id)
        _SESSION_DATASET_IDS[req.session_id] = req.agent_dataset_id

    if not GROQ_API_KEY.strip():
        raise HTTPException(
            503,
            "Groq API key is not configured. Please set GROQ_API_KEY in your environment or .env file.",
        )

    store = get_agent_store()

    agent = T1DMAgent(store=store)

    try:
        result = await agent.run(
            message=req.message,
            session_id=req.session_id,
            context=context,
        )
        return ChatResponse(
            answer=result["answer"],
            tool_calls=[ToolCallLog(**tc) for tc in result["tool_calls"]],
            disclaimer=result["disclaimer"],
            session_id=result["session_id"],
            latency_ms=result.get("latency_ms"),
            tokens=result.get("tokens"),
        )
    except Exception as exc:
        logger.error("Agent chat failed: %s", exc)
        raise HTTPException(502, f"Agent failed to complete reasoning: {str(exc)}")


@router.post("/reset")
async def agent_reset(req: ResetSessionRequest) -> dict[str, str]:
    """Reset the agent conversation history for a session."""
    reset_session(req.session_id)
    _SESSION_DATASET_IDS.pop(req.session_id, None)
    return {"status": "ok", "message": f"Session '{req.session_id}' reset successfully."}


@router.get("/chat/stream")
@router.post("/chat/stream")
async def agent_chat_stream(
    message: str = Query(..., description="User query"),
    session_id: str = Query("default", description="Session ID"),
    agent_dataset_id: str = Query(..., min_length=32, max_length=128, description="Opaque ID from a successful patient upload"),
) -> StreamingResponse:
    """SSE streaming variant that emits tool events followed by synthesized response tokens."""
    context = _context_for_dataset(agent_dataset_id)
    previous_dataset = _SESSION_DATASET_IDS.get(session_id)
    if previous_dataset != agent_dataset_id:
        reset_session(session_id)
        _SESSION_DATASET_IDS[session_id] = agent_dataset_id

    if not GROQ_API_KEY.strip():
        async def err_gen() -> AsyncIterator[str]:
            yield f"data: {json.dumps({'error': 'GROQ_API_KEY is not configured'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    store = get_agent_store()
    agent = T1DMAgent(store=store)

    async def event_generator() -> AsyncIterator[str]:
        try:
            # Run the agent tool execution loop
            result = await agent.run(message=message, session_id=session_id, context=context)

            # Stream each executed tool call first
            for tc in result["tool_calls"]:
                yield f"data: {json.dumps({'type': 'tool_call', 'name': tc['name'], 'args': tc['args'], 'result': tc['result']})}\n\n"

            # Stream answer in word chunks to simulate smooth delivery
            words = result["answer"].split(" ")
            for i, word in enumerate(words):
                sep = " " if i > 0 else ""
                yield f"data: {json.dumps({'type': 'token', 'token': sep + word})}\n\n"

            # Final completion event
            yield f"data: {json.dumps({'type': 'done', 'disclaimer': result['disclaimer'], 'latency_ms': result.get('latency_ms')})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
