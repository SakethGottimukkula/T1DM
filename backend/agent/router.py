"""FastAPI router for the Groq-powered T1DM forecasting agent."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.predictor import ModelStore
from app.schemas import GlucoseReading

from .agent import T1DMAgent, reset_session
from .config import GROQ_API_KEY, GROQ_MODEL, GROQ_REASONING_EFFORT
from .guardrails import RESEARCH_DISCLAIMER
from .llm import GroqClientWrapper
from .tools import PatientContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


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
    session_id: str = Field(default="default", description="Conversation session ID")
    message: str = Field(..., min_length=1, max_length=2000, description="User question")
    patient_id: Optional[str] = Field(default="559", description="Ohio patient ID")
    readings: Optional[list[GlucoseReading]] = Field(default=None, description="Recent CGM readings")
    forecast_context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Active pipeline forecast context (predictions, current glucose, rate of change, patient-specific RMSE, etc.)",
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


@router.post("/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest) -> ChatResponse:
    """Send a query to the T1DM forecasting agent."""
    if not GROQ_API_KEY.strip():
        raise HTTPException(
            503,
            "Groq API key is not configured. Please set GROQ_API_KEY in your environment or .env file.",
        )

    store = get_agent_store()

    # Build patient context if custom readings were supplied
    if req.readings and len(req.readings) >= 24:
        dict_readings = [{"timestamp": r.timestamp, "glucose": r.glucose} for r in req.readings]
        context = PatientContext.from_readings(
            dict_readings,
            patient_id=req.patient_id or "559",
            forecast_context=req.forecast_context,
        )
    else:
        context = PatientContext.default(forecast_context=req.forecast_context)

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
    return {"status": "ok", "message": f"Session '{req.session_id}' reset successfully."}


@router.get("/chat/stream")
@router.post("/chat/stream")
async def agent_chat_stream(
    message: str = Query(..., description="User query"),
    session_id: str = Query("default", description="Session ID"),
) -> StreamingResponse:
    """SSE streaming variant that emits tool events followed by synthesized response tokens."""
    if not GROQ_API_KEY.strip():
        async def err_gen() -> AsyncIterator[str]:
            yield f"data: {json.dumps({'error': 'GROQ_API_KEY is not configured'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    store = get_agent_store()
    context = PatientContext.default()
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
