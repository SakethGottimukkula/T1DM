"""Agent loop: coordinates LLM interactions, tool calls, and session memory."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from app.predictor import ModelStore

from .config import MAX_AGENT_ITERATIONS, MEMORY_TURNS
from .guardrails import (
    RESEARCH_DISCLAIMER,
    SYSTEM_PROMPT,
    post_process_response,
)
from .llm import GroqClientWrapper, parse_tool_args
from .tools import (
    TOOL_DEFINITIONS,
    PatientContext,
    execute_tool_call,
)

logger = logging.getLogger(__name__)

# In-memory session memory: session_id -> list of {"role": str, "content": str}
_SESSIONS: dict[str, list[dict[str, str]]] = {}


def get_session_history(session_id: str) -> list[dict[str, str]]:
    """Return recent conversation messages for a session."""
    return _SESSIONS.get(session_id, [])


def append_session_message(session_id: str, role: str, content: str) -> None:
    """Append a message and trim to MEMORY_TURNS (each turn is user + assistant)."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = []
    _SESSIONS[session_id].append({"role": role, "content": content})
    max_msgs = MEMORY_TURNS * 2
    if len(_SESSIONS[session_id]) > max_msgs:
        _SESSIONS[session_id] = _SESSIONS[session_id][-max_msgs:]


def reset_session(session_id: str) -> None:
    """Clear memory for a given session."""
    if session_id in _SESSIONS:
        del _SESSIONS[session_id]


def select_reasoning_effort(message: str) -> str:
    """Choose reasoning effort based on question depth.

    High effort for explanations, causality, risk evaluations, meal plans, workout routines, and hypo/hyper triage.
    Low effort for direct forecast lookups and single values.
    """
    lower = message.lower()
    high_keywords = (
        "why", "explain", "reason", "cause", "factor", "risk", "dangerous",
        "compare", "how come", "meal", "workout", "routine", "exercise",
        "hypo", "hyper", "what to do", "what should i do", "protocol",
    )
    if any(k in lower for k in high_keywords):
        return "high"

    low_keywords = ("what will", "what is", "current", "forecast", "reading", "value", "in 1 hour", "in 30", "in 60")
    if any(k in lower for k in low_keywords):
        return "low"

    return "medium"


class T1DMAgent:
    """Autonomous agent orchestrating Groq LLM and calibrated XGBoost tools."""

    def __init__(self, store: ModelStore, client_wrapper: Optional[GroqClientWrapper] = None):
        self.store = store
        self.client = client_wrapper or GroqClientWrapper()

    async def run(
        self,
        message: str,
        session_id: str = "default",
        context: Optional[PatientContext] = None,
    ) -> dict[str, Any]:
        """Execute the agent loop up to MAX_AGENT_ITERATIONS."""
        start_time = time.perf_counter()
        active_context = context or PatientContext.default()
        effort = select_reasoning_effort(message)

        # Prepare messages
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Inject active dashboard context guidance if available
        fc = getattr(active_context, "forecast_context", None)
        if fc:
            pt_id = fc.get("patient_id", active_context.patient_id)
            selected_pt_id = fc.get("selected_patient_id", pt_id)
            has_fc = fc.get("has_forecast", True)
            if not has_fc:
                ctx_note = (
                    f"[ACTIVE DASHBOARD CONTEXT: Selected patient {selected_pt_id}. Note: No forecast has been run yet "
                    f"for this patient/case. When answering forecast questions, verify tool data and inform the user "
                    f"that no active forecast is currently available. Suggest clicking 'Run Forecast' or uploading CGM data.]"
                )
            else:
                src = fc.get("source", "pipeline")
                ctx_note = (
                    f"[ACTIVE DASHBOARD CONTEXT: Selected patient {selected_pt_id}; forecast patient/source ID {pt_id}; source: {src}. "
                    f"{'This is the previously used forecast for this selected patient. ' if fc.get('previous_forecast') else ''}"
                    f"This context is authoritative for the current turn and replaces any older patient's context in the chat history. "
                    f"Use tools to retrieve exact predictions and metrics from the active forecast.]"
                )
            upload_quality = fc.get("upload_quality")
            if upload_quality:
                ctx_note += (
                    " [CSV REVIEW DATA: "
                    f"filename={upload_quality.get('filename')}; model={upload_quality.get('model')}; "
                    f"headers={upload_quality.get('headers')}; rows={upload_quality.get('row_count')}; "
                    f"validation_warnings={upload_quality.get('warnings')}; "
                    f"meal_defaulted_to_zero={upload_quality.get('meal_defaulted_to_zero')}. "
                    "When asked to improve the upload, base suggestions on these observed details, distinguish confirmed issues from general recommendations, and do not claim to have inspected data beyond the supplied summary. ]"
                )
            messages.append({"role": "system", "content": ctx_note})

        # Inject session history
        history = get_session_history(session_id)
        for hist_msg in history:
            messages.append({"role": hist_msg["role"], "content": hist_msg["content"]})

        # Append current user prompt
        messages.append({"role": "user", "content": message})

        tool_calls_executed: list[dict[str, Any]] = []
        final_answer = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0

        logger.info("Agent started: session=%s effort=%s query='%s'", session_id, effort, message[:80])

        for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
            iter_start = time.perf_counter()
            response_data = await self.client.chat(
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                reasoning_effort=effort,
            )

            usage = response_data.get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

            tool_calls = response_data.get("tool_calls", [])
            raw_text = response_data.get("text", "")

            # If no tools were called, this is the final answer
            if not tool_calls:
                final_answer = raw_text
                logger.info(
                    "Agent concluded in iteration %d (%.2fs)",
                    iteration,
                    time.perf_counter() - start_time,
                )
                break

            # Process tool calls (parallel where independent)
            tool_msg_items = []
            logger.info("Iteration %d: executing %d tool(s)", iteration, len(tool_calls))

            # Helper for executing single tool call
            def _run_single(tc: Any) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
                tc_id = getattr(tc, "id", f"call_{iteration}")
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", "")
                raw_args = getattr(fn, "arguments", "{}")
                try:
                    args = parse_tool_args(raw_args)
                except Exception as exc:
                    args = {}
                    result = {"error": str(exc)}
                    return tc_id, name, args, result

                result = execute_tool_call(name, args, active_context, self.store)
                return tc_id, name, args, result

            # Run all tool calls
            for tc in tool_calls:
                tc_id, name, args, result = _run_single(tc)
                tool_calls_executed.append({
                    "id": tc_id,
                    "name": name,
                    "args": args,
                    "result": result,
                })
                tool_msg_items.append((tc_id, json.dumps(result)))

            # Append assistant message with tool calls to conversation
            messages.append({
                "role": "assistant",
                "content": raw_text or None,
                "tool_calls": tool_calls,
            })

            # Append tool response messages
            for tc_id, result_json in tool_msg_items:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_json,
                })

        else:
            # Reached max iterations without finishing
            logger.warning("Agent reached max iterations (%d)", MAX_AGENT_ITERATIONS)
            final_answer = (
                "I have retrieved your glucose forecast and context from the models, but reached the maximum tool reasoning steps. "
                "Here is a summary of the model values obtained:\n"
            )
            for tc in tool_calls_executed:
                final_answer += f"- **{tc['name']}**: {json.dumps(tc['result'])}\n"

        # Apply safety guardrails post-processing
        guarded_answer = post_process_response(final_answer, tool_calls_executed)

        # Update in-memory session history
        append_session_message(session_id, "user", message)
        append_session_message(session_id, "assistant", guarded_answer)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        logger.info(
            "Agent finished: session=%s latency=%0.1fms tools_count=%d tokens=(prompt:%d, completion:%d)",
            session_id,
            latency_ms,
            len(tool_calls_executed),
            total_prompt_tokens,
            total_completion_tokens,
        )

        return {
            "answer": guarded_answer,
            "tool_calls": tool_calls_executed,
            "disclaimer": RESEARCH_DISCLAIMER,
            "session_id": session_id,
            "latency_ms": latency_ms,
            "tokens": {
                "prompt": total_prompt_tokens,
                "completion": total_completion_tokens,
            },
        }
