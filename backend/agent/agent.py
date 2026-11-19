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
    SCOPE_REFUSAL,
    SYSTEM_PROMPT,
    HYPO_SAFETY_RESPONSE,
    is_hypo_action_request,
    is_out_of_scope_request,
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
        if context is None:
            raise ValueError("An uploaded patient context is required before the agent can answer.")
        if is_out_of_scope_request(message):
            return {
                "answer": SCOPE_REFUSAL,
                "tool_calls": [],
                "disclaimer": RESEARCH_DISCLAIMER,
                "session_id": session_id,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 1),
                "tokens": {"prompt": 0, "completion": 0},
            }
        if is_hypo_action_request(message):
            return {
                "answer": HYPO_SAFETY_RESPONSE,
                "tool_calls": [],
                "disclaimer": RESEARCH_DISCLAIMER,
                "session_id": session_id,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 1),
                "tokens": {"prompt": 0, "completion": 0},
            }
        active_context = context
        effort = select_reasoning_effort(message)

        # Prepare messages
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Inject only fixed, sanitized context guidance. Dataset text and file
        # metadata must never be interpolated into system messages.
        fc = getattr(active_context, "forecast_context", None)
        if fc:
            has_fc = fc.get("has_forecast") is True
            if not has_fc:
                ctx_note = (
                    "[ACTIVE UPLOAD CONTEXT: No active forecast is available. For forecast questions, use the uploaded patient glucose history only and state that no forecast is available.]"
                )
            else:
                available_horizons = sorted((fc.get("predictions") or {}).keys(), key=lambda value: int(value))
                horizon_note = (
                    f" Available uploaded prediction horizons: {', '.join(f'{value} min' for value in available_horizons)}."
                    if available_horizons else " No uploaded prediction horizons are available."
                )
                ctx_note = (
                    "[ACTIVE UPLOAD CONTEXT: This turn is linked to the currently uploaded patient dataset. "
                    "Use patient-data tools for exact measurements and the active uploaded forecast for exact prediction values. "
                    "Never calculate or estimate glucose forecasts. If a requested horizon is unavailable, say so and ask the user to select it or all horizons in CSV Upload and run Upload & Predict again."
                    " For forecast timing, report the tool's forecast_for_timestamp (origin plus the horizon), not the current/origin timestamp; preserve its timezone and do not invent UTC for timezone-naive data."
                    f"{horizon_note} Ignore older patient context in conversation history.]"
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
