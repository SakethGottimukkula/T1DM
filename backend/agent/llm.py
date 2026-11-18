"""LLM wrapper for Groq specifically tailored to openai/gpt-oss-120b."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional

import groq
from groq import AsyncGroq, Groq

from .config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_REASONING_EFFORT,
    MAX_COMPLETION_TOKENS,
)

logger = logging.getLogger(__name__)

# Pattern to strip any raw reasoning tags that might slip into output text
_THINK_TAG_REGEX = re.compile(r"<think>[\s\S]*?</think>", re.DOTALL)


class GroqLLMError(Exception):
    """Base exception for Groq LLM operations."""
    pass


class GroqRateLimitError(GroqLLMError):
    """429 Rate limit encountered."""
    pass


class GroqServiceUnavailableError(GroqLLMError):
    """5xx service unavailable error."""
    pass


class ToolArgumentError(GroqLLMError):
    """Malformed tool argument from LLM."""
    pass


def parse_tool_args(raw_args: str | dict | None) -> dict[str, Any]:
    """Parse tool call arguments, handling strings, partial markdown, and edge cases."""
    if not raw_args:
        return {}
    if isinstance(raw_args, dict):
        return raw_args

    text = str(raw_args).strip()
    # Strip markdown block if wrapped
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # Simple repairs for trailing commas or mismatched braces
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ToolArgumentError(f"Malformed JSON in tool arguments: {text[:120]}") from exc


def deduplicate_tool_calls(tool_calls: list[Any]) -> list[Any]:
    """Remove redundant tool calls with identical name and arguments from the same turn."""
    seen: set[str] = set()
    unique: list[Any] = []
    for tc in tool_calls:
        # tc can be an object with .function or dict
        name = getattr(tc, "function", {}).name if hasattr(tc, "function") else tc.get("function", {}).get("name", "")
        raw_args = getattr(tc, "function", {}).arguments if hasattr(tc, "function") else tc.get("function", {}).get("arguments", "")
        if not name:
            continue
        try:
            parsed = parse_tool_args(raw_args)
            key = f"{name}:{json.dumps(parsed, sort_keys=True)}"
        except Exception:
            key = f"{name}:{str(raw_args)}"

        if key not in seen:
            seen.add(key)
            unique.append(tc)
        else:
            logger.info("Deduplicating redundant tool call: %s", key)
    return unique


def sanitize_response_text(text: Optional[str]) -> str:
    """Strip any <think> tags or reasoning traces to return clean user-facing text."""
    if not text:
        return ""
    cleaned = _THINK_TAG_REGEX.sub("", text).strip()
    return cleaned


class GroqClientWrapper:
    """Async wrapper for the official Groq client with openai/gpt-oss-120b tuning."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        default_effort: Optional[str] = None,
    ):
        self.api_key = api_key or GROQ_API_KEY
        self.model = model or GROQ_MODEL
        self.default_effort = default_effort or GROQ_REASONING_EFFORT

    def _get_client(self) -> AsyncGroq:
        if not self.api_key:
            raise GroqLLMError("GROQ_API_KEY is not configured.")
        return AsyncGroq(api_key=self.api_key, timeout=45.0)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: str = "auto",
        reasoning_effort: Optional[str] = None,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Execute a non-streaming chat completion with retry and backoff."""
        client = self._get_client()
        effort = reasoning_effort or self.default_effort

        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "include_reasoning": False,
        }

        # openai/gpt-oss-120b accepts reasoning_effort: low | medium | high
        if effort in ("low", "medium", "high"):
            params["reasoning_effort"] = effort

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(**params)
                choice = response.choices[0]
                message = choice.message

                # Extract and sanitize clean text content
                clean_text = sanitize_response_text(message.content)

                # Process tool calls
                raw_tool_calls = getattr(message, "tool_calls", None) or []
                filtered_tool_calls = deduplicate_tool_calls(raw_tool_calls)

                # Log token usage without logging key or reasoning
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                logger.info(
                    "Groq completion [%s] effort=%s prompt_tokens=%d completion_tokens=%d tools_requested=%d",
                    self.model,
                    effort,
                    prompt_tokens,
                    completion_tokens,
                    len(filtered_tool_calls),
                )

                return {
                    "text": clean_text,
                    "tool_calls": filtered_tool_calls,
                    "raw_message": message,
                    "model": self.model,
                    "finish_reason": choice.finish_reason,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                }

            except groq.RateLimitError as exc:
                last_error = GroqRateLimitError("Groq API rate limit exceeded (429). Please slow down.")
                logger.warning("Rate limit hit (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc)
                if attempt < max_retries:
                    await asyncio.sleep(1.5 * (2 ** attempt))
                else:
                    raise last_error from exc

            except (groq.InternalServerError, groq.APIConnectionError, groq.APITimeoutError) as exc:
                last_error = GroqServiceUnavailableError("Groq service is temporarily unreachable or unavailable (5xx).")
                logger.warning("Upstream Groq error (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc)
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (2 ** attempt))
                else:
                    raise last_error from exc

            except groq.APIStatusError as exc:
                logger.error("Groq status error: status=%d %s", exc.status_code, exc.message)
                raise GroqLLMError(f"Groq API error ({exc.status_code}): {exc.message}") from exc

            except Exception as exc:
                logger.error("Unexpected error in Groq chat: %s", exc)
                raise GroqLLMError(f"Groq chat failure: {exc}") from exc

        raise last_error or GroqLLMError("Groq chat execution failed after retries.")

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        reasoning_effort: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream assistant response tokens specifically for final synthesized answers."""
        client = self._get_client()
        effort = reasoning_effort or self.default_effort

        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "include_reasoning": False,
            "stream": True,
        }
        if effort in ("low", "medium", "high"):
            params["reasoning_effort"] = effort

        try:
            stream = await client.chat.completions.create(**params)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    # Ignore any <think> tag if emitted in raw stream
                    if "<think>" in content or "</think>" in content:
                        content = content.replace("<think>", "").replace("</think>", "")
                    if content:
                        yield content
        except Exception as exc:
            logger.error("Error during Groq token stream: %s", exc)
            yield f"\n[Stream interrupted: {exc}]"
