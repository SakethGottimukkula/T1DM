"""Groq-powered wellness suggestion agent for Type 1 Diabetes."""

from __future__ import annotations

import logging
import os
import re

import httpx
from fastapi import HTTPException

from .config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL
from .schemas import WellnessResponse

logger = logging.getLogger(__name__)

WELLNESS_DISCLAIMER: str = (
    "Disclaimer: This is only a suggestion, not medical advice. "
    "Please consult your doctor before following any meal or exercise plan."
)

WELLNESS_SYSTEM_PROMPT: str = """\
You are a helpful wellness assistant for people managing Type 1 Diabetes.

Guidelines:
- Offer practical, general meal ideas, meal plans, and activity or workout routines.
- Clearly state any assumptions or missing information.
- Do NOT diagnose, prescribe, recommend medication changes, calculate insulin doses, or override the user's diabetes care plan.
- Do NOT present glucose forecasts or model outputs as medical advice.
- If a request requires personalized medical decisions, explain that the user should consult their diabetes care team.
- For urgent symptoms or a possible emergency, advise contacting local emergency services or the user's clinician promptly.
- Ask only for information useful to the request. Do NOT request names or other identifying information.
- Do NOT claim that suggestions are clinically validated or guaranteed to be safe.

Always keep your response focused, helpful, and clearly framed as a general suggestion."""


async def get_wellness_suggestion(message: str) -> WellnessResponse:
    """Generate a wellness suggestion using Groq's chat completions API."""
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY).strip()
    model = os.getenv("GROQ_MODEL", GROQ_MODEL).strip()
    base_url = os.getenv("GROQ_BASE_URL", GROQ_BASE_URL).strip()

    if not api_key:
        raise HTTPException(
            503,
            "Wellness agent is not configured. Set the GROQ_API_KEY environment variable.",
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": WELLNESS_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.TimeoutException:
        raise HTTPException(504, "Groq API request timed out. Please try again.")
    except httpx.RequestError as exc:
        logger.error("Groq request error: %s", exc)
        raise HTTPException(502, "Could not reach the Groq API. Please try again later.")

    if resp.status_code == 401:
        logger.error("Groq API authentication failed (401)")
        raise HTTPException(502, "Groq authentication failed. Please verify GROQ_API_KEY.")
    elif resp.status_code != 200:
        logger.error("Groq API returned %d: %s", resp.status_code, resp.text[:500])
        raise HTTPException(
            502,
            "The Groq API returned an error. Please try again later.",
        )

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        logger.error("Unexpected Groq response structure: %s", exc)
        raise HTTPException(502, "Received an unexpected response from Groq.")

    # Remove internal thought/reasoning tags if present (e.g. from gpt-oss or reasoning models)
    content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()

    # Prepend disclaimer, avoiding duplication if the model already included it.
    trimmed = content.strip()
    if trimmed.startswith(WELLNESS_DISCLAIMER):
        final_content = trimmed
    elif WELLNESS_DISCLAIMER.lower() in trimmed.lower():
        # Avoid duplicating disclaimer if model placed it elsewhere
        cleaned = trimmed.replace(WELLNESS_DISCLAIMER, "").strip()
        final_content = f"{WELLNESS_DISCLAIMER}\n\n{cleaned}"
    else:
        final_content = f"{WELLNESS_DISCLAIMER}\n\n{trimmed}"

    return WellnessResponse(
        response=final_content,
        model=model,
        disclaimer=WELLNESS_DISCLAIMER,
    )
