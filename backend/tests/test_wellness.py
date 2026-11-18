"""Tests for the Groq wellness suggestion agent."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.wellness import (
    WELLNESS_DISCLAIMER,
    WELLNESS_SYSTEM_PROMPT,
    get_wellness_suggestion,
)


def test_wellness_missing_key_raises_503(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(get_wellness_suggestion("Suggest a low-carb breakfast."))
    assert exc_info.value.status_code == 503
    assert "not configured" in exc_info.value.detail.lower()


def test_wellness_success_prepends_disclaimer(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_fake_key_12345")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "A vegetable omelet with spinach and mushrooms is a great option."
                }
            }
        ]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = asyncio.run(get_wellness_suggestion("Suggest breakfast"))
        assert res.response.startswith(WELLNESS_DISCLAIMER)
        assert "vegetable omelet" in res.response
        assert res.model == "llama-3.3-70b-versatile"


def test_wellness_no_disclaimer_duplication(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_fake_key_12345")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": f"{WELLNESS_DISCLAIMER}\n\nHere are gentle walking routines."
                }
            }
        ]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = asyncio.run(get_wellness_suggestion("Suggest walk"))
        assert res.response.count(WELLNESS_DISCLAIMER) == 1


def test_wellness_gpt_oss_reasoning_tags_stripped(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_fake_key_12345")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "<think>Reasoning through gentle 20-min walking routine...</think>A 20-minute post-meal walk helps steady glucose."
                }
            }
        ]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = asyncio.run(get_wellness_suggestion("Post-meal workout"))
        assert "<think>" not in res.response
        assert "</think>" not in res.response
        assert "post-meal walk" in res.response
        assert res.model == "openai/gpt-oss-120b"


def test_wellness_groq_upstream_error_raises_502(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_fake_key_12345")

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Groq Error"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(get_wellness_suggestion("Snack ideas"))
        assert exc_info.value.status_code == 502

