"""Agent configuration and environment loading."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load .env from the project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Groq API configuration
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Reasoning effort: 'low' | 'medium' | 'high'
_RAW_EFFORT: str = os.getenv("GROQ_REASONING_EFFORT", "medium").lower().strip()
if _RAW_EFFORT not in ("low", "medium", "high"):
    _RAW_EFFORT = "medium"
GROQ_REASONING_EFFORT: str = _RAW_EFFORT

# Token and agent limits
MAX_COMPLETION_TOKENS: int = int(os.getenv("MAX_COMPLETION_TOKENS", "4096"))
MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "5"))
MEMORY_TURNS: int = int(os.getenv("MEMORY_TURNS", "10"))
