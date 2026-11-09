"""Test OpenAPI contract frozen snapshot consistency."""

from __future__ import annotations

import json
from pathlib import Path

from app.main import app


def test_openapi_contract_matches_app_schema():
    """Verify backend/openapi.json matches app.openapi() exactly."""
    openapi_file = Path(__file__).resolve().parent.parent / "openapi.json"
    assert openapi_file.exists(), (
        f"Missing {openapi_file}. Run 'python scripts/export_openapi.py' to generate it."
    )

    with open(openapi_file, "r", encoding="utf-8") as f:
        file_schema = json.load(f)

    app_schema = json.loads(json.dumps(app.openapi()))
    assert file_schema == app_schema, (
        "backend/openapi.json differs from app.openapi(). "
        "Run 'python scripts/export_openapi.py' to update it."
    )
