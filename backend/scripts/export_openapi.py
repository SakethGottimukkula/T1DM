#!/usr/bin/env python
"""Export or check the OpenAPI contract JSON for the frontend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from app.main import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Export or verify backend/openapi.json")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if backend/openapi.json matches app.openapi() without writing",
    )
    args = parser.parse_args()

    schema = json.loads(json.dumps(app.openapi()))
    out_file = _BACKEND_DIR / "openapi.json"

    if args.check:
        if not out_file.exists():
            print(
                "ERROR: backend/openapi.json does not exist. Run 'python scripts/export_openapi.py' to generate it.",
                file=sys.stderr,
            )
            sys.exit(1)
        with open(out_file, "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        if on_disk != schema:
            print(
                "ERROR: backend/openapi.json is out of date. Run 'python scripts/export_openapi.py' to update it.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("backend/openapi.json is up to date.")
        sys.exit(0)

    formatted = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with open(out_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(formatted)
    print(f"Exported OpenAPI schema to {out_file}")


if __name__ == "__main__":
    main()
