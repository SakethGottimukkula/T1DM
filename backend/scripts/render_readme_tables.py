#!/usr/bin/env python
"""Render Markdown tables from backend/models/metrics.json for root README.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent


def build_table_1(d: dict) -> str:
    """T1: XGBoost performance (test set, N rows)."""
    n_eval = d.get("n_eval_rows", 0)
    n_str = f"{n_eval:,}" if n_eval else "N"
    lines = [
        f"### T1: XGBoost performance (test set, {n_str} rows)",
        "",
        "| Horizon | RMSE (mg/dL) | MAE (mg/dL) | Clarke A (%) | Clarke B (%) | Clarke C (%) | Clarke D (%) | Clarke E (%) | A+B (%) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    horizons = sorted(d.get("horizons", {}).keys(), key=lambda x: int(x))
    for hz in horizons:
        ov = d["horizons"][hz]["overall"]
        rmse = ov["rmse"]
        mae = ov["mae"]
        zp = ov.get("zone_percentages", {})
        ab = ov["a_plus_b_pct"]
        lines.append(
            f"| {hz} min | {rmse:.2f} | {mae:.2f} | {zp.get('A', 0.0):.2f} | {zp.get('B', 0.0):.2f} | {zp.get('C', 0.0):.2f} | {zp.get('D', 0.0):.2f} | {zp.get('E', 0.0):.2f} | {ab:.2f} |"
        )
    return "\n".join(lines)


def build_table_2(d: dict) -> str:
    """T2: Baseline comparison per horizon and per model."""
    lines = [
        "### T2: Baseline comparison",
        "",
        "| Horizon | Model | RMSE (mg/dL) | MAE (mg/dL) | A+B (%) |",
        "|:---:|:---|:---:|:---:|:---:|",
    ]
    horizons = sorted(d.get("horizons", {}).keys(), key=lambda x: int(x))
    baselines = d.get("baselines", {})
    for hz in horizons:
        ov = d["horizons"][hz]["overall"]
        lines.append(
            f"| {hz} min | XGBoost | {ov['rmse']:.2f} | {ov['mae']:.2f} | {ov['a_plus_b_pct']:.2f} |"
        )
        if "persistence" in baselines and hz in baselines["persistence"]:
            pm = baselines["persistence"][hz]
            lines.append(
                f"| {hz} min | Persistence | {pm['rmse']:.2f} | {pm['mae']:.2f} | {pm['a_plus_b_pct']:.2f} |"
            )
        if "ridge" in baselines and hz in baselines["ridge"]:
            rm = baselines["ridge"][hz]
            lines.append(
                f"| {hz} min | Ridge | {rm['rmse']:.2f} | {rm['mae']:.2f} | {rm['a_plus_b_pct']:.2f} |"
            )
        if "lasso" in baselines and hz in baselines["lasso"]:
            lm = baselines["lasso"][hz]
            lines.append(
                f"| {hz} min | Lasso | {lm['rmse']:.2f} | {lm['mae']:.2f} | {lm['a_plus_b_pct']:.2f} |"
            )
    return "\n".join(lines)


def build_table_3(d: dict) -> str:
    """T3: Per-patient RMSE and A+B (%)."""
    horizons = sorted(d.get("horizons", {}).keys(), key=lambda x: int(x))
    hz_headers = " | ".join(f"{h} min" for h in horizons)
    hz_align = " | ".join(":---:" for _ in horizons)
    lines = [
        "### T3: Per-patient RMSE and A+B (%)",
        "",
        f"| Patient | {hz_headers} |",
        f"|:---:| {hz_align} |",
    ]
    patients = d.get("patients", [])
    if not patients and horizons:
        patients = list(d["horizons"][horizons[0]].get("per_patient", {}).keys())
    sorted_patients = sorted(patients, key=lambda x: int(x) if str(x).isdigit() else str(x))

    for p in sorted_patients:
        pid_str = str(p)
        row_cells = []
        for hz in horizons:
            pm = d["horizons"][hz]["per_patient"].get(pid_str, {})
            rmse = pm.get("rmse", 0.0)
            ab = pm.get("a_plus_b_pct", 0.0)
            row_cells.append(f"{rmse:.2f} / {ab:.2f}")
        lines.append(f"| {pid_str} | {' | '.join(row_cells)} |")
    return "\n".join(lines)


def build_table_4(d: dict) -> str:
    """T4: Feature importance (gain), top 5 per horizon."""
    horizons = sorted(d.get("horizons", {}).keys(), key=lambda x: int(x))
    top5_per_hz = {}
    for hz in horizons:
        fi = d.get("feature_importance", {}).get(hz, {})
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_per_hz[hz] = sorted_fi

    lines = [
        "### T4: Feature importance (gain), top 5 per horizon",
        "",
        f"| Rank | {' | '.join(f'{h} min' for h in horizons)} |",
        f"|:---:| {' | '.join(':---' for _ in horizons)} |",
    ]
    for rank_idx in range(5):
        rank_num = rank_idx + 1
        rank_cells = []
        for hz in horizons:
            items = top5_per_hz.get(hz, [])
            if rank_idx < len(items):
                feat, val = items[rank_idx]
                rank_cells.append(f"`{feat}` ({val:.2f})")
            else:
                rank_cells.append("-")
        lines.append(f"| {rank_num} | {' | '.join(rank_cells)} |")
    return "\n".join(lines)


def render_all_tables(d: dict) -> str:
    """Render all four markdown tables separated by blank lines."""
    t1 = build_table_1(d)
    t2 = build_table_2(d)
    t3 = build_table_3(d)
    t4 = build_table_4(d)
    return f"{t1}\n\n{t2}\n\n{t3}\n\n{t4}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render metrics tables from metrics.json")
    parser.add_argument(
        "--metrics",
        default=str(_BACKEND_DIR / "models" / "metrics.json"),
        help="Path to metrics.json (default: backend/models/metrics.json)",
    )
    parser.add_argument(
        "--readme",
        default=str(_REPO_ROOT / "README.md"),
        help="Path to root README.md (default: repo root README.md)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Update root README.md between <!-- METRICS:START --> and <!-- METRICS:END -->",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    rendered_tables = render_all_tables(metrics_data)

    if not args.write:
        print(rendered_tables)
        return

    readme_path = Path(args.readme)
    if not readme_path.exists():
        print(f"ERROR: {readme_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()

    is_crlf = "\r\n" in readme_content
    nl = "\r\n" if is_crlf else "\n"

    start_marker = "<!-- METRICS:START -->"
    end_marker = "<!-- METRICS:END -->"

    if start_marker not in readme_content or end_marker not in readme_content:
        # Markers not found; must add them where old table was
        print(f"ERROR: Markers {start_marker} and {end_marker} not found in {readme_path}.", file=sys.stderr)
        sys.exit(1)

    start_idx = readme_content.index(start_marker) + len(start_marker)
    end_idx = readme_content.index(end_marker)

    rendered_block = nl + rendered_tables.replace("\n", nl) + nl

    new_readme = readme_content[:start_idx] + rendered_block + readme_content[end_idx:]

    with open(readme_path, "wb") as f:
        f.write(new_readme.encode("utf-8"))

    print(f"Updated {readme_path} between {start_marker} and {end_marker}")


if __name__ == "__main__":
    main()
