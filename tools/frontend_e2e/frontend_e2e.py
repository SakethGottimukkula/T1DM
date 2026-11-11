"""Live end-to-end Playwright verification for Glucose Horizon frontend (FE-1, FE-2, FE-3).

Expects the FastAPI backend (with TEST_CSV_PATH set) and static frontend server
to be running already. Prints PASS/FAIL checks and aggregate counts only
(no raw patient glucose values, timestamps, or dataset directory paths).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from playwright.sync_api import sync_playwright

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://127.0.0.1:5500").rstrip("/")
HORIZONS = [30, 60, 90, 120]
ZONE_COLORS = {
    "A": "#218566",
    "B": "#087f8c",
    "C": "#ba7918",
    "D": "#e8825a",
    "E": "#d35d57",
}
ZONE_LABEL_MAP = {
    "low": ("Below range", "bad"),
    "in-range": ("In range", "ok"),
    "high": ("Above range", "warn"),
}


def js_round(val: float) -> int:
    """Match JavaScript Math.round for non-negative glucose / error values."""
    return int(math.floor(float(val) + 0.5))


def js_num(val: float) -> str:
    """Match JavaScript default Number -> String formatting."""
    f = float(val)
    return str(int(f)) if f.is_integer() else str(f)


def format_size(num_bytes: int) -> str:
    kb = 1024.0
    mb = kb * kb
    if num_bytes < mb:
        return f"{num_bytes / kb:.1f} KB"
    return f"{num_bytes / mb:.1f} MB"


def api_get(path: str) -> Any:
    req = urllib.request.Request(f"{API_URL}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_post_json(path: str, payload: dict[str, Any]) -> Any:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_post_form(path: str, fields: dict[str, str]) -> Any:
    boundary = "----E2EFormBoundary7MA4YWxkTrZu0gW"
    parts = []
    for k, v in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{k}"\r\n\r\n'
            f"{v}\r\n"
        )
    parts.append(f"--{boundary}--\r\n")
    body = "".join(parts).encode("utf-8")
    req = urllib.request.Request(
        f"{API_URL}{path}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


class Runner:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        tag = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        suffix = f" ({detail})" if detail else ""
        print(f"{tag}: {name}{suffix}", flush=True)
        return ok


def wait_for_ready(page: Any, timeout_ms: int = 20000) -> None:
    page.wait_for_function(
        """() => {
            const cards = document.querySelectorAll('#fcards .stat b');
            return cards.length === 4 && document.querySelectorAll('.sel-case option').length > 0;
        }""",
        timeout=timeout_ms,
    )


def wait_for_case_id(page: Any, expected_id: str, timeout_ms: int = 15000) -> None:
    page.wait_for_function(
        "(id) => typeof appState !== 'undefined' && appState.caseId === id",
        arg=expected_id,
        timeout=timeout_ms,
    )


def get_active_case_id(page: Any) -> str:
    return str(page.evaluate("() => appState.caseId"))


def verify_case_views(
    runner: Runner,
    page: Any,
    case_id: str,
    prefix: str,
) -> None:
    detail = api_get(f"/v1/cases/{urllib.request.quote(case_id)}")
    pred_resp = api_post_json(
        "/v1/predict", {"readings": detail["history"], "horizon": "all"}
    )
    preds_by_h = {int(p["horizon"]): p for p in pred_resp["predictions"]}
    actual_by_h = {int(k): float(v) for k, v in detail["actual_at"].items()}
    cur_g = float(pred_resp["context"]["current_glucose"])

    # Forecast cards & Risk trajectory zone labels (Overview)
    page.click("button[data-view='overview']")
    fcards = page.evaluate(
        """() => Array.from(document.querySelectorAll('#fcards .stat')).map(el => ({
            small: el.querySelector('small')?.textContent?.trim() || '',
            val: el.querySelector('b')?.textContent?.trim() || '',
            span: el.querySelector('span')?.textContent?.trim() || '',
            sel: el.classList.contains('sel')
        }))"""
    )
    fcards_ok = len(fcards) == 4
    if fcards_ok:
        for h, card in zip(HORIZONS, fcards):
            p_val = js_round(preds_by_h[h]["predicted_glucose"])
            a_val = js_round(actual_by_h[h])
            e_val = js_round(abs(preds_by_h[h]["predicted_glucose"] - actual_by_h[h]))
            if (
                card["small"] != f"{h} min forecast"
                or card["val"] != f"{p_val} mg/dL"
                or card["span"] != f"Actual {a_val} · error {e_val}"
            ):
                fcards_ok = False
                break
    runner.record(f"{prefix} forecast cards match /v1/predict", fcards_ok, f"count={len(fcards)}")

    # Risk trajectory zone_label check per horizon + #interp zone label
    traj = page.evaluate(
        """() => Array.from(document.querySelectorAll('#riskTrajectory > div')).map(row => {
            const spans = row.querySelectorAll('span');
            return {
                left: spans[0]?.textContent?.trim() || '',
                label: spans[1]?.textContent?.trim() || '',
                cls: spans[1]?.className || ''
            };
        })"""
    )
    zone_ok = len(traj) == 4
    if zone_ok:
        for h, row in zip(HORIZONS, traj):
            z = preds_by_h[h]["zone_label"]
            exp_label, exp_cls = ZONE_LABEL_MAP[z]
            if row["label"] != exp_label or exp_cls not in row["cls"].split():
                zone_ok = False
                break
    active_h = int(page.evaluate("() => appState.horizon"))
    interp_text = page.locator("#interp").inner_text()
    exp_active_zone = ZONE_LABEL_MAP[preds_by_h[active_h]["zone_label"]][0].lower()
    zone_ok = zone_ok and (f"({exp_active_zone})" in interp_text)
    runner.record(f"{prefix} zone labels match /v1/predict zone_label", zone_ok, f"horizons={len(traj)}")

    # Chart point counts 37 / 25 / 5
    chart_counts = page.evaluate(
        """() => ({
            cMain: charts.cMain.data.datasets.map(d => d.data.length),
            cDetail: charts.cDetail.data.datasets.map(d => d.data.length)
        })"""
    )
    expected_counts = [len(detail["history"]), len(detail["future"]) + 1, 5]
    counts_ok = (
        expected_counts == [37, 25, 5]
        and chart_counts["cMain"] == [37, 25, 5]
        and chart_counts["cDetail"] == [37, 25, 5]
    )
    runner.record(
        f"{prefix} chart point counts equal 37/25/5",
        counts_ok,
        f"cMain={chart_counts['cMain']}, cDetail={chart_counts['cDetail']}",
    )

    # Error table (Forecast view)
    page.click("button[data-view='forecast']")
    rows = page.evaluate(
        """() => Array.from(document.querySelectorAll('#errTable tr')).slice(1).map(tr =>
            Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim())
        )"""
    )
    err_ok = len(rows) == 4
    if err_ok:
        for h, cells in zip(HORIZONS, rows):
            p_raw = preds_by_h[h]["predicted_glucose"]
            a_raw = actual_by_h[h]
            exp = [
                f"{h} min",
                str(js_round(cur_g)),
                str(js_round(p_raw)),
                str(js_round(a_raw)),
                f"{js_round(abs(p_raw - a_raw))} mg/dL",
            ]
            if cells != exp:
                err_ok = False
                break
    runner.record(f"{prefix} error table matches /v1/predict", err_ok, f"rows={len(rows)}")

    # Explorer context cards (#results & #ctx)
    page.click("button[data-view='explorer']")
    results_cards = page.evaluate(
        """() => Array.from(document.querySelectorAll('#results .stat')).map(el => ({
            label: el.querySelector('small')?.textContent?.trim() || '',
            val: el.querySelector('b')?.textContent?.trim() || '',
            span: el.querySelector('span')?.textContent?.trim() || ''
        }))"""
    )
    ctx_data = pred_resp["context"]
    sel_pred = preds_by_h[active_h]["predicted_glucose"]
    sel_act = actual_by_h[active_h]
    sel_err = abs(sel_pred - sel_act)
    res_ok = (
        len(results_cards) == 5
        and results_cards[0]["label"] == "Model window"
        and results_cards[0]["val"] == f"{ctx_data['window_readings']} readings"
        and results_cards[1]["val"] == f"{js_round(sel_pred)} mg/dL"
        and results_cards[2]["val"] == f"{js_round(sel_act)} mg/dL"
        and results_cards[3]["val"] == f"{js_round(sel_err)} mg/dL"
    )
    runner.record(f"{prefix} explorer #results matches /v1/predict", res_ok, f"cards={len(results_cards)}")

    ctx_cards = page.evaluate(
        """() => Object.fromEntries(Array.from(document.querySelectorAll('#ctx .stat')).map(el => [
            el.querySelector('small')?.textContent?.trim() || '',
            el.querySelector('b')?.textContent?.trim() || ''
        ]))"""
    )
    ctx_ok = (
        len(ctx_cards) == 9
        and ctx_cards.get("Current glucose") == f"{js_num(ctx_data['current_glucose'])} mg/dL"
        and ctx_cards.get("CGM readings in model window") == f"{js_num(ctx_data['window_readings'])} readings"
        and ctx_cards.get("Model window") == f"{js_num(ctx_data['window_span_minutes'])} min"
        and ctx_cards.get("Minimum glucose") == f"{js_num(ctx_data['window_min'])} mg/dL"
        and ctx_cards.get("Maximum glucose") == f"{js_num(ctx_data['window_max'])} mg/dL"
        and ctx_cards.get("Mean glucose") == f"{js_num(ctx_data['window_mean'])} mg/dL"
        and ctx_cards.get("Recent 5-minute change") == f"{js_num(ctx_data['change_5min'])} mg/dL"
        and ctx_cards.get("Recent 30-minute change") == f"{js_num(ctx_data['change_30min'])} mg/dL"
        and len(ctx_cards.get("Forecast timestamp", "")) >= 10
    )
    runner.record(f"{prefix} explorer #ctx matches /v1/predict context", ctx_ok, f"cards={len(ctx_cards)}")


def run_fe1_checks(runner: Runner, page: Any) -> None:
    cases_resp = api_get("/v1/cases?per_patient=3")
    patients_api = [str(p["patient_id"]) for p in cases_resp["patients"]]
    cases_api = cases_resp["cases"]

    # 6 patients
    patient_opts = page.evaluate(
        "() => Array.from(document.querySelector('.sel-patient').options).map(o => o.value)"
    )
    runner.record(
        "FE-1 6 patients loaded matching /v1/cases",
        len(patient_opts) == 6 and patient_opts == patients_api,
        f"count={len(patient_opts)}",
    )

    # 3 cases for first patient
    first_pid = patients_api[0]
    first_patient_cases = [c["case_id"] for c in cases_api if str(c["patient_id"]) == first_pid]
    case_opts = page.evaluate(
        "() => Array.from(document.querySelector('.sel-case').options).map(o => o.value)"
    )
    runner.record(
        "FE-1 3 cases for first patient",
        len(case_opts) == 3 and case_opts == first_patient_cases,
        f"count={len(case_opts)}",
    )

    initial_case_id = get_active_case_id(page)
    verify_case_views(runner, page, initial_case_id, "FE-1 initial case")

    # Horizon change -> 90, verify cards & prediction, then restore 60
    page.click("button[data-view='explorer']")
    page.select_option("#selH", "90")
    pred_90_text = page.locator("#results .stat").nth(1).locator("b").inner_text().strip()
    pred_90_span = page.locator("#results .stat").nth(1).locator("span").inner_text().strip()
    detail_init = api_get(f"/v1/cases/{urllib.request.quote(initial_case_id)}")
    pred_init = api_post_json("/v1/predict", {"readings": detail_init["history"], "horizon": "all"})
    p90 = next(p for p in pred_init["predictions"] if int(p["horizon"]) == 90)
    page.click("button[data-view='overview']")
    sel_small = page.locator("#fcards .stat.sel small").inner_text().strip()
    horizon_ok = (
        pred_90_text == f"{js_round(p90['predicted_glucose'])} mg/dL"
        and pred_90_span == "90 min ahead"
        and sel_small == "90 min forecast"
    )
    runner.record("FE-1 horizon change updates Explorer and Overview", horizon_ok, "horizon=90")
    page.click("button[data-view='explorer']")
    page.select_option("#selH", "60")

    # Case change -> second case of first patient
    second_case_id = first_patient_cases[1]
    page.select_option(".sel-case >> nth=0", second_case_id)
    wait_for_case_id(page, second_case_id)
    verify_case_views(runner, page, second_case_id, "FE-1 case change")

    # Patient change -> second patient
    second_pid = patients_api[1]
    second_patient_cases = [c["case_id"] for c in cases_api if str(c["patient_id"]) == second_pid]
    page.select_option(".sel-patient >> nth=0", second_pid)
    wait_for_case_id(page, second_patient_cases[0])
    page.click("button[data-view='overview']")
    p2_opts = page.evaluate(
        "() => Array.from(document.querySelector('.sel-case').options).map(o => o.value)"
    )
    p2_summary = page.locator("#summary .stat").nth(2).locator("b").inner_text().strip()
    runner.record(
        "FE-1 patient change updates cases and patient summary",
        p2_opts == second_patient_cases and p2_summary == f"Patient {second_pid}",
        f"cases={len(p2_opts)}",
    )

    # Random Test Case button (#rand)
    page.click("button[data-view='explorer']")
    before_rand = get_active_case_id(page)
    page.click("#rand")
    page.wait_for_function("(prev) => appState.caseId !== prev", arg=before_rand, timeout=15000)
    after_rand = get_active_case_id(page)
    runner.record(
        "FE-1 Random Test Case switches to another case",
        after_rand != before_rand and after_rand in [c["case_id"] for c in cases_api],
    )

    # Run Forecast button (#run) triggers POST /v1/predict
    with page.expect_response(
        lambda r: r.url.endswith("/v1/predict") and r.request.method == "POST" and r.status == 200,
        timeout=15000,
    ) as resp_info:
        page.click("#run")
    pred_http_resp = resp_info.value
    runner.record(
        "FE-1 Run Forecast calls POST /v1/predict",
        pred_http_resp.status == 200 and pred_http_resp.request.method == "POST",
        f"status={pred_http_resp.status}",
    )

    # Stale-response guard during rapid case switching: clear cache first in page, then race two uncached requests
    target_slow = first_patient_cases[2]
    target_fast = second_patient_cases[1]
    if target_fast == get_active_case_id(page):
        target_fast = second_patient_cases[2]
    page.evaluate("() => { caseCache.clear(); }")

    slow_encoded = urllib.request.quote(target_slow)
    slow_pattern = f"**/v1/cases/{target_slow}"

    def delay_slow_case(route: Any) -> None:
        time.sleep(0.6)
        route.continue_()

    page.route(slow_pattern, delay_slow_case)
    # Trigger slow case then immediately trigger fast case without waiting
    page.evaluate(
        """([slowId, fastId]) => {
            selectCaseById(slowId);
            selectCaseById(fastId);
        }""",
        [target_slow, target_fast],
    )
    wait_for_case_id(page, target_fast)
    # Wait past the 600ms delayed response to make sure stale response does NOT overwrite
    page.wait_for_timeout(900)
    page.unroute(slow_pattern)
    page.click("button[data-view='overview']")
    final_id = get_active_case_id(page)
    summary_patient = page.locator("#summary .stat").nth(2).locator("b").inner_text().strip()
    expected_fast_pid = next(str(c["patient_id"]) for c in cases_api if c["case_id"] == target_fast)
    runner.record(
        "FE-1 stale-response guard ignores slower case response",
        final_id == target_fast and summary_patient == f"Patient {expected_fast_pid}",
        f"match={final_id == target_fast}",
    )


def run_fe2_checks(runner: Runner, page: Any) -> None:
    metrics_resp = api_get("/v1/metrics")
    m = metrics_resp["metrics"]

    page.click("button[data-view='performance']")
    # Performance cards
    perf_cards = page.evaluate(
        """() => Array.from(document.querySelectorAll('#perfCards .stat')).map(el => ({
            small: el.querySelector('small')?.textContent?.trim() || '',
            val: el.querySelector('b')?.textContent?.trim() || '',
            span: el.querySelector('span')?.textContent?.trim() || ''
        }))"""
    )
    pc_ok = len(perf_cards) == 4
    if pc_ok:
        for h, c in zip(HORIZONS, perf_cards):
            ov = m["horizons"][str(h)]["overall"]
            if (
                c["small"] != f"{h} min RMSE"
                or c["val"] != f"{ov['rmse']:.2f} mg/dL"
                or c["span"] != f"MAE {ov['mae']:.2f} · Clarke A+B {ov['a_plus_b_pct']:.2f}%"
            ):
                pc_ok = False
                break
    runner.record("FE-2 #perfCards match /v1/metrics", pc_ok, f"cards={len(perf_cards)}")

    # #hTable
    h_rows = page.evaluate(
        """() => Array.from(document.querySelectorAll('#hTable tr')).slice(1).map(tr =>
            Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim())
        )"""
    )
    ht_ok = len(h_rows) == 4
    if ht_ok:
        for h, cells in zip(HORIZONS, h_rows):
            ov = m["horizons"][str(h)]["overall"]
            exp = [
                f"{h} min",
                f"{ov['rmse']:.2f} mg/dL",
                f"{ov['mae']:.2f} mg/dL",
                f"{ov['a_plus_b_pct']:.2f}%",
            ]
            if cells != exp:
                ht_ok = False
                break
    runner.record("FE-2 #hTable matches /v1/metrics", ht_ok, f"rows={len(h_rows)}")

    # cErr chart data
    cerr_data = page.evaluate(
        """() => charts.cErr.data.datasets.map(ds => ({ label: ds.label, data: ds.data }))"""
    )
    exp_xgb = [m["horizons"][str(h)]["overall"]["rmse"] for h in HORIZONS]
    exp_pers = [m["baselines"]["persistence"][str(h)]["rmse"] for h in HORIZONS]
    exp_ridge = [m["baselines"]["ridge"][str(h)]["rmse"] for h in HORIZONS]
    cerr_ok = (
        len(cerr_data) == 3
        and cerr_data[0]["data"] == exp_xgb
        and cerr_data[1]["data"] == exp_pers
        and cerr_data[2]["data"] == exp_ridge
    )
    runner.record("FE-2 cErr chart matches /v1/metrics RMSE", cerr_ok, f"series={len(cerr_data)}")

    # #mTable + summary sentence consistency
    m_rows = page.evaluate(
        """() => Array.from(document.querySelectorAll('#mTable tr')).slice(1).map(tr =>
            Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim())
        )"""
    )
    mt_ok = len(m_rows) >= 6
    if mt_ok:
        sources = [
            ("XGBoost", lambda h: m["horizons"][str(h)]["overall"]),
            ("Persistence", lambda h: m["baselines"]["persistence"][str(h)]),
            ("Ridge", lambda h: m["baselines"]["ridge"][str(h)]),
            ("Lasso", lambda h: m["baselines"]["lasso"][str(h)]),
        ]
        for idx, (name, getter) in enumerate(sources):
            exp_rmse = " / ".join(f"{getter(h)['rmse']:.2f}" for h in HORIZONS)
            exp_ab = " / ".join(f"{getter(h)['a_plus_b_pct']:.2f}" for h in HORIZONS)
            if m_rows[idx] != [name, exp_rmse, exp_ab]:
                mt_ok = False
                break

    lowest_rmse_all = all(
        m["horizons"][str(h)]["overall"]["rmse"]
        <= min(
            m["baselines"]["persistence"][str(h)]["rmse"],
            m["baselines"]["ridge"][str(h)]["rmse"],
            m["baselines"]["lasso"][str(h)]["rmse"],
        )
        for h in HORIZONS
    )
    margins = [
        round(m["baselines"]["ridge"][str(h)]["rmse"] - m["horizons"][str(h)]["overall"]["rmse"], 2)
        for h in HORIZONS
    ]
    summary_cell = m_rows[4][0] if len(m_rows) > 4 else ""
    meta_cell = m_rows[5][0] if len(m_rows) > 5 else ""
    summary_ok = (
        ("lowest RMSE across all horizons" in summary_cell if lowest_rmse_all else "lowest RMSE" in summary_cell)
        and f"{min(margins):.2f} mg/dL to {max(margins):.2f} mg/dL" in summary_cell
        and f"Evaluated on {m['n_eval_rows']} test rows from {len(m['patients'])} patients" in meta_cell
    )
    runner.record("FE-2 #mTable models & summary consistent with /v1/metrics", mt_ok and summary_ok, f"rows={len(m_rows)}")

    # Clinical view across all 4 horizons
    for h in HORIZONS:
        page.click("button[data-view='explorer']")
        page.select_option("#selH", str(h))
        page.click("button[data-view='clinical']")
        page.wait_for_function(
            "(hz) => typeof clinicalState !== 'undefined' && clinicalState.phase === 'ready' && clinicalState.horizon === hz",
            arg=h,
            timeout=30000,
        )
        bt = api_post_form("/v1/backtest", {"horizon": str(h), "max_points": "2000"})

        c_info = page.evaluate(
            """() => ({
                sampleLen: charts.cClarke.data.datasets[0].data.length,
                bgColors: charts.cClarke.data.datasets[0].pointBackgroundColor,
                selCaseLen: charts.cClarke.data.datasets[1].data.length
            })"""
        )
        sample_ok = c_info["sampleLen"] == len(bt["actual"]) and c_info["selCaseLen"] == 4

        exp_color_counts: dict[str, int] = {}
        for z in bt["zones"]:
            col = ZONE_COLORS.get(z, "#718391")
            exp_color_counts[col] = exp_color_counts.get(col, 0) + 1
        got_color_counts: dict[str, int] = {}
        for col in c_info["bgColors"]:
            got_color_counts[col] = got_color_counts.get(col, 0) + 1
        colors_ok = got_color_counts == exp_color_counts

        zrows = page.evaluate(
            """() => Array.from(document.querySelectorAll('#zones .zrow')).map(row => ({
                letter: row.querySelector('b')?.textContent?.trim() || '',
                pct: row.querySelector('span')?.textContent?.trim() || ''
            }))"""
        )
        exp_zones = [
            {"letter": k, "pct": f"{float(v):.2f}%"}
            for k, v in bt["zone_percentages"].items()
        ]
        zones_ok = zrows == exp_zones

        markers = page.evaluate(
            """() => Array.from(document.querySelectorAll('#range .rbar .mk')).map(el => ({
                text: el.textContent.trim(),
                left: el.style.left
            }))"""
        )
        cur_case_pid = page.evaluate("() => String(getCurrentCase().patient)")
        case_cur_val = page.evaluate("() => Math.round(getCurrentCase().cur)")
        case_pred_val = page.evaluate("(hz) => Math.round(getCurrentCase().preds[hz].pred)", h)
        markers_ok = (
            len(markers) == 2
            and markers[0]["text"] == f"Now {case_cur_val}"
            and markers[1]["text"] == f"Pred {case_pred_val}"
        )

        zones_note = page.locator("#zonesNote").inner_text()
        p_info = bt["per_patient"][cur_case_pid]
        exp_pat_sentence = (
            f"Patient {cur_case_pid} at this horizon: RMSE {float(p_info['rmse']):.2f} mg/dL, "
            f"A+B {float(p_info['a_plus_b_pct']):.2f} %, {p_info['n_rows']} rows evaluated."
        )
        note_ok = (
            f"Evaluated {bt['n_rows']} rows (A+B = {float(bt['a_plus_b_pct']):.2f}%)." in zones_note
            and exp_pat_sentence in zones_note
        )

        ov_h = m["horizons"][str(h)]["overall"]
        metrics_consistent = (
            abs(float(bt["rmse"]) - float(ov_h["rmse"])) < 0.05
            and abs(float(bt["a_plus_b_pct"]) - float(ov_h["a_plus_b_pct"])) < 0.05
            and int(bt["n_rows"]) == int(m["n_eval_rows"])
        )

        all_h_ok = sample_ok and colors_ok and zones_ok and markers_ok and note_ok and metrics_consistent
        runner.record(
            f"FE-2 Clinical {h} min matches /v1/backtest and /v1/metrics",
            all_h_ok,
            f"sample={c_info['sampleLen']}, colors_ok={colors_ok}, zones_ok={zones_ok}, markers={len(markers)}, note_ok={note_ok}, metrics_match={metrics_consistent}",
        )

    # Rapid horizon switching stale-response guard on Clinical
    page.evaluate("() => { backtestCache.clear(); }")

    def delay_30_backtest(route: Any) -> None:
        post_data = route.request.post_data or ""
        if 'name="horizon"\r\n\r\n30' in post_data or "horizon=30" in post_data:
            time.sleep(0.6)
        route.continue_()

    page.route("**/v1/backtest", delay_30_backtest)
    page.evaluate(
        """() => {
            const sel = document.getElementById('selH');
            sel.value = '30';
            sel.onchange({ target: sel });
            sel.value = '120';
            sel.onchange({ target: sel });
        }"""
    )
    page.wait_for_function(
        "() => clinicalState.phase === 'ready' && clinicalState.horizon === 120",
        timeout=30000,
    )
    page.wait_for_timeout(900)
    page.unroute("**/v1/backtest")
    range_note = page.locator("#rangeNote").inner_text()
    bt_120_ab = float(m["horizons"]["120"]["overall"]["a_plus_b_pct"])
    rapid_ok = (
        page.evaluate("() => clinicalState.horizon") == 120
        and "120 min" in range_note
        and f"{bt_120_ab:.2f}%" in range_note
    )
    runner.record(
        "FE-2 Clinical rapid horizon switching ignores stale 30 min response",
        rapid_ok,
        f"horizon={page.evaluate('() => clinicalState.horizon')}",
    )


def run_fe3_checks(runner: Runner, page: Any) -> None:
    models_resp = api_get("/v1/models")
    page.click("button[data-view='model']")

    # 1. Feature groups (18 features, prefix counts summing to feature_columns length)
    cols = list(models_resp["feature_columns"])
    prefix_defs = [
        ("lag_", "Glucose lags"),
        ("roll_mean_", "Rolling mean"),
        ("roll_std_", "Rolling standard deviation"),
        ("delta_", "Rate of change"),
        ("hour", "Time-of-day encoding"),
    ]
    exp_groups = []
    for pfx, title in prefix_defs:
        items = [c for c in cols if c.startswith(pfx)]
        if items:
            exp_groups.append((title, items))

    cards = page.evaluate(
        """() => Array.from(document.querySelectorAll('#modelCards .stat')).map(el => ({
            small: el.querySelector('small')?.textContent?.trim() || '',
            val: el.querySelector('b')?.textContent?.trim() || '',
            span: el.querySelector('span')?.textContent?.trim() || ''
        }))"""
    )
    total_ok = (
        len(cards) == 1 + len(exp_groups)
        and cards[0]["small"] == "Engineered features"
        and cards[0]["val"] == f"{len(cols)} total"
    )
    sum_counts = 0
    groups_ok = total_ok
    if groups_ok:
        for (title, items), card in zip(exp_groups, cards[1:]):
            sum_counts += len(items)
            if (
                card["small"] != title
                or card["val"] != f"{len(items)} features"
                or card["span"] != ", ".join(items)
            ):
                groups_ok = False
                break
    groups_ok = groups_ok and sum_counts == len(cols)
    runner.record(
        "FE-3 Model view feature groups match /v1/models",
        groups_ok,
        f"total={len(cols)}, group_sum={sum_counts}, cards={len(cards)}",
    )

    # 2. Four model rows with hash prefixes equal to /v1/models
    file_rows = page.evaluate(
        """() => Array.from(document.querySelectorAll('#modelFilesTable tr')).slice(1).map(tr =>
            Array.from(tr.querySelectorAll('td')).map(td => ({
                text: td.textContent.trim(),
                title: td.getAttribute('title') || ''
            }))
        )"""
    )
    api_models = models_resp["models"]
    files_ok = len(file_rows) == 4 and len(api_models) == 4
    if files_ok:
        for m_info, row in zip(api_models, file_rows):
            loaded_str = "yes" if m_info["loaded"] else "no"
            exp_h = f"{m_info['horizon']} min"
            if m_info["loaded"]:
                full_sha = m_info["sha256"] or ""
                exp_sha = full_sha[:12]
                exp_size = format_size(int(m_info["size_bytes"]))
                if (
                    row[0]["text"] != exp_h
                    or row[1]["text"] != loaded_str
                    or row[2]["text"] != exp_sha
                    or row[2]["title"] != full_sha
                    or row[3]["text"] != exp_size
                ):
                    files_ok = False
                    break
            else:
                if row[0]["text"] != exp_h or row[1]["text"] != loaded_str or row[2]["text"] != "not loaded":
                    files_ok = False
                    break
    runner.record(
        "FE-3 Model view 4 model rows match /v1/models hashes and sizes",
        files_ok,
        f"rows={len(file_rows)}",
    )

    # 3. Metrics status line equals metrics_status
    status_el = page.locator("#modelMetricsStatus")
    status_text = status_el.inner_text().strip()
    status_cls = status_el.get_attribute("class") or ""
    exp_status = models_resp["metrics_status"]
    warns = models_resp["metrics_warnings"]
    status_ok = f"Metrics status: {exp_status}" in status_text
    if exp_status != "ok":
        status_ok = status_ok and ("warn" in status_cls.split())
    for w in warns:
        if w not in status_text:
            status_ok = False
    runner.record(
        "FE-3 Model view metrics_status line matches /v1/models",
        status_ok,
        f"status={exp_status}",
    )

    # 4. Feature importance table matches /v1/models.feature_importance at 1 decimal
    fi = models_resp["feature_importance"]
    imp_card_text = page.locator("#modelImpCard").inner_text()
    imp_type = fi.get("importance_type", "gain")
    feat_names = sorted(
        {k for h in HORIZONS for k in fi[str(h)].keys()}
    )
    ranked = sorted(
        feat_names,
        key=lambda n: (
            -sum(float(fi[str(h)].get(n, 0.0)) for h in HORIZONS) / len(HORIZONS),
            n,
        ),
    )
    top10 = ranked[:10]
    rest = ranked[10:]
    imp_rows = page.evaluate(
        """() => Array.from(document.querySelectorAll('#modelImpTable tr')).slice(1).map(tr =>
            Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim())
        )"""
    )
    exp_imp_rows = [
        [name] + [f"{float(fi[str(h)].get(name, 0.0)) * 100:.1f}%" for h in HORIZONS]
        for name in top10
    ]
    if rest:
        exp_imp_rows.append(
            [f"Other {len(rest)} features"]
            + [
                f"{sum(float(fi[str(h)].get(n, 0.0)) for n in rest) * 100:.1f}%"
                for h in HORIZONS
            ]
        )
    fi_ok = (imp_type in imp_card_text) and (imp_rows == exp_imp_rows)
    runner.record(
        "FE-3 Model view feature importance table matches /v1/models",
        fi_ok,
        f"rows={len(imp_rows)}",
    )


def main() -> int:
    runner = Runner()
    console_errors: list[str] = []
    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()

        page.on(
            "console",
            lambda msg: console_errors.append(msg.text)
            if msg.type == "error"
            else None,
        )
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        page.goto(FRONTEND_URL, wait_until="domcontentloaded")
        wait_for_ready(page)

        run_fe1_checks(runner, page)
        run_fe2_checks(runner, page)
        run_fe3_checks(runner, page)

        browser.close()

    runner.record(
        "Zero console/page errors during live run",
        len(console_errors) == 0 and len(page_errors) == 0,
        f"console_errors={len(console_errors)}, page_errors={len(page_errors)}",
    )

    print(f"\nSummary: {runner.passed} passed, {runner.failed} failed", flush=True)
    return 0 if runner.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
