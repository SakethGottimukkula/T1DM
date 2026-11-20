# Groq-Powered T1D Forecast & Wellness Agent (`openai/gpt-oss-120b`)

Autonomous clinical-research forecasting and wellness assistant for Type 1 Diabetes (T1D). It allows users to ask natural-language questions about upcoming glucose projections, hypoglycemia/hyperglycemia risk, model accuracy, predictive factors, balanced meal plans, safe workout routines, and immediate step-by-step triage for out-of-range excursions.

The agent operates with strict grounding: **it never invents numbers**. Every figure, rate of change, RMSE, and predicted glucose value is produced by executing calibrated machine-learning models and tools in real time.

---

## Core Capabilities

1. **Model Forecasts & Grounded Explanations**: 30, 60, 90, and 120-minute predictions via calibrated XGBoost models or direct pipeline integration, with feature-importance breakdowns.
2. **Actual Pipeline Forecast Integration**: The agent dynamically reasons over actual model outputs from the dashboard (Ohio test cases and uploaded CGM CSVs): current blood glucose, rate of change, trend arrows, predicted horizons, relative changes, held-out actuals, Clarke zones, and patient-specific RMSE.
3. **No-Forecast-Available Grounding**: If no forecast has been run yet for the active patient/case, the agent explicitly states that no patient-specific forecast is available and prompts the user to click "Run Forecast" or upload CGM data, eliminating hallucination.
4. **Meal & Nutritional Strategy**: In-depth carbohydrate pacing, low-GI balancing, protein/fiber pairing, and sample meal/snack ideas to prevent postprandial spikes.
5. **Workout & Activity Routines**: Safe aerobic vs resistance pacing, pre-exercise glucose checks, mid-workout precautions, and delayed nocturnal hypoglycemia awareness.
6. **Immediate Hypo Triage (<70 mg/dL or Severe <54 mg/dL)**: Step-by-step first-aid protocol following the **"Rule of 15"** (15g fast simple carbs, 15-minute wait, capillary re-check) and emergency glucagon/care instructions for severe lows.
7. **Immediate Hyper Triage (>180 mg/dL or Severe >250 mg/dL)**: Aggressive water hydration (16-24 oz), ketone screening protocols for DKA prevention, exercise safety warnings, and physician care-plan guidance.
8. **Sentence-Level Guardrail Sanitization**: Actionable insulin dosing directives (e.g., *"take 5 units of insulin"*) are stripped, while the model's own refusal sentences and surrounding lifestyle/meal advice are preserved.
9. **Clinical Report Markdown Presentation**: Enhanced Markdown rendering eliminates raw markdown artifacts, displaying styled clinical tables (`.report-table`), responsive metric cards (`.report-metric-card`), section hierarchy, dividers, and callouts (`.report-callout`).
10. **Modern Chat UX & Sticky Disclaimer**: Compact, auto-expanding composer (up to 120px max-height; Enter submits, Shift+Enter adds newline); independent conversation thread scrolling; prominent initial welcome disclaimer smoothly transitioning to a pinned sticky banner during active dialogue; and upload-page research disclaimer.

---

## Architecture

Located in `backend/agent/`:

```text
backend/agent/
  config.py       - Loads GROQ_API_KEY, GROQ_MODEL, GROQ_REASONING_EFFORT, MAX_COMPLETION_TOKENS from root .env
  llm.py          - Official Groq SDK wrapper (openai/gpt-oss-120b) with include_reasoning=False & <think> tag stripping
  tools.py        - 6 Pydantic-validated tools backed by OhioT1DM XGBoost models & performance benchmarks
  guardrails.py   - Narrow dosing patterns, sentence-level sanitize_answer(), emergency advisory for excursions
  agent.py        - Autonomous agent loop with short-term session memory (last 10 turns) & max 5 iterations
  router.py       - FastAPI router mounted at /agent (/chat, /chat/stream, /reset, /health)
```

### Available Tools
1. `get_forecast(horizon_min)`: Runs matching XGBoost model for 30, 60, 90, or 120-min horizons.
2. `get_all_forecasts()`: Runs all 4 horizons simultaneously.
3. `get_recent_glucose(minutes)`: Summarizes recent readings, 15-min rate of change (ROC), min, max, mean, and time in range.
4. `assess_risk()`: Categorizes hypoglycemia (<70, <54 mg/dL) and hyperglycemia (>180, >250 mg/dL) excursion risk across horizons.
5. `get_model_performance(horizon_min)`: Retrieves empirical RMSE and Clarke Error Grid zone A+B percentages from stored test cohorts.
6. `explain_forecast(horizon_min)`: Computes top XGBoost feature gains and maps them to the patient's current glucose lags and trends.

---

## Setup & Configuration

1. Obtain an API key from [Groq Console](https://console.groq.com).
2. Set your environment variables in `.env` at the project root (see `.env.example`):

```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_REASONING_EFFORT=medium
MAX_COMPLETION_TOKENS=4096
```

- `GROQ_MODEL`: Defaults to `openai/gpt-oss-120b`.
- `GROQ_REASONING_EFFORT`: `low` for direct lookups, `medium` default, or `high` for causal/risk/lifestyle inquiries.
- `MAX_COMPLETION_TOKENS`: Defaults to `4096` to provide ample room for comprehensive structured plans.
- The official `groq` SDK automatically routes requests to Groq Cloud; no `base_url` configuration is needed.
- *.env is ignored by Git to keep API keys secure.*

---

## Running the Backend

From the repository root:

```powershell
uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

Interactive OpenAPI docs: `http://127.0.0.1:8000/docs`.

---

## Endpoint Examples with `curl`

### 1. Check Agent Health & Model Configuration
```powershell
curl.exe http://127.0.0.1:8000/agent/health
```

### 2. Ask a Forecast or Lifestyle Question (`POST /agent/chat`)
```powershell
curl.exe -X POST http://127.0.0.1:8000/agent/chat `
  -H "Content-Type: application/json" `
  -d '{
    "session_id": "session_123",
    "message": "Suggest meals and workouts to keep my glucose stable"
  }'
```

### 3. Reset Session History (`POST /agent/reset`)
```powershell
curl.exe -X POST http://127.0.0.1:8000/agent/reset `
  -H "Content-Type: application/json" `
  -d '{"session_id": "session_123"}'
```

### 4. Stream Response Tokens via SSE (`GET /agent/chat/stream`)
```powershell
curl.exe "http://127.0.0.1:8000/agent/chat/stream?session_id=session_123&message=Why+this+prediction"
```

---

## Safety & Clinical Boundaries

- **Investigational Prototype**: This software is a clinical research prototype, not an approved medical device.
- **CGM-Only Context**: Forecast models predict trajectory from CGM history alone without insulin or meal inputs; unrecorded insulin or carbohydrates will alter real-world trajectory.
- **Zero Hallucination**: Every numeric prediction, trend rate, or metric is sourced directly from tool calls.
- **Strict Clinical Boundaries**: The agent does not calculate or recommend insulin doses, medication adjustments, carb ratios, or correction factors. It also does not create personalized hypo/hyper treatment plans or provide step-by-step regimens or carbohydrate amounts. Immediate low/high concerns receive brief, non-prescriptive guidance to follow the user's established care plan and seek emergency help for severe symptoms.

## Patient dataset access boundary (2026-11-19)

Patient-specific answers must use only a successfully uploaded patient dataset. Do not use a default/test patient when no upload exists; state that patient context cannot be determined. The uploaded dataset query capability is for patient measurements and clinical readings, including timestamped lookups when available. Do not expose code, model internals, project files, or unrelated system information through agent tools or answers. Uploaded context is sanitized and kept in temporary backend memory; it is not used for model training.

## Forecast grounding and response behavior (2026-11-19)

- The agent can retrieve only forecast horizons already present in the active upload context. It must not run a model, estimate a missing horizon, or substitute another horizon. Re-uploading the same readings with additional horizons accumulates those values; a different dataset or model family starts a new prediction set.
- For forecast timing, tools provide the forecast origin and `forecast_for_timestamp` (origin plus horizon). The agent uses the latter for “when” questions, preserves the dataset timezone, and distinguishes the predicted-for time from the most recent measured reading.
- Exact measurement, timestamp, and range questions should use the patient-data query tool. If a value, field, or horizon is unavailable, the agent says so instead of guessing. Short follow-ups should be interpreted using recent conversation context.
- Agent replies have a two-second minimum display delay in the frontend. User prompts can be copied from chat; animated typing dots appear while a response is running.
