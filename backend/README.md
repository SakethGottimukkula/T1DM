# Glucose Horizon Backend

## Patient dataset access and context (2026-11-19)

The Forecast Agent uses only the successfully uploaded patient dataset as its patient-specific source. It does not fall back to a built-in/default test patient. Before an upload, the UI states that patient context cannot be determined and there is no active uploaded-patient context.

The upload path stores a sanitized, in-memory dataset under an opaque identifier for the agent's dataset query tool. This supports timestamped questions about available patient measurements (such as glucose or heart rate) while excluding code, model internals, project files, and unrelated system information from agent tools/context. Uploads are not used to train or fine-tune models. This context is temporary and does not persist across backend restarts.

Forecast answers are retrieval-only: the agent can return prediction horizons included in the active upload context, but cannot run a model or invent missing horizons. Repeated uploads with the same patient readings and model family accumulate selected horizon predictions in the frontend context (for example, 30 min followed by 60 min). Selecting a different dataset starts a fresh context. Forecast tools return both the forecast origin and the forecast-for timestamp (origin plus horizon), and instructions require the agent to use the latter when asked when a prediction is expected.

Hypoglycemia action-plan requests are handled with a consistent, non-prescriptive response. The agent does not produce a personalized plan, carbohydrate amount, or step-by-step treatment regimen; it directs users to their established care plan and urgent assistance for severe symptoms.

## Frontend updates (2026-11-19)

The dashboard uses a light warm-white, navy, blue-teal, marigold, and sky visual system with responsive layouts. The sidebar has animated active/hover states, the cursor particles glow blue, and reduced-motion preferences are supported. Test Explorer aligns Ohio and SimGlucose layouts and typography; their forecast loading animations match. CSV horizon selection, EGA horizon selection, performance/overview typography, and upload mismatch banners are documented in the root README.

FastAPI service for the project's two separate blood-glucose forecasting paths:

- OhioT1DM CGM-only models, which remain the existing default models and backtest path.
- SimGlucose CGM + insulin models, which accept optional CHO and use their own model files and feature pipeline.

An uploaded CSV is used for prediction only. Uploads do not retrain or fine-tune either model family.

## Install and run

From this `backend` directory:

```powershell
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive API documentation: `http://127.0.0.1:8000/docs`.

### Agent Configuration (Groq & openai/gpt-oss-120b)

The backend features an autonomous forecasting and wellness agent located in `backend/agent/`. Set these in your `.env` file or environment:

```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
GROQ_REASONING_EFFORT=medium
MAX_COMPLETION_TOKENS=4096
```

- `GROQ_MODEL`: default `openai/gpt-oss-120b`.
- `GROQ_REASONING_EFFORT`: `low` (direct lookups), `medium` (default), or `high` (explanatory, risk, and lifestyle questions).
- `MAX_COMPLETION_TOKENS`: default `4096` to provide ample room for comprehensive structured meal and exercise plans.
- The agent loop suppresses internal model reasoning from logs and responses, strictly executes real XGBoost tools for all numerical values, and enforces sentence-level clinical safety guardrails (stripping actionable dosing directives while preserving safe educational and lifestyle advice).
- The `/agent/chat` endpoint accepts an optional `forecast_context` payload from the dashboard containing real pipeline predictions (current glucose, rate of change, 30/60/90/120 min forecasts, actuals, and patient-specific RMSE). When no forecast has been run, the agent explicitly informs the user that no active forecast is available, completely preventing hallucinations. Full documentation is in [AGENT_README.md](../AGENT_README.md).

## Model training

### OhioT1DM CGM-only models

Ohio training data is not distributed here and requires authorized access to the OhioT1DM dataset.

```powershell
python scripts/train_models.py --train path/to/ohio_training_combined.csv --test path/to/ohio_testing_combined.csv
```

This updates `models/model_30.json`, `model_60.json`, `model_90.json`, `model_120.json`, and `models/feature_columns.json`.

### SimGlucose CGM + insulin models

```powershell
python scripts/train_simglucose_models.py
```

The default simulation is 30 virtual patients (10 each in adult, adolescent, and child groups) for 35 days at five-minute sampling, about 10,081 raw rows per patient. Set a different duration with `--days N`. The generated CSV is written to `data/simglucose_training.csv`; the four horizon models and feature list are written to `models/simglucose/`.

The script splits by virtual-patient ID to measure generalization: 24 patients are used for a held-out evaluation on six patients. It saves these evaluation weights under `models/simglucose/evaluation/` for Test Explorer. After scoring them, it refits the deployment weights on **all 30 patients** for upload predictions. Input features use past CGM, insulin, and CHO. `BG` is the target; simulator risk fields are excluded. Held-out results measure performance on simulated virtual patients and do not establish real-patient accuracy.

If the generated CSV already exists and you only need to rebuild the models, skip simulation with:

```powershell
python scripts/train_simglucose_models.py --reuse-existing
```

**Current artifacts:** `data/simglucose_training.csv` currently has 302,430 rows: 30 patients with 10,081 rows each, from a 35-day simulation. The saved `models/simglucose/` deployment weights have now been refit using all 30 patients. Restart the API after regenerating weights because models are loaded at backend startup.

## CSV prediction formats

### Ohio CGM-only

`POST /v1/predict/csv` accepts a CSV with timestamp and glucose history (the parser supports common `Time`, `CGM`, or `BG` aliases). At least 25 valid, sequential readings at roughly five-minute intervals are expected by the upload interface. It uses the Ohio CGM-only models.

### SimGlucose multimodal

`POST /v1/predict/simglucose/csv` requires timestamp/`Time`, `CGM`, and `insulin`. `CHO` is optional and defaults to zero. `BG` is optional; when absent, CGM is used as the BG reference. The backend resamples records to five-minute intervals and requires at least 25 valid readings. If SimGlucose model files are unavailable, the endpoint returns HTTP 503.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health and Ohio model inventory |
| `GET` | `/v1/models` | Ohio horizons, feature columns, and loaded models |
| `GET` | `/v1/models/simglucose` | SimGlucose model load status and feature columns |
| `GET` | `/v1/simglucose/test/patients` | List six held-out virtual patients available in Test Explorer |
| `GET` | `/v1/simglucose/test/patient/{patient_id}?horizon=30&case_index=2` | Evaluate a selected case for one held-out patient with models trained on the other 24 patients |
| `POST` | `/v1/predict` | Ohio prediction from JSON CGM readings |
| `POST` | `/v1/predict/csv` | Ohio prediction from a CGM CSV |
| `POST` | `/v1/predict/simglucose/csv` | SimGlucose prediction from a multimodal CSV |
| `POST` | `/v1/wellness` | Groq-powered general wellness suggestion agent (meal/workout ideas) |
| `GET` | `/agent/health` | Check Groq forecast agent health, model ID, and configuration |
| `POST` | `/agent/chat` | Groq-powered T1D forecast agent with autonomous tool execution |
| `POST` | `/agent/reset` | Reset conversational session memory for a given session ID |
| `GET/POST` | `/agent/chat/stream` | SSE streaming of tool events and synthesized answer tokens |
| `POST` | `/v1/backtest` | Ohio backtest using an uploaded CSV and chosen horizon |
| `GET` | `/v1/backtest/overall` | Overall Ohio held-out metrics at all horizons |
| `GET` | `/v1/backtest/patient/{patient_id}` | Ohio held-out metrics for one patient at all horizons |

The backtest endpoints are for OhioT1DM and use the Ohio models and feature generation. They do not evaluate SimGlucose models. `/v1/backtest` expects multipart form fields `file` and `horizon`; the CSV must contain `timestamp`, `glucose`, and `patient_id`. Its timestamps should use `%d-%m-%Y %H:%M:%S`. Supported horizons are 30, 60, 90, and 120 minutes.

The SimGlucose Test Explorer lists the six held-out IDs from `models/simglucose/evaluation/patient_ids.json`. It offers five case windows per patient (early through latest) and all four horizons. The selected case returns actual and predicted BG, single-case absolute error, and Clarke EGA zone; patient-wide RMSE/MAE are calculated over that patient's valid held-out forecast rows. CGM context cards include recent CGM trend, mean, variability, time in range, and 60-minute insulin/CHO totals. The upload endpoint continues to use the deployment weights trained on all 30 virtual patients.

Example uploaded-file backtest:

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/backtest `
  -F "file=@path/to/ohio_testing_combined.csv" `
  -F "horizon=30"
```

Overall and patient-level metrics are computed from the configured Ohio held-out test CSV. Patient IDs must match IDs present in that test file. The patient-specific route is:

```text
GET /v1/backtest/patient/559
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `backend/models` | Ohio `.json` model directory |
| `OHIO_TEST_CSV` | `ohio_testing_combined.csv` at repository root | Ohio held-out CSV used by the overall and patient-level metric endpoints |
| `CORS_ORIGINS` | `http://127.0.0.1:5500,http://localhost:5500` | Comma-separated allowed frontend origins |
| `TRAIN_CSV` | unset | Optional Ohio training CSV path used by `scripts/train_models.py` when `--train` is not supplied |
| `GROQ_API_KEY` | unset | Groq Cloud API key for the forecasting agent and wellness assistant |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq chat completion model ID (e.g. `openai/gpt-oss-120b`, `llama-3.3-70b-versatile`) |
| `GROQ_REASONING_EFFORT` | `medium` | Groq reasoning effort: `low`, `medium`, or `high` |
| `MAX_COMPLETION_TOKENS` | `4096` | Maximum completion token budget for detailed responses |

## Wellness Suggestion Agent (Groq)

The wellness suggestion agent provides general meal ideas, meal plans, and physical activity or workout routines for individuals managing Type 1 Diabetes.

> **Disclaimer:** Suggestions are general wellness ideas only and do not constitute individualized medical care, diagnosis, prescription, or insulin dosing recommendations. Every response prepends a mandatory clinical disclaimer.

### Key Configuration and Security

In compliance with Groq's security recommendations:
- **Never commit your API key to source control.** Local secret files (`.env`, `*.env`) are gitignored.
- Set `GROQ_API_KEY` in your environment or in a local `.env` file at the repository root.
- The backend communicates directly with Groq via the official SDK; the frontend never receives or handles the API key.

#### Setting Environment Variables

On Windows (PowerShell):
```powershell
$env:GROQ_API_KEY = "gsk_your_actual_key_here"
$env:GROQ_MODEL = "openai/gpt-oss-120b"  # Optional, defaults to openai/gpt-oss-120b
$env:GROQ_REASONING_EFFORT = "medium"
```

On Linux/macOS:
```bash
export GROQ_API_KEY="gsk_your_actual_key_here"
export GROQ_MODEL="openai/gpt-oss-120b"
export GROQ_REASONING_EFFORT="medium"
```

Or create a `.env` file in the project root:
```ini
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL=openai/gpt-oss-120b
GROQ_REASONING_EFFORT=medium
MAX_COMPLETION_TOKENS=4096
```

### Trying the Feature

#### Via CLI:
```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/wellness `
  -H "Content-Type: application/json" `
  -d '{"message": "Suggest 3 balanced low-glycemic lunch ideas."}'
```

#### Via Frontend:
1. Open the dashboard in your browser (e.g., via VS Code Live Server at `http://127.0.0.1:5500/Frontend/index.html`).
2. Click **Wellness Agent** in the sidebar navigation.
3. Type a meal or exercise inquiry (or click one of the quick suggestions) and click **Get Suggestions**.
4. If `GROQ_API_KEY` is not set, the UI clearly displays a configuration notice with instructions.

The Ohio feature pipeline uses 18 CGM-derived columns: `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`; rolling mean and standard deviation for 30, 60, 90, and 120 minutes; `delta_5`, `delta_30`; and `hour`, `hour_sin`, `hour_cos`. SimGlucose has a separate feature specification stored at `models/simglucose/feature_columns.json`.

## Tests

Run the backend tests from this directory with:

```powershell
pytest tests/ -v
```
