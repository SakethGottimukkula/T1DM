# Glucose Horizon — Blood Glucose Forecasting (CGM-Only)

**Glucose Horizon** is a multi-horizon blood glucose forecasting and clinical safety evaluation system for Type 1 Diabetes (T1D). It uses continuous glucose monitoring (CGM) sensor streams alone—without requiring manual insulin dosing or meal logs—to predict glucose trajectories 30, 60, 90, and 120 minutes into the future.

The system combines:
1. **Machine Learning Core**: 4 XGBoost regressors trained on OhioT1DM CGM data.
2. **Production Backend**: A FastAPI REST microservice supporting real-time inference, file-based forecasting, and batch backtesting with Clarke Error Grid Analysis (EGA).
3. **Interactive Frontend**: A research dashboard for held-out test case visualization, patient switching, risk alerts, Clarke error grid analysis, benchmark performance metrics, and model metadata inspection.

---

## Problem Statement & Clinical Motivation

Type 1 diabetes management requires continuous vigilance to prevent two life-threatening acute extremes:
- **Hypoglycemia (< 70 mg/dL)**: Can lead to loss of consciousness, seizures, or diabetic coma within minutes.
- **Hyperglycemia (> 180 mg/dL)**: Leads over time to diabetic ketoacidosis (DKA) and systemic microvascular/macrovascular damage.

Existing multi-modal models often depend on patient self-reported carb intake and bolus insulin data. In real-world conditions, meal logging is frequently skipped, delayed, or inaccurate. This project assesses **how accurate, reliable, and clinically safe multi-horizon forecasts can be when relying strictly on CGM streams**.

---

## Approach & Features

### Feature Engineering (18 features derived exclusively from CGM)
Features are computed strictly backwards from each reading with zero forward lookahead:
- **Lag Readings**: `lag_1` (t-5m), `lag_2` (t-10m), `lag_3` (t-15m), `lag_6` (t-30m), `lag_12` (t-60m)
- **Rolling Statistics**: Rolling mean and standard deviation over 30, 60, 90, and 120-minute windows (`roll_mean_30`, `roll_std_30`, `roll_mean_60`, `roll_std_60`, `roll_mean_90`, `roll_std_90`, `roll_mean_120`, `roll_std_120`)
- **Rate of Change (Deltas)**: 5-minute velocity (`delta_5 = glucose - lag_1`) and 30-minute velocity (`delta_30 = glucose - lag_6`)
- **Circadian Encoding**: Cyclical time-of-day features (`hour`, `hour_sin`, `hour_cos`)

### Clinical Safety Metric: Clarke Error Grid Analysis (EGA)
Every forecast is evaluated not only on statistical metrics (RMSE, MAE) but also against the clinical standard:
- **Zone A**: Clinically accurate (leads to correct treatment decisions)
- **Zone B**: Benign errors (leads to no treatment or safe treatment)
- **Clinically Safe**: Zone A + Zone B (target: > 95% for safe automated alert systems)
- **Zones C, D, E**: Overcorrection, failure to detect hypo/hyperglycemia, or erroneous treatment

---

## Benchmark Results (OhioT1DM Test Cohort)

Tested across all patients in the OhioT1DM test set (the number of evaluated rows is given in the first table):

<!-- METRICS:START -->
### T1: XGBoost performance (test set, 15,688 rows)

| Horizon | RMSE (mg/dL) | MAE (mg/dL) | Clarke A (%) | Clarke B (%) | Clarke C (%) | Clarke D (%) | Clarke E (%) | A+B (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 30 min | 20.94 | 14.57 | 88.57 | 9.72 | 0.04 | 1.67 | 0.00 | 98.29 |
| 60 min | 33.67 | 24.90 | 72.37 | 24.35 | 0.22 | 3.06 | 0.00 | 96.72 |
| 90 min | 42.44 | 32.10 | 61.04 | 34.10 | 0.44 | 4.39 | 0.04 | 95.14 |
| 120 min | 48.08 | 37.04 | 54.21 | 39.66 | 0.48 | 5.53 | 0.12 | 93.87 |

### T2: Baseline comparison

| Horizon | Model | RMSE (mg/dL) | MAE (mg/dL) | A+B (%) |
|:---:|:---|:---:|:---:|:---:|
| 30 min | XGBoost | 20.94 | 14.57 | 98.29 |
| 30 min | Persistence | 23.15 | 16.62 | 99.04 |
| 30 min | Ridge | 20.99 | 14.85 | 98.44 |
| 30 min | Lasso | 21.27 | 15.24 | 98.27 |
| 60 min | XGBoost | 33.67 | 24.90 | 96.72 |
| 60 min | Persistence | 37.08 | 27.62 | 97.23 |
| 60 min | Ridge | 33.75 | 25.45 | 96.72 |
| 60 min | Lasso | 34.06 | 25.83 | 96.58 |
| 90 min | XGBoost | 42.44 | 32.10 | 95.14 |
| 90 min | Persistence | 47.32 | 35.72 | 95.28 |
| 90 min | Ridge | 42.51 | 32.90 | 95.51 |
| 90 min | Lasso | 42.87 | 33.30 | 95.41 |
| 120 min | XGBoost | 48.08 | 37.04 | 93.87 |
| 120 min | Persistence | 55.22 | 42.16 | 93.38 |
| 120 min | Ridge | 48.52 | 38.13 | 94.30 |
| 120 min | Lasso | 48.90 | 38.51 | 94.14 |

### T3: Per-patient RMSE and A+B (%)

| Patient | 30 min | 60 min | 90 min | 120 min |
|:---:| :---: | :---: | :---: | :---: |
| 559 | 21.88 / 98.50 | 34.23 / 96.76 | 43.30 / 94.65 | 49.37 / 93.47 |
| 563 | 18.79 / 99.29 | 30.66 / 98.49 | 38.86 / 96.91 | 44.23 / 95.60 |
| 570 | 18.80 / 99.67 | 31.56 / 99.41 | 43.16 / 97.59 | 51.32 / 95.63 |
| 575 | 24.13 / 96.19 | 38.21 / 93.16 | 46.00 / 90.88 | 50.76 / 90.41 |
| 588 | 19.38 / 99.89 | 32.03 / 98.62 | 40.51 / 97.34 | 46.39 / 95.70 |
| 591 | 22.22 / 96.17 | 34.91 / 93.81 | 42.54 / 93.25 | 46.06 / 92.26 |

### T4: Feature importance (gain), top 5 per horizon

| Rank | 30 min | 60 min | 90 min | 120 min |
|:---:| :--- | :--- | :--- | :--- |
| 1 | `lag_1` (0.80) | `lag_1` (0.60) | `lag_1` (0.45) | `lag_1` (0.29) |
| 2 | `delta_5` (0.08) | `delta_5` (0.12) | `delta_5` (0.13) | `delta_5` (0.13) |
| 3 | `delta_30` (0.02) | `delta_30` (0.05) | `delta_30` (0.07) | `delta_30` (0.08) |
| 4 | `roll_mean_30` (0.01) | `hour` (0.02) | `hour` (0.04) | `hour` (0.06) |
| 5 | `hour` (0.01) | `hour_cos` (0.02) | `hour_cos` (0.03) | `roll_mean_30` (0.05) |
<!-- METRICS:END -->

While XGBoost achieves the lowest RMSE and MAE across all four forecasting horizons, its performance margin over linear Ridge regression is narrow (RMSE difference of 0.05 mg/dL at 30 min, 0.08 mg/dL at 60 min, 0.07 mg/dL at 90 min, and 0.44 mg/dL at 120 min). On clinical safety (Clarke Zone A+B percentage), baseline models match or exceed XGBoost at several horizons: persistence yields higher A+B% at 30 min (99.04% vs 98.29%), 60 min (97.23% vs 96.72%), and 90 min (95.28% vs 95.14%), while Ridge matches or exceeds XGBoost at all four horizons (98.44% at 30 min, 96.72% at 60 min, 95.51% at 90 min, and 94.30% at 120 min), and Lasso exceeds XGBoost at 90 min (95.41%) and 120 min (94.14%).

---

## Repository Structure

```
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── casedata.py          # Case extraction & deterministic segment sampling
│   │   ├── config.py            # Environment configuration & validation thresholds
│   │   ├── ega.py               # Clarke Error Grid classification logic
│   │   ├── features.py          # 18-column feature pipeline (grouped & single-patient)
│   │   ├── main.py              # FastAPI application, lifespan, routes & CORS
│   │   ├── metrics.py           # In-memory backtest evaluation & metrics calculation
│   │   ├── metricsfile.py       # metrics.json loader & schema validator
│   │   ├── predictor.py         # XGBoost model loader & inference manager
│   │   ├── schemas.py           # Pydantic v2 request/response schemas
│   │   └── timeutil.py          # Timestamp parsing and cadence utilities
│   ├── models/                  # Exported metrics & model directory (.gitkeep)
│   │   ├── .gitkeep
│   │   └── metrics.json         # Canonical test-set evaluation metrics (tracked)
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── evaluate.py          # Independent evaluation against CSV datasets
│   │   ├── export_openapi.py    # OpenAPI JSON schema exporter and check tool
│   │   ├── render_readme_tables.py # Renders Markdown metrics tables into README
│   │   └── train_models.py      # CLI script for multi-horizon training & eval
│   ├── tests/                   # Pytest test suite
│   │   ├── __init__.py
│   │   ├── test_api.py          # Endpoint tests, validation & 422 error cases
│   │   ├── test_backtest.py     # Backtest endpoint tests & fileless backtest
│   │   ├── test_cases.py        # Case extraction & midpoint selection tests
│   │   ├── test_context.py      # Context & state initialization tests
│   │   ├── test_ega.py          # Hand-checked Clarke EGA boundary tests
│   │   ├── test_features.py     # Feature correctness & leakage tests
│   │   ├── test_metrics.py      # Metrics calculation tests
│   │   ├── test_metrics_file.py # Metrics file loading & verification tests
│   │   ├── test_metricsfile.py  # Metricsfile schema validation tests
│   │   ├── test_openapi_contract.py # OpenAPI JSON contract drift test
│   │   ├── test_policy.py       # Clinical policy & safety rule tests
│   │   ├── test_render_tables.py# Markdown table rendering tests
│   │   └── test_timeutil.py     # Timestamp and cadence utility tests
│   ├── Dockerfile               # Container definition (python:3.11-slim + libgomp1)
│   ├── docker-compose.yml       # Production Compose configuration with models mount
│   ├── openapi.json             # Committed OpenAPI schema contract
│   ├── pytest.ini               # Pytest configuration
│   ├── requirements-dev.txt     # Development and testing dependencies
│   ├── requirements.txt         # Pinned backend dependencies
│   └── README.md                # Detailed backend documentation & curl examples
├── Frontend/
│   ├── index.html               # Clinical dashboard markup & multi-view panels
│   ├── style.css                # Dashboard styling, typography & responsive layouts
│   └── script.js                # Chart.js visualization, API client & backend-driven views
├── Model/
│   └── XGBoost.ipynb            # Research notebook: feature extraction, training & plots
├── requirements.txt             # Pinned root dependencies
├── .gitignore                   # Excludes CSV datasets, checkpoints & artifacts
└── README.md                    # Project README
```

---

## Quick Start — Running the Full System

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
```

### 2. Model Training (Optional if pre-trained JSON models exist)
If retraining from OhioT1DM CSVs:
```bash
python backend/scripts/train_models.py \
  --train path/to/ohio_training_combined.csv \
  --test path/to/ohio_testing_combined.csv
```
This saves `model_30.json`, `model_60.json`, `model_90.json`, `model_120.json`, and `feature_columns.json` into `backend/models/`.

### 3. Launch the Backend API
Optionally configure `TEST_CSV_PATH` to enable patient case exploration (`/v1/cases`) and fileless backtesting:
```bash
# Windows PowerShell
$env:TEST_CSV_PATH = "C:\path\to\ohio_testing_combined.csv"

# Linux / macOS
export TEST_CSV_PATH="/path/to/ohio_testing_combined.csv"
```

Start the FastAPI server via Uvicorn:
```bash
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI Specification: `http://127.0.0.1:8000/openapi.json`

> **Privacy Note**: Patient CGM streams contain sensitive health information. Never commit CSV files, patient identifiers, or raw readings to version control; CSV datasets and trained model weights (`backend/models/*.json`) remain strictly outside git and are excluded via `.gitignore`.

Alternatively, run via Docker Compose:
```bash
cd backend
docker compose up --build
```

### 4. Running the dashboard
The dashboard requires the running FastAPI backend at `http://127.0.0.1:8000` and does not fall back to synthetic or demo data. Neither the OhioT1DM CSV datasets nor trained model files (`backend/models/*.json`) are stored in the repository.

Start the backend from `backend/` with `TEST_CSV_PATH` set to your local testing CSV:
```powershell
# PowerShell (Windows)
$env:TEST_CSV_PATH = "C:\path\to\ohio_testing_combined.csv"
cd backend
..\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
```bash
# Bash (Linux / macOS)
export TEST_CSV_PATH="/path/to/ohio_testing_combined.csv"
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, serve the `Frontend/` directory and open `http://127.0.0.1:5500`:
```bash
cd Frontend
python -m http.server 5500 --bind 127.0.0.1
```

Dashboard states shown in the UI:
- **Loading**: Initial requests to `/health`, `/v1/metrics`, `/v1/models`, and `/v1/cases` are in flight.
- **Backend offline**: The API at `http://127.0.0.1:8000` is unreachable (start the backend and reload the page).
- **Models not loaded**: The backend is reachable, but fewer than the expected 4 horizon models are loaded; `Model` and `Performance` views reflect model/metrics status.
- **No test data configured**: `TEST_CSV_PATH` is unset or provides no eligible cases; case-dependent views (`Overview`, `Glucose Forecast`, `Test Explorer`, `Clinical Analysis`) prompt for test data while `Performance` and `Model` still display backend data.
- **Request error**: An API endpoint returned a non-OK error response; the backend's `detail` message is displayed safely.

---

## Backend API Specification

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status and loaded model inventory |
| `GET` | `/v1/models` | Loaded models with file hashes and sizes, feature columns, metrics status and feature importance |
| `GET` | `/v1/metrics` | Returns pre-computed canonical test-set metrics from `backend/models/metrics.json` |
| `GET` | `/v1/cases` | Representative clinical evaluation cases sampled across patients from `TEST_CSV_PATH` |
| `GET` | `/v1/cases/{case_id}` | Detailed case inspection with 3h history, 2h future, and actual horizon values |
| `POST` | `/v1/predict` | JSON prediction given historical CGM readings (minimum 24 readings) |
| `POST` | `/v1/predict/csv` | Multipart CSV upload; optional patient_id selects the patient; forecast from that patient's most recent reading |
| `POST` | `/v1/backtest` | Batch evaluation (uploaded CSV, or the server test data when no file is sent) returning RMSE, MAE, Clarke EGA zones, per-patient results, gap warnings and a representative plot sample |

### Example Prediction Request (`POST /v1/predict`)
```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [
      {"timestamp": "2026-11-03T08:00:00", "glucose": 120.0},
      {"timestamp": "2026-11-03T08:05:00", "glucose": 122.0},
      ... 24+ consecutive 5-minute CGM readings ...
    ],
    "horizon": "all"
  }'
```

### Example Backtest Request (`POST /v1/backtest`)
```bash
curl -X POST http://127.0.0.1:8000/v1/backtest \
  -F "file=@ohio_testing_combined.csv" \
  -F "horizon=30"
```

---

## Running the Automated Test Suite

The test suite validates Clarke EGA classification boundaries, feature transformations (ensuring zero data leakage and exact lag shifts), API routes, OpenAPI schema contracts, and metrics reporting:
```bash
cd backend
pytest tests/ -v
```
Test suite coverage:
- `tests/test_api.py`: Endpoint health, loaded models inventory, predict history/glucose validation, and basic backtest route execution
- `tests/test_backtest.py`: Batch evaluation canonical row counts across all horizons, per-patient breakdowns, plot sampling, and summary metric alignment
- `tests/test_cases.py`: Vectorized case eligibility, deterministic segment-midpoint sampling, case detail endpoints, log hygiene, and fileless backtest execution
- `tests/test_context.py`: Prediction context summary statistics, model file hash reporting, and `/v1/metrics` loading status
- `tests/test_ega.py`: Hand-checked Clarke Error Grid Analysis boundary values and zone classifications (Zones A through E)
- `tests/test_features.py`: 18-feature extraction pipeline, lag shifts, rolling windows, cyclical circadian encoding, and target calculation without forward leakage
- `tests/test_metrics.py`: Statistical (RMSE, MAE) and clinical safety (Clarke zones, A+B%) summarization logic on known samples and empty inputs
- `tests/test_metrics_file.py`: Structural verification and key presence of canonical `backend/models/metrics.json`
- `tests/test_metricsfile.py`: Validation and status checks for `metrics.json` against loaded model store hashes, feature columns, and schema versions
- `tests/test_openapi_contract.py`: Frozen snapshot synchronization between `backend/openapi.json` and live FastAPI schema (`app.openapi()`)
- `tests/test_policy.py`: Input validation order, timestamp uniqueness, gap tolerance, multi-patient CSV routing, and day-first date formatting
- `tests/test_render_tables.py`: Markdown table formatting (T1–T4), numerical rounding, patient sorting, and absence of deprecated models
- `tests/test_timeutil.py`: Flexible timestamp parsing (ISO 8601 and day-first), timezone offset stripping, and timezone-naive standardization

---

## Data Notice

This project utilizes the **OhioT1DM Dataset** (2020 cohort). Access to the OhioT1DM dataset requires an executed Data Use Agreement with Ohio University and is therefore **not distributed** in this repository. All CSV datasets are explicitly gitignored.
