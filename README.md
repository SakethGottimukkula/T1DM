# Glucose Horizon — Blood Glucose Forecasting (CGM-Only)

**Glucose Horizon** is a multi-horizon blood glucose forecasting and clinical safety evaluation system for Type 1 Diabetes (T1D). It uses continuous glucose monitoring (CGM) sensor streams alone—without requiring manual insulin dosing or meal logs—to predict glucose trajectories 30, 60, 90, and 120 minutes into the future.

The system combines:
1. **Machine Learning Core**: 4 XGBoost regressors trained on OhioT1DM CGM data.
2. **Production Backend**: A FastAPI REST microservice supporting real-time inference, file-based forecasting, and batch backtesting with Clarke Error Grid Analysis (EGA).
3. **Interactive Frontend**: A clinical dashboard for real-time visualization, patient switching, risk stratification, and error grid analysis.

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

Tested across all patients in the OhioT1DM test set:

| Horizon | Prediction Shift | RMSE (mg/dL) | MAE (mg/dL) | Clinically Safe (A + B) | Zone A | Zone B |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30 min** | 6 steps ahead (t+30m) | **20.94** | **14.58** | **98.30%** | 88.60% | 9.70% |
| **60 min** | 12 steps ahead (t+60m) | **33.67** | **24.32** | **96.72%** | 76.54% | 20.18% |
| **90 min** | 18 steps ahead (t+90m) | **42.44** | **31.76** | **95.14%** | 68.21% | 26.93% |
| **120 min** | 24 steps ahead (t+120m) | **48.08** | **36.98** | **93.87%** | 62.80% | 31.07% |

---

## Repository Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application, lifespan, routes & CORS
│   │   ├── config.py            # Environment configuration & validation thresholds
│   │   ├── schemas.py           # Pydantic v2 request/response schemas
│   │   ├── features.py          # 18-column feature pipeline (grouped & single-patient)
│   │   ├── predictor.py         # XGBoost model loader & inference manager
│   │   └── ega.py               # Clarke Error Grid classification logic
│   ├── models/                  # Exported XGBoost JSON models & feature spec
│   │   ├── model_30.json
│   │   ├── model_60.json
│   │   ├── model_90.json
│   │   ├── model_120.json
│   │   └── feature_columns.json
│   ├── scripts/
│   │   └── train_models.py      # CLI script for multi-horizon training & eval
│   ├── tests/                   # Pytest test suite (38 tests)
│   │   ├── test_api.py          # Endpoint tests, validation & 422 error cases
│   │   ├── test_ega.py          # Hand-checked Clarke EGA boundary tests
│   │   └── test_features.py     # Feature correctness & leakage tests
│   ├── Dockerfile               # Container definition (python:3.11-slim + libgomp1)
│   ├── docker-compose.yml       # Production Compose configuration with models mount
│   ├── requirements.txt         # Pinned backend dependencies
│   └── README.md                # Detailed backend documentation & curl examples
├── Frontend/
│   ├── index.html               # Clinical dashboard markup & multi-view panels
│   ├── style.css                # Dashboard styling, typography & responsive layouts
│   └── script.js                # Chart.js visualization, API client & graceful fallback
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
Start the FastAPI server via Uvicorn:
```bash
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI Specification: `http://127.0.0.1:8000/openapi.json`

Alternatively, run via Docker Compose:
```bash
cd backend
docker compose up --build
```

### 4. Run the Frontend Dashboard
Serve the `Frontend/` directory using VS Code Live Server (port 5500) or any local HTTP server:
```bash
# Example with Python's built-in HTTP server:
cd Frontend
python -m http.server 5500
```
Open `http://127.0.0.1:5500` in your browser.
The dashboard automatically fetches predictions from the FastAPI service at `http://127.0.0.1:8000`. If the backend is offline, it gracefully falls back to synthetic demonstration data.

---

## Backend API Specification

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status and loaded model inventory |
| `GET` | `/v1/models` | List of supported horizons, feature columns, and file paths |
| `POST` | `/v1/predict` | JSON prediction given historical CGM readings (minimum 24 readings) |
| `POST` | `/v1/predict/csv` | Multipart CSV upload returning forecast from the most recent row |
| `POST` | `/v1/backtest` | Batch evaluation returning RMSE, MAE, Clarke EGA, gap warnings, and capped plot series |

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

The test suite validates Clarke EGA classification boundaries, feature transformations (ensuring zero data leakage and exact lag shifts), and API routes:
```bash
cd backend
pytest tests/ -v
```
All 38 test cases pass:
- `tests/test_ega.py`: 13 EGA boundary & zone mapping tests
- `tests/test_features.py`: 13 feature calculation & leakage validation tests
- `tests/test_api.py`: 12 API endpoint, validation, and backtest tests

---

## Data Notice

This project utilizes the **OhioT1DM Dataset** (2020 cohort). Access to the OhioT1DM dataset requires an executed Data Use Agreement with Ohio University and is therefore **not distributed** in this repository. All CSV datasets are explicitly gitignored.
