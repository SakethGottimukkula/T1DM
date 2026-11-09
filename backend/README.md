# Glucose Horizon — Backend

FastAPI backend for CGM-only blood glucose forecasting (Type 1 Diabetes, OhioT1DM dataset).

## Quick Start

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Train models

You need the OhioT1DM training CSV (requires a [data use agreement](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html)).

```bash
python scripts/train_models.py --train path/to/ohio_training_combined.csv
```

With optional evaluation on the test set:

```bash
python scripts/train_models.py --train path/to/ohio_training_combined.csv --test path/to/ohio_testing_combined.csv
```

This produces `models/model_30.json`, `model_60.json`, `model_90.json`, `model_120.json`, and `models/feature_columns.json`.

### 3. Run the server

```bash
cd backend
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs at `/docs`.

### 4. Run tests

```bash
cd backend
pytest tests/ -v
```

## Docker

```bash
cd backend
docker compose up --build
```

Models are mounted from `./models` via the volume in `docker-compose.yml`.

## API Endpoints

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

### `GET /v1/models`

```bash
curl http://127.0.0.1:8000/v1/models
```

- **Model file information**: Each model in `models` includes `sha256` (SHA-256 hash in lowercase hex) and `size_bytes` (file size in bytes), or null if not loaded.
- **Metrics consistency**: `metrics_status` reports consistency between loaded models and `metrics.json` (`ok`, `models_not_loaded`, `model_mismatch`, `feature_mismatch`, `missing`, `unreadable`, or `unsupported_schema`).
- **Warnings & importance**: `metrics_warnings` lists human-readable discrepancy descriptions, and `feature_importance` contains the full feature importance block from `metrics.json` (or null if missing).

### `GET /v1/metrics`

```bash
curl http://127.0.0.1:8000/v1/metrics
```

- **Response**: Returns `{status, warnings, metrics}` with the full contents of `metrics.json`.
- **Status values**: `ok` (hashes and features match), `models_not_loaded` (one or more horizon models not loaded), `model_mismatch` (model hash differs from `metrics.json`), `feature_mismatch` (feature columns differ), `missing` (file not found), `unreadable` (invalid JSON/non-object), `unsupported_schema` (schema_version != 1).
- **Error handling**: Returns HTTP 503 if `metrics.json` cannot be loaded (`missing`, `unreadable`, or `unsupported_schema`), advising to run `python scripts/evaluate.py`.

### `POST /v1/predict`

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [
      {"timestamp": "2020-01-15T08:00:00", "glucose": 120},
      {"timestamp": "2020-01-15T08:05:00", "glucose": 122},
      {"timestamp": "2020-01-15T08:10:00", "glucose": 118},
      {"timestamp": "2020-01-15T08:15:00", "glucose": 115},
      {"timestamp": "2020-01-15T08:20:00", "glucose": 112},
      {"timestamp": "2020-01-15T08:25:00", "glucose": 110},
      {"timestamp": "2020-01-15T08:30:00", "glucose": 108},
      {"timestamp": "2020-01-15T08:35:00", "glucose": 106},
      {"timestamp": "2020-01-15T08:40:00", "glucose": 105},
      {"timestamp": "2020-01-15T08:45:00", "glucose": 107},
      {"timestamp": "2020-01-15T08:50:00", "glucose": 110},
      {"timestamp": "2020-01-15T08:55:00", "glucose": 113},
      {"timestamp": "2020-01-15T09:00:00", "glucose": 116},
      {"timestamp": "2020-01-15T09:05:00", "glucose": 119},
      {"timestamp": "2020-01-15T09:10:00", "glucose": 122},
      {"timestamp": "2020-01-15T09:15:00", "glucose": 125},
      {"timestamp": "2020-01-15T09:20:00", "glucose": 128},
      {"timestamp": "2020-01-15T09:25:00", "glucose": 130},
      {"timestamp": "2020-01-15T09:30:00", "glucose": 132},
      {"timestamp": "2020-01-15T09:35:00", "glucose": 134},
      {"timestamp": "2020-01-15T09:40:00", "glucose": 136},
      {"timestamp": "2020-01-15T09:45:00", "glucose": 138},
      {"timestamp": "2020-01-15T09:50:00", "glucose": 140},
      {"timestamp": "2020-01-15T09:55:00", "glucose": 142}
    ],
    "horizon": "all"
  }'
```

### `POST /v1/predict/csv`

```bash
curl -X POST http://127.0.0.1:8000/v1/predict/csv \
  -F "file=@path/to/data.csv" \
  -F "horizon=60"
```

### Prediction Context (`context` block in `/v1/predict` and `/v1/predict/csv`)

Both prediction endpoints return a `context` block detailing the input history window:
- `readings_submitted`: Total number of CGM readings submitted in the request or parsed from the CSV after patient selection.
- `window_readings`: Number of readings in the feature calculation window, always equal to 24 (2 hours of history).
- `window_span_minutes`: Time span between the first and last reading in the window in minutes (rounded to 2 decimal places).
- `window_start`: ISO 8601 timestamp without timezone of the earliest reading in the 24-reading window.
- `window_end`: ISO 8601 timestamp without timezone of the most recent reading in the window (equals the response timestamp).
- `current_glucose`: Most recent blood glucose value in mg/dL (rounded to 2 decimal places).
- `window_min`: Minimum blood glucose value observed across the 24-reading window in mg/dL (rounded to 2 decimal places).
- `window_max`: Maximum blood glucose value observed across the 24-reading window in mg/dL (rounded to 2 decimal places).
- `window_mean`: Arithmetic mean of blood glucose values across the 24-reading window in mg/dL (rounded to 2 decimal places).
- `change_5min`: Rate of glucose change over the last 5 minutes (`glucose - lag_1`) in mg/dL (rounded to 2 decimal places).
- `change_30min`: Rate of glucose change over the last 30 minutes (`glucose - lag_6`) in mg/dL (rounded to 2 decimal places).
- `patient_id`: Identifier of the selected patient when provided in CSV input, or null for JSON predict requests without a patient column.
- `features`: Dictionary containing all 18 engineered feature values supplied to the models in canonical order (rounded to 4 decimal places).

### `POST /v1/backtest`

```bash
curl -X POST http://127.0.0.1:8000/v1/backtest \
  -F "file=@path/to/ohio_testing_combined.csv" \
  -F "horizon=30" \
  -F "max_points=2000"
```

Running backtest without file upload (using server-configured `TEST_CSV_PATH`):

```bash
curl -X POST http://127.0.0.1:8000/v1/backtest \
  -F "horizon=30" \
  -F "max_points=2000"
```

- **Evaluation rule**: Evaluates only rows where all 18 features and all four targets (30, 60, 90, 120 min) are non-null ($N - 47$ rows per patient). The exact same evaluated row set is used across all four horizons.
- **Parameters**:
  - `file`: Optional uploaded CSV with `timestamp`, `glucose`, `patient_id`. If omitted, server falls back to `TEST_CSV_PATH`. Returns HTTP 422 if neither is provided.
  - `horizon`: Forecast horizon (30, 60, 90, or 120).
  - `max_points`: Optional int, default 2000 (range 50..5000).
- **Plot sampling**: When evaluated rows $N > \text{max\_points}$, samples at most `max_points` evenly spaced rows via deterministic `np.linspace(0, n - 1, max_points)` rounded and de-duplicated. If $N \le \text{max\_points}$, all evaluated rows are returned.
- **Response fields**: Includes overall metrics (`n_rows`, `rmse`, `mae`, `zone_counts`, `zone_percentages`, `a_plus_b_pct`), `n_rows_raw`, `evaluation_rule`, `per_patient` (per-patient metric breakdown), `data_source` (`"upload"` or `"server"`), and parallel plotted arrays capped at `max_points` (`timestamps`, `actual`, `predicted`, `patient_ids`, `zones`).

### `GET /v1/cases`

```bash
curl http://127.0.0.1:8000/v1/cases?per_patient=3
```

- **Query parameters**: `per_patient` (optional int, default 3, range 1–10).
- **Eligibility criteria**: A reading at index $i$ is eligible as a test case if and only if it has:
  - At least 36 historical readings ($i-36 \ge 0$, spanning 3 hours),
  - At least 24 future readings ($i+24 \le n-1$, spanning 2 hours),
  - All 60 consecutive reading intervals between index $i-36$ and $i+24$ are strictly between 3 and 7 minutes (180s to 420s).
- **Case selection**: Deterministically selects up to `per_patient` eligible cases per patient evenly spaced across their eligible readings via linear interpolation.
- **Response**: Returns `data_source` summary (`configured`, `file_name`, `status`, `total_readings`, `total_patients`, `total_eligible_cases`), per-patient statistics (`n_readings`, `n_eligible_cases`, `first_timestamp`, `last_timestamp`), and a list of `cases` with `case_id`, `patient_id`, `timestamp`, and `glucose`. Returns HTTP 503 if server test data is not configured or missing.

### `GET /v1/cases/{case_id}`

```bash
curl http://127.0.0.1:8000/v1/cases/101-20220115T120000
```

- **Path parameters**: `case_id` formatted as `{patient_id}-{timestamp:%Y%m%dT%H%M%S}`.
- **Response**: Returns full case context including:
  - `history`: 37 readings ($i-36$ to $i$ inclusive, representing 3 hours of CGM history up to the decision point),
  - `future`: 24 readings ($i+1$ to $i+24$ inclusive, representing 2 hours of actual subsequent CGM observations),
  - `actual_at`: Ground truth glucose values at each model horizon (`"30"`, `"60"`, `"90"`, `"120"`).
- **Errors**: Returns HTTP 404 if the case ID does not match any eligible case. Returns HTTP 503 if server test data is not configured.

## Timestamps and gap policy

- **Accepted formats**: ISO 8601 (e.g. `YYYY-MM-DDTHH:MM:SS`) and Day-first (`DD-MM-YYYY HH:MM:SS`).
- **Offsets**: Any UTC offset or `Z` suffix is ignored; written wall-clock time is used directly.
- **Output**: Predictions and backtests return timestamps in ISO 8601 format without timezone.
- **Duplicate timestamps**: Rejected with HTTP 422 (`Timestamps must be unique.`).
- **Gap policy for predict**: Only the last 24 readings (2 hours) must be continuous (3–7 min spacing); gaps before this window are ignored.
- **Patient selection (/v1/predict/csv)**: Optional `patient_id` form field selects a specific patient from multi-patient CSVs.
- **Backtest behaviour**: `/v1/backtest` only logs non-5-minute gaps as warnings and computes features by row position without rejection.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `backend/models` | Directory containing model `.json` files |
| `CORS_ORIGINS` | `http://127.0.0.1:5500,http://localhost:5500` | Comma-separated CORS origins |
| `TRAIN_CSV` | — | Alternative to `--train` CLI arg |
| `TEST_CSV_PATH` | — | Path to server-side test dataset CSV (e.g. ohio_testing_combined.csv) |

### Setting `TEST_CSV_PATH`

PowerShell:
```powershell
$env:TEST_CSV_PATH = "C:\path\to\ohio_testing_combined.csv"
uvicorn app.main:app --reload
```

Bash / macOS / Linux:
```bash
export TEST_CSV_PATH="/path/to/ohio_testing_combined.csv"
uvicorn app.main:app --reload
```

### Dataset Privacy

Patient CGM data is strictly private and covered by data use agreements. The server accepts `TEST_CSV_PATH` pointing to local server-side files, but patient records are never logged, stored in source control, or exposed outside their intended inspection endpoints. Error messages and status endpoints reference only base file names (e.g., `ohio_testing_combined.csv`) and never divulge directory paths.

## Feature Columns (18)

Lag features: `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`
Rolling stats: `roll_mean_30`, `roll_std_30`, `roll_mean_60`, `roll_std_60`, `roll_mean_90`, `roll_std_90`, `roll_mean_120`, `roll_std_120`
Rate of change: `delta_5`, `delta_30`
Time encoding: `hour`, `hour_sin`, `hour_cos`
