# Glucose Horizon Backend

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

Docker Compose is also available:

```powershell
docker compose up --build
```

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

The script splits by virtual-patient ID: 24 patients fit the models, while six held-out patients are used only for evaluation. Input features use past CGM, insulin, and CHO. `BG` is the target; simulator risk fields are excluded. The held-out results measure performance on simulated virtual patients and do not establish real-patient accuracy.

**Current artifact caveat:** the CSV currently in `data/simglucose_training.csv` has 14,430 rows: 30 patients with 481 rows each, corresponding to a one-day simulation. The currently saved `models/simglucose/` weights were trained from that one-day dataset. The script's 35-day duration is its current default, but existing artifacts do not change until the script completes successfully. Simulation/training may take several minutes. Restart the API after regenerating weights because models are loaded at backend startup.

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
| `POST` | `/v1/predict` | Ohio prediction from JSON CGM readings |
| `POST` | `/v1/predict/csv` | Ohio prediction from a CGM CSV |
| `POST` | `/v1/predict/simglucose/csv` | SimGlucose prediction from a multimodal CSV |
| `POST` | `/v1/backtest` | Ohio backtest using an uploaded CSV and chosen horizon |
| `GET` | `/v1/backtest/overall` | Overall Ohio held-out metrics at all horizons |
| `GET` | `/v1/backtest/patient/{patient_id}` | Ohio held-out metrics for one patient at all horizons |

The backtest endpoints are for OhioT1DM and use the Ohio models and feature generation. They do not evaluate SimGlucose models. `/v1/backtest` expects multipart form fields `file` and `horizon`; the CSV must contain `timestamp`, `glucose`, and `patient_id`. Its timestamps should use `%d-%m-%Y %H:%M:%S`. Supported horizons are 30, 60, 90, and 120 minutes.

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

The Ohio feature pipeline uses 18 CGM-derived columns: `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`; rolling mean and standard deviation for 30, 60, 90, and 120 minutes; `delta_5`, `delta_30`; and `hour`, `hour_sin`, `hour_cos`. SimGlucose has a separate feature specification stored at `models/simglucose/feature_columns.json`.

## Tests

Run the backend tests from this directory with:

```powershell
pytest tests/ -v
```
