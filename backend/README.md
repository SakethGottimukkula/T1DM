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

### `POST /v1/backtest`

```bash
curl -X POST http://127.0.0.1:8000/v1/backtest \
  -F "file=@path/to/ohio_testing_combined.csv" \
  -F "horizon=30"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `backend/models` | Directory containing model `.json` files |
| `CORS_ORIGINS` | `http://127.0.0.1:5500,http://localhost:5500` | Comma-separated CORS origins |
| `TRAIN_CSV` | — | Alternative to `--train` CLI arg |

## Feature Columns (18)

Lag features: `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`
Rolling stats: `roll_mean_30`, `roll_std_30`, `roll_mean_60`, `roll_std_60`, `roll_mean_90`, `roll_std_90`, `roll_mean_120`, `roll_std_120`
Rate of change: `delta_5`, `delta_30`
Time encoding: `hour`, `hour_sin`, `hour_cos`
