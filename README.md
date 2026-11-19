# Glucose Horizon

Glucose Horizon is a multi-horizon blood-glucose forecasting project for Type 1 diabetes. It includes a FastAPI backend and a browser dashboard. Forecast horizons are 30, 60, 90, and 120 minutes.

The app has two prediction paths:

- **CGM only (OhioT1DM):** uses the existing Ohio-trained models for glucose-history CSVs.
- **CGM with insulin (SimGlucose):** uses separate XGBoost models trained on virtual-patient traces produced with SimGlucose. CHO (meal carbohydrate) is optional and is treated as zero when omitted.

An uploaded file is used for inference only; uploading does not train or fine-tune a model. The two model families have separate weights and feature pipelines.

## Repository layout

```text
backend/
  app/                 FastAPI routes, predictors, feature engineering, EGA
  data/                Generated SimGlucose training data (not user upload data)
  models/              Ohio models and separate models/simglucose weights
  scripts/             Ohio and SimGlucose training scripts
  tests/               Backend tests
Frontend/
  index.html           Dashboard views, including Models, Performance, and Upload
  style.css            Responsive dashboard styles and animations
  script.js            Navigation, charts, API calls, upload, cursor effect
Model/                 Research notebooks
```

## Run locally

Install the project dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

Start the backend from the repository root:

```powershell
uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

API documentation is available at `http://127.0.0.1:8000/docs`.

Serve the frontend in a second terminal:

```powershell
cd Frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500`. The frontend calls the backend at `http://127.0.0.1:8000`. If the API is unavailable, some dashboard views can use synthetic demonstration data; such data is for UI demonstration and is not a patient-specific forecast.

## Models and training data

### OhioT1DM: CGM-only

The Ohio pipeline retains its existing models and behavior. The model predicts future glucose from past CGM-derived lags, rolling statistics, rates of change, and time-of-day features. Ohio data is not included in the repository; access is subject to the dataset's data-use terms. Train or evaluate the Ohio models with:

```powershell
python backend/scripts/train_models.py --train path/to/ohio_training_combined.csv --test path/to/ohio_testing_combined.csv
```

### SimGlucose: CGM plus insulin, optional CHO

Generate virtual-patient data and train the separate multimodal models with:

```powershell
python backend/scripts/train_simglucose_models.py
```

The script defaults to simulating 30 virtual patients (10 each in adult, adolescent, and child groups) for 35 days at five-minute sampling, about 10,081 raw readings per patient. Override the duration with `--days N`. It saves the generated raw CSV to `backend/data/simglucose_training.csv` and the four horizon models and feature specification to `backend/models/simglucose/`.

Patients are split by ID for evaluation: 24 patients are used to fit evaluation models and six are held out. After reporting held-out RMSE, the script refits the deployment models on all 30 patients, so the upload endpoint uses weights trained on the full simulated dataset. The held-out scores describe the 24-patient evaluation models, not the final all-patient weights. Input features use past CGM, insulin, and CHO; `BG` is the prediction target, and simulator risk fields are excluded. The training script's 35-day setting is a default, not a guarantee that existing generated artifacts were made with that setting. Use `python backend/scripts/train_simglucose_models.py --reuse-existing` to rebuild models from the existing generated CSV without running simulation again. Simulation can take several minutes.

The SimGlucose evaluation split is patient-held-out but simulator-based. It does not establish performance on real people or replace clinical validation.

## Frontend sections

- **Normal View and Patient View:** a checkbox at the bottom of the sidebar switches between the full dashboard and a simpler patient-facing mode. Patient View limits navigation to Overview, Upload CGM Data, and Forecast Agent; it hides the detailed forecast cards and risk timeline on Overview and the Prediction Models card on the upload page. Turning the checkbox off restores the full navigation.
- **Overview:** patient glucose values stay blank until an uploaded CSV has completed prediction. The normal view includes the actual-versus-predicted trajectory card; Patient View hides that chart. Both modes include a **Get tips from your Forecast Agent** action that opens a patient-aware prompt, using upload context when available.
- **Glucose Forecast:** the plots and evaluation table stay empty until a CSV prediction is available. For uploaded data, forward predictions are shown at the selected horizons and historical actual-versus-predicted evaluation is shown when there are enough future readings in the file.
- **Test Explorer:** OhioT1DM and SimGlucose use separate panels with distinct heading, controls, results, and context boxes. The SimGlucose panel shows CGM context for the selected patient, case, and horizon before **Run Forecast**; prediction results appear only after that button is pressed. The Ohio and SimGlucose controls remain independent.
- **Clinical Analysis:** the glucose range bar remains visible as an empty baseline when no patient data is selected. Uploaded patient values populate it after prediction. Clarke EGA plots are hidden by default and appear after enabling the reference-patient checkbox and choosing an OhioT1DM or SimGlucose patient. The reference selector does not replace upload data in other sections.
- **Performance:** an uploaded CSV's historical accuracy and forward forecasts appear automatically after prediction. The test-patient checkbox and dropdown allow choosing an OhioT1DM or SimGlucose patient; an explicit test-patient selection takes precedence in this panel. With neither upload results nor a selected test patient, the metrics area stays empty.
- **Models:** descriptions of the Ohio CGM-only and SimGlucose multimodal model families, their horizons, and data paths. This section is part of Normal View.
- **Upload CGM Data:** upload history is validated and routed through the chosen CGM-only or multimodal model. The app flags obvious model/data mismatches while still displaying predictions where available; failed incompatible uploads show a prominent error. After prediction, the Forecast Agent shortcut asks for meal and physical-activity ideas grounded in the uploaded glucose history and forecast. The Prediction Models information card is shown in Normal View and hidden in Patient View.
- **Forecast Agent:** the Groq assistant (`openai/gpt-oss-120b`) uses the active uploaded forecast context and remembers the conversation's selected patient context. The chat has a collapsible suggested-question drawer, a clearable composer that remains editable while a response is running, an in-progress reasoning indicator, and a full-screen expand control. Full-screen mode closes with Escape or a backdrop click. Internal tool-call details are not displayed in chat replies.
- **Safety and presentation:** the forecasting and wellness features are research aids, not treatment or insulin-dosing advice. The interface keeps clinical disclaimers visible, supports reduced-motion preferences, and uses responsive layouts for smaller screens.

## CSV upload formats

Choose the prediction mode that matches the data:

| Mode | Required columns | Optional columns | Notes |
|---|---|---|---|
| CGM only (Ohio) | Timestamp and glucose | — | Use at least 25 sequential readings at roughly five-minute intervals. Accepted aliases include `Time`/`timestamp` and `CGM`/`BG` as supported by the backend parser. |
| CGM + insulin (SimGlucose) | Timestamp/`Time`, `CGM`, `insulin` | `CHO`; `BG` | CHO is set to zero if absent. If `BG` is absent, CGM is used as the target-side reference for features/output. Data is resampled to five-minute intervals; at least 25 valid readings are required. |

The SimGlucose route needs trained weights present in `backend/models/simglucose/`. If they are absent, the endpoint returns HTTP 503 until the SimGlucose training script completes. Uploaded records are not added to training.

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Backend and Ohio model health/inventory |
| `GET` | `/v1/models` | Ohio model horizons and features |
| `GET` | `/v1/models/simglucose` | SimGlucose model load status and features |
| `GET` | `/v1/simglucose/test/patients` | List held-out SimGlucose patient IDs for Test Explorer |
| `GET` | `/v1/simglucose/test/patient/{patient_id}?horizon=30&case_index=2` | Return selected case prediction, EGA, and patient-wide metrics from the held-out SimGlucose model |
| `POST` | `/v1/predict` | Ohio CGM-only prediction from JSON readings |
| `POST` | `/v1/predict/csv` | Ohio CGM-only prediction from uploaded CSV |
| `POST` | `/v1/predict/simglucose/csv` | SimGlucose prediction from uploaded CGM/insulin/optional CHO CSV |
| `POST` | `/v1/wellness` | Groq-powered general wellness suggestion agent |
| `GET` | `/agent/health` | Groq forecast agent health & model status |
| `POST` | `/agent/chat` | Autonomous T1D forecast agent chat with tool execution |
| `POST` | `/agent/reset` | Reset conversational session memory |
| `GET/POST` | `/agent/chat/stream` | Server-Sent Events (SSE) token and tool streaming |
| `POST` | `/v1/backtest` | Ohio backtest for an uploaded CSV and chosen horizon |
| `GET` | `/v1/backtest/overall` | Ohio held-out overall metrics across horizons |
| `GET` | `/v1/backtest/patient/{patient_id}` | Ohio held-out metrics for one patient across horizons |

The backtest routes use the Ohio feature pipeline and Ohio models; they are not a SimGlucose backtest. The patient-specific route reports metrics for the requested patient from the configured Ohio test CSV. The uploaded-file `/v1/backtest` requires `timestamp`, `glucose`, and `patient_id` columns and a supported horizon (30, 60, 90, or 120).

## Clinical evaluation

Ohio forecasts are reported with RMSE and MAE and classified with Clarke Error Grid Analysis (EGA). Zones A and B are commonly grouped as clinically acceptable for this analysis. These metrics are model-evaluation aids and are not medical advice or a substitute for clinical validation.

The benchmark figures shown in the research materials describe the Ohio test cohort and depend on the model/data version used for that run; reproduce them with the corresponding data and training/evaluation workflow rather than treating them as a guarantee for new uploads.

## Data and artifact notes

- OhioT1DM data is not bundled in this repository. All users must follow its access and data-use terms.
- `backend/data/simglucose_training.csv` contains synthetic SimGlucose traces generated by the training script.
- Model JSON files are runtime artifacts. The backend loads the Ohio family and the SimGlucose family from separate directories; restart the backend after replacing model files so it reloads the weights.
- Synthetic frontend demo data is only a fallback for demonstrating the interface.

More backend-specific request examples and configuration notes are in [backend/README.md](backend/README.md).
