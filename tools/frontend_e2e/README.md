# Frontend End-to-End Regression Tool

`frontend_e2e.py` verifies the live Glucose Horizon dashboard against independent FastAPI responses across **FE-1** (Overview, Forecast, Explorer, stale-response guard), **FE-2** (Performance, Clinical Error Grid across all 4 horizons, rapid horizon switching), and **FE-3** (Model metadata view, feature groups, SHA-256 hashes, feature importance table, backend zone labels).

The script never prints raw patient glucose values, timestamps, or dataset file paths—only `PASS`/`FAIL` status lines and counts.

## Prerequisites

Create a separate virtual environment **outside** the repository for Playwright (so browser automation dependencies stay separate from backend runtime dependencies):

```powershell
# PowerShell (Windows)
python -m venv C:\temp\pw_env
C:\temp\pw_env\Scripts\pip.exe install playwright
C:\temp\pw_env\Scripts\playwright.exe install chromium
```

```bash
# Bash (Linux / macOS)
python -m venv /tmp/pw_env
/tmp/pw_env/bin/pip install playwright
/tmp/pw_env/bin/playwright install chromium
```

## Starting the Servers

1. Start the FastAPI backend from `backend/` using the repo's virtual environment, with `TEST_CSV_PATH` pointing to your local testing CSV:
   ```powershell
   $env:TEST_CSV_PATH = "C:\path\to\ohio_testing_combined.csv"
   cd backend
   ..\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
2. Start the static server for `Frontend/` on port `5500`:
   ```bash
   cd Frontend
   python -m http.server 5500 --bind 127.0.0.1
   ```

## Running the Regression Script

With both servers running, execute the script using the Playwright virtual environment's Python interpreter:

```powershell
C:\temp\pw_env\Scripts\python.exe tools\frontend_e2e\frontend_e2e.py
```

Optional environment variables:
- `API_URL` (default: `http://127.0.0.1:8000`)
- `FRONTEND_URL` (default: `http://127.0.0.1:5500`)

The script exits with code `0` only if every check passes.

## Manual State Checks

Offline, missing test data (`TEST_CSV_PATH` unset), missing/empty models (`MODEL_DIR` empty or corrupted model file), and XSS payload injection checks require restarting or reconfiguring the backend between runs and are verified manually (not covered by this script).
