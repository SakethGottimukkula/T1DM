/* Demo / research prototype using CGM-only OhioT1DM replay data. */

const API_BASE_URL = 'http://127.0.0.1:8000';
const HZ = [30, 60, 90, 120];

const COLORS = {
  accent: '#087f8c',
  blue: '#537cdb',
  ok: '#218566',
  warn: '#ba7918',
  bad: '#d35d57',
  mute: '#718391',
  grid: '#e1e9ed'
};

const ZONE_COLORS = {
  A: COLORS.ok,
  B: COLORS.accent,
  C: COLORS.warn,
  D: '#e8825a',
  E: COLORS.bad
};

const EVENTS = [
  'CGM sampled every 5 minutes',
  'Historical glucose window used for forecasting',
  'Lag features derived from glucose history',
  'Rolling statistics derived from the CGM stream',
  '5-minute and 30-minute rate-of-change features',
  'Hour sine/cosine time encoding'
];

// ---------------------------------------------------------------------------
// HTML escape helper
// ---------------------------------------------------------------------------

function escapeHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ---------------------------------------------------------------------------
// API client and data state
// ---------------------------------------------------------------------------

/** Track whether the backend is reachable and models are loaded. */
let apiState = {
  connected: false,
  modelsLoaded: 0,
  modelsExpected: 4,
  status: 'unknown',
  featureColumns: []
};

/** UI state for honest status representation across case views (Overview, Forecast, Explorer). */
let uiState = {
  phase: 'loading', // 'loading' | 'ready' | 'offline' | 'degraded' | 'no_data' | 'error'
  detail: ''
};

/** State for Performance view (depends solely on GET /v1/metrics). */
let perfState = {
  phase: 'loading', // 'loading' | 'ready' | 'offline' | 'unavailable'
  detail: '',
  data: null
};

/** State for Clinical view (depends on case markers and POST /v1/backtest). */
let clinicalState = {
  phase: 'loading', // 'loading' | 'ready' | 'offline' | 'degraded' | 'no_data' | 'error'
  detail: '',
  horizon: null,
  data: null
};

let caseList = [];             // from /v1/cases: [{ id, patient, ts, cur }]
let patientList = [];          // from /v1/cases: patient ids as strings
let caseCache = new Map();     // case id -> loaded case object
let backtestCache = new Map(); // horizon (number) -> BacktestResponse

let appState = {
  caseId: null,
  horizon: 60
};

let currentLoadCaseId = null;       // Stale-response guard for asynchronous case loading
let currentBacktestHorizon = null;  // Stale-response guard for asynchronous backtest loading

/**
 * Fetch wrapper for GET requests.
 * Returns { ok, status, data, detail } and never throws.
 */
async function apiGet(path) {
  try {
    const res = await fetch(`${API_BASE_URL}${path}`);
    let data = null;
    let detail = '';
    try {
      data = await res.json();
      if (!res.ok && data) {
        if (typeof data.detail === 'string') {
          detail = data.detail;
        } else if (Array.isArray(data.detail)) {
          detail = data.detail.map(d => d.msg || JSON.stringify(d)).join('; ');
        } else if (data.detail) {
          detail = JSON.stringify(data.detail);
        }
      }
    } catch {
      detail = res.statusText || 'Response parse failed';
    }
    return { ok: res.ok, status: res.status, data, detail };
  } catch (err) {
    return { ok: false, status: 0, data: null, detail: err.message || 'Network error' };
  }
}

/**
 * Fetch wrapper for POST requests with JSON body.
 * Returns { ok, status, data, detail } and never throws.
 */
async function apiPost(path, body) {
  try {
    const res = await fetch(`${API_BASE_URL}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    let data = null;
    let detail = '';
    try {
      data = await res.json();
      if (!res.ok && data) {
        if (typeof data.detail === 'string') {
          detail = data.detail;
        } else if (Array.isArray(data.detail)) {
          detail = data.detail.map(d => d.msg || JSON.stringify(d)).join('; ');
        } else if (data.detail) {
          detail = JSON.stringify(data.detail);
        }
      }
    } catch {
      detail = res.statusText || 'Response parse failed';
    }
    return { ok: res.ok, status: res.status, data, detail };
  } catch (err) {
    return { ok: false, status: 0, data: null, detail: err.message || 'Network error' };
  }
}

/**
 * Fetch wrapper for POST requests with FormData body (multipart/form-data).
 * Do NOT set Content-Type header so the browser sets the boundary automatically.
 */
async function apiPostFormData(path, formData) {
  try {
    const res = await fetch(`${API_BASE_URL}${path}`, {
      method: 'POST',
      body: formData
    });
    let data = null;
    let detail = '';
    try {
      data = await res.json();
      if (!res.ok && data) {
        if (typeof data.detail === 'string') {
          detail = data.detail;
        } else if (Array.isArray(data.detail)) {
          detail = data.detail.map(d => d.msg || JSON.stringify(d)).join('; ');
        } else if (data.detail) {
          detail = JSON.stringify(data.detail);
        }
      }
    } catch {
      detail = res.statusText || 'Response parse failed';
    }
    return { ok: res.ok, status: res.status, data, detail };
  } catch (err) {
    return { ok: false, status: 0, data: null, detail: err.message || 'Network error' };
  }
}

/**
 * Format ISO timestamp string to YYYY-MM-DD HH:MM via string manipulation only.
 * Never uses new Date() to avoid local timezone shifts.
 */
function formatTs(iso) {
  if (!iso || typeof iso !== 'string') return '';
  const s = iso.replace('T', ' ');
  return s.length >= 16 ? s.slice(0, 16) : s;
}

/** Render a connection status badge in the sidebar. */
function renderApiStatus() {
  let el = document.getElementById('api-status');
  if (!el) {
    el = document.createElement('div');
    el.id = 'api-status';
    el.style.cssText = 'padding:6px 12px;font-size:11px;display:flex;align-items:center;gap:6px;margin-top:auto;border-top:1px solid var(--line,#e1e9ed);color:var(--mute,#718391);';
    const sidebar = document.querySelector('.side');
    if (sidebar) sidebar.appendChild(el);
  }
  const dotColor = apiState.status === 'ok' ? '#218566'
    : apiState.status === 'degraded' ? '#ba7918' : '#d35d57';
  const label = apiState.status === 'ok' ? 'Backend connected'
    : apiState.status === 'degraded'
      ? `Backend degraded (${apiState.modelsLoaded}/${apiState.modelsExpected} models)`
      : 'Backend offline';
  el.innerHTML = `<span style="width:8px;height:8px;border-radius:50%;background:${dotColor};display:inline-block;"></span>${escapeHtml(label)}`;
}

let charts = {};

function selectElement(selector, root = document) {
  return root.querySelector(selector);
}

function selectAll(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function getCurrentCase() {
  if (!appState.caseId) return null;
  return caseCache.get(appState.caseId) || null;
}

function rangeOf(glucoseValue) {
  if (glucoseValue < 70) {
    return ['Below range', 'bad'];
  }
  if (glucoseValue > 180) {
    return ['Above range', 'warn'];
  }
  return ['In range', 'ok'];
}

function trendText() {
  const currentCase = getCurrentCase();
  if (!currentCase || !currentCase.curve || currentCase.curve.length < 4) {
    return ['—', 'No data'];
  }
  const glucoseCurve = currentCase.curve;
  const indexOfNow = glucoseCurve.findIndex(point => point.x === 0);
  if (indexOfNow < 3 || indexOfNow === -1) {
    return ['—', 'No data'];
  }
  const delta = (glucoseCurve[indexOfNow].y - glucoseCurve[indexOfNow - 3].y) / 15;
  const signPrefix = delta >= 0 ? '+' : '';

  if (delta > 1) {
    return ['↑ Rising', `${signPrefix}${delta.toFixed(1)} mg/dL/min`];
  }
  if (delta < -1) {
    return ['↓ Falling', `${signPrefix}${delta.toFixed(1)} mg/dL/min`];
  }
  return ['→ Stable', `${signPrefix}${delta.toFixed(1)} mg/dL/min`];
}

function stat(label, valueText, detail, cssClass = '') {
  return `<div class="stat"><small>${escapeHtml(label)}</small><b class="${escapeHtml(cssClass)}">${escapeHtml(valueText)}</b><span>${escapeHtml(detail)}</span></div>`;
}

function errorStatus(errorValue) {
  if (errorValue <= 15) {
    return ['Close match', 'ok'];
  }
  if (errorValue <= 30) {
    return ['Moderate deviation', 'warn'];
  }
  return ['Large deviation', 'bad'];
}

/** Build a standardized case object preserving the shape expected by renderers. */
function buildCaseObject(detail, predictData) {
  // curve: history readings with x = (index-36)*5 (so -180..0), then future readings with x = (index+1)*5 (5..120); y = glucose.
  // Nominal 5-minute grid, the same convention the backend uses for actual_at.
  const curve = [];
  (detail.history || []).forEach((point, idx) => {
    curve.push({ x: (idx - 36) * 5, y: point.glucose });
  });
  (detail.future || []).forEach((point, idx) => {
    curve.push({ x: (idx + 1) * 5, y: point.glucose });
  });

  const preds = {};
  HZ.forEach(h => {
    const predItem = (predictData.predictions || []).find(p => Number(p.horizon) === h);
    const pred = predItem ? predItem.predicted_glucose : null;
    const actual = detail.actual_at ? detail.actual_at[String(h)] : null;
    const err = (pred !== null && actual !== null) ? Math.abs(pred - actual) : null;
    preds[h] = { pred, actual, err };
  });

  const ctxData = predictData.context || {};
  const currentGlucose = ctxData.current_glucose != null ? ctxData.current_glucose : detail.glucose;
  const submittedCount = ctxData.readings_submitted != null ? ctxData.readings_submitted : (detail.history ? detail.history.length : 37);
  const windowReadings = ctxData.window_readings != null ? ctxData.window_readings : 24;
  const windowSpan = ctxData.window_span_minutes != null ? ctxData.window_span_minutes : 115;
  const windowMin = ctxData.window_min != null ? ctxData.window_min : '';
  const windowMax = ctxData.window_max != null ? ctxData.window_max : '';
  const windowMean = ctxData.window_mean != null ? ctxData.window_mean : '';
  const change5 = ctxData.change_5min != null ? ctxData.change_5min : '';
  const change30 = ctxData.change_30min != null ? ctxData.change_30min : '';
  const windowEnd = ctxData.window_end ? formatTs(ctxData.window_end) : formatTs(detail.timestamp);

  const ctx = {
    'Current glucose': [currentGlucose, 'mg/dL', 'at forecast time'],
    'CGM readings in model window': [windowReadings, 'readings', `of ${submittedCount} submitted`],
    'Model window': [windowSpan, 'min', 'CGM history used by the model'],
    'Minimum glucose': [windowMin, 'mg/dL', 'in model window'],
    'Maximum glucose': [windowMax, 'mg/dL', 'in model window'],
    'Mean glucose': [windowMean, 'mg/dL', 'in model window'],
    'Recent 5-minute change': [change5, 'mg/dL', 'from t-5 to now'],
    'Recent 30-minute change': [change30, 'mg/dL', 'from t-30 to now'],
    'Forecast timestamp': [windowEnd, '', '']
  };

  return {
    id: detail.case_id,
    patient: String(detail.patient_id),
    ts: detail.timestamp,
    cur: detail.glucose,
    curve,
    preds,
    ctx,
    rawCtx: ctxData,
    events: EVENTS,
    history: detail.history,
    future: detail.future,
    actual_at: detail.actual_at
  };
}

/** Update an existing case object with fresh prediction and context data. */
function updateCasePredictions(caseObj, predictData) {
  HZ.forEach(h => {
    const predItem = (predictData.predictions || []).find(p => Number(p.horizon) === h);
    if (predItem && caseObj.preds[h]) {
      caseObj.preds[h].pred = predItem.predicted_glucose;
      if (caseObj.preds[h].actual !== null && caseObj.preds[h].actual !== undefined) {
        caseObj.preds[h].err = Math.abs(caseObj.preds[h].pred - caseObj.preds[h].actual);
      }
    }
  });

  const ctxData = predictData.context || {};
  caseObj.rawCtx = ctxData;
  if (ctxData.current_glucose != null) {
    caseObj.ctx['Current glucose'] = [ctxData.current_glucose, 'mg/dL', 'at forecast time'];
  }
  if (ctxData.window_readings != null) {
    caseObj.ctx['CGM readings in model window'] = [ctxData.window_readings, 'readings', `of ${ctxData.readings_submitted} submitted`];
  }
  if (ctxData.window_span_minutes != null) {
    caseObj.ctx['Model window'] = [ctxData.window_span_minutes, 'min', 'CGM history used by the model'];
  }
  if (ctxData.window_min != null) {
    caseObj.ctx['Minimum glucose'] = [ctxData.window_min, 'mg/dL', 'in model window'];
  }
  if (ctxData.window_max != null) {
    caseObj.ctx['Maximum glucose'] = [ctxData.window_max, 'mg/dL', 'in model window'];
  }
  if (ctxData.window_mean != null) {
    caseObj.ctx['Mean glucose'] = [ctxData.window_mean, 'mg/dL', 'in model window'];
  }
  if (ctxData.change_5min != null) {
    caseObj.ctx['Recent 5-minute change'] = [ctxData.change_5min, 'mg/dL', 'from t-5 to now'];
  }
  if (ctxData.change_30min != null) {
    caseObj.ctx['Recent 30-minute change'] = [ctxData.change_30min, 'mg/dL', 'from t-30 to now'];
  }
  if (ctxData.window_end) {
    caseObj.ctx['Forecast timestamp'] = [formatTs(ctxData.window_end), '', ''];
  }
}

/**
 * Fetch case list from /v1/cases?per_patient=3.
 * Populates caseList and patientList or sets uiState on failure.
 */
async function loadCaseList() {
  const res = await apiGet('/v1/cases?per_patient=3');
  if (!res.ok || !res.data) {
    if (res.status === 503) {
      uiState.phase = 'no_data';
      uiState.detail = res.detail || 'Test data not configured on the server';
    } else {
      uiState.phase = 'error';
      uiState.detail = res.detail || 'Failed to fetch case list';
    }
    return false;
  }

  const data = res.data;
  patientList = (data.patients || []).map(p => String(p.patient_id));
  caseList = (data.cases || []).map(c => ({
    id: c.case_id,
    patient: String(c.patient_id),
    ts: c.timestamp,
    cur: c.glucose
  }));
  return true;
}

/**
 * Load a case by case_id.
 * Returns cached object, or fetches from /v1/cases/{id} and POST /v1/predict.
 * Implements a stale-response guard to avoid overwriting newer selections.
 */
async function loadCase(id) {
  if (!id) return null;
  if (caseCache.has(id)) {
    return caseCache.get(id);
  }
  currentLoadCaseId = id;

  const caseRes = await apiGet(`/v1/cases/${encodeURIComponent(id)}`);
  if (currentLoadCaseId !== id) return null; // Stale-response guard

  if (!caseRes.ok || !caseRes.data) {
    uiState.phase = 'error';
    uiState.detail = caseRes.detail || 'Failed to fetch case detail';
    renderAll();
    return null;
  }

  const detail = caseRes.data;

  // Predict using historical CGM readings exactly as received
  const predRes = await apiPost('/v1/predict', {
    readings: detail.history,
    horizon: 'all'
  });
  if (currentLoadCaseId !== id) return null; // Stale-response guard

  if (!predRes.ok || !predRes.data) {
    uiState.phase = 'error';
    uiState.detail = predRes.detail || 'Failed to fetch predictions';
    renderAll();
    return null;
  }

  const caseObj = buildCaseObject(detail, predRes.data);
  caseCache.set(id, caseObj);
  return caseObj;
}

/** Helper to select and load a case by ID. */
async function selectCaseById(id) {
  currentLoadCaseId = id;
  appState.caseId = id;
  const c = await loadCase(id);
  if (currentLoadCaseId !== id) {
    return;
  }
  if (c) {
    uiState.phase = 'ready';
  }
  renderAll();
}

/**
 * Load /v1/metrics once and cache it for the Performance view.
 * Performance view works independently of case data configuration.
 */
async function loadMetrics() {
  perfState.phase = 'loading';
  renderPerformance();

  const res = await apiGet('/v1/metrics');
  if (!res.ok || !res.data) {
    if (res.status === 0) {
      perfState.phase = 'offline';
      perfState.detail = 'Start the API (see README) and reload this page';
    } else {
      perfState.phase = 'unavailable';
      perfState.detail = res.detail || 'Metrics unavailable';
    }
    renderPerformance();
    return false;
  }

  perfState.phase = 'ready';
  perfState.data = res.data;
  renderPerformance();
  return true;
}

/**
 * Load backtest for a specific forecast horizon.
 * Cached per horizon. Uses a stale-response guard so older requests don't overwrite newer ones.
 */
async function loadBacktest(horizon) {
  const h = Number(horizon);
  if (backtestCache.has(h)) {
    clinicalState.phase = 'ready';
    clinicalState.data = backtestCache.get(h);
    clinicalState.horizon = h;
    renderClinical();
    return clinicalState.data;
  }

  currentBacktestHorizon = h;
  clinicalState.phase = 'loading';
  clinicalState.horizon = h;
  renderClinical();

  const fd = new FormData();
  fd.append('horizon', String(h));
  fd.append('max_points', '2000');

  const res = await apiPostFormData('/v1/backtest', fd);
  if (currentBacktestHorizon !== h) return null; // Stale-response guard

  if (!res.ok || !res.data) {
    if (res.status === 0) {
      clinicalState.phase = 'offline';
      clinicalState.detail = 'Start the API (see README) and reload this page';
    } else if (res.status === 422 || res.status === 503) {
      const d = String(res.detail || '');
      if (d.includes('TEST_CSV_PATH') || d.includes('test data') || d.includes('Server test data')) {
        clinicalState.phase = 'no_data';
        clinicalState.detail = res.detail;
      } else if (apiState.modelsLoaded < apiState.modelsExpected) {
        clinicalState.phase = 'degraded';
        clinicalState.detail = 'Train or provide the model files, then restart the API';
      } else {
        clinicalState.phase = 'error';
        clinicalState.detail = res.detail || 'Failed to fetch clinical backtest';
      }
    } else if (apiState.modelsLoaded < apiState.modelsExpected) {
      clinicalState.phase = 'degraded';
      clinicalState.detail = 'Train or provide the model files, then restart the API';
    } else {
      clinicalState.phase = 'error';
      clinicalState.detail = res.detail || 'Failed to fetch clinical backtest';
    }
    renderClinical();
    return null;
  }

  backtestCache.set(h, res.data);
  clinicalState.phase = 'ready';
  clinicalState.data = res.data;
  clinicalState.horizon = h;
  renderClinical();
  return res.data;
}

/** Check if the Clinical view section is currently active. */
function isClinicalActive() {
  const el = document.getElementById('v-clinical');
  return el ? el.classList.contains('active') : false;
}

/**
 * Bootstrap application on startup:
 * 1. Check /health
 * 2. Load metrics for Performance view
 * 3. Fetch case list and initial case
 * 4. Load backtest if Clinical view is active
 * 5. Render views
 */
async function bootstrap() {
  uiState.phase = 'loading';
  uiState.detail = 'Fetching test cases from the backend';
  renderAll();

  const healthRes = await apiGet('/health');
  if (!healthRes.ok || !healthRes.data) {
    uiState.phase = 'offline';
    uiState.detail = 'Start the API (see README) and reload this page';
    perfState.phase = 'offline';
    perfState.detail = 'Start the API (see README) and reload this page';
    clinicalState.phase = 'offline';
    clinicalState.detail = 'Start the API (see README) and reload this page';
    apiState.connected = false;
    apiState.status = 'offline';
    renderApiStatus();
    renderAll();
    renderPerformance();
    renderClinical();
    return;
  }

  const hData = healthRes.data;
  apiState.connected = true;
  apiState.status = hData.status;
  apiState.modelsLoaded = hData.models_loaded;
  apiState.modelsExpected = hData.models_expected;

  // Load metrics independently (Performance view works even if test data is not configured)
  await loadMetrics();

  if (hData.models_loaded < hData.models_expected) {
    uiState.phase = 'degraded';
    uiState.detail = 'Train or provide the model files, then restart the API';
    clinicalState.phase = 'degraded';
    clinicalState.detail = 'Train or provide the model files, then restart the API';
    renderApiStatus();
    renderAll();
    renderClinical();
    return;
  }

  const listOk = await loadCaseList();
  if (!listOk || uiState.phase !== 'loading') {
    clinicalState.phase = uiState.phase;
    clinicalState.detail = uiState.detail;
    renderApiStatus();
    renderAll();
    renderClinical();
    return;
  }

  if (caseList.length > 0) {
    appState.caseId = caseList[0].id;
    const firstCase = await loadCase(appState.caseId);
    if (firstCase) {
      uiState.phase = 'ready';
    }
  } else {
    uiState.phase = 'no_data';
    uiState.detail = 'No test cases returned by the server';
  }

  renderApiStatus();
  renderAll();

  if (isClinicalActive()) {
    await loadBacktest(appState.horizon);
  }
}

/** Render honest unavailable state across Overview, Forecast, and Explorer views. */
function renderUnavailable() {
  let label = 'Request failed';
  let value = 'Error';
  let detail = uiState.detail ? String(uiState.detail).slice(0, 200) : 'An error occurred';
  let cssClass = 'bad';

  if (uiState.phase === 'loading') {
    label = 'Loading';
    value = '…';
    detail = 'Fetching test cases from the backend';
    cssClass = '';
  } else if (uiState.phase === 'offline') {
    label = 'Backend offline';
    value = 'Unavailable';
    detail = 'Start the API (see README) and reload this page';
    cssClass = 'bad';
  } else if (uiState.phase === 'degraded') {
    label = 'Models not loaded';
    value = `${apiState.modelsLoaded || 0}/${apiState.modelsExpected || 4}`;
    detail = 'Train or provide the model files, then restart the API';
    cssClass = 'warn';
  } else if (uiState.phase === 'no_data') {
    label = 'No test data configured';
    value = 'Unavailable';
    detail = uiState.detail || 'Test data is not available on the server';
    cssClass = 'warn';
  } else if (uiState.phase === 'error') {
    label = 'Request failed';
    value = 'Error';
    detail = uiState.detail ? String(uiState.detail).slice(0, 200) : 'An error occurred';
    cssClass = 'bad';
  }

  const summaryEl = selectElement('#summary');
  if (summaryEl) {
    summaryEl.innerHTML = stat(label, value, detail, cssClass);
  }

  const errTableEl = selectElement('#errTable');
  if (errTableEl) {
    errTableEl.innerHTML = `<tr><td colspan="5">${stat(label, value, detail, cssClass)}</td></tr>`;
  }

  const resultsEl = selectElement('#results');
  if (resultsEl) {
    resultsEl.innerHTML = stat(label, value, detail, cssClass);
  }

  const emptyIds = ['#fcards', '#riskSummary', '#riskTrajectory', '#riskAlert', '#interp', '#ctx', '#events'];
  emptyIds.forEach(id => {
    const el = selectElement(id);
    if (el) el.innerHTML = '';
  });

  if (charts.cMain) {
    charts.cMain.data.datasets.forEach(d => { d.data = []; });
    charts.cMain.update();
  }
  if (charts.cDetail) {
    charts.cDetail.data.datasets.forEach(d => { d.data = []; });
    charts.cDetail.update();
  }

  selectAll('.sel-patient').forEach(el => { el.innerHTML = ''; });
  selectAll('.sel-case').forEach(el => { el.innerHTML = ''; });
}

let nowPlugin = {
  id: 'now',
  beforeDatasetsDraw(chart) {
    const chartArea = chart.chartArea;
    const xScale = chart.scales.x;
    const xZeroPosition = xScale.getPixelForValue(0);
    const context = chart.ctx;

    context.save();
    context.fillStyle = 'rgba(8,127,140,.06)';
    context.fillRect(xZeroPosition, chartArea.top, chartArea.right - xZeroPosition, chartArea.bottom - chartArea.top);

    context.strokeStyle = COLORS.mute;
    context.setLineDash([4, 4]);
    context.beginPath();
    context.moveTo(xZeroPosition, chartArea.top);
    context.lineTo(xZeroPosition, chartArea.bottom);
    context.stroke();

    context.setLineDash([]);
    context.fillStyle = COLORS.mute;
    context.font = '11px IBM Plex Sans';
    context.fillText('Now', xZeroPosition + 5, chartArea.top + 12);
    context.fillText('History', chartArea.left + 6, chartArea.top + 12);
    context.fillText('Future', chartArea.right - 40, chartArea.top + 12);
    context.restore();
  }
};

function buildAxisConfig(titleText = '') {
  return {
    grid: { color: COLORS.grid },
    ticks: { color: COLORS.mute },
    title: {
      display: Boolean(titleText),
      text: titleText,
      color: COLORS.mute
    }
  };
}

function createLineDataset(label, color, extraOptions = {}) {
  const baseDataset = {
    label,
    borderColor: color,
    backgroundColor: color,
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.3,
    data: []
  };

  return { ...baseDataset, ...extraOptions };
}

function createLineChart(chartId) {
  const chartConfig = {
    type: 'line',
    plugins: [nowPlugin],
    data: {
      datasets: [
        createLineDataset('Historical glucose', COLORS.blue),
        createLineDataset('Actual future (test case)', COLORS.ok),
        createLineDataset('Forecast', COLORS.accent, {
          borderDash: [6, 4],
          pointRadius: 5,
          pointHoverRadius: 7
        })
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 350 },
      interaction: { mode: 'nearest', intersect: false },
      scales: {
        x: {
          type: 'linear',
          min: -180,
          max: 120,
          ...buildAxisConfig('Minutes from forecast time'),
          ticks: {
            color: COLORS.mute,
            stepSize: 30
          }
        },
        y: {
          min: 40,
          max: 360,
          ...buildAxisConfig('Glucose (mg/dL)')
        }
      },
      plugins: {
        legend: {
          labels: {
            color: COLORS.mute,
            boxWidth: 12
          }
        },
        tooltip: {
          callbacks: {
            label(context) {
              const xValue = context.parsed.x;
              const yValue = context.parsed.y;
              const offsetText = xValue >= 0 ? '+' : '';
              return `${context.dataset.label}: ${yValue} mg/dL (t ${offsetText}${xValue} min)`;
            }
          }
        }
      }
    }
  };

  charts[chartId] = new Chart(selectElement(`#${chartId}`), chartConfig);
}

function setLine(chartId) {
  const currentCase = getCurrentCase();
  const chart = charts[chartId];
  if (!chart || !currentCase) return;

  chart.data.datasets[0].data = currentCase.curve.filter(point => point.x <= 0);
  chart.data.datasets[1].data = currentCase.curve.filter(point => point.x >= 0);
  chart.data.datasets[2].data = [{ x: 0, y: Math.round(currentCase.cur) }].concat(
    HZ.map(hour => ({ x: hour, y: Math.round(currentCase.preds[hour].pred) }))
  );

  chart.data.datasets[2].pointRadius = function(context) {
    if (context.dataIndex === 0) {
      return 0;
    }
    if (HZ[context.dataIndex - 1] === appState.horizon) {
      return 8;
    }
    return 4;
  };

  chart.update();
}

function thresholdState(glucoseValue) {
  if (glucoseValue < 70) {
    return { label: 'Below range', className: 'bad' };
  }
  if (glucoseValue > 180) {
    return { label: 'Above range', className: 'warn' };
  }
  return { label: 'In range', className: 'ok' };
}

function renderRiskAlerts() {
  const currentCase = getCurrentCase();
  if (!currentCase) return;
  const currentStatus = thresholdState(currentCase.cur);

  const trajectoryItems = HZ.map(hour => {
    const forecastValue = Math.round(currentCase.preds[hour].pred);
    const state = thresholdState(forecastValue);
    return `<div style="display:flex; align-items:center; justify-content:space-between; gap:10px; padding:8px 0; border-bottom:1px solid var(--line);"><span>${hour} min — ${forecastValue} mg/dL</span><span class="${escapeHtml(state.className)}" style="font-weight:600;">${escapeHtml(state.label)}</span></div>`;
  }).join('');

  const earliestPrediction = HZ.map(hour => ({
    h: hour,
    value: Math.round(currentCase.preds[hour].pred)
  })).find(item => item.value < 70 || item.value > 180);

  let alertHtml = 'No threshold excursion predicted within the selected forecast horizon.';

  if (earliestPrediction) {
    const excursionType = earliestPrediction.value < 70
      ? 'Potential hypoglycemic excursion'
      : 'Potential hyperglycemic excursion';

    alertHtml = `<strong>${escapeHtml(excursionType)}</strong> — earliest horizon: ${earliestPrediction.h} min — predicted glucose: ${earliestPrediction.value} mg/dL`;
  }

  const summaryHtml = stat('Current glucose', `${Math.round(currentCase.cur)} mg/dL`, currentStatus.label, currentStatus.className)
    + stat('Current status', currentStatus.label, 'Threshold band', currentStatus.className)
    + stat('Selected patient', `Patient ${currentCase.patient}`, 'OhioT1DM')
    + stat('Selected horizon', `${appState.horizon} min`, 'Current forecast setting');

  selectElement('#riskSummary').innerHTML = summaryHtml;
  selectElement('#riskTrajectory').innerHTML = trajectoryItems;
  selectElement('#riskAlert').innerHTML = `<div class="stat" style="padding:14px 16px;"><small>Earliest predicted excursion</small><b>${alertHtml}</b></div>`;
}

function renderOverview() {
  const currentCase = getCurrentCase();
  if (!currentCase) return;
  const currentForecast = currentCase.preds[appState.horizon];
  if (!currentForecast) return;
  const [currentRangeLabel, currentRangeClass] = rangeOf(currentCase.cur);
  const [trendLabel, trendValue] = trendText();

  const summaryHtml = stat('Current Glucose', `${Math.round(currentCase.cur)} mg/dL`, currentRangeLabel, currentRangeClass)
    + stat('Current Trend', trendLabel, trendValue)
    + stat('Selected Patient', `Patient ${currentCase.patient}`, 'OhioT1DM')
    + stat('Forecast Horizon', `${appState.horizon} min`, 'CGM-only forecast');

  selectElement('#summary').innerHTML = summaryHtml;

  const futureCardsHtml = HZ.map(hour => {
    const forecastItem = currentCase.preds[hour];
    const selectedClass = hour === appState.horizon ? 'sel' : '';
    const predRounded = Math.round(forecastItem.pred);
    const actualRounded = Math.round(forecastItem.actual);
    const errRounded = Math.round(forecastItem.err);
    return `<div class="stat ${escapeHtml(selectedClass)}"><small>${hour} min forecast</small><b>${predRounded} mg/dL</b><span>Actual ${actualRounded} · error ${errRounded}</span></div>`;
  }).join('');

  selectElement('#fcards').innerHTML = futureCardsHtml;

  const [predictionRangeLabel] = rangeOf(currentForecast.pred);
  const [forecastErrorText] = errorStatus(currentForecast.err);

  selectElement('#interp').innerHTML = `At ${appState.horizon} min the model projects <b>${Math.round(currentForecast.pred)} mg/dL</b> (${escapeHtml(predictionRangeLabel.toLowerCase())}); the held-out value was <b>${Math.round(currentForecast.actual)} mg/dL</b>, an absolute error of ${Math.round(currentForecast.err)} mg/dL (${escapeHtml(forecastErrorText.toLowerCase())}). This CGM-only replay uses the OhioT1DM cohort with 6 patients, 5-minute sampling, and 30/60/90/120 min forecasting. It is a research prototype and not treatment guidance.`;

  renderRiskAlerts();
  setLine('cMain');
}

function renderForecast() {
  const currentCase = getCurrentCase();
  if (!currentCase) return;
  let tableHtml = '<tr><th>Horizon</th><th>Current</th><th>Predicted</th><th>Actual</th><th>Absolute error</th></tr>';

  HZ.forEach(hour => {
    const forecastItem = currentCase.preds[hour];
    const [errorText, errorClass] = errorStatus(forecastItem.err);
    tableHtml += `<tr><td>${hour} min</td><td>${Math.round(currentCase.cur)}</td><td>${Math.round(forecastItem.pred)}</td><td>${Math.round(forecastItem.actual)}</td><td class="${escapeHtml(errorClass)}">${Math.round(forecastItem.err)} mg/dL</td></tr>`;
  });

  selectElement('#errTable').innerHTML = tableHtml;
  setLine('cDetail');
}

function renderExplorer() {
  const currentCase = getCurrentCase();
  if (!currentCase) return;
  const selectedForecast = currentCase.preds[appState.horizon];
  if (!selectedForecast) return;
  const [errorText, errorClass] = errorStatus(selectedForecast.err);

  selectElement('#selH').value = String(appState.horizon);

  // Explorer results: replace 'History Window' by 'Model window'
  // showing ${window_readings} readings with detail ${window_min}–${window_max} mg/dL · about ${window_span_minutes} min (from context)
  const ctx = currentCase.rawCtx || {};
  const modelWindowVal = ctx.window_readings ? `${ctx.window_readings} readings` : '24 readings';
  const modelWindowDetail = (ctx.window_min != null && ctx.window_max != null && ctx.window_span_minutes != null)
    ? `${ctx.window_min}–${ctx.window_max} mg/dL · about ${ctx.window_span_minutes} min`
    : 'in model window';

  const resultsHtml = stat('Model window', modelWindowVal, modelWindowDetail)
    + stat('Prediction', `${Math.round(selectedForecast.pred)} mg/dL`, `${appState.horizon} min ahead`)
    + stat('Actual Value', `${Math.round(selectedForecast.actual)} mg/dL`, 'Held-out')
    + stat('Absolute Error', `${Math.round(selectedForecast.err)} mg/dL`, '')
    + stat('Forecast Status', errorText, 'Error ≤15 close, ≤30 moderate', errorClass);

  selectElement('#results').innerHTML = resultsHtml;

  const contextHtml = Object.entries(currentCase.ctx).map(([key, details]) => {
    const value = details[0];
    const unit = details[1];
    const note = details[2];
    const displayValue = `${value}${unit ? ` ${unit}` : ''}`;

    return stat(key, displayValue, note || '');
  }).join('');

  selectElement('#ctx').innerHTML = contextHtml;
  selectElement('#events').innerHTML = currentCase.events.map(eventText => `<li>${escapeHtml(eventText)}</li>`).join('');
}

function renderClinicalUnavailable() {
  let label = 'Request failed';
  let value = 'Error';
  let detail = clinicalState.detail ? String(clinicalState.detail).slice(0, 200) : 'An error occurred';
  let cssClass = 'bad';

  if (clinicalState.phase === 'loading') {
    label = 'Loading';
    value = '…';
    detail = 'Fetching clinical backtest from the backend';
    cssClass = '';
  } else if (clinicalState.phase === 'offline') {
    label = 'Backend offline';
    value = 'Unavailable';
    detail = 'Start the API (see README) and reload this page';
    cssClass = 'bad';
  } else if (clinicalState.phase === 'degraded') {
    label = 'Models not loaded';
    value = `${apiState.modelsLoaded || 0}/${apiState.modelsExpected || 4}`;
    detail = 'Train or provide the model files, then restart the API';
    cssClass = 'warn';
  } else if (clinicalState.phase === 'no_data') {
    label = 'No test data configured';
    value = 'Unavailable';
    detail = clinicalState.detail || 'Test data is not available on the server';
    cssClass = 'warn';
  } else if (clinicalState.phase === 'error') {
    label = 'Request failed';
    value = 'Error';
    detail = clinicalState.detail ? String(clinicalState.detail).slice(0, 200) : 'An error occurred';
    cssClass = 'bad';
  }

  const rangeEl = selectElement('#range');
  if (rangeEl) rangeEl.innerHTML = '';
  const rangeNoteEl = selectElement('#rangeNote');
  if (rangeNoteEl) rangeNoteEl.textContent = '';

  const zonesEl = selectElement('#zones');
  if (zonesEl) {
    zonesEl.innerHTML = stat(label, value, detail, cssClass);
  }
  const zonesNoteEl = selectElement('#zonesNote');
  if (zonesNoteEl) zonesNoteEl.textContent = '';

  if (charts.cClarke) {
    charts.cClarke.data.datasets[0].data = [];
    charts.cClarke.data.datasets[1].data = [];
    charts.cClarke.update();
  }
}

function positionOnRangeScale(glucoseValue) {
  return clamp((glucoseValue - 40) / 360 * 100, 0, 100);
}

function makeMarker(value, labelText, color, markerTop) {
  return `<span class="mk" style="left:${positionOnRangeScale(value)}%;color:${color};top:${markerTop}px">${escapeHtml(labelText)} ${value}</span>`;
}

function renderClinical() {
  if (clinicalState.phase !== 'ready' || !clinicalState.data) {
    renderClinicalUnavailable();
    return;
  }

  const backtest = clinicalState.data;
  const currentCase = getCurrentCase();

  // Target range bar (#range)
  const rangeEl = selectElement('#range');
  const rangeNoteEl = selectElement('#rangeNote');
  if (currentCase && currentCase.preds && currentCase.preds[appState.horizon]) {
    const selectedForecast = currentCase.preds[appState.horizon];
    const rangeBarHtml = `<div class="rbar"><i style="width:${positionOnRangeScale(70)}%;background:${COLORS.bad}"></i><i style="width:${positionOnRangeScale(180) - positionOnRangeScale(70)}%;background:${COLORS.ok}"></i><i style="flex:1;background:${COLORS.warn}"></i>${makeMarker(Math.round(currentCase.cur), 'Now', '#fff', -26)}${makeMarker(Math.round(selectedForecast.pred), 'Pred', COLORS.accent, -26)}</div>`;
    if (rangeEl) rangeEl.innerHTML = rangeBarHtml;
    if (rangeNoteEl) rangeNoteEl.textContent = `Clarke Error Grid Analysis is used to evaluate clinical acceptability of prediction errors. This research prototype does not provide treatment or insulin recommendations. For the selected horizon, ${appState.horizon} min, clinically safe A+B = ${Number(backtest.a_plus_b_pct).toFixed(2)}%.`;
  } else {
    if (rangeEl) rangeEl.innerHTML = '';
    if (rangeNoteEl) rangeNoteEl.textContent = '';
  }

  // Clarke error grid scatter chart (cClarke)
  const clarkeChart = charts.cClarke;
  if (clarkeChart) {
    // Dataset 0: Test-set sample from POST /v1/backtest
    clarkeChart.data.datasets[0].label = 'Test-set sample';
    const samplePoints = (backtest.actual || []).map((act, i) => ({
      x: act,
      y: backtest.predicted[i],
      z: backtest.zones[i]
    }));
    clarkeChart.data.datasets[0].data = samplePoints;
    const sampleColors = (backtest.zones || []).map(z => ZONE_COLORS[z] || COLORS.mute);
    clarkeChart.data.datasets[0].backgroundColor = sampleColors;
    clarkeChart.data.datasets[0].pointBackgroundColor = sampleColors;

    // Dataset 1: Selected case 4 horizon predictions vs actuals
    clarkeChart.data.datasets[1].label = 'Selected case';
    if (currentCase && currentCase.preds) {
      clarkeChart.data.datasets[1].data = HZ.map(h => ({
        x: Math.round(currentCase.preds[h].actual),
        y: Math.round(currentCase.preds[h].pred)
      }));
    } else {
      clarkeChart.data.datasets[1].data = [];
    }
    clarkeChart.update();
  }

  // Zone distribution breakdown (#zones)
  const zonesEl = selectElement('#zones');
  if (zonesEl && backtest.zone_percentages) {
    const zoneRowsHtml = Object.entries(backtest.zone_percentages).map(([zoneName, zonePercentage]) => {
      const pct = Number(zonePercentage);
      const color = ZONE_COLORS[zoneName] || COLORS.mute;
      return `<div class="zrow"><b>${escapeHtml(zoneName)}</b><div><i style="width:${pct}%;background:${color}"></i></div><span>${pct.toFixed(2)}%</span></div>`;
    }).join('');
    zonesEl.innerHTML = zoneRowsHtml;
  }

  // Zone note paragraph (#zonesNote)
  const zonesNoteEl = selectElement('#zonesNote');
  if (zonesNoteEl) {
    let noteText = `Zones A and B are generally treated as clinically acceptable in the literature. Points are a deterministic sample across all test patients; zone letters come from the backend's Clarke implementation. Dashed lines: perfect agreement, ±20 %, and the 70 and 180 mg/dL thresholds. Evaluated ${backtest.n_rows} rows (A+B = ${Number(backtest.a_plus_b_pct).toFixed(2)}%).`;
    if (currentCase && backtest.per_patient && backtest.per_patient[String(currentCase.patient)]) {
      const pData = backtest.per_patient[String(currentCase.patient)];
      noteText += ` Patient ${currentCase.patient} at this horizon: RMSE ${Number(pData.rmse).toFixed(2)} mg/dL, A+B ${Number(pData.a_plus_b_pct).toFixed(2)} %, ${pData.n_rows} rows evaluated.`;
    }
    zonesNoteEl.textContent = noteText;
  }
}

function renderPerformanceUnavailable() {
  let label = 'Request failed';
  let value = 'Error';
  let detail = perfState.detail ? String(perfState.detail).slice(0, 200) : 'An error occurred';
  let cssClass = 'bad';

  if (perfState.phase === 'loading') {
    label = 'Loading';
    value = '…';
    detail = 'Fetching model metrics from the backend';
    cssClass = '';
  } else if (perfState.phase === 'offline') {
    label = 'Backend offline';
    value = 'Unavailable';
    detail = 'Start the API (see README) and reload this page';
    cssClass = 'bad';
  } else if (perfState.phase === 'unavailable') {
    label = 'Metrics unavailable';
    value = 'Unavailable';
    detail = perfState.detail || 'Metrics file is missing or unreadable on the server';
    cssClass = 'warn';
  }

  const perfCardsEl = selectElement('#perfCards');
  if (perfCardsEl) {
    perfCardsEl.innerHTML = stat(label, value, detail, cssClass);
  }

  const hTableEl = selectElement('#hTable');
  if (hTableEl) hTableEl.innerHTML = '';

  const mTableEl = selectElement('#mTable');
  if (mTableEl) mTableEl.innerHTML = '';

  if (charts.cErr) {
    charts.cErr.data.datasets.forEach(d => { d.data = []; });
    charts.cErr.update();
  }
}

function renderPerformance() {
  if (perfState.phase !== 'ready' || !perfState.data || !perfState.data.metrics) {
    renderPerformanceUnavailable();
    return;
  }

  const data = perfState.data;
  const metrics = data.metrics;

  // 1. #perfCards: four cards, one per horizon
  const perfCardsHtml = HZ.map(h => {
    const overall = (metrics.horizons && metrics.horizons[String(h)]) ? metrics.horizons[String(h)].overall : {};
    const rmse = Number(overall.rmse != null ? overall.rmse : 0).toFixed(2);
    const mae = Number(overall.mae != null ? overall.mae : 0).toFixed(2);
    const ab = Number(overall.a_plus_b_pct != null ? overall.a_plus_b_pct : 0).toFixed(2);
    return stat(`${h} min RMSE`, `${rmse} mg/dL`, `MAE ${mae} · Clarke A+B ${ab}%`);
  }).join('');

  selectElement('#perfCards').innerHTML = perfCardsHtml;

  // 2. #hTable: header Horizon | RMSE | MAE | Clinically safe (A+B)
  let horizonTableHtml = '<tr><th>Horizon</th><th>RMSE</th><th>MAE</th><th>Clinically safe (A+B)</th></tr>';
  HZ.forEach(h => {
    const overall = (metrics.horizons && metrics.horizons[String(h)]) ? metrics.horizons[String(h)].overall : {};
    const rmse = Number(overall.rmse != null ? overall.rmse : 0).toFixed(2);
    const mae = Number(overall.mae != null ? overall.mae : 0).toFixed(2);
    const ab = Number(overall.a_plus_b_pct != null ? overall.a_plus_b_pct : 0).toFixed(2);
    horizonTableHtml += `<tr><td>${h} min</td><td>${rmse} mg/dL</td><td>${mae} mg/dL</td><td>${ab}%</td></tr>`;
  });
  selectElement('#hTable').innerHTML = horizonTableHtml;

  // 3. Chart cErr: Error vs horizon
  const cErrChart = charts.cErr;
  if (cErrChart && metrics.horizons && metrics.baselines) {
    const xgbRmse = HZ.map(h => (metrics.horizons[String(h)] ? metrics.horizons[String(h)].overall.rmse : 0));
    const persRmse = HZ.map(h => (metrics.baselines.persistence && metrics.baselines.persistence[String(h)] ? metrics.baselines.persistence[String(h)].rmse : 0));
    const ridgeRmse = HZ.map(h => (metrics.baselines.ridge && metrics.baselines.ridge[String(h)] ? metrics.baselines.ridge[String(h)].rmse : 0));

    cErrChart.data.labels = HZ.map(hour => `${hour} min`);
    cErrChart.data.datasets = [
      {
        label: 'XGBoost',
        data: xgbRmse,
        borderColor: COLORS.accent,
        backgroundColor: COLORS.accent,
        tension: 0.3
      },
      {
        label: 'Persistence',
        data: persRmse,
        borderColor: COLORS.mute,
        backgroundColor: COLORS.mute,
        tension: 0.3
      },
      {
        label: 'Ridge',
        data: ridgeRmse,
        borderColor: COLORS.blue,
        backgroundColor: COLORS.blue,
        tension: 0.3
      }
    ];

    // Dynamic y-axis max: ceil(maxValue / 10) * 10 + 10
    const allVals = [...xgbRmse, ...persRmse, ...ridgeRmse].filter(v => typeof v === 'number' && !isNaN(v));
    const maxVal = allVals.length > 0 ? Math.max(...allVals) : 50;
    const yMax = Math.ceil(maxVal / 10) * 10 + 10;
    cErrChart.options.scales.y.max = yMax;
    cErrChart.update();
  }

  // 4. #mTable: Model comparison
  let modelTableHtml = '<tr><th>Model</th><th>RMSE (30/60/90/120 min)</th><th>Clinically safe A+B % (30/60/90/120 min)</th></tr>';

  // Rows: XGBoost, Persistence, Ridge, Lasso
  const modelsList = [
    { name: 'XGBoost', getH: h => metrics.horizons[String(h)].overall },
    { name: 'Persistence', getH: h => (metrics.baselines.persistence ? metrics.baselines.persistence[String(h)] : {}) },
    { name: 'Ridge', getH: h => (metrics.baselines.ridge ? metrics.baselines.ridge[String(h)] : {}) },
    { name: 'Lasso', getH: h => (metrics.baselines.lasso ? metrics.baselines.lasso[String(h)] : {}) }
  ];

  modelsList.forEach(m => {
    const rmseVals = HZ.map(h => Number(m.getH(h).rmse || 0).toFixed(2)).join(' / ');
    const abVals = HZ.map(h => Number(m.getH(h).a_plus_b_pct || 0).toFixed(2)).join(' / ');
    modelTableHtml += `<tr><td>${escapeHtml(m.name)}</td><td>${rmseVals}</td><td>${abVals}</td></tr>`;
  });

  // Generated summary sentence row from data
  // Check lowest RMSE and MAE
  const lowestRmseHz = HZ.filter(h => {
    const m = metrics.horizons[String(h)].overall.rmse;
    return m <= (metrics.baselines.persistence[String(h)].rmse || 999) &&
           m <= (metrics.baselines.ridge[String(h)].rmse || 999) &&
           m <= (metrics.baselines.lasso[String(h)].rmse || 999);
  });

  const lowestMaeHz = HZ.filter(h => {
    const m = metrics.horizons[String(h)].overall.mae;
    return m <= (metrics.baselines.persistence[String(h)].mae || 999) &&
           m <= (metrics.baselines.ridge[String(h)].mae || 999) &&
           m <= (metrics.baselines.lasso[String(h)].mae || 999);
  });

  const rmsePhrase = lowestRmseHz.length === HZ.length ? 'all horizons' : lowestRmseHz.map(h => `${h} min`).join(', ');
  const maePhrase = lowestMaeHz.length === HZ.length ? 'all horizons' : lowestMaeHz.map(h => `${h} min`).join(', ');

  const ridgeMargins = HZ.map(h => Number((metrics.baselines.ridge[String(h)].rmse - metrics.horizons[String(h)].overall.rmse).toFixed(2)));
  const minMargin = Math.min(...ridgeMargins).toFixed(2);
  const maxMargin = Math.max(...ridgeMargins).toFixed(2);

  const beatingBaselines = [];
  ['Persistence', 'Ridge', 'Lasso'].forEach(b => {
    const key = b.toLowerCase();
    if (metrics.baselines[key]) {
      const beatingHz = HZ.filter(h => metrics.baselines[key][String(h)].a_plus_b_pct > metrics.horizons[String(h)].overall.a_plus_b_pct);
      if (beatingHz.length > 0) {
        beatingBaselines.push(`${b} at ${beatingHz.map(h => `${h} min`).join(', ')}`);
      }
    }
  });
  const beatingPhrase = beatingBaselines.length > 0 ? beatingBaselines.join('; ') : 'none';

  modelTableHtml += `<tr><td colspan="3"><strong>Summary:</strong> XGBoost achieves the lowest RMSE across ${rmsePhrase} and lowest MAE across ${maePhrase}, with an RMSE margin over Ridge ranging from ${minMargin} mg/dL to ${maxMargin} mg/dL across horizons. Baselines with a higher clinically safe A+B percentage than XGBoost: ${beatingPhrase}.</td></tr>`;

  // Second full-width row: baselines.note, eval rows, patients, date
  const evalRows = metrics.n_eval_rows || 0;
  const numPatients = (metrics.patients || []).length;
  const genDate = metrics.generated_at_utc ? metrics.generated_at_utc.slice(0, 10) : '';
  const noteText = metrics.baselines && metrics.baselines.note ? metrics.baselines.note : '';
  modelTableHtml += `<tr><td colspan="3">${escapeHtml(noteText)} · Evaluated on ${evalRows} test rows from ${numPatients} patients · Generated ${escapeHtml(genDate)}</td></tr>`;

  // Warnings row if status is not 'ok' or warnings non-empty
  if (data.status !== 'ok' || (data.warnings && data.warnings.length > 0)) {
    const warnList = (data.warnings || []).map(escapeHtml).join('; ');
    const statusMsg = `Status: ${escapeHtml(data.status)}${warnList ? ' · Warnings: ' + warnList : ''}`;
    modelTableHtml += `<tr class="warn"><td colspan="3" class="warn">${statusMsg}</td></tr>`;
  }

  selectElement('#mTable').innerHTML = modelTableHtml;
}

function renderModel() {
  const cards = [
    ['Glucose Lags', 'Lag features at 5, 10, 15, 30, and 60 minutes capture short- and medium-range CGM dependencies.'],
    ['Rolling Mean / Standard Deviation', '30-, 60-, 90-, and 120-minute rolling statistics summarize recent glucose stability and variation.'],
    ['Rate of Change', '5-minute and 30-minute change features quantify direction and acceleration in the glucose trajectory.'],
    ['Time-of-Day Encoding', 'Hour sine/cosine terms encode cyclical daily patterns in the CGM signal.'],
    ['XGBoost', 'Strong for small-to-mid-sized tabular data, captures nonlinear relationships, and provides feature-importance interpretability for engineered CGM features.'],
    ['Four Horizon-Specific Models', 'Four separate XGBoost models are trained, one per forecast horizon: 30, 60, 90, and 120 minutes.'],
    ['Reference vs Our Approach', 'Multimodal reference model: CGM + insulin + meal composition + other patient/time context with a Transformer + BiLSTM. Our approach: CGM only, CGM-derived features, XGBoost, four horizon-specific models.']
  ];

  const cardsHtml = cards.map(([title, text]) => stat(title, '', text)).join('');
  selectElement('#modelCards').innerHTML = cardsHtml;
}

function fillCases(caseSelect) {
  const currentCase = getCurrentCase();
  if (!currentCase) {
    caseSelect.innerHTML = '';
    return;
  }
  const selectedPatientId = currentCase.patient;
  const patientCases = caseList.filter(c => c.patient === selectedPatientId);
  caseSelect.innerHTML = patientCases
    .map(c => `<option value="${escapeHtml(c.id)}">${escapeHtml(formatTs(c.ts))}</option>`)
    .join('');
}

function renderAll() {
  if (uiState.phase !== 'ready' || !getCurrentCase()) {
    renderUnavailable();
    renderClinical();
    return;
  }

  const currentCase = getCurrentCase();

  selectAll('.sel-patient').forEach(patientSelect => {
    if (patientSelect.options.length !== patientList.length) {
      patientSelect.innerHTML = patientList.map(p => `<option value="${escapeHtml(p)}">Patient ${escapeHtml(p)}</option>`).join('');
    }
    patientSelect.value = currentCase.patient;
  });

  selectAll('.sel-case').forEach(caseSelect => {
    fillCases(caseSelect);
    caseSelect.value = currentCase.id;
  });

  renderOverview();
  renderForecast();
  renderExplorer();
  renderClinical();
}

async function runForecast() {
  const currentCase = getCurrentCase();
  if (!currentCase || !currentCase.history) return;
  const resultsBox = selectElement('#results');
  if (resultsBox) resultsBox.classList.add('loading');

  try {
    const res = await apiPost('/v1/predict', {
      readings: currentCase.history,
      horizon: 'all'
    });
    if (res.ok && res.data) {
      updateCasePredictions(currentCase, res.data);
      renderAll();
    } else {
      if (resultsBox) {
        resultsBox.innerHTML = stat('Request failed', 'Error', res.detail ? String(res.detail).slice(0, 200) : 'Prediction failed', 'bad');
      }
    }
  } finally {
    if (resultsBox) resultsBox.classList.remove('loading');
  }
}

function init() {
  selectAll('.sel-patient').forEach(patientSelect => {
    patientSelect.onchange = async () => {
      const chosenPatient = patientSelect.value;
      const matchingCase = caseList.find(c => c.patient === chosenPatient);
      if (matchingCase) {
        await selectCaseById(matchingCase.id);
      }
    };
  });

  selectAll('.sel-case').forEach(caseSelect => {
    caseSelect.onchange = async () => {
      const chosenCaseId = caseSelect.value;
      if (chosenCaseId) {
        await selectCaseById(chosenCaseId);
      }
    };
  });

  selectElement('#selH').onchange = async event => {
    appState.horizon = Number(event.target.value);
    renderAll();
    if (isClinicalActive()) {
      await loadBacktest(appState.horizon);
    }
  };

  selectElement('#run').onclick = runForecast;

  selectElement('#rand').onclick = async () => {
    if (caseList.length <= 1) return;
    const otherCases = caseList.filter(c => c.id !== appState.caseId);
    if (otherCases.length > 0) {
      const chosen = otherCases[Math.floor(Math.random() * otherCases.length)];
      await selectCaseById(chosen.id);
    }
  };

  selectElement('#nav').onclick = async event => {
    const clickedButton = event.target.closest('button');
    if (!clickedButton) return;

    selectAll('#nav button').forEach(navButton => {
      navButton.classList.toggle('active', navButton === clickedButton);
    });

    selectAll('.view').forEach(viewElement => {
      const viewId = `v-${clickedButton.dataset.view}`;
      viewElement.classList.toggle('active', viewElement.id === viewId);
    });

    const activeView = clickedButton.dataset.view;
    if (activeView === 'clinical') {
      await loadBacktest(appState.horizon);
    } else if (activeView === 'performance') {
      renderPerformance();
    }

    Object.keys(charts).forEach(chartName => {
      if (charts[chartName]) {
        charts[chartName].resize();
      }
    });
  };

  createLineChart('cMain');
  createLineChart('cDetail');

  function createGuideLine(startPoint, endPoint, color = COLORS.grid) {
    return {
      type: 'line',
      data: [startPoint, endPoint],
      borderColor: color,
      borderWidth: 1,
      pointRadius: 0,
      borderDash: [4, 4]
    };
  }

  charts.cClarke = new Chart(selectElement('#cClarke'), {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'Test-set sample', data: [], pointRadius: 3.5 },
        { label: 'Selected case', data: [], pointRadius: 7, pointStyle: 'rectRot', borderColor: COLORS.accent, backgroundColor: 'transparent', borderWidth: 2 },
        createGuideLine({ x: 0, y: 0 }, { x: 400, y: 400 }, COLORS.mute),
        createGuideLine({ x: 0, y: 0 }, { x: 400, y: 320 }),
        createGuideLine({ x: 0, y: 0 }, { x: 333, y: 400 }),
        createGuideLine({ x: 0, y: 70 }, { x: 400, y: 70 }),
        createGuideLine({ x: 0, y: 180 }, { x: 400, y: 180 }),
        createGuideLine({ x: 70, y: 0 }, { x: 70, y: 400 }),
        createGuideLine({ x: 180, y: 0 }, { x: 180, y: 400 })
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      scales: {
        x: {
          min: 0,
          max: 400,
          ...buildAxisConfig('Reference glucose (mg/dL)')
        },
        y: {
          min: 0,
          max: 400,
          ...buildAxisConfig('Predicted glucose (mg/dL)')
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          filter(item) {
            return item.datasetIndex < 2;
          },
          callbacks: {
            label(context) {
              const p = context.raw;
              const zonePart = p && p.z ? `, zone ${p.z}` : '';
              return `Ref ${context.parsed.x}, pred ${context.parsed.y}${zonePart}`;
            }
          }
        }
      }
    }
  });

  charts.cErr = new Chart(selectElement('#cErr'), {
    type: 'line',
    data: {
      labels: HZ.map(hour => `${hour} min`),
      datasets: []
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: buildAxisConfig(),
        y: {
          ...buildAxisConfig('mg/dL'),
          min: 0,
          max: 60
        }
      },
      plugins: {
        legend: {
          labels: { color: COLORS.mute }
        }
      }
    }
  });

  renderModel();
  lucide.createIcons();

  // Initial bootstrap
  bootstrap();
}

document.addEventListener('DOMContentLoaded', init);
