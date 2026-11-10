/* Demo / research prototype using CGM-only OhioT1DM replay data. */

const API_BASE_URL = 'http://127.0.0.1:8000';

// ---------------------------------------------------------------------------
// API client and data state
// ---------------------------------------------------------------------------

/** Track whether the backend is reachable and models are loaded. */
const apiState = {
  connected: false,
  modelsLoaded: 0,
  modelsExpected: 4,
  status: 'unknown',
  featureColumns: []
};

/** UI state for honest status representation across views. */
const uiState = {
  phase: 'loading', // 'loading' | 'ready' | 'offline' | 'degraded' | 'no_data' | 'error'
  message: '',
  detail: ''
};

let caseList = [];        // from /v1/cases: [{ id, patient, ts, cur }]
let patientList = [];     // from /v1/cases: patient ids as strings
const caseCache = new Map();   // case id -> loaded case object
const appState = {
  caseId: null,
  horizon: 60
};
let currentLoadCaseId = null;  // Stale-response guard for asynchronous case loading

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
 * Fetch wrapper for POST requests.
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
  el.innerHTML = `<span style="width:8px;height:8px;border-radius:50%;background:${dotColor};display:inline-block;"></span>${label}`;
}

const HZ = [30, 60, 90, 120];
const RMSE = { 30: 20.94, 60: 33.67, 90: 42.44, 120: 48.08 };
const SAFE = { 30: 98.29, 60: 96.72, 90: 95.14, 120: 93.87 };
const REFERENCE = { 30: 99.6, 60: 99.0, 90: 97.6, 120: 96.4 };

const MODEL_ROWS = [
  { model: 'Ridge/Lasso', note: 'Linear baselines considered', comparison: '—' },
  { model: 'ARIMA', note: 'Classical time-series baseline considered', comparison: '—' },
  { model: 'ARX/ARIMAX', note: 'Ruled out because required insulin/carb inputs are unavailable', comparison: '—' },
  { model: 'XGBoost', note: 'Selected primary model: strong for small-to-mid-sized tabular data, captures nonlinear patterns in engineered CGM features, and provides interpretable feature importance for this dataset scale.', comparison: 'Selected' },
  { model: 'Four horizon-specific models', note: 'One XGBoost model trained per forecast horizon (30/60/90/120 min)', comparison: '—' },
  { model: 'Xiong et al. (2025) — multimodal reference model', note: 'CGM + insulin + meal composition + other patient/time context; Transformer + BiLSTM', comparison: '30 min 99.6%; 60 min 99.0%; 90 min 97.6%; 120 min 96.4%' }
];

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

function selectElement(selector, root = document) {
  return root.querySelector(selector);
}

function selectAll(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function clarkeZone(referenceGlucose, predictedGlucose) {
  if ((referenceGlucose < 70 && predictedGlucose < 70) || Math.abs(predictedGlucose - referenceGlucose) <= 0.2 * referenceGlucose) {
    return 'A';
  }

  if ((referenceGlucose >= 180 && predictedGlucose <= 70) || (referenceGlucose <= 70 && predictedGlucose >= 180)) {
    return 'E';
  }

  if ((referenceGlucose >= 70 && referenceGlucose <= 290 && predictedGlucose >= referenceGlucose + 110) ||
      (referenceGlucose >= 130 && referenceGlucose <= 180 && predictedGlucose <= (7 / 5) * referenceGlucose - 182)) {
    return 'C';
  }

  if ((referenceGlucose >= 240 && predictedGlucose >= 70 && predictedGlucose <= 180) ||
      (referenceGlucose <= 175 / 3 && predictedGlucose >= 70 && predictedGlucose <= 180) ||
      (referenceGlucose > 175 / 3 && referenceGlucose <= 70 && predictedGlucose >= 1.2 * referenceGlucose)) {
    return 'D';
  }

  return 'B';
}

const CLARKE = [{"x": 65, "y": 78, "z": "A"}, {"x": 138, "y": 142, "z": "A"}, {"x": 355, "y": 384, "z": "A"}, {"x": 224, "y": 239, "z": "A"}, {"x": 228, "y": 257, "z": "A"}, {"x": 300, "y": 289, "z": "A"}, {"x": 341, "y": 354, "z": "A"}, {"x": 290, "y": 327, "z": "A"}, {"x": 60, "y": 64, "z": "A"}, {"x": 190, "y": 192, "z": "A"}, {"x": 348, "y": 329, "z": "A"}, {"x": 187, "y": 209, "z": "A"}, {"x": 187, "y": 173, "z": "A"}, {"x": 90, "y": 103, "z": "A"}, {"x": 256, "y": 253, "z": "A"}, {"x": 229, "y": 213, "z": "A"}, {"x": 153, "y": 146, "z": "A"}, {"x": 193, "y": 186, "z": "A"}, {"x": 251, "y": 264, "z": "A"}, {"x": 349, "y": 373, "z": "A"}, {"x": 337, "y": 378, "z": "A"}, {"x": 213, "y": 227, "z": "A"}, {"x": 278, "y": 266, "z": "A"}, {"x": 173, "y": 194, "z": "A"}, {"x": 306, "y": 287, "z": "A"}, {"x": 196, "y": 198, "z": "A"}, {"x": 247, "y": 224, "z": "A"}, {"x": 56, "y": 52, "z": "A"}, {"x": 297, "y": 325, "z": "A"}, {"x": 313, "y": 346, "z": "A"}, {"x": 217, "y": 198, "z": "A"}, {"x": 88, "y": 83, "z": "A"}, {"x": 156, "y": 154, "z": "A"}, {"x": 350, "y": 385, "z": "A"}, {"x": 180, "y": 198, "z": "A"}, {"x": 355, "y": 358, "z": "A"}, {"x": 169, "y": 165, "z": "A"}, {"x": 112, "y": 100, "z": "A"}, {"x": 246, "y": 243, "z": "A"}, {"x": 158, "y": 144, "z": "A"}, {"x": 235, "y": 254, "z": "A"}, {"x": 148, "y": 145, "z": "A"}, {"x": 86, "y": 87, "z": "A"}, {"x": 74, "y": 65, "z": "A"}, {"x": 265, "y": 253, "z": "A"}, {"x": 119, "y": 116, "z": "A"}, {"x": 57, "y": 52, "z": "A"}, {"x": 206, "y": 215, "z": "A"}, {"x": 69, "y": 74, "z": "A"}, {"x": 188, "y": 167, "z": "A"}, {"x": 128, "y": 144, "z": "A"}, {"x": 161, "y": 159, "z": "A"}, {"x": 79, "y": 93, "z": "A"}, {"x": 297, "y": 275, "z": "A"}, {"x": 96, "y": 98, "z": "A"}, {"x": 247, "y": 231, "z": "A"}, {"x": 218, "y": 219, "z": "A"}, {"x": 238, "y": 266, "z": "A"}, {"x": 275, "y": 274, "z": "A"}, {"x": 295, "y": 278, "z": "A"}, {"x": 349, "y": 376, "z": "A"}, {"x": 272, "y": 267, "z": "A"}, {"x": 58, "y": 47, "z": "A"}, {"x": 223, "y": 235, "z": "A"}, {"x": 163, "y": 139, "z": "A"}, {"x": 239, "y": 209, "z": "A"}, {"x": 271, "y": 255, "z": "A"}, {"x": 243, "y": 258, "z": "A"}, {"x": 325, "y": 353, "z": "A"}, {"x": 309, "y": 344, "z": "A"}, {"x": 127, "y": 141, "z": "A"}, {"x": 107, "y": 106, "z": "A"}, {"x": 109, "y": 127, "z": "A"}, {"x": 235, "y": 222, "z": "A"}, {"x": 211, "y": 198, "z": "A"}, {"x": 180, "y": 204, "z": "A"}, {"x": 242, "y": 234, "z": "A"}, {"x": 132, "y": 126, "z": "A"}, {"x": 209, "y": 216, "z": "A"}, {"x": 346, "y": 358, "z": "A"}, {"x": 158, "y": 136, "z": "A"}, {"x": 196, "y": 179, "z": "A"}, {"x": 165, "y": 184, "z": "A"}, {"x": 116, "y": 114, "z": "A"}, {"x": 127, "y": 129, "z": "A"}, {"x": 206, "y": 230, "z": "A"}, {"x": 55, "y": 44, "z": "A"}, {"x": 190, "y": 204, "z": "A"}, {"x": 196, "y": 176, "z": "A"}, {"x": 96, "y": 93, "z": "A"}, {"x": 259, "y": 270, "z": "A"}, {"x": 218, "y": 198, "z": "A"}, {"x": 140, "y": 122, "z": "A"}, {"x": 304, "y": 316, "z": "A"}, {"x": 65, "y": 72, "z": "A"}, {"x": 83, "y": 76, "z": "A"}, {"x": 316, "y": 283, "z": "A"}, {"x": 141, "y": 156, "z": "A"}, {"x": 265, "y": 235, "z": "A"}, {"x": 247, "y": 238, "z": "A"}, {"x": 252, "y": 270, "z": "A"}, {"x": 196, "y": 203, "z": "A"}, {"x": 81, "y": 74, "z": "A"}, {"x": 283, "y": 263, "z": "A"}, {"x": 285, "y": 271, "z": "A"}, {"x": 185, "y": 179, "z": "A"}, {"x": 248, "y": 254, "z": "A"}, {"x": 180, "y": 193, "z": "A"}, {"x": 252, "y": 227, "z": "A"}, {"x": 163, "y": 186, "z": "A"}, {"x": 218, "y": 248, "z": "A"}, {"x": 180, "y": 183, "z": "A"}, {"x": 156, "y": 137, "z": "A"}, {"x": 143, "y": 132, "z": "A"}, {"x": 208, "y": 200, "z": "A"}, {"x": 77, "y": 76, "z": "A"}, {"x": 184, "y": 182, "z": "A"}, {"x": 167, "y": 185, "z": "A"}, {"x": 141, "y": 121, "z": "A"}, {"x": 156, "y": 141, "z": "A"}, {"x": 157, "y": 148, "z": "A"}, {"x": 136, "y": 149, "z": "A"}, {"x": 55, "y": 49, "z": "A"}, {"x": 131, "y": 135, "z": "A"}, {"x": 214, "y": 217, "z": "A"}, {"x": 212, "y": 225, "z": "A"}, {"x": 223, "y": 214, "z": "A"}, {"x": 340, "y": 382, "z": "A"}, {"x": 97, "y": 112, "z": "A"}, {"x": 271, "y": 242, "z": "A"}, {"x": 127, "y": 122, "z": "A"}, {"x": 78, "y": 85, "z": "A"}, {"x": 328, "y": 340, "z": "A"}, {"x": 103, "y": 116, "z": "A"}, {"x": 313, "y": 304, "z": "A"}, {"x": 73, "y": 73, "z": "A"}, {"x": 283, "y": 247, "z": "A"}, {"x": 109, "y": 123, "z": "A"}, {"x": 194, "y": 175, "z": "A"}, {"x": 77, "y": 89, "z": "A"}, {"x": 298, "y": 259, "z": "A"}, {"x": 270, "y": 298, "z": "A"}, {"x": 138, "y": 129, "z": "A"}, {"x": 308, "y": 342, "z": "A"}, {"x": 200, "y": 190, "z": "A"}, {"x": 319, "y": 286, "z": "A"}, {"x": 304, "y": 336, "z": "A"}, {"x": 333, "y": 303, "z": "A"}, {"x": 277, "y": 275, "z": "A"}, {"x": 194, "y": 189, "z": "A"}, {"x": 194, "y": 220, "z": "A"}, {"x": 87, "y": 72, "z": "A"}, {"x": 181, "y": 188, "z": "A"}, {"x": 345, "y": 324, "z": "A"}, {"x": 264, "y": 241, "z": "A"}, {"x": 303, "y": 328, "z": "A"}, {"x": 263, "y": 245, "z": "A"}, {"x": 188, "y": 166, "z": "A"}, {"x": 294, "y": 323, "z": "A"}, {"x": 280, "y": 249, "z": "A"}, {"x": 317, "y": 284, "z": "A"}, {"x": 302, "y": 302, "z": "A"}, {"x": 79, "y": 67, "z": "A"}, {"x": 101, "y": 111, "z": "A"}, {"x": 268, "y": 279, "z": "A"}, {"x": 58, "y": 58, "z": "A"}, {"x": 109, "y": 113, "z": "A"}, {"x": 281, "y": 257, "z": "A"}, {"x": 283, "y": 320, "z": "A"}, {"x": 189, "y": 170, "z": "A"}, {"x": 268, "y": 275, "z": "A"}, {"x": 90, "y": 98, "z": "A"}, {"x": 204, "y": 212, "z": "A"}, {"x": 138, "y": 124, "z": "A"}, {"x": 72, "y": 63, "z": "A"}, {"x": 274, "y": 272, "z": "A"}, {"x": 228, "y": 205, "z": "A"}, {"x": 278, "y": 312, "z": "A"}, {"x": 222, "y": 220, "z": "A"}, {"x": 335, "y": 356, "z": "A"}];

const charts = {};

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
  return `<div class="stat"><small>${label}</small><b class="${cssClass}">${valueText}</b><span>${detail}</span></div>`;
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
  appState.caseId = id;
  const c = await loadCase(id);
  if (c && currentLoadCaseId === id) {
    uiState.phase = 'ready';
  }
  renderAll();
}

/**
 * Bootstrap application on startup:
 * 1. Check /health
 * 2. Fetch case list
 * 3. Load initial case
 * 4. Render states
 */
async function bootstrap() {
  uiState.phase = 'loading';
  uiState.message = 'Loading';
  uiState.detail = 'Fetching test cases from the backend';
  renderAll();

  const healthRes = await apiGet('/health');
  if (!healthRes.ok || !healthRes.data) {
    uiState.phase = 'offline';
    uiState.message = 'Backend offline';
    uiState.detail = 'Start the API (see README) and reload this page';
    apiState.connected = false;
    apiState.status = 'offline';
    renderApiStatus();
    renderAll();
    return;
  }

  const hData = healthRes.data;
  apiState.connected = true;
  apiState.status = hData.status;
  apiState.modelsLoaded = hData.models_loaded;
  apiState.modelsExpected = hData.models_expected;

  if (hData.models_loaded < hData.models_expected) {
    uiState.phase = 'degraded';
    uiState.message = 'Models not loaded';
    uiState.detail = 'Train or provide the model files, then restart the API';
    renderApiStatus();
    renderAll();
    return;
  }

  const listOk = await loadCaseList();
  if (!listOk || uiState.phase !== 'loading') {
    renderApiStatus();
    renderAll();
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

const nowPlugin = {
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
    return `<div style="display:flex; align-items:center; justify-content:space-between; gap:10px; padding:8px 0; border-bottom:1px solid var(--line);"><span>${hour} min — ${forecastValue} mg/dL</span><span class="${state.className}" style="font-weight:600;">${state.label}</span></div>`;
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

    alertHtml = `<strong>${excursionType}</strong> — earliest horizon: ${earliestPrediction.h} min — predicted glucose: ${earliestPrediction.value} mg/dL`;
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
    return `<div class="stat ${selectedClass}"><small>${hour} min forecast</small><b>${predRounded} mg/dL</b><span>Actual ${actualRounded} · error ${errRounded}</span></div>`;
  }).join('');

  selectElement('#fcards').innerHTML = futureCardsHtml;

  const [predictionRangeLabel] = rangeOf(currentForecast.pred);
  const [forecastErrorText] = errorStatus(currentForecast.err);

  selectElement('#interp').innerHTML = `At ${appState.horizon} min the model projects <b>${Math.round(currentForecast.pred)} mg/dL</b> (${predictionRangeLabel.toLowerCase()}); the held-out value was <b>${Math.round(currentForecast.actual)} mg/dL</b>, an absolute error of ${Math.round(currentForecast.err)} mg/dL (${forecastErrorText.toLowerCase()}). This CGM-only replay uses the OhioT1DM cohort with 6 patients, 5-minute sampling, and 30/60/90/120 min forecasting. It is a research prototype and not treatment guidance.`;

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
    tableHtml += `<tr><td>${hour} min</td><td>${Math.round(currentCase.cur)}</td><td>${Math.round(forecastItem.pred)}</td><td>${Math.round(forecastItem.actual)}</td><td class="${errorClass}">${Math.round(forecastItem.err)} mg/dL</td></tr>`;
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
  selectElement('#events').innerHTML = currentCase.events.map(eventText => `<li>${eventText}</li>`).join('');
}

function zoneDistribution(horizon) {
  const values = {
    30: { A: 86.50, B: 11.79, C: 1.35, D: 0.30, E: 0.06 },
    60: { A: 82.47, B: 14.25, C: 2.64, D: 0.52, E: 0.12 },
    90: { A: 79.31, B: 15.83, C: 3.82, D: 0.79, E: 0.25 },
    120: { A: 75.72, B: 18.15, C: 4.76, D: 1.04, E: 0.33 }
  };

  return values[horizon];
}

function renderClinical() {
  const currentCase = getCurrentCase();
  if (!currentCase) {
    if (charts.cClarke) {
      charts.cClarke.data.datasets[0].data = CLARKE;
      charts.cClarke.data.datasets[1].data = [];
      charts.cClarke.update();
    }
    return;
  }
  const selectedForecast = currentCase.preds[appState.horizon];

  function positionOnRangeScale(glucoseValue) {
    return clamp((glucoseValue - 40) / 360 * 100, 0, 100);
  }

  function makeMarker(value, labelText, color, markerTop) {
    return `<span class="mk" style="left:${positionOnRangeScale(value)}%;color:${color};top:${markerTop}px">${labelText} ${value}</span>`;
  }

  const rangeBarHtml = `<div class="rbar"><i style="width:${positionOnRangeScale(70)}%;background:${COLORS.bad}"></i><i style="width:${positionOnRangeScale(180) - positionOnRangeScale(70)}%;background:${COLORS.ok}"></i><i style="flex:1;background:${COLORS.warn}"></i>${makeMarker(Math.round(currentCase.cur), 'Now', '#fff', -26)}${makeMarker(Math.round(selectedForecast.pred), 'Pred', COLORS.accent, -26)}</div>`;

  selectElement('#range').innerHTML = rangeBarHtml;
  selectElement('#rangeNote').textContent = `Clarke Error Grid Analysis is used to evaluate clinical acceptability of prediction errors. This research prototype does not provide treatment or insulin recommendations. For the selected horizon, ${appState.horizon} min, clinically safe A+B = ${SAFE[appState.horizon]}%.`;

  const zoneValues = zoneDistribution(appState.horizon);
  const zoneRowsHtml = Object.entries(zoneValues).map(([zoneName, zonePercentage]) => {
    return `<div class="zrow"><b>${zoneName}</b><div><i style="width:${zonePercentage}%;background:${ZONE_COLORS[zoneName]}"></i></div><span>${zonePercentage.toFixed(2)}%</span></div>`;
  }).join('');

  selectElement('#zones').innerHTML = zoneRowsHtml;

  const clarkeChart = charts.cClarke;
  if (clarkeChart) {
    clarkeChart.data.datasets[0].data = CLARKE;
    clarkeChart.data.datasets[0].backgroundColor = CLARKE.map(point => ZONE_COLORS[point.z]);
    clarkeChart.data.datasets[1].data = HZ.map(hour => ({
      x: Math.round(currentCase.preds[hour].actual),
      y: Math.round(currentCase.preds[hour].pred)
    }));
    clarkeChart.update();
  }
}

function renderPerformance() {
  const perfCardsHtml = stat('30 min RMSE', '20.94 mg/dL', 'Clarke A+B 98.29%')
    + stat('60 min RMSE', '33.67 mg/dL', 'Clarke A+B 96.72%')
    + stat('90 min RMSE', '42.44 mg/dL', 'Clarke A+B 95.14%')
    + stat('120 min RMSE', '48.08 mg/dL', 'Clarke A+B 93.87%');

  selectElement('#perfCards').innerHTML = perfCardsHtml;

  let horizonTableHtml = '<tr><th>Horizon</th><th>RMSE</th><th>Clinically safe (A+B)</th></tr>';

  HZ.forEach(hour => {
    horizonTableHtml += `<tr><td>${hour} min</td><td>${RMSE[hour]} mg/dL</td><td>${SAFE[hour]}%</td></tr>`;
  });

  selectElement('#hTable').innerHTML = horizonTableHtml;

  const modelRowsHtml = '<tr><th>Model / reference</th><th>Comparison</th><th>Notes</th></tr>'
    + MODEL_ROWS.map(row => `<tr><td>${row.model}</td><td>${row.comparison}</td><td>${row.note}</td></tr>`).join('')
    + '<tr><td colspan="3"><strong>Why XGBoost?</strong> It is well suited to small-to-mid-sized tabular forecasting tasks with engineered CGM features, captures nonlinear relationships in glucose dynamics, and provides interpretable feature importance without requiring the larger deep-learning architecture used in the multimodal reference model.</td></tr>';

  selectElement('#mTable').innerHTML = modelRowsHtml;
}

function renderModel() {
  const cards = [
    ['Glucose Lags', 'Lag features at 5, 10, 15, 30, and 60 minutes capture short- and medium-range CGM dependencies.'],
    ['Rolling Mean / Standard Deviation', '30-, 60-, 90-, and 120-minute rolling statistics summarize recent glucose stability and variation.'],
    ['Rate of Change', '5-minute and 30-minute change features quantify direction and acceleration in the glucose trajectory.'],
    ['Time-of-Day Encoding', 'Hour sine/cosine terms encode cyclical daily patterns in the CGM signal.'],
    ['XGBoost', 'Strong for small-to-mid-sized tabular data, captures nonlinear relationships, and provides feature-importance interpretability for engineered CGM features.'],
    ['Four Horizon-Specific Models', 'Four separate XGBoost models are trained, one per forecast horizon: 30, 60, 90, and 120 minutes.'],
    ['Reference vs Our Approach', 'Xiong et al. (2025): CGM + insulin + meal composition + other patient/time context with a Transformer + BiLSTM. Our approach: CGM only, CGM-derived features, XGBoost, four horizon-specific models.']
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
    .map(c => `<option value="${c.id}">${formatTs(c.ts)}</option>`)
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
      patientSelect.innerHTML = patientList.map(p => `<option value="${p}">Patient ${p}</option>`).join('');
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

  selectElement('#selH').onchange = event => {
    appState.horizon = Number(event.target.value);
    renderAll();
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

  selectElement('#nav').onclick = event => {
    const clickedButton = event.target.closest('button');

    if (!clickedButton) {
      return;
    }

    selectAll('#nav button').forEach(navButton => {
      navButton.classList.toggle('active', navButton === clickedButton);
    });

    selectAll('.view').forEach(viewElement => {
      const viewId = `v-${clickedButton.dataset.view}`;
      viewElement.classList.toggle('active', viewElement.id === viewId);
    });

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
        { label: 'Demo points', data: [], pointRadius: 3.5 },
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
              return `Ref ${context.parsed.x}, pred ${context.parsed.y}`;
            }
          }
        }
      }
    }
  });

  renderPerformance();
  renderModel();

  charts.cErr = new Chart(selectElement('#cErr'), {
    type: 'line',
    data: {
      labels: HZ.map(hour => `${hour} min`),
      datasets: [{
        label: 'RMSE',
        data: HZ.map(hour => RMSE[hour]),
        borderColor: COLORS.accent,
        backgroundColor: COLORS.accent,
        tension: 0.3
      }]
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

  lucide.createIcons();

  // Initial bootstrap: load health, case list, and first case
  bootstrap();
}

document.addEventListener('DOMContentLoaded', init);
