/* =============================================================================
   Glucose Horizon — CGM-only forecasting dashboard
   Plain browser JS. Charts: Chart.js 4.4.1. Icons: Lucide.

   Change API_BASE_URL or ENDPOINTS if the backend moves. predict() accepts
   either the live FastAPI payload or a future shape with history, timestamps,
   predicted, actual, horizon, error, clarke_zone, and context.
   ============================================================================= */

const API_BASE_URL = 'http://localhost:8000';

const ENDPOINTS = {
  health: '/health',
  patients: '/v1/patients',
  cases: '/v1/cases',
  predict: '/v1/predict',
  overallMetrics: '/v1/backtest/overall',
  patientMetrics: '/v1/backtest/patient',
  uploadCsv: '/v1/predict/csv',
  multimodalCsv: '/v1/predict/simglucose/csv',
  multimodalModels: '/v1/models/simglucose',
  simTestPatients: '/v1/simglucose/test/patients',
  simTestPatient: '/v1/simglucose/test/patient',
  wellness: '/v1/wellness',
  agentChat: '/agent/chat',
  agentReset: '/agent/reset',
  agentHealth: '/agent/health',
};

const HZ = [30, 60, 90, 120];
const SAFE = { 30: 98.29, 60: 96.72, 90: 95.14, 120: 93.87 };
const ZONE_SHARE = {
  30: { A: 86.5, B: 11.79, C: 1.35, D: 0.3, E: 0.06 },
  60: { A: 82.47, B: 14.25, C: 2.64, D: 0.52, E: 0.12 },
  90: { A: 79.31, B: 15.83, C: 3.82, D: 0.79, E: 0.25 },
  120: { A: 75.72, B: 18.15, C: 4.76, D: 1.04, E: 0.33 },
};

const COLORS = {
  accent: '#E5573B',
  blue: '#1F2A44',
  ok: '#285C95',
  warn: '#946000',
  bad: '#D64545',
  mute: '#6B7391',
  grid: 'rgba(31, 42, 68, 0.10)',
  text: '#1F2A44',
};

const ZONE_COLORS = {
  A: '#4DA3E0',
  B: '#F6D98D',
  C: '#F5B83D',
  D: '#ED824B',
  E: '#D64545',
};

const ZONE_RGBA = {
  A: [77, 163, 224, 45],
  B: [219, 231, 239, 55],
  C: [245, 184, 61, 58],
  D: [237, 130, 75, 54],
  E: [214, 69, 69, 48],
};

const LOAD_STEPS = [
  'Reading CGM window...',
  'Engineering features...',
  'Running XGBoost model...',
  'Computing risk...',
];

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const mobileQuery = window.matchMedia('(max-width: 860px)');

const charts = {};
let clarkeImage = null;
let statusTimer = 0;
const patientMetricsCache = new Map();
const patientClinicalCache = new Map();
const lastForecastContexts = new Map();
const patientMetricsLoading = new Set();
const patientMetricsFailed = new Set();
const simReferenceCache = new Map();
const simReferenceLoading = new Set();
const simReferenceFailed = new Set();
let simTestPatientIds = [];
let simDashboardRequestId = 0;
let overallMetrics = null;
let overallMetricsLoading = false;

const state = {
  view: 'overview',
  dataset: 'ohio',
  patient: '',
  simPatient: '',
  simSample: null,
  caseId: '',
  horizon: 60,
  running: false,
  checked: false,
  connected: false,
  health: 'unknown',
  usingDemo: true,
  // caseId whose forecast has been run and may be shown. null = nothing run yet.
  forecastFor: null,
  // Cached active uploaded CSV prediction response
  lastUploadForecast: null,
  lastUploadFileName: '',
};

/* -----------------------------------------------------------------------------
   Small helpers
   ----------------------------------------------------------------------------- */

function $(selector, root = document) {
  return root.querySelector(selector);
}

function $all(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function round1(value) {
  return Math.round(Number(value) * 10) / 10;
}

function mg(value) {
  const rounded = round1(value);
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
}

function esc(value) {
  return String(value).replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[ch]));
}

function hasUploadedPatientContext() {
  return Boolean(
    state.lastUploadForecast?.agent_dataset_id &&
    state.lastUploadForecast?.predictions?.length &&
    state.lastUploadForecast?.readings?.length >= 24
  );
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function initCursorDots() {
  const canvas = $('#cursorDots');
  if (!canvas || reduceMotion || window.matchMedia('(pointer: coarse)').matches) return;
  const ctx = canvas.getContext('2d');
  const spacing = 22;
  const radius = 190;
  const pointer = { x: -1000, y: -1000, targetX: -1000, targetY: -1000, opacity: 0, targetOpacity: 0 };
  let width = 0;
  let height = 0;
  let frame = 0;

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  };

  const draw = () => {
    ctx.clearRect(0, 0, width, height);
    pointer.x += (pointer.targetX - pointer.x) * 0.18;
    pointer.y += (pointer.targetY - pointer.y) * 0.18;
    pointer.opacity += (pointer.targetOpacity - pointer.opacity) * 0.12;
    if (pointer.opacity > 0.005) {
      for (let y = spacing / 2; y < height; y += spacing) {
        for (let x = spacing / 2; x < width; x += spacing) {
          const distance = Math.hypot(x - pointer.x, y - pointer.y);
          if (distance >= radius) continue;
          const fade = 1 - distance / radius;
          ctx.globalAlpha = pointer.opacity * fade * fade * 0.56;
          ctx.fillStyle = '#1F426B';
          ctx.shadowColor = 'rgba(31, 66, 107, 0.52)';
          ctx.shadowBlur = 7;
          ctx.beginPath();
          ctx.arc(x, y, 1.15, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
    if (pointer.opacity > 0.005 || pointer.targetOpacity > 0) frame = requestAnimationFrame(draw);
    else frame = 0;
  };

  window.addEventListener('pointermove', (event) => {
    if (event.pointerType === 'touch') return;
    pointer.targetX = event.clientX;
    pointer.targetY = event.clientY;
    pointer.targetOpacity = 1;
    if (!frame) frame = requestAnimationFrame(draw);
  }, { passive: true });
  window.addEventListener('pointerleave', () => { pointer.targetOpacity = 0; }, { passive: true });
  window.addEventListener('blur', () => { pointer.targetOpacity = 0; }, { passive: true });
  window.addEventListener('resize', resize, { passive: true });
  resize();
}

function mulberry32(seed) {
  let stateSeed = seed >>> 0;
  return function next() {
    stateSeed = (stateSeed + 0x6d2b79f5) >>> 0;
    let t = Math.imul(stateSeed ^ (stateSeed >>> 15), 1 | stateSeed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function isoAt(year, monthIndex, day, hour, minute, plusMinutes) {
  return new Date(Date.UTC(year, monthIndex, day, hour, minute + plusMinutes, 0))
    .toISOString()
    .slice(0, 19);
}

function paintIcons() {
  if (window.lucide && typeof window.lucide.createIcons === 'function') {
    window.lucide.createIcons();
  }
}

function showToast(message, kind = 'info') {
  const host = $('#toastHost');
  if (!host) return;
  const toast = document.createElement('div');
  toast.className = `toast ${kind}`;
  toast.textContent = message;
  host.appendChild(toast);
  setTimeout(() => toast.remove(), 4200);
}

/* -----------------------------------------------------------------------------
   Clinical helpers
   ----------------------------------------------------------------------------- */

function clarkeZone(reference, predicted) {
  const ref = Number(reference);
  const pred = Number(predicted);
  if ((ref < 70 && pred < 70) || Math.abs(pred - ref) <= 0.2 * ref) return 'A';
  if ((ref >= 180 && pred <= 70) || (ref <= 70 && pred >= 180)) return 'E';
  if (
    (ref >= 70 && ref <= 290 && pred >= ref + 110) ||
    (ref >= 130 && ref <= 180 && pred <= (7 / 5) * ref - 182)
  ) {
    return 'C';
  }
  if (
    (ref >= 240 && pred >= 70 && pred <= 180) ||
    (ref <= 175 / 3 && pred >= 70 && pred <= 180) ||
    (ref > 175 / 3 && ref <= 70 && pred >= 1.2 * ref)
  ) {
    return 'D';
  }
  return 'B';
}

function rangeOf(glucose) {
  if (glucose < 70) return { label: 'Below range', cls: 'bad', zone: 'low' };
  if (glucose > 180) return { label: 'Above range', cls: 'warn', zone: 'high' };
  return { label: 'In range', cls: 'ok', zone: 'in' };
}

function riskOf(glucose) {
  if (glucose < 70) return { label: 'Hypoglycemia risk', cls: 'bad' };
  if (glucose > 180) return { label: 'Hyperglycemia risk', cls: 'warn' };
  return { label: 'Low risk', cls: 'ok' };
}

function forecastRiskOf(forecast) {
  if (!forecast.clarke_zone) {
    const risk = riskOf(forecast.predicted);
    return { ...risk, detail: `${rangeOf(forecast.predicted).label} · EGA unavailable for uploaded forecasts` };
  }
  if (!['A', 'B'].includes(forecast.clarke_zone)) {
    return {
      label: 'Unreliable forecast',
      detail: `Clarke zone ${forecast.clarke_zone} · outside clinically acceptable agreement`,
      cls: 'warn',
    };
  }
  const risk = riskOf(forecast.predicted);
  return { ...risk, detail: rangeOf(forecast.predicted).label };
}

function errorStatus(error) {
  if (error <= 15) return { label: 'Close match', cls: 'ok' };
  if (error <= 30) return { label: 'Moderate deviation', cls: 'warn' };
  return { label: 'Large deviation', cls: 'bad' };
}

function valueAt(sample, minute) {
  const point = sample.series.find((item) => item.x === minute);
  return point ? point.y : sample.series[0].y;
}

function trendOf(sample) {
  const delta = (valueAt(sample, 0) - valueAt(sample, -15)) / 15;
  const signed = `${delta >= 0 ? '+' : ''}${delta.toFixed(1)} mg/dL/min`;
  if (delta > 1) return { label: 'Rising', arrow: '↑', detail: signed };
  if (delta < -1) return { label: 'Falling', arrow: '↓', detail: signed };
  return { label: 'Stable', arrow: '→', detail: signed };
}

function buildContext(sample) {
  const history = sample.history;
  const current = history[history.length - 1];
  const roc = (current - history[history.length - 4]) / 15;
  const mean = history.reduce((sum, value) => sum + value, 0) / history.length;
  const variance = history.reduce((sum, value) => sum + (value - mean) ** 2, 0) / history.length;
  const inRange = history.filter((value) => value >= 70 && value <= 180).length;
  return {
    current,
    roc,
    mean,
    sd: Math.sqrt(variance),
    tir: (100 * inRange) / history.length,
  };
}

function buildEvents(sample, context) {
  const events = ['CGM sampled every 5 minutes across a 3-hour window'];
  if (context.roc > 1) events.push('Glucose is rising into the forecast');
  else if (context.roc < -1) events.push('Glucose is falling into the forecast');
  else events.push('Glucose is stable across the last 15 minutes');

  const lows = sample.history.filter((value) => value < 70).length;
  const highs = sample.history.filter((value) => value > 180).length;
  if (lows) events.push(`${lows} reading${lows === 1 ? '' : 's'} below 70 mg/dL in the window`);
  if (highs) events.push(`${highs} reading${highs === 1 ? '' : 's'} above 180 mg/dL in the window`);
  events.push('Lag, rolling, delta, and time-of-day features come from this window');
  return events;
}

/* -----------------------------------------------------------------------------
   Built-in held-out samples
   These are the only selectable inputs. Predictions are replaced when /v1/predict
   succeeds; actual held-out glucose stays with the sample.
   ----------------------------------------------------------------------------- */

function buildCases() {
  const patients = ['559', '563', '570', '575', '588', '591'];
  const times = [
    [7, 35],
    [11, 10],
    [14, 50],
    [18, 25],
    [21, 45],
    [0, 20],
  ];
  const cases = [];

  patients.forEach((patient, patientIndex) => {
    for (let copy = 0; copy < 2; copy += 1) {
      const id = `${patient}-${copy + 1}`;
      const random = mulberry32(patientIndex * 50 + copy * 17 + 11);
      const clock = times[(patientIndex + copy * 3) % times.length];
      const day = 2 + patientIndex * 3 + copy * 2;
      const base = 100 + patientIndex * 8 + random() * 24;
      const amplitude = 20 + random() * 36;
      const phase = (patientIndex * 40 + copy * 25) % 180;
      const series = [];

      for (let minute = -180; minute <= 120; minute += 5) {
        const wave = Math.sin((minute + phase) / 34) * amplitude;
        const second = Math.cos((minute + phase) / 68) * amplitude * 0.42;
        const drift = (minute / 180) * (random() - 0.5) * 16;
        const noise = (random() - 0.5) * 7;
        series.push({
          x: minute,
          y: Math.round(clamp(base + wave + second + drift + noise, 58, 250)),
        });
      }

      const preds = {};
      HZ.forEach((horizon) => {
        const actual = series.find((point) => point.x === horizon).y;
        const bias = Math.sin((patientIndex + copy + horizon) / 8) * 7 + (random() - 0.5) * 8;
        const predicted = Math.round(clamp(actual + bias, 55, 270));
        preds[horizon] = {
          predicted,
          actual,
          error: Math.abs(predicted - actual),
          clarke_zone: clarkeZone(actual, predicted),
        };
      });

      const history = series.filter((point) => point.x <= 0).map((point) => point.y);
      const timestamps = series.map((point) =>
        isoAt(2020, 2, day, clock[0], clock[1], point.x)
      );
      const sample = {
        id,
        patient,
        label: `2020-03-${String(day).padStart(2, '0')} ${String(clock[0]).padStart(2, '0')}:${String(clock[1]).padStart(2, '0')}`,
        series,
        history,
        timestamps,
        preds,
      };
      sample.context = buildContext(sample);
      sample.events = buildEvents(sample, sample.context);
      cases.push(sample);
    }
  });

  return cases;
}

const CASES = buildCases();
const livePredictions = new Map();

function getCase(caseId) {
  return CASES.find((item) => item.id === caseId) || CASES[0];
}

function patients() {
  return [...new Set(CASES.map((item) => item.patient))];
}

function casesFor(patient) {
  return CASES.filter((item) => item.patient === patient);
}

function horizonsFor(caseId) {
  if (state.lastUploadForecast?.predictions?.length) return currentSample().preds;
  if (state.dataset === 'sim' && state.simSample) return state.simSample.preds;
  return livePredictions.get(caseId) || getCase(caseId).preds;
}

function currentSample() {
  if (state.lastUploadForecast?.readings?.length) {
    const upload = state.lastUploadForecast;
    const rows = upload.readings;
    const last = rows[rows.length - 1];
    const values = rows.map((row) => Number(row.glucose));
    const current = Number(upload.latest_glucose ?? last.glucose);
    const roc = values.length >= 4 ? (values.at(-1) - values.at(-4)) / 15 : 0;
    const history = rows.map((row) => ({
      x: (Date.parse(row.timestamp) - Date.parse(last.timestamp)) / 60_000,
      y: Number(row.glucose),
    }));
    const preds = Object.fromEntries(upload.predictions.map((item) => [Number(item.horizon), {
      predicted: Number(item.predicted_glucose), actual: null, error: null, clarke_zone: null,
      zone_label: item.zone_label,
    }]));
    const series = [...history, ...upload.predictions.map((item) => ({ x: Number(item.horizon), y: Number(item.predicted_glucose) }))];
    return {
      id: 'upload-patient', patient: upload.patient_used || 'Uploaded patient',
      label: state.lastUploadFileName || 'Uploaded CGM history', series,
      timestamps: [...rows.map((row) => row.timestamp), ...upload.predictions.map(() => last.timestamp)],
      preds, history,
      context: {
        current, roc, mean: values.reduce((sum, value) => sum + value, 0) / values.length,
        sd: Math.sqrt(values.reduce((sum, value) => sum + (value - values.reduce((total, reading) => total + reading, 0) / values.length) ** 2, 0) / values.length),
        tir: values.filter((value) => value >= 70 && value <= 180).length * 100 / values.length,
      },
      events: [`Uploaded file: ${state.lastUploadFileName || 'CGM CSV'}`, `${rows.length} recent readings supplied to the forecast`],
      isUpload: true,
    };
  }
  if (state.dataset === 'sim' && state.simSample) return state.simSample;
  return getCase(state.caseId);
}

// Predictions are only visible after Run Forecast completes for the selected case.
// Changing patient, case, or horizon clears them again.
function hasForecast() {
  if (state.lastUploadForecast?.predictions?.length) return true;
  if (state.dataset === 'sim') return Boolean(state.simSample);
  return state.forecastFor === state.caseId;
}

function currentForecast(horizon = state.horizon) {
  if (state.lastUploadForecast?.predictions?.length) return currentSample().preds[horizon] || null;
  if (state.dataset === 'sim' && state.simSample) return state.simSample.preds[horizon] || null;
  return hasForecast() ? horizonsFor(state.caseId)[horizon] : null;
}

function patientRmse(patient, horizon) {
  const metrics = patientMetricsCache.get(String(patient));
  const value = metrics && metrics[String(horizon)] && metrics[String(horizon)].rmse;
  return Number.isFinite(value) ? value : null;
}

function patientRmseText(patient, horizon) {
  const metrics = patientMetricsCache.get(String(patient));
  const item = metrics && metrics[String(horizon)];
  const value = item && item.rmse;
  if (Number.isFinite(value)) return `${value.toFixed(2)} mg/dL`;
  return patientMetricsLoading.has(String(patient)) ? 'Loading…' : 'Unavailable';
}

function overallRmseText(horizon) {
  const item = overallMetrics && overallMetrics[String(horizon)];
  const value = item && item.rmse;
  if (Number.isFinite(value)) return `${value.toFixed(2)} mg/dL`;
  return overallMetricsLoading ? 'Loading…' : 'Unavailable';
}

function patientMaeText(patient, horizon) {
  const item = patientMetricsCache.get(String(patient))?.[String(horizon)];
  if (Number.isFinite(item?.mae)) return `${item.mae.toFixed(2)} mg/dL`;
  return patientMetricsLoading.has(String(patient)) ? 'Loading…' : 'Unavailable';
}

function overallMaeText(horizon) {
  const item = overallMetrics && overallMetrics[String(horizon)];
  if (Number.isFinite(item?.mae)) return `${item.mae.toFixed(2)} mg/dL`;
  return overallMetricsLoading ? 'Loading…' : 'Unavailable';
}

async function loadPatientMetrics(patient) {
  const id = String(patient);
  if (patientMetricsCache.has(id) || patientMetricsLoading.has(id)) return;
  patientMetricsLoading.add(id);
  patientMetricsFailed.delete(id);
  renderAll({ animate: false, draw: false });
  try {
    const result = await api.getPatientMetrics(id);
    patientMetricsCache.set(id, result.metrics || {});
    patientClinicalCache.set(id, result.clarke || {});
  } catch (error) {
    patientMetricsFailed.add(id);
    console.warn(`Could not load RMSE for patient ${id}.`, error.message);
  } finally {
    patientMetricsLoading.delete(id);
    if (String(state.patient) === id) renderAll({ animate: false, draw: false });
    if ($('#showReferenceEga')?.checked && $('#referencePatient')?.value === `ohio|${id}`) {
      renderClinical();
      requestAnimationFrame(resizeActiveCharts);
    }
    if ($('#showPerformancePatient')?.checked && $('#performancePatient')?.value === `ohio|${id}`) renderPerformance();
  }
}

async function loadOverallMetrics() {
  if (overallMetrics || overallMetricsLoading) return;
  overallMetricsLoading = true;
  renderAll({ animate: false, draw: false });
  try {
    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.overallMetrics}`);
    if (!response.ok) throw new Error(`Overall metrics request failed (${response.status})`);
    const result = await response.json();
    overallMetrics = result.metrics || {};
  } catch (error) {
    console.warn('Could not load overall RMSE.', error.message);
  } finally {
    overallMetricsLoading = false;
    renderAll({ animate: false, draw: false });
  }
}

function getActiveForecastContext() {
  // 1. Check if there is an active uploaded CSV forecast
  if (state.lastUploadForecast && state.lastUploadForecast.predictions && state.lastUploadForecast.predictions.length > 0) {
    const upload = state.lastUploadForecast;
    const predsMap = {};
    const latestGlucose = upload.latest_glucose ?? null;
    const timestamp = upload.timestamp || null;
    const agentPredictions = upload.agent_predictions || upload.predictions;
    agentPredictions.forEach((p) => {
      predsMap[p.horizon] = {
        predicted: p.predicted_glucose,
        change: p.change_from_latest,
        zone: p.zone_label === 'low' ? 'hypoglycemia' : (p.zone_label === 'high' ? 'hyperglycemia' : 'in_range'),
      };
    });

    const context = {
      has_forecast: true,
      source: 'upload',
      patient_id: upload.patient_used || upload.patient_id || 'Uploaded-CSV',
      selected_patient_id: String(upload.patient_used || upload.patient_id || 'Uploaded patient'),
      test_case_id: 'csv_upload',
      timestamp: timestamp || new Date().toISOString(),
      current_glucose: latestGlucose,
      predictions: predsMap,
      range_classification: latestGlucose != null ? (latestGlucose < 70 ? 'hypoglycemia' : (latestGlucose > 180 ? 'hyperglycemia' : 'in_range')) : null,
      patient_rmse: null,
      upload_quality: {
        filename: state.lastUploadFileName || null,
        model: upload.selected_model || $('#uploadMode')?.value || 'cgm',
        headers: upload.headers || [],
        row_count: upload.row_count ?? null,
        warnings: upload.warnings || {},
        meal_defaulted_to_zero: Boolean(upload.meal_defaulted_to_zero),
      },
      context_readings: upload.readings || [],
    };
    lastForecastContexts.set(`${state.dataset}|${context.selected_patient_id}`, context);
    return context;
  }

  // 2. Check held-out Ohio test cases or sim
  const sample = currentSample();
  const hasRun = hasForecast();
  const ptId = state.dataset === 'sim'
    ? String(state.simPatient || (sample && sample.patient) || 'unknown')
    : String(state.patient || (sample && sample.patient) || '559');
  const caseId = state.dataset === 'sim' ? (sample && sample.id) || null : state.caseId || (sample && sample.id) || null;

  // Calculate current glucose & rate of change from sample readings
  const sampleReadings = readingsFor(sample);
  let latestGlucose = null;
  let roc = null;
  let trend = '→';
  let timestamp = sample ? sample.timestamp : null;

  if (sampleReadings && sampleReadings.length > 0) {
    latestGlucose = sampleReadings[sampleReadings.length - 1].glucose;
    timestamp = sampleReadings[sampleReadings.length - 1].timestamp;
    if (sampleReadings.length >= 4) {
      const g0 = sampleReadings[sampleReadings.length - 4].glucose;
      const g1 = latestGlucose;
      roc = Math.round(((g1 - g0) / 15.0) * 100) / 100;
    } else if (sampleReadings.length >= 2) {
      const g0 = sampleReadings[0].glucose;
      const g1 = latestGlucose;
      roc = Math.round(((g1 - g0) / ((sampleReadings.length - 1) * 5.0)) * 100) / 100;
    }
    if (roc != null) {
      if (roc > 1.0) trend = '↑';
      else if (roc < -1.0) trend = '↓';
      else trend = '→';
    }
  }

  // Patient-specific RMSE (never cohort-wide)
  const ptMetrics = patientMetricsCache.get(ptId);
  const patientRmseMap = {};
  if (ptMetrics) {
    [30, 60, 90, 120].forEach((hz) => {
      const m = ptMetrics[String(hz)];
      if (m && Number.isFinite(m.rmse)) {
        patientRmseMap[hz] = m.rmse;
      }
    });
  }

  if (!hasRun) {
    const previousContext = lastForecastContexts.get(`${state.dataset}|${ptId}`);
    if (previousContext) {
      return { ...previousContext, selected_patient_id: ptId, previous_forecast: true };
    }
    return {
      has_forecast: false,
      source: 'held_out',
      patient_id: ptId,
      test_case_id: caseId,
      timestamp,
      current_glucose: latestGlucose,
      rate_of_change: roc,
      trend,
      range_classification: latestGlucose != null ? (latestGlucose < 70 ? 'hypoglycemia' : (latestGlucose > 180 ? 'hyperglycemia' : 'in_range')) : null,
      predictions: null,
      patient_rmse: Object.keys(patientRmseMap).length > 0 ? patientRmseMap : null,
    };
  }

  const rawPreds = horizonsFor(caseId);
  const predsMap = {};
  [30, 60, 90, 120].forEach((hz) => {
    const p = rawPreds && rawPreds[hz];
    if (p) {
      predsMap[hz] = {
        predicted: p.predicted,
        change: latestGlucose != null ? Math.round((p.predicted - latestGlucose) * 10) / 10 : null,
        actual: p.actual != null ? p.actual : null,
        error: p.error != null ? p.error : null,
        clarke_zone: p.clarke_zone || null,
        zone: p.predicted < 54 ? 'severe_hypoglycemia' : (p.predicted < 70 ? 'hypoglycemia' : (p.predicted > 250 ? 'severe_hyperglycemia' : (p.predicted > 180 ? 'hyperglycemia' : 'in_range'))),
      };
    }
  });

  const context = {
    has_forecast: true,
    source: 'held_out',
    patient_id: ptId,
    test_case_id: caseId,
    timestamp,
    current_glucose: latestGlucose,
    rate_of_change: roc,
    trend,
    range_classification: latestGlucose != null ? (latestGlucose < 70 ? 'hypoglycemia' : (latestGlucose > 180 ? 'hyperglycemia' : 'in_range')) : null,
    predictions: predsMap,
    patient_rmse: Object.keys(patientRmseMap).length > 0 ? patientRmseMap : null,
    context_readings: sampleReadings,
  };
  lastForecastContexts.set(`${state.dataset}|${ptId}`, context);
  return context;
}

/* -----------------------------------------------------------------------------
   API layer — the only place that knows endpoint paths
   ----------------------------------------------------------------------------- */

const api = {
  async getPatients() {
    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.patients}`);
    if (!response.ok) return null;
    const data = await response.json();
    const list = data.patients || data;
    return Array.isArray(list) ? list.map(String) : null;
  },

  async getCases(patient) {
    const response = await fetch(
      `${API_BASE_URL}${ENDPOINTS.cases}?patient=${encodeURIComponent(patient)}`
    );
    if (!response.ok) return null;
    const data = await response.json();
    const list = data.cases || data;
    return Array.isArray(list) ? list : null;
  },

  async getPatientMetrics(patient) {
    const response = await fetch(
      `${API_BASE_URL}${ENDPOINTS.patientMetrics}/${encodeURIComponent(patient)}`
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Patient metrics failed (${response.status})`);
    return data;
  },

  async predictCsv(file, horizon, multimodal = false) {
    const form = new FormData();
    form.append('file', file);
    form.append('horizon', horizon);
    const endpoint = multimodal ? ENDPOINTS.multimodalCsv : ENDPOINTS.uploadCsv;
    const response = await fetch(`${API_BASE_URL}${endpoint}`, { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Upload failed (${response.status})`);
    return data;
  },

  async getMultimodalModels() {
    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.multimodalModels}`);
    if (!response.ok) return null;
    return response.json();
  },

  async getWellness(message) {
    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.wellness}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const err = new Error(data && data.detail ? data.detail : `Request failed (${response.status})`);
      err.status = response.status;
      err.detail = data && data.detail;
      throw err;
    }
    return data;
  },

  async agentHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}${ENDPOINTS.agentHealth}`);
      if (!response.ok) return null;
      return response.json();
    } catch {
      return null;
    }
  },

  async agentChat(message, sessionId = 'default') {
    const upload = state.lastUploadForecast;
    if (!hasUploadedPatientContext()) {
      throw new Error('No patient upload has been completed. Patient context cannot be determined yet.');
    }
    const forecastContext = getActiveForecastContext();
    if (forecastContext?.source !== 'upload') {
      throw new Error('Uploaded patient context is unavailable. Please upload a patient CSV first.');
    }
    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.agentChat}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        message,
        agent_dataset_id: upload.agent_dataset_id,
        forecast_context: forecastContext,
      }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const err = new Error(data && data.detail ? data.detail : `Agent request failed (${response.status})`);
      err.status = response.status;
      err.detail = data && data.detail;
      throw err;
    }
    return data;
  },

  async agentReset(sessionId = 'default') {
    try {
      const response = await fetch(`${API_BASE_URL}${ENDPOINTS.agentReset}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
      return response.json();
    } catch {
      return { status: 'error' };
    }
  },

  /**
   * Run the horizon models for one held-out sample.
   * Returns { source, horizons } where each horizon has
   * predicted, actual, error, and clarke_zone.
   */
  async predict(patient, caseId, horizon) {
    const sample = getCase(caseId);
    if (String(sample.patient) !== String(patient)) {
      throw new Error('Patient and test case do not match');
    }
    const requested = Number(horizon);
    if (!HZ.includes(requested)) {
      throw new Error('Horizon must be 30, 60, 90, or 120');
    }

    const signal = typeof AbortSignal !== 'undefined' && AbortSignal.timeout
      ? AbortSignal.timeout(8000)
      : undefined;

    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.predict}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        readings: readingsFor(sample),
        // Each horizon has its own model. "all" runs every loaded model.
        horizon: 'all',
      }),
      signal,
    });

    if (!response.ok) {
      throw new Error(`Predict failed (${response.status})`);
    }

    const data = await response.json();
    return normalizePrediction(data, sample, horizon);
  },
};

function readingsFor(sample) {
  if (!sample?.series) return [];
  return sample.series
    .filter((point) => point.x <= 0)
    .map((point) => ({
      timestamp: sample.timestamps[sample.series.indexOf(point)],
      glucose: point.y,
    }));
}

function normalizePrediction(data, sample, horizon) {
  if (data && data.predicted != null && data.actual != null && Array.isArray(data.history)) {
    const key = Number(data.horizon || horizon);
    const horizons = data.horizons
      ? { ...data.horizons }
      : {
          [key]: {
            predicted: round1(data.predicted),
            actual: round1(data.actual),
            error: round1(data.error ?? Math.abs(data.predicted - data.actual)),
            clarke_zone: data.clarke_zone || clarkeZone(data.actual, data.predicted),
          },
        };
    return { source: 'api', horizons };
  }

  const horizons = {};
  (data.predictions || []).forEach((item) => {
    const actual = sample.preds[item.horizon] ? sample.preds[item.horizon].actual : data.actual;
    if (actual == null) return;
    const predicted = round1(item.predicted_glucose);
    horizons[item.horizon] = {
      predicted,
      actual,
      error: round1(Math.abs(predicted - actual)),
      clarke_zone: clarkeZone(actual, predicted),
    };
  });

  if (!Object.keys(horizons).length) {
    throw new Error('Prediction response was empty');
  }
  return { source: 'api', horizons };
}

function mockPrediction(caseId) {
  const sample = getCase(caseId);
  return { source: 'mock', horizons: sample.preds };
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.health}`);
    if (!response.ok) throw new Error('Health check failed');
    const data = await response.json();
    state.connected = true;
    state.health = data.status || 'ok';
  } catch {
    state.connected = false;
    state.health = 'offline';
    state.usingDemo = true;
    showDemoBadge(true);
    showToast('Backend unreachable. Showing demo data.');
  } finally {
    state.checked = true;
    renderStatus();
  }
}

function showDemoBadge(show) {
  ['demoBadge', 'demoBadgeMobile'].forEach((id) => {
    const badge = document.getElementById(id);
    if (badge) badge.hidden = !show;
  });
}

function renderStatus() {
  const el = $('#apiStatus');
  if (!el) return;
  const pending = !state.checked;
  const degraded = state.connected && state.health === 'degraded';
  const cls = pending ? 'pending' : state.connected ? (degraded ? 'degraded' : 'ok') : '';
  const label = pending
    ? 'Checking backend…'
    : state.connected
      ? degraded
        ? 'Backend degraded'
        : 'Backend connected'
      : 'Backend offline';
  el.innerHTML = `<span class="dot ${cls}"></span>${esc(label)}`;
}

/* -----------------------------------------------------------------------------
   Charts
   ----------------------------------------------------------------------------- */

function configureChartDefaults() {
  if (typeof Chart === 'undefined') return;
  Chart.defaults.font.family = '"DM Sans", system-ui, sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.color = COLORS.mute;
  Chart.defaults.borderColor = COLORS.grid;
  Chart.defaults.plugins.tooltip.backgroundColor = getComputedStyle(document.documentElement).getPropertyValue('--bg-elevated').trim();
  Chart.defaults.plugins.tooltip.borderColor = getComputedStyle(document.documentElement).getPropertyValue('--line-strong').trim();
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.titleColor = COLORS.text;
  Chart.defaults.plugins.tooltip.bodyColor = COLORS.text;
}

const rangeBandPlugin = {
  id: 'rangeBand',
  beforeDatasetsDraw(chart) {
    const { ctx, chartArea, scales } = chart;
    if (!chartArea || !scales.y) return;
    const top = scales.y.getPixelForValue(180);
    const bottom = scales.y.getPixelForValue(70);
    ctx.save();
    ctx.fillStyle = 'rgba(77, 163, 224, 0.12)';
    ctx.fillRect(chartArea.left, top, chartArea.right - chartArea.left, bottom - top);
    ctx.strokeStyle = 'rgba(77, 163, 224, 0.32)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(chartArea.left, top);
    ctx.lineTo(chartArea.right, top);
    ctx.moveTo(chartArea.left, bottom);
    ctx.lineTo(chartArea.right, bottom);
    ctx.stroke();
    ctx.restore();
  },
};

const nowPlugin = {
  id: 'nowMarker',
  beforeDatasetsDraw(chart) {
    const { ctx, chartArea, scales } = chart;
    if (!chartArea || !scales.x) return;
    const xNow = scales.x.getPixelForValue(0);
    ctx.save();
    ctx.strokeStyle = COLORS.mute;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(xNow, chartArea.top);
    ctx.lineTo(xNow, chartArea.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLORS.mute;
    ctx.font = '11px "DM Sans", sans-serif';
    ctx.fillText('History', chartArea.left + 6, chartArea.top + 14);
    ctx.fillText('Now', xNow + 6, chartArea.top + 14);
    ctx.fillText('Future', chartArea.right - 48, chartArea.top + 14);
    ctx.restore();
  },
};

function axis(title) {
  return {
    grid: { color: COLORS.grid, lineWidth: 0.6 },
    ticks: { color: COLORS.mute, font: { size: 11 } },
    title: {
      display: Boolean(title),
      text: title,
      color: COLORS.mute,
      font: { size: 12, weight: '500' },
    },
  };
}

function trajectoryConfig() {
  return {
    type: 'line',
    plugins: [rangeBandPlugin, nowPlugin],
    data: {
      datasets: [
        {
          label: 'Historical glucose',
          borderColor: '#1F2A44',
          backgroundColor: '#1F2A44',
          borderWidth: 1.8,
          pointRadius: 0,
          tension: 0.3,
          data: [],
        },
        {
          label: 'Actual future (held out)',
          borderColor: '#4DA3E0',
          backgroundColor: '#4DA3E0',
          borderWidth: 1.8,
          pointRadius: 0,
          tension: 0.3,
          data: [],
        },
        {
          label: 'Forecast',
          borderColor: '#24697C',
          backgroundColor: '#24697C',
          borderWidth: 1.8,
          borderDash: [6, 4],
          pointRadius: 4,
          pointHoverRadius: 6,
          tension: 0.25,
          data: [],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: reduceMotion ? 0 : 900, easing: 'easeOutQuart' },
      interaction: { mode: 'nearest', intersect: false },
      scales: {
        x: {
          type: 'linear',
          min: -180,
          max: 120,
          ...axis('Minutes from forecast time'),
          ticks: { color: COLORS.mute, stepSize: 30 },
        },
        y: {
          min: 40,
          max: 320,
          ...axis('Glucose (mg/dL)'),
        },
      },
      plugins: {
        legend: { labels: { color: COLORS.mute, boxWidth: 10, usePointStyle: true, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label(context) {
              const minute = context.parsed.x;
              const sign = minute > 0 ? '+' : '';
              return `${context.dataset.label}: ${mg(context.parsed.y)} mg/dL (t${sign}${minute} min)`;
            },
          },
        },
      },
    },
  };
}

function ensureChart(id, factory) {
  const canvas = document.getElementById(id);
  if (!canvas || typeof Chart === 'undefined') return null;
  if (charts[id] && charts[id].canvas !== canvas) destroyChart(id);
  if (charts[id]) return charts[id];
  charts[id] = factory(canvas);
  return charts[id];
}

function destroyChart(id) {
  if (!charts[id]) return;
  charts[id].destroy();
  charts[id] = null;
}

function setTrajectory(id, opts = {}) {
  const chart = charts[id];
  const sample = opts.sample || currentSample();
  if (!chart || !sample) return;

  // Upload samples are reconstructed on each lookup, so compare their stable ID
  // rather than object identity before deciding whether to show the forecast.
  const ready = opts.ready ?? (sample.id === currentSample().id && hasForecast());
  const horizons = ready ? (opts.horizons || horizonsFor(sample.id)) : null;
  const visible = ready ? sample.series : sample.series.filter((point) => point.x <= 0);
  const values = visible.map((point) => point.y)
    .concat(ready ? HZ.map((horizon) => horizons[horizon].predicted) : []);
  const yMin = Math.min(50, ...values) - 15;
  const yMax = Math.max(200, ...values) + 15;
  chart.options.scales.y.min = Math.max(30, Math.floor(yMin / 10) * 10);
  chart.options.scales.y.max = Math.min(400, Math.ceil(yMax / 10) * 10);

  chart.data.datasets[0].data = sample.series.filter((point) => point.x <= 0);
  // Held-out actuals and the forecast stay hidden until Run Forecast has finished.
  chart.data.datasets[1].data = ready && !sample.isUpload ? sample.series.filter((point) => point.x >= 0) : [];
  chart.data.datasets[2].data = ready
    ? [{ x: 0, y: sample.context.current }].concat(
        HZ.map((horizon) => ({ x: horizon, y: horizons[horizon].predicted }))
      )
    : [];
  chart.data.datasets[2].pointRadius = (context) => {
    if (context.dataIndex === 0) return 0;
    return HZ[context.dataIndex - 1] === state.horizon ? 7 : 3;
  };
  chart.data.datasets[2].pointBackgroundColor = (context) => {
    if (context.dataIndex === 0) return '#24697C';
    return HZ[context.dataIndex - 1] === state.horizon ? '#24697C' : 'rgba(31, 42, 68, 0.52)';
  };

  const duration = reduceMotion ? 0 : opts.draw ? (opts.animate ? 1400 : 700) : 0;
  chart.options.animation = { duration, easing: 'easeOutQuart' };
  chart.update();
}

function clarkeBackground() {
  if (clarkeImage) return clarkeImage;
  const canvas = document.createElement('canvas');
  canvas.width = 400;
  canvas.height = 400;
  const context = canvas.getContext('2d');
  const image = context.createImageData(400, 400);
  const pixels = image.data;
  for (let py = 0; py < 400; py += 1) {
    const predicted = 399 - py;
    for (let px = 0; px < 400; px += 1) {
      const rgba = ZONE_RGBA[clarkeZone(px, predicted)];
      const index = (py * 400 + px) * 4;
      pixels[index] = rgba[0];
      pixels[index + 1] = rgba[1];
      pixels[index + 2] = rgba[2];
      pixels[index + 3] = rgba[3];
    }
  }
  context.putImageData(image, 0, 0);
  clarkeImage = canvas;
  return canvas;
}

const clarkePlugin = {
  id: 'clarkeZones',
  beforeDatasetsDraw(chart) {
    const { ctx, chartArea } = chart;
    if (!chartArea) return;
    ctx.save();
    ctx.drawImage(
      clarkeBackground(),
      chartArea.left,
      chartArea.top,
      chartArea.right - chartArea.left,
      chartArea.bottom - chartArea.top
    );
    ctx.restore();
  },
  afterDatasetsDraw(chart) {
    const { ctx, chartArea, scales } = chart;
    if (!chartArea) return;
    ctx.save();
    ctx.font = '600 12px "DM Sans", sans-serif';
    ctx.fillStyle = '#1F2A44';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    [
      ['A', 140, 140],
      ['B', 250, 210],
      ['C', 300, 355],
      ['D', 300, 120],
      ['E', 36, 280],
    ].forEach(([label, x, y]) => {
      const px = scales.x.getPixelForValue(x);
      const py = scales.y.getPixelForValue(y);
      if (px < chartArea.left || px > chartArea.right || py < chartArea.top || py > chartArea.bottom) return;
      ctx.fillText(label, px, py);
    });
    ctx.restore();
  },
};

function buildClarkePoints() {
  const random = mulberry32(4242);
  const points = [];
  for (let index = 0; index < 140; index += 1) {
    const reference = Math.round(50 + random() * 300);
    const predicted = Math.round(clamp(reference + (random() - 0.5) * 2 * (0.12 * reference + 8), 40, 390));
    points.push({ x: reference, y: predicted, z: clarkeZone(reference, predicted) });
  }
  return points;
}

const CLARKE_POINTS = buildClarkePoints();

function ensureClarke() {
  return ensureChart('cClarke', (canvas) => new Chart(canvas, {
    type: 'scatter',
    plugins: [clarkePlugin],
    data: {
      datasets: [
        {
          label: 'Demo points',
          data: CLARKE_POINTS,
          pointRadius: 3,
          pointHoverRadius: 5,
          pointBackgroundColor: '#1F2A44',
          pointBorderWidth: 0,
        },
        {
          label: 'Selected case',
          data: [],
          pointStyle: 'rectRot',
          pointRadius: 8,
          pointHoverRadius: 9,
          borderColor: COLORS.accent,
          backgroundColor: 'transparent',
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: reduceMotion ? 0 : 500 },
      scales: {
        x: { min: 0, max: 400, ...axis('Reference glucose (mg/dL)') },
        y: { min: 0, max: 400, ...axis('Predicted glucose (mg/dL)') },
      },
      plugins: {
        legend: { labels: { color: COLORS.mute, boxWidth: 12, usePointStyle: true } },
        tooltip: {
          callbacks: {
            label(context) {
              const horizon = context.raw && context.raw.horizon;
              const prefix = horizon ? `${horizon} min · ` : '';
              return `${prefix}Ref ${mg(context.parsed.x)}, pred ${mg(context.parsed.y)}`;
            },
          },
        },
      },
    },
  }));
}

function ensureErrorChart() {
  return ensureChart('cErr', (canvas) => new Chart(canvas, {
    data: {
      labels: HZ.map((horizon) => `${horizon} min`),
      datasets: [
        {
          type: 'bar',
          label: `Patient ${state.patient} RMSE (mg/dL)`,
          data: HZ.map((horizon) => patientRmse(state.patient, horizon)),
          backgroundColor: ['#4DA3E0', '#B8D9EE', '#F5B83D', '#E5573B'],
          borderColor: ['#4DA3E0', '#B8D9EE', '#F5B83D', '#E5573B'],
          borderWidth: 1,
          borderRadius: 6,
          yAxisID: 'y',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: reduceMotion ? 0 : 800 },
      scales: {
        x: axis(''),
        y: { ...axis('RMSE (mg/dL)'), min: 0, max: 60 },
      },
      plugins: {
        legend: { labels: { color: COLORS.mute, boxWidth: 12, usePointStyle: true } },
      },
    },
  }));
}

function ensureUploadBacktestChart() {
  return ensureChart('cUploadBacktest', (canvas) => new Chart(canvas, {
    type: 'scatter',
    data: { datasets: [
      { label: 'Uploaded patient readings', data: [], borderColor: '#1F2A44', backgroundColor: '#1F2A44', pointRadius: 3, pointHoverRadius: 5 },
      { label: 'Perfect agreement', data: [{ x: 40, y: 40 }, { x: 400, y: 400 }], type: 'line', borderColor: 'rgba(31,42,68,.38)', borderDash: [6, 5], pointRadius: 0, borderWidth: 1.2 },
    ] },
    options: { responsive: true, maintainAspectRatio: false, aspectRatio: 1,
      plugins: { legend: { labels: { color: COLORS.mute } }, tooltip: { callbacks: { label: (context) => {
        const origin = context.raw.timestamp ? ` · forecast origin ${context.raw.timestamp}` : '';
        return `Actual ${mg(context.parsed.x)} · predicted ${mg(context.parsed.y)} mg/dL${origin}`;
      } } } },
      scales: { x: { min: 40, max: 400, ticks: { color: COLORS.mute }, grid: { color: COLORS.grid }, title: { display: true, text: 'Actual glucose (mg/dL)', color: COLORS.mute } },
        y: { min: 40, max: 400, ticks: { color: COLORS.mute }, grid: { color: COLORS.grid }, title: { display: true, text: 'Predicted glucose (mg/dL)', color: COLORS.mute } } } },
  }));
}

function resizeActiveCharts() {
  const map = {
    overview: ['cMain'],
    forecast: ['cDetail', 'cUploadBacktest'],
    clinical: ['cClarke'],
    performance: ['cErr'],
  };
  (map[state.view] || []).forEach((id) => {
    if (charts[id]) charts[id].resize();
  });
}

/* -----------------------------------------------------------------------------
   Rendering
   ----------------------------------------------------------------------------- */

function stat(label, value, detail, valueClass = '', opts = {}) {
  const classes = ['stat', opts.reveal ? 'reveal' : '', opts.sel ? 'sel' : '', opts.idle ? 'idle' : ''].filter(Boolean).join(' ');
  const delay = opts.delay ? ` style="animation-delay:${opts.delay}ms"` : '';
  const countAttrs = opts.count == null
    ? ''
    : ` data-count="${opts.count}" data-decimals="${opts.decimals || 0}" data-suffix="${esc(opts.suffix || '')}"`;
  const shown = opts.count == null ? value : '0';
  return `<div class="${classes}"${delay}><small>${esc(label)}</small><b class="${valueClass}"${countAttrs}>${shown}</b><span>${detail}</span></div>`;
}

function animateCounts(root) {
  const nodes = [...root.querySelectorAll('[data-count]')];
  if (!nodes.length) return;
  if (reduceMotion) {
    nodes.forEach((node) => {
      node.textContent = `${Number(node.dataset.count).toFixed(Number(node.dataset.decimals || 0))}${node.dataset.suffix || ''}`;
    });
    return;
  }
  const started = performance.now();
  const duration = 800;
  function frame(now) {
    const progress = Math.min(1, (now - started) / duration);
    const eased = 1 - (1 - progress) ** 3;
    nodes.forEach((node) => {
      const target = Number(node.dataset.count);
      const decimals = Number(node.dataset.decimals || 0);
      node.textContent = `${(target * eased).toFixed(decimals)}${node.dataset.suffix || ''}`;
    });
    if (progress < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function reveal(opts, index) {
  if (!opts.animate) return {};
  return { reveal: true, delay: index * 70 };
}

function renderOverview(opts = {}) {
  if (!state.lastUploadForecast?.predictions?.length) {
    $('#overview-title').textContent = 'Patient Glucose Overview';
    $('#overviewDescription').textContent = 'Upload a patient’s CGM CSV and run a prediction to populate this overview with their records.';
    $('#summary').innerHTML = [
      stat('Current glucose', '—', 'Awaiting patient upload', '', { idle: true }),
      stat('Current trend', '—', 'Awaiting patient upload', '', { idle: true }),
      stat('Patient data', '—', 'Upload a CSV to begin', '', { idle: true }),
      stat('Forecast horizon', '—', 'Awaiting prediction', '', { idle: true }),
    ].join('');
    $('#fcards').innerHTML = HZ.map((horizon) =>
      stat(`${horizon} min forecast`, '—', 'Awaiting patient upload', '', { idle: true })
    ).join('');
    $('#riskSummary').innerHTML = '';
    $('#riskTrajectory').innerHTML = '';
    const banner = $('#riskAlert');
    banner.className = 'alert-banner idle';
    banner.textContent = 'Upload a patient CSV and run a prediction to view patient-specific risk.';
    $('#interp').textContent = 'Patient-specific glucose history, forecasts, and evaluations will appear here after a CSV upload and prediction.';
    $('#overviewTrajectoryCard').hidden = document.body.classList.contains('patient-view');
    if (state.view === 'overview' && !document.body.classList.contains('patient-view')) {
      const chart = ensureChart('cMain', (canvas) => new Chart(canvas, trajectoryConfig()));
      if (chart) {
        chart.data.datasets.forEach((dataset) => { dataset.data = []; });
        chart.update(reduceMotion ? 'none' : undefined);
      }
    }
    return;
  }

  const sample = currentSample();
  $('#overviewTrajectoryCard').hidden = document.body.classList.contains('patient-view');
  $('#overview-title').textContent = 'Patient Glucose Overview';
  $('#overviewDescription').textContent = 'Patient glucose records and forecasts from the uploaded CSV.';
  const forecast = currentForecast();
  const ready = Boolean(forecast);
  const trend = trendOf(sample);
  const currentRange = rangeOf(sample.context.current);
  const summary = $('#summary');
  summary.innerHTML = [
    stat('Current glucose', `${mg(sample.context.current)} mg/dL`, currentRange.label, currentRange.cls, {
      ...reveal(opts, 0),
      count: opts.animate ? sample.context.current : null,
      suffix: ' mg/dL',
    }),
    stat('Current trend', `${trend.arrow} ${trend.label}`, trend.detail, '', reveal(opts, 1)),
    stat(sample.isUpload ? 'Active data' : 'Selected patient', sample.isUpload ? 'Uploaded patient' : (state.dataset === 'sim' ? sample.patient : `Patient ${sample.patient}`), sample.isUpload ? (state.lastUploadFileName || 'Uploaded CGM CSV') : (state.dataset === 'sim' ? 'SimGlucose held-out virtual patient' : 'OhioT1DM demo case'), '', reveal(opts, 2)),
    stat('Forecast horizon', `${state.horizon} min`, sample.isUpload ? 'Uploaded patient history' : state.dataset === 'sim' ? 'SimGlucose evaluation model' : 'Ohio CGM-only model', '', reveal(opts, 3)),
  ].join('');

  $('#fcards').innerHTML = HZ.map((horizon, index) => {
    if (!ready) {
      return stat(`${horizon} min forecast`, '—', 'Awaiting forecast', '', {
        idle: true,
        sel: horizon === state.horizon,
      });
    }
    const item = horizonsFor(sample.id)[horizon];
    const assessment = forecastRiskOf(item);
    return stat(
      `${horizon} min forecast`,
      `${mg(item.predicted)} mg/dL`,
      sample.isUpload ? `Patient upload · ${assessment.label}` : `Actual ${mg(item.actual)} · error ${mg(item.error)} · ${assessment.label}`,
      assessment.cls,
      {
        ...reveal(opts, index),
        sel: horizon === state.horizon,
        count: opts.animate ? item.predicted : null,
        decimals: Number.isInteger(round1(item.predicted)) ? 0 : 1,
        suffix: ' mg/dL',
      }
    );
  }).join('');

  renderRisk(opts);

  if (state.view === 'overview' && !document.body.classList.contains('patient-view')) {
    ensureChart('cMain', (canvas) => new Chart(canvas, trajectoryConfig()));
    setTrajectory('cMain', opts);
  }

  if (ready) {
    const assessment = forecastRiskOf(forecast);
    const status = sample.isUpload ? { label: 'upload forecast' } : errorStatus(forecast.error);
    const predictionNote = sample.isUpload
      ? 'The estimate is based on the uploaded patient’s CGM history.'
      : assessment.label === 'Unreliable forecast'
      ? `This prediction should be treated as unreliable (${assessment.detail.toLowerCase()}).`
      : `The estimate indicates ${assessment.detail.toLowerCase()}.`;
    const source = sample.isUpload
      ? `the uploaded patient data in ${esc(state.lastUploadFileName || 'the CSV')}`
      : state.dataset === 'sim'
      ? `the held-out SimGlucose virtual patient ${esc(sample.patient)}`
      : `an illustrative OhioT1DM demo scenario for patient ${esc(sample.patient)}`;
    $('#interp').innerHTML = sample.isUpload
      ? `At ${state.horizon} minutes the model projects <b>${mg(forecast.predicted)} mg/dL</b>. ${predictionNote} The upload does not contain future reference readings, so error and Clarke EGA cannot be calculated. This readout uses ${source}.`
      : `At ${state.horizon} minutes the model projects <b>${mg(forecast.predicted)} mg/dL</b>. ${predictionNote} The reference glucose was <b>${mg(forecast.actual)} mg/dL</b>, an absolute error of ${mg(forecast.error)} mg/dL (${status.label.toLowerCase()}, Clarke zone ${forecast.clarke_zone}). This readout uses ${source}. It is a research interpretation, not a treatment recommendation.`;
  } else {
    $('#interp').innerHTML = `No forecast has been run for patient ${esc(sample.patient)} (${esc(sample.label)}) yet. Pick a horizon in the <button type="button" class="link-btn" data-go="explorer">Test Explorer</button> and press <b>Run Forecast</b> to see the prediction, reference value, and error.`;
  }

  if (opts.animate) animateCounts(summary);
  if (opts.animate) animateCounts($('#fcards'));
}

function renderRisk(opts = {}) {
  const sample = currentSample();
  const current = rangeOf(sample.context.current);
  const banner = $('#riskAlert');
  banner.className = 'alert-banner';

  if (!hasForecast()) {
    $('#riskSummary').innerHTML = [
      stat('Current status', current.label, `${mg(sample.context.current)} mg/dL`, current.cls),
      stat('Selected risk', '—', 'Awaiting forecast', '', { idle: true }),
      stat('Lowest forecast', '—', 'Awaiting forecast', '', { idle: true }),
      stat('Highest forecast', '—', 'Awaiting forecast', '', { idle: true }),
    ].join('');
    $('#riskTrajectory').innerHTML = HZ.map((horizon) =>
      `<div class="risk-chip"><span class="hz">${horizon} min</span><span class="val idle-val">—</span><span class="zone idle-val">Not run</span></div>`
    ).join('');
    banner.classList.add('idle');
    banner.innerHTML = '<strong>No forecast yet.</strong> Run a forecast in the Test Explorer to see risk across all four horizons.';
    return;
  }

  const horizons = horizonsFor(sample.id);
  const values = HZ.map((horizon) => horizons[horizon].predicted);
  const lowest = Math.min(...values);
  const highest = Math.max(...values);
  const selected = horizons[state.horizon];
  const selectedRisk = forecastRiskOf(selected);
  const lowestItem = HZ.map((horizon) => horizons[horizon]).find((item) => item.predicted === lowest);
  const highestItem = HZ.map((horizon) => horizons[horizon]).find((item) => item.predicted === highest);
  const lowestAssessment = forecastRiskOf(lowestItem);
  const highestAssessment = forecastRiskOf(highestItem);

  $('#riskSummary').innerHTML = [
    stat('Current status', current.label, `${mg(sample.context.current)} mg/dL`, current.cls, reveal(opts, 0)),
    stat('Selected risk', selectedRisk.label, `${state.horizon} min · ${mg(selected.predicted)} mg/dL · ${selectedRisk.detail}`, selectedRisk.cls, reveal(opts, 1)),
    stat('Lowest forecast', `${mg(lowest)} mg/dL`, lowestAssessment.label, lowestAssessment.cls, reveal(opts, 2)),
    stat('Highest forecast', `${mg(highest)} mg/dL`, highestAssessment.label, highestAssessment.cls, reveal(opts, 3)),
  ].join('');

  $('#riskTrajectory').innerHTML = HZ.map((horizon) => {
    const item = horizons[horizon];
    const assessment = forecastRiskOf(item);
    const zoneClass = sample.isUpload ? `zone-${rangeOf(item.predicted).zone}` : ['A', 'B'].includes(item.clarke_zone) ? `zone-${rangeOf(item.predicted).zone}` : 'zone-review';
    return `<div class="risk-chip ${zoneClass}"><span class="hz">${horizon} min</span><span class="val ${assessment.cls}">${mg(item.predicted)}</span><span class="zone ${assessment.cls}">${assessment.label}</span></div>`;
  }).join('');

  const unreliable = sample.isUpload ? null : HZ.map((horizon) => ({ horizon, item: horizons[horizon] }))
    .find(({ item }) => !['A', 'B'].includes(item.clarke_zone));
  const excursion = HZ.map((horizon) => ({ horizon, item: horizons[horizon] }))
    .filter(({ item }) => sample.isUpload || ['A', 'B'].includes(item.clarke_zone))
    .map(({ horizon, item }) => ({ horizon, value: item.predicted }))
    .find((item) => item.value < 70 || item.value > 180);
  if (excursion) {
    const message = excursion.value < 70 ? 'Low glucose risk signal' : 'High glucose risk signal';
    const reliabilityNote = unreliable
      ? ` Another forecast at ${unreliable.horizon} min is unreliable (Clarke zone ${unreliable.item.clarke_zone}).`
      : '';
    banner.classList.add(excursion.value < 70 ? 'danger' : 'warn');
    banner.innerHTML = `<strong>${message}${sample.isUpload ? '.' : ' (zone A/B).'}</strong> Earliest horizon outside range: ${excursion.horizon} min, predicted ${mg(excursion.value)} mg/dL.${reliabilityNote}`;
  } else if (unreliable) {
    banner.classList.add('warn');
    banner.innerHTML = `<strong>Forecast needs review.</strong> The ${unreliable.horizon} min prediction is outside Clarke zones A/B (zone ${unreliable.item.clarke_zone}), so it is too unreliable to label as high or low glucose.`;
  } else {
    banner.classList.add('ok');
    banner.innerHTML = '<strong>No threshold excursion predicted.</strong> Forecasts from 30 to 120 minutes stay inside 70–180 mg/dL.';
  }
}

function renderForecast(opts = {}) {
  if (!state.lastUploadForecast?.predictions?.length) {
    $('#forecastDescription').textContent = 'Upload a patient’s CGM CSV and run a prediction to view their glucose history and forecast.';
    $('#errTable').innerHTML = '<tr><th>Horizon</th><th>Forward forecast</th><th>Scored pairs</th><th>RMSE</th><th>Clarke A+B</th><th>Status</th></tr><tr><td colspan="6" class="muted">Patient forecast data will appear here after a CSV upload and prediction.</td></tr>';
    $('#uploadBacktestCard').hidden = true;
    if (charts.cUploadBacktest) {
      charts.cUploadBacktest.data.datasets[0].data = [];
      charts.cUploadBacktest.update(reduceMotion ? 'none' : undefined);
    }
    if (state.view === 'forecast') {
      const chart = ensureChart('cDetail', (canvas) => new Chart(canvas, trajectoryConfig()));
      if (chart) {
        chart.data.datasets.forEach((dataset) => { dataset.data = []; });
        chart.update(reduceMotion ? 'none' : undefined);
      }
    }
    return;
  }

  const sample = currentSample();
  $('#forecastDescription').textContent = 'Uploaded patient glucose history, forecasts, and actual-versus-predicted evaluation.';
  const header = sample.isUpload
    ? '<tr><th>Horizon</th><th>Forward forecast</th><th>Scored pairs</th><th>RMSE</th><th>Clarke A+B</th><th>Status</th></tr>'
    : '<tr><th>Horizon</th><th>Predicted</th><th>Actual</th><th>Absolute error</th><th>Clarke</th><th>Status</th></tr>';
  if (hasForecast()) {
    const rows = HZ.map((horizon) => {
      const item = horizonsFor(sample.id)[horizon];
      if (sample.isUpload) {
        const metric = state.lastUploadForecast.evaluation?.[String(horizon)];
        return `<tr${horizon === state.horizon ? ' class="is-selected"' : ''}><td>${horizon} min</td><td>${mg(item.predicted)}</td><td>${metric ? `${metric.n_rows} historical pairs` : '—'}</td><td>${metric ? `${metric.rmse.toFixed(2)} mg/dL RMSE` : '—'}</td><td>${metric ? `${(metric.zone_percentages.A + metric.zone_percentages.B).toFixed(2)}% A+B` : '—'}</td><td>${metric ? 'Scored on upload' : 'Insufficient history'}</td></tr>`;
      }
      const status = errorStatus(item.error);
      const selected = horizon === state.horizon ? ' class="is-selected"' : '';
      return sample.isUpload
        ? `<tr${selected}><td>${horizon} min</td><td>${mg(item.predicted)}</td><td colspan="3" class="muted">Future reference unavailable; EGA not calculated</td><td>${forecastRiskOf(item).label}</td></tr>`
        : `<tr${selected}><td>${horizon} min</td><td>${mg(item.predicted)}</td><td>${mg(item.actual)}</td><td class="${status.cls}">${mg(item.error)} mg/dL</td><td>${item.clarke_zone}</td><td class="${status.cls}">${status.label}</td></tr>`;
    }).join('');
    $('#errTable').innerHTML = header + rows;
  } else {
    $('#errTable').innerHTML = `${header}<tr><td colspan="6" class="muted">No forecast yet. Run one in the <button type="button" class="link-btn" data-go="explorer">Test Explorer</button> to fill this table.</td></tr>`;
  }

  const backtestCard = $('#uploadBacktestCard');
  if (backtestCard) {
    const evaluation = sample.isUpload ? state.lastUploadForecast.evaluation?.[String(state.horizon)] : null;
    backtestCard.hidden = !sample.isUpload || !evaluation;
    if (sample.isUpload && state.view === 'forecast' && evaluation) {
      const chart = ensureUploadBacktestChart();
      $('#uploadBacktestTitle').textContent = `Uploaded history: actual vs predicted at ${state.horizon} min`;
      chart.data.datasets[0].data = evaluation.actual.map((actual, index) => ({ x: actual, y: evaluation.predicted[index], timestamp: evaluation.timestamps[index] }));
      chart.update(reduceMotion ? 'none' : undefined);
    }
  }

  if (state.view === 'forecast') {
    ensureChart('cDetail', (canvas) => new Chart(canvas, trajectoryConfig()));
    setTrajectory('cDetail', opts);
  }
}

function renderExplorer(opts = {}) {
  if (state.running) return;
  const sample = getCase(state.caseId);
  const ohioHorizons = livePredictions.get(sample.id) || sample.preds;
  const forecast = state.forecastFor === sample.id ? ohioHorizons[state.horizon] : null;
  const results = $('#results');
  results.classList.remove('is-loading');

  if (forecast) {
    const status = errorStatus(forecast.error);
    const risk = forecastRiskOf(forecast);
    results.innerHTML = [
      stat('Predicted', `${mg(forecast.predicted)} mg/dL`, `${state.horizon} min ahead`, rangeOf(forecast.predicted).cls, {
        ...reveal(opts, 0),
        count: opts.animate ? forecast.predicted : null,
        decimals: Number.isInteger(round1(forecast.predicted)) ? 0 : 1,
        suffix: ' mg/dL',
      }),
      stat('Actual', `${mg(forecast.actual)} mg/dL`, 'Held-out value', '', {
        ...reveal(opts, 1),
        count: opts.animate ? forecast.actual : null,
        suffix: ' mg/dL',
      }),
      stat('Case RMSE', `${mg(forecast.error)} mg/dL`, `Single held-out case · ${state.horizon} min · ${status.label.toLowerCase()}`, '', {
        ...reveal(opts, 2),
        count: opts.animate ? forecast.error : null,
        decimals: Number.isInteger(round1(forecast.error)) ? 0 : 1,
        suffix: ' mg/dL',
      }),
      stat('Clarke zone', forecast.clarke_zone, 'Selected horizon', ZONE_CLASS[forecast.clarke_zone] || '', reveal(opts, 3)),
      stat('Risk level', risk.label, risk.detail, risk.cls, reveal(opts, 4)),
    ].join('');
  } else {
    results.innerHTML = [
      stat('Predicted', '—', `${state.horizon} min ahead · press Run Forecast`, '', { idle: true }),
      stat('Actual', '—', 'Held-out value · press Run Forecast', '', { idle: true }),
      stat('Case RMSE', '—', `${state.horizon} min · Run Forecast to compare predicted and actual glucose`, '', { idle: true }),
      stat('Clarke zone', '—', 'Selected horizon · press Run Forecast', '', { idle: true }),
      stat('Risk level', '—', 'From the forecast · press Run Forecast', '', { idle: true }),
    ].join('');
  }

  const context = sample.context;
  const roc = `${context.roc >= 0 ? '+' : ''}${context.roc.toFixed(1)} mg/dL/min`;
  $('#ctx').innerHTML = [
    stat('Current glucose', `${mg(context.current)} mg/dL`, 'At forecast time', rangeOf(context.current).cls),
    stat('Rate of change', roc, 'Last 15 minutes'),
    stat('Recent mean', `${mg(context.mean)} mg/dL`, '3-hour window'),
    stat('Variability', `${mg(context.sd)} mg/dL`, 'Standard deviation'),
    stat('Time in range', `${context.tir.toFixed(0)}%`, '70–180 mg/dL in window'),
  ].join('');
  $('#events').innerHTML = sample.events.map((event) => `<li>${esc(event)}</li>`).join('');
  $('#selH').value = String(state.horizon);
  if (opts.animate && forecast) animateCounts(results);

}

const ZONE_CLASS = { A: 'ok', B: 'ok', C: 'warn', D: 'warn', E: 'bad' };

function renderClinical() {
  const sample = currentSample();
  const forecast = currentForecast();
  const scale = (glucose) => clamp(((glucose - 40) / 360) * 100, 0, 100);
  let rangeNow = null;
  let rangePred = null;
  let rangeLabel = '';
  if (sample.isUpload) {
    rangeNow = sample.context.current;
    rangePred = forecast?.predicted ?? null;
    rangeLabel = 'Uploaded patient';
  } else if (state.dataset === 'sim' && state.simSample) {
    rangeNow = state.simSample.context.current;
    rangePred = state.simSample.preds[state.horizon]?.predicted ?? null;
    rangeLabel = `SimGlucose patient ${state.simPatient}`;
  } else if (state.forecastFor === state.caseId) {
    rangeNow = sample.context.current;
    rangePred = forecast?.predicted ?? null;
    rangeLabel = `OhioT1DM patient ${sample.patient}`;
  }
  const nowMarker = rangeNow == null ? '' : `<span class="mk" style="left:${scale(rangeNow)}%;color:${COLORS.text}">Now ${mg(rangeNow)}</span>`;
  const predMarker = rangePred == null ? '' : `<span class="mk below" style="left:${scale(rangePred)}%;color:${COLORS.accent}">Pred ${mg(rangePred)}</span>`;
  $('#range').innerHTML = `<div class="rbar" role="img" aria-label="${esc(rangeLabel || 'Glucose range; no patient selected')} glucose range and forecast">
      <i style="width:${scale(70)}%;background:${COLORS.bad}"></i>
      <i style="width:${scale(180) - scale(70)}%;background:${COLORS.ok}"></i>
      <i style="flex:1;background:${COLORS.warn}"></i>
      ${nowMarker}
      ${predMarker}
    </div>`;
  const referenceEnabled = $('#showReferenceEga')?.checked;
  const referenceValue = $('#referencePatient')?.value || '';
  const [referenceDataset, referencePatient] = referenceValue.split('|');
  if (referenceEnabled && referenceDataset && referencePatient) {
    const referenceEgaControl = $('#uploadEgaHorizonControl');
    const referenceEgaSelect = $('#uploadEgaHorizon');
    if (referenceEgaControl) referenceEgaControl.hidden = false;
    if (referenceEgaSelect) referenceEgaSelect.value = String(state.horizon);
    $('#clarkePlotCard').hidden = false;
    $('#clarkeZonesCard').hidden = false;
    const simReference = referenceDataset === 'sim' ? simReferenceCache.get(referencePatient) : null;
    const reference = referenceDataset === 'sim'
      ? (simReference ? { percentages: simReference.metrics[String(state.horizon)]?.zone_percentages, points: simReference.scatter[String(state.horizon)] } : null)
      : patientClinicalCache.get(String(referencePatient))?.[String(state.horizon)];
    if (referenceDataset === 'sim' && simReference) {
      rangeNow = simReference.context.current;
      rangePred = simReference.predictions[String(state.horizon)]?.predicted ?? null;
      rangeLabel = `SimGlucose patient ${referencePatient}`;
    } else if (referenceDataset === 'ohio' && reference?.points?.length) {
      rangeNow = reference.points.at(-1).x;
      rangePred = reference.points.at(-1).y;
      rangeLabel = `OhioT1DM patient ${referencePatient}`;
    }
    const referenceNow = rangeNow == null ? '' : `<span class="mk" style="left:${scale(rangeNow)}%;color:${COLORS.text}">Now ${mg(rangeNow)}</span>`;
    const referencePred = rangePred == null ? '' : `<span class="mk below" style="left:${scale(rangePred)}%;color:${COLORS.accent}">Pred ${mg(rangePred)}</span>`;
    $('#range').innerHTML = `<div class="rbar" role="img" aria-label="${esc(rangeLabel || 'Glucose range')} glucose range and forecast"><i style="width:${scale(70)}%;background:${COLORS.bad}"></i><i style="width:${scale(180) - scale(70)}%;background:${COLORS.ok}"></i><i style="flex:1;background:${COLORS.warn}"></i>${referenceNow}${referencePred}</div>`;
    $('#zones').innerHTML = reference?.percentages
      ? Object.entries(reference.percentages).map(([zone, percent]) => `<div class="zrow"><b>${zone}</b><div><i style="width:${percent}%;background:${ZONE_COLORS[zone]}"></i></div><span>${percent.toFixed(2)}%</span></div>`).join('')
      : (referenceDataset === 'sim' ? simReferenceFailed.has(referencePatient) : patientMetricsFailed.has(String(referencePatient)))
        ? '<p class="muted">Could not load reference patient EGA. Check the backend connection and select the patient again to retry.</p>'
        : '<p class="muted">Loading reference patient EGA…</p>';
    $('#rangeNote').textContent = `${rangeLabel} reference patient range at ${state.horizon} min.`;
    $('#clarkeNote').textContent = `Held-out ${referenceDataset === 'sim' ? 'SimGlucose' : 'OhioT1DM'} test readings for reference patient ${referencePatient}.`;
    if (state.view === 'clinical') {
      const chart = ensureClarke();
      if (chart) {
        chart.data.datasets[0].label = `${referenceDataset === 'sim' ? 'SimGlucose' : 'OhioT1DM'} reference patient ${referencePatient}`;
        chart.data.datasets[0].data = reference?.points || [];
        chart.data.datasets[0].pointBackgroundColor = '#1F2A44';
        chart.data.datasets[1].data = [];
        chart.update(reduceMotion ? 'none' : undefined);
      }
    }
    if (referenceDataset === 'ohio' && !reference && !patientMetricsLoading.has(referencePatient) && !patientMetricsFailed.has(referencePatient)) loadPatientMetrics(referencePatient);
    if (referenceDataset === 'sim' && !simReference && !simReferenceLoading.has(referencePatient) && !simReferenceFailed.has(referencePatient)) loadSimReference(referencePatient);
    return;
  }
  const inactiveEgaControl = $('#uploadEgaHorizonControl');
  if (inactiveEgaControl) inactiveEgaControl.hidden = true;
  if (sample.isUpload) {
    const uploadEgaControl = $('#uploadEgaHorizonControl');
    const uploadEgaSelect = $('#uploadEgaHorizon');
    const allUploadHorizons = state.lastUploadForecast.requested_horizon === 'all';
    if (uploadEgaControl) uploadEgaControl.hidden = !allUploadHorizons;
    if (uploadEgaSelect) uploadEgaSelect.value = String(state.horizon);
    const evaluation = state.lastUploadForecast.evaluation?.[String(state.horizon)];
    $('#clarkePlotCard').hidden = !evaluation;
    $('#clarkeZonesCard').hidden = !evaluation;
    $('#rangeNote').textContent = `Now is the uploaded patient’s final reading (${mg(sample.context.current)} mg/dL); Pred is the ${state.horizon}-minute forecast (${mg(forecast?.predicted ?? sample.context.current)} mg/dL).`;
    $('#zones').innerHTML = evaluation?.zone_percentages
      ? Object.entries(evaluation.zone_percentages).map(([zone, percent]) => `<div class="zrow"><b>${zone}</b><div><i style="width:${percent}%;background:${ZONE_COLORS[zone]}"></i></div><span>${percent.toFixed(2)}%</span></div>`).join('')
      : '<p class="muted">Not enough future readings in this CSV to calculate EGA for the selected horizon.</p>';
    $('#clarkeNote').textContent = evaluation
      ? `Uploaded patient history scored at ${state.horizon} minutes; points compare actual future glucose with model predictions.`
      : 'More historical readings are needed to score this horizon.';
    if (state.view === 'clinical' && evaluation) {
      const chart = ensureClarke();
      if (chart) {
        chart.data.datasets[0].data = evaluation?.points || [];
        chart.data.datasets[0].pointBackgroundColor = '#1F2A44';
        chart.data.datasets[1].data = [];
        chart.data.datasets[0].label = `Uploaded patient · ${state.horizon} min historical EGA`;
        chart.update(reduceMotion ? 'none' : undefined);
      }
    }
    return;
  }
  $('#clarkePlotCard').hidden = true;
  $('#clarkeZonesCard').hidden = true;
  $('#rangeNote').textContent = rangeLabel ? `${rangeLabel}: current glucose ${mg(rangeNow)} mg/dL${rangePred == null ? '' : ` · forecast ${state.horizon} min ahead: ${mg(rangePred)} mg/dL`}.` : 'The range bar is ready for an uploaded or selected patient.';
  $('#zones').innerHTML = '';
  $('#clarkeNote').textContent = '';
  if (state.view === 'clinical') {
    const chart = ensureClarke();
    if (chart) {
      chart.data.datasets[0].data = [];
      chart.data.datasets[1].data = [];
      chart.data.datasets[0].label = 'No EGA data selected';
      chart.update(reduceMotion ? 'none' : undefined);
    }
  }
  return;
}

function renderPerformance() {
  const selected = $('#showPerformancePatient')?.checked ? ($('#performancePatient')?.value || '') : '';
  if (!selected && state.lastUploadForecast?.predictions?.length) {
    $('#performanceMetricsLayout').hidden = false;
    $('#performanceChartCard').hidden = false;
    $('#modelComparisonCard').hidden = false;
    const upload = currentSample();
    const evaluation = state.lastUploadForecast.evaluation || {};
    $('#performanceDescription').textContent = 'Historical forecast accuracy measured on the uploaded patient’s own records, plus forward predictions from their latest readings.';
    $('#perfCards').innerHTML = HZ.map((horizon) => {
      const metric = evaluation[String(horizon)];
      return stat(`${horizon} min · Uploaded RMSE`, metric ? `${metric.rmse.toFixed(2)} mg/dL` : 'Unavailable', metric ? `MAE ${metric.mae.toFixed(2)} mg/dL · ${metric.n_rows} scored readings · A+B ${(metric.zone_percentages.A + metric.zone_percentages.B).toFixed(2)}%` : `Forward forecast ${mg(upload.preds[horizon].predicted)} mg/dL · insufficient future labels`);
    }).join('');
    $('#hTable').innerHTML = `<tr><th>Horizon</th><th>Forward forecast</th><th>Historical RMSE</th><th>MAE</th><th>Clarke A+B</th></tr>${HZ.map((horizon) => {
      const metric = evaluation[String(horizon)];
      return `<tr class="${horizon === state.horizon ? 'is-selected' : ''}"><td>${horizon} min</td><td>${mg(upload.preds[horizon].predicted)} mg/dL</td><td>${metric ? `${metric.rmse.toFixed(2)} mg/dL` : '—'}</td><td>${metric ? `${metric.mae.toFixed(2)} mg/dL` : '—'}</td><td>${metric ? `${(metric.zone_percentages.A + metric.zone_percentages.B).toFixed(2)}%` : '—'}</td></tr>`;
    }).join('')}`;
    $('#mTable').innerHTML = `<tr><th>Upload context</th><th>Value</th></tr><tr><td>File</td><td>${esc(state.lastUploadFileName || 'Uploaded CSV')}</td></tr><tr><td>Readings used</td><td>${upload.history.length}</td></tr><tr><td>Model</td><td>${upload.isUpload ? esc(state.lastUploadForecast.selected_model || 'CGM') : 'CGM'}</td></tr>`;
    if (state.view === 'performance') {
      const chart = ensureErrorChart();
      chart.data.datasets[0].label = 'Uploaded patient RMSE (mg/dL)';
      chart.data.datasets[0].data = HZ.map((horizon) => evaluation[String(horizon)]?.rmse ?? null);
      chart.update(reduceMotion ? 'none' : undefined);
    }
    return;
  }
  if (!selected) {
    $('#performanceMetricsLayout').hidden = true;
    $('#performanceChartCard').hidden = true;
    $('#modelComparisonCard').hidden = true;
    $('#performanceDescription').textContent = 'Upload a patient CSV or select a test patient to view patient-specific forecast performance.';
    $('#perfCards').innerHTML = '';
    $('#hTable').innerHTML = '';
    $('#mTable').innerHTML = '';
    if (charts.cErr) {
      charts.cErr.data.datasets[0].data = [];
      charts.cErr.update(reduceMotion ? 'none' : undefined);
    }
    return;
  }
  const [dataset, patient] = selected.split('|');
  const simData = dataset === 'sim' ? simReferenceCache.get(patient) : null;
  const metrics = dataset === 'sim'
    ? simData?.metrics
    : patientMetricsCache.get(patient);
  if (!metrics) {
    $('#performanceMetricsLayout').hidden = true;
    $('#performanceChartCard').hidden = true;
    $('#modelComparisonCard').hidden = true;
    $('#performanceDescription').textContent = simReferenceLoading.has(patient) || patientMetricsLoading.has(patient)
      ? `Loading performance for ${dataset === 'sim' ? 'SimGlucose' : 'OhioT1DM'} patient ${patient}…`
      : `Performance is unavailable for ${dataset === 'sim' ? 'SimGlucose' : 'OhioT1DM'} patient ${patient}.`;
    if (dataset === 'sim' && !simReferenceLoading.has(patient) && !simReferenceFailed.has(patient)) loadSimReference(patient);
    if (dataset === 'ohio' && !patientMetricsLoading.has(patient) && !patientMetricsFailed.has(patient)) loadPatientMetrics(patient);
    return;
  }
  $('#performanceMetricsLayout').hidden = false;
  $('#performanceChartCard').hidden = false;
  $('#modelComparisonCard').hidden = false;
  const clinical = dataset === 'sim' ? null : patientClinicalCache.get(patient);
  const patientName = `${dataset === 'sim' ? 'SimGlucose' : 'OhioT1DM'} patient ${patient}`;
  $('#performanceDescription').textContent = `Historical RMSE, MAE, and Clarke zones for ${patientName}.`;
  $('#perfCards').innerHTML = HZ.map((horizon) => {
    const metric = metrics[String(horizon)];
    const zoneShares = metric.zone_percentages || clinical?.[String(horizon)]?.percentages;
    const ab = zoneShares ? zoneShares.A + zoneShares.B : null;
    return stat(`${horizon} min · RMSE`, `${metric.rmse.toFixed(2)} mg/dL`, `MAE ${metric.mae.toFixed(2)} mg/dL · ${metric.n_rows} scored readings · A+B ${ab == null ? '—' : `${ab.toFixed(2)}%`}`);
  }).join('');
  $('#hTable').innerHTML = `<tr><th>Horizon</th><th>RMSE</th><th>MAE</th><th>Scored readings</th><th>Clarke A+B</th></tr>${HZ.map((horizon) => {
    const metric = metrics[String(horizon)];
    const zones = metric.zone_percentages || clinical?.[String(horizon)]?.percentages;
    const ab = zones ? `${(zones.A + zones.B).toFixed(2)}%` : '—';
    return `<tr class="${horizon === state.horizon ? 'is-selected' : ''}"><td>${horizon} min</td><td>${metric.rmse.toFixed(2)} mg/dL</td><td>${metric.mae.toFixed(2)} mg/dL</td><td>${metric.n_rows}</td><td>${ab}</td></tr>`;
  }).join('')}`;
  $('#mTable').innerHTML = `<tr><th>Metric · ${esc(patientName)}</th>${HZ.map((horizon) => `<th>${horizon} min</th>`).join('')}</tr>
    <tr><td>RMSE (mg/dL)</td>${HZ.map((horizon) => `<td>${metrics[String(horizon)].rmse.toFixed(2)}</td>`).join('')}</tr>
    <tr><td>MAE (mg/dL)</td>${HZ.map((horizon) => `<td>${metrics[String(horizon)].mae.toFixed(2)}</td>`).join('')}</tr>
    <tr><td>Scored readings</td>${HZ.map((horizon) => `<td>${metrics[String(horizon)].n_rows}</td>`).join('')}</tr>`;
  if (state.view === 'performance') {
    const chart = ensureErrorChart();
    chart.data.datasets[0].label = `${patientName} RMSE (mg/dL)`;
    chart.data.datasets[0].data = HZ.map((horizon) => metrics[String(horizon)]?.rmse ?? null);
    chart.update(reduceMotion ? 'none' : undefined);
  }
}

function renderModel() {
  const cards = [
    ['Ohio model', '18 CGM features', 'Shared XGBoost models trained on pooled OhioT1DM patient records. Patient groups are kept separate while building time-series features.'],
    ['SimGlucose model', '28 past-only features', 'Shared models trained across 30 virtual patients: CGM history, past insulin, and past CHO. Missing CHO is zero-filled.'],
    ['Forecast horizons', '4 models per path', 'Separate 30, 60, 90, and 120 minute models. Current selection: ' + state.horizon + ' minutes.'],
    ['At prediction time', 'Patient history', 'Each uploaded file supplies the patient-specific input history; model weights are shared across training patients.'],
  ];
  $('#modelCards').innerHTML = cards.map(([label, value, detail]) => stat(label, value, detail)).join('');
  $all('[data-model-horizon]').forEach((node) => { node.textContent = String(state.horizon); });
  paintIcons();
}

function renderAll(opts = {}) {
  renderOverview(opts);
  renderForecast(opts);
  renderExplorer(opts);
  renderClinical();
  renderPerformance();
  renderModel();
  requestAnimationFrame(resizeActiveCharts);
}

/* -----------------------------------------------------------------------------
   Selects, navigation, predict interaction
   ----------------------------------------------------------------------------- */

function syncSelects() {
  $all('.sel-patient').forEach((select) => {
    select.value = state.patient;
  });
  $all('.sel-case').forEach((select) => {
    select.innerHTML = casesFor(state.patient)
      .map((item) => `<option value="${esc(item.id)}">${esc(item.label)}</option>`)
      .join('');
    select.value = state.caseId;
  });
  const horizonSelect = $('#selH');
  if (horizonSelect) horizonSelect.value = String(state.horizon);
  const sidebarPatient = $('#sidebarPatient');
  if (sidebarPatient) sidebarPatient.value = state.dataset === 'sim'
    ? `sim|${state.simPatient}`
    : `ohio|${state.patient}`;
}

function setBusy(busy) {
  const run = $('#run');
  const random = $('#rand');
  run.disabled = busy;
  random.disabled = busy;
  run.setAttribute('aria-busy', busy ? 'true' : 'false');
  if (busy) {
    run.innerHTML = '<span class="spinner" aria-hidden="true"></span><span>Running…</span>';
  } else {
    run.innerHTML = '<i data-lucide="play" aria-hidden="true"></i><span>Run Forecast</span>';
    random.innerHTML = '<i data-lucide="shuffle" aria-hidden="true"></i><span>Random Test Case</span>';
    paintIcons();
  }
}

function showExplorerLoading() {
  const results = $('#results');
  results.classList.add('is-loading');
  results.innerHTML = `<div class="skeleton-grid">${Array.from({ length: 5 }, () => '<div class="skeleton-card" aria-hidden="true"></div>').join('')}</div><div class="loading-status" role="status">${LOAD_STEPS[0]}</div>`;
}

function cycleStatus() {
  let step = 0;
  clearInterval(statusTimer);
  statusTimer = setInterval(() => {
    step = Math.min(step + 1, LOAD_STEPS.length - 1);
    const status = $('#results .loading-status');
    if (status) status.textContent = LOAD_STEPS[step];
  }, 480);
  return () => clearInterval(statusTimer);
}

function applyPrediction(result) {
  const merged = { ...getCase(state.caseId).preds, ...result.horizons };
  state.forecastFor = state.caseId;
  if (result.source === 'api') {
    livePredictions.set(state.caseId, merged);
    state.usingDemo = false;
    state.connected = true;
    showDemoBadge(false);
  } else {
    livePredictions.delete(state.caseId);
    state.usingDemo = true;
    showDemoBadge(true);
  }
  getActiveForecastContext();
  renderStatus();
}

async function runForecast() {
  if (state.running) return;
  state.dataset = 'ohio';
  // Hide any earlier result so nothing is visible until this run finishes.
  state.forecastFor = null;
  renderAll({ animate: false, draw: false });
  state.running = true;
  setBusy(true);
  showExplorerLoading();
  const stopCycle = cycleStatus();
  const started = performance.now();
  let result;
  let failed = false;

  try {
    result = await api.predict(state.patient, state.caseId, state.horizon);
  } catch (error) {
    failed = true;
    result = mockPrediction(state.caseId);
    console.warn('Glucose Horizon predict fell back to demo data.', error.message);
  }

  const delay = 1500 + Math.random() * 1000;
  const remaining = delay - (performance.now() - started);
  if (remaining > 0) await sleep(remaining);

  stopCycle();
  applyPrediction(result);
  state.running = false;
  setBusy(false);
  renderAll({ animate: true, draw: true });
  if (failed) {
    showToast(state.checked && !state.connected
      ? 'Backend unreachable. Showing demo data.'
      : 'Forecast request failed. Showing demo data.');
  }
}

function chooseRandomCase() {
  if (state.running) return;
  state.dataset = 'ohio';
  const ids = CASES.map((item) => item.id);
  let next = ids[Math.floor(Math.random() * ids.length)];
  if (ids.length > 1) {
    while (next === state.caseId) next = ids[Math.floor(Math.random() * ids.length)];
  }
  const sample = getCase(next);
  state.patient = sample.patient;
  state.caseId = sample.id;
  syncSelects();
  loadPatientMetrics(state.patient);
  runForecast();
}

function showView(name) {
  state.view = name;
  $all('.view').forEach((view) => {
    const active = view.id === `v-${name}`;
    view.classList.toggle('active', active);
    view.hidden = !active;
  });
  $all('#nav button').forEach((button) => {
    const active = button.dataset.view === name;
    button.classList.toggle('active', active);
    if (active) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  if (mobileQuery.matches) setDrawer(false);
  const view = $(`#v-${name}`);
  if (view) void view.offsetWidth;
  renderAll({ animate: false, draw: true });
}

function setDrawer(open) {
  const sidebar = $('#sidebar');
  const backdrop = $('#backdrop');
  const toggle = $('#menuToggle');
  const shell = $('.shell');
  const mobile = mobileQuery.matches;
  sidebar.classList.toggle('open', open && mobile);
  backdrop.hidden = !(open && mobile);
  toggle.setAttribute('aria-expanded', open && mobile ? 'true' : 'false');
  toggle.setAttribute('aria-label', open && mobile ? 'Close navigation menu' : 'Open navigation menu');
  if ('inert' in sidebar) {
    sidebar.inert = mobile && !open;
    if (shell) shell.inert = mobile && open;
  }
  if (open && mobile) {
    const current = $('#nav button.active') || $('#nav button');
    if (current) current.focus();
  }
}

function bindNavigation() {
  $('#nav').addEventListener('click', (event) => {
    const button = event.target.closest('button');
    if (!button) return;
    showView(button.dataset.view);
  });

  $('#nav').addEventListener('keydown', (event) => {
    const buttons = $all('#nav button');
    const index = buttons.indexOf(document.activeElement);
    if (index < 0) return;
    let next = index;
    if (event.key === 'ArrowDown' || event.key === 'ArrowRight') next = (index + 1) % buttons.length;
    else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') next = (index - 1 + buttons.length) % buttons.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = buttons.length - 1;
    else return;
    event.preventDefault();
    buttons[next].focus();
    showView(buttons[next].dataset.view);
  });
}

function bindSelects() {
  $all('.sel-patient').forEach((select) => {
    select.addEventListener('change', () => {
      if (state.running) {
        syncSelects();
        return;
      }
      state.dataset = 'ohio';
      state.patient = select.value;
      state.caseId = casesFor(state.patient)[0].id;
      state.forecastFor = null;
      state.lastUploadForecast = null;
      syncSelects();
      renderAll({ animate: false, draw: true });
      loadPatientMetrics(state.patient);
    });
  });

  $all('.sel-case').forEach((select) => {
    select.addEventListener('change', () => {
      if (state.running) {
        syncSelects();
        return;
      }
      state.dataset = 'ohio';
      state.caseId = select.value;
      state.patient = getCase(state.caseId).patient;
      state.forecastFor = null;
      state.lastUploadForecast = null;
      syncSelects();
      renderAll({ animate: false, draw: true });
      loadPatientMetrics(state.patient);
    });
  });

  $('#selH').addEventListener('change', () => {
    if (state.running) {
      $('#selH').value = String(state.horizon);
      return;
    }
    state.dataset = 'ohio';
    state.horizon = Number($('#selH').value);
    state.forecastFor = null;
    state.lastUploadForecast = null;
    renderAll({ animate: false, draw: true });
  });
  $('#uploadEgaHorizon')?.addEventListener('change', (event) => {
    state.horizon = Number(event.currentTarget.value);
    renderAll({ animate: false, draw: true });
  });
}

function bindChrome() {
  const patientViewToggle = $('#patientViewToggle');
  patientViewToggle?.addEventListener('change', () => {
    document.body.classList.toggle('patient-view', patientViewToggle.checked);
    showView('overview');
  });
  $('#overviewAgentBtn')?.addEventListener('click', () => {
    showView('agent');
    const input = $('#agentInput');
    input.value = hasUploadedPatientContext()
      ? 'Please explain my uploaded glucose history and forecasts, and suggest safe next steps based on the current readings.'
      : '';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.focus();
  });
  $('#menuToggle').addEventListener('click', () => {
    const open = !$('#sidebar').classList.contains('open');
    setDrawer(open);
  });
  $('#backdrop').addEventListener('click', () => setDrawer(false));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') setDrawer(false);
  });
  mobileQuery.addEventListener('change', () => setDrawer(false));
  document.addEventListener('click', (event) => {
    const link = event.target.closest('[data-go]');
    if (link) showView(link.dataset.go);
  });
  const referenceToggle = $('#showReferenceEga');
  const referenceControl = $('#referencePatientControl');
  const referenceSelect = $('#referencePatient');
  if (referenceToggle && referenceControl && referenceSelect) {
    populateAnalysisPatientOptions();
    referenceToggle.addEventListener('change', () => {
      referenceControl.hidden = !referenceToggle.checked;
      if (!referenceToggle.checked) referenceSelect.value = '';
      renderClinical();
    });
    referenceSelect.addEventListener('change', () => {
      const [dataset, patient] = referenceSelect.value.split('|');
      if (dataset === 'ohio' && patient) loadPatientMetrics(patient);
      if (dataset === 'sim' && patient) loadSimReference(patient);
      renderClinical();
    });
  }
  const perfToggle = $('#showPerformancePatient');
  const perfControl = $('#performancePatientControl');
  const perfSelect = $('#performancePatient');
  if (perfToggle && perfControl && perfSelect) {
    populateAnalysisPatientOptions();
    perfToggle.addEventListener('change', () => {
      perfControl.hidden = !perfToggle.checked;
      if (!perfToggle.checked) perfSelect.value = '';
      renderPerformance();
    });
    perfSelect.addEventListener('change', () => {
      const [dataset, patient] = perfSelect.value.split('|');
      if (dataset === 'ohio' && patient) loadPatientMetrics(patient);
      if (dataset === 'sim' && patient) loadSimReference(patient);
      renderPerformance();
    });
  }
  $('#run').addEventListener('click', runForecast);
  $('#rand').addEventListener('click', chooseRandomCase);
  $('#runSimTest').addEventListener('click', runSimTest);
  ['simTestPatient', 'simTestCase', 'simTestHorizon'].forEach((id) => {
    $(`#${id}`).addEventListener('change', () => {
      state.dataset = 'sim';
      state.horizon = Number($('#simTestHorizon').value);
      state.simPatient = $('#simTestPatient').value;
      $('#simTestResults').innerHTML = '';
      $('#simTestContext').innerHTML = '';
      $('#simTestEvents').innerHTML = '';
      loadSimTestContext();
    });
  });
  $('#uploadMode').addEventListener('change', updateUploadHelp);
  $('#chooseCsv').addEventListener('click', () => $('#uploadCsv').click());
  $('#uploadCsv').addEventListener('change', () => {
    $('#uploadFileName').textContent = $('#uploadCsv').files[0]?.name || 'No file selected';
    state.lastUploadForecast = null;
    state.lastUploadFileName = $('#uploadCsv').files[0]?.name || '';
    $('#askCsvAgent').disabled = true;
    document.dispatchEvent(new Event('agent-upload-context-change'));
    renderAll({ animate: false, draw: true });
  });
  $('#uploadRun').addEventListener('click', uploadAndPredict);
  $('#askCsvAgent').addEventListener('click', () => {
    if (!state.lastUploadForecast) return;
    showView('agent');
    const input = $('#agentInput');
    input.value = `Using the uploaded patient’s glucose history and latest forecast, suggest practical meal ideas and safe physical activity options. Explain how the suggestions relate to the patient’s glucose trend, and do not provide insulin dosing advice.`;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    $('#agentChatForm').requestSubmit();
  });
  renderUploadModels();
}

async function loadSimTestPatients() {
  const select = $('#simTestPatient');
  try {
    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.simTestPatients}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Held-out SimGlucose models unavailable');
    simTestPatientIds = data.patients || [];
    select.innerHTML = simTestPatientIds.map((id) => `<option value="${esc(id)}">${esc(id)}</option>`).join('');
    fillSidebarPatients();
    populateAnalysisPatientOptions();
    select.addEventListener('change', () => {
      state.dataset = 'sim';
      state.simPatient = select.value;
      state.horizon = Number($('#simTestHorizon').value || state.horizon);
      loadSimTestContext();
    });
    loadSimTestContext();
  } catch (error) {
    simTestPatientIds = [];
    fillSidebarPatients();
    populateAnalysisPatientOptions();
    select.innerHTML = '<option value="">Unavailable</option>';
    $('#simTestResults').innerHTML = `<div class="stat"><small>SimGlucose evaluation</small><b>Unavailable</b><span>${esc(error.message)}</span></div>`;
  }
}

function populateAnalysisPatientOptions() {
  const markup = [
    '<option value="">Choose a patient…</option>',
    `<optgroup label="OhioT1DM test patients">${patients().map((id) => `<option value="ohio|${esc(id)}">Patient ${esc(id)}</option>`).join('')}</optgroup>`,
    `<optgroup label="SimGlucose test patients">${simTestPatientIds.map((id) => `<option value="sim|${esc(id)}">${esc(id)}</option>`).join('')}</optgroup>`,
  ].join('');
  ['#referencePatient', '#performancePatient'].forEach((selector) => {
    const select = $(selector);
    if (!select) return;
    const value = select.value;
    select.innerHTML = markup;
    if (Array.from(select.options).some((option) => option.value === value)) select.value = value;
  });
}

async function loadSimReference(patientId) {
  if (simReferenceCache.has(patientId) || simReferenceLoading.has(patientId)) return;
  simReferenceLoading.add(patientId);
  simReferenceFailed.delete(patientId);
  try {
    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.simTestPatient}/${encodeURIComponent(patientId)}?horizon=${state.horizon}&case_index=2`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Could not load SimGlucose patient metrics');
    simReferenceCache.set(patientId, data);
  } catch (error) {
    simReferenceFailed.add(patientId);
    console.warn(`Could not load SimGlucose reference patient ${patientId}.`, error.message);
  } finally {
    simReferenceLoading.delete(patientId);
    const refSelected = $('#showReferenceEga')?.checked && $('#referencePatient')?.value === `sim|${patientId}`;
    const perfSelected = $('#showPerformancePatient')?.checked && $('#performancePatient')?.value === `sim|${patientId}`;
    if (refSelected || perfSelected) renderAll({ animate: false, draw: false });
  }
}

function buildSimDashboardSample(data) {
  const id = `sim|${data.patient_id}|${$('#simTestCase').value}`;
  const predictions = Object.fromEntries(Object.entries(data.predictions).map(([horizon, item]) => [Number(horizon), item]));
  const series = [...data.series.history, ...data.series.actual_future];
  const referenceTime = Date.parse(data.timestamp);
  return {
    id,
    patient: data.patient_id,
    label: `Held-out case · ${data.timestamp}`,
    series,
    timestamps: series.map((point) => new Date(referenceTime + point.x * 60_000).toISOString()),
    preds: predictions,
    metrics: data.metrics,
    scatter: data.scatter,
    context: data.context,
    events: [`Insulin in the last 60 minutes: ${data.context.insulin_60}`, `CHO in the last 60 minutes: ${data.context.cho_60} g`],
  };
}

async function loadSimDashboard(options = {}) {
  const patient = state.simPatient || $('#simTestPatient').value;
  if (!patient) return;
  state.dataset = 'sim';
  state.simPatient = patient;
  const horizon = Number($('#simTestHorizon').value || state.horizon);
  state.horizon = horizon;
  const caseIndex = $('#simTestCase').value || 2;
  const requestId = ++simDashboardRequestId;
  const simResults = $('#simTestResults');
  if (!simResults.classList.contains('is-loading')) simResults.innerHTML = '<div class="stat"><small>SimGlucose data</small><b>Loading…</b><span>Updating charts for this virtual patient</span></div>';
  try {
    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.simTestPatient}/${encodeURIComponent(patient)}?horizon=${horizon}&case_index=${caseIndex}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Could not load SimGlucose patient data');
    if (requestId !== simDashboardRequestId) return;
    if (options.startedAt !== undefined && options.delayMs) {
      const remaining = options.delayMs - (performance.now() - options.startedAt);
      if (remaining > 0) await sleep(remaining);
      if (requestId !== simDashboardRequestId) return;
    }
    simResults.classList.remove('is-loading');
    state.simSample = buildSimDashboardSample(data);
    getActiveForecastContext();
    renderSimTestResults(data);
    renderAll({ animate: true, draw: true });
  } catch (error) {
    if (requestId !== simDashboardRequestId) return;
    simResults.classList.remove('is-loading');
    $('#simTestResults').innerHTML = `<div class="stat"><small>SimGlucose data</small><b>Unavailable</b><span>${esc(error.message)}</span></div>`;
  }
}

let simContextRequestId = 0;
async function loadSimTestContext() {
  const patient = $('#simTestPatient')?.value;
  if (!patient) return;
  const requestId = ++simContextRequestId;
  const horizon = Number($('#simTestHorizon').value || state.horizon);
  const caseIndex = $('#simTestCase').value || 2;
  $('#simTestContext').innerHTML = '<div class="stat idle"><small>CGM context</small><b>Loading…</b><span>For the selected patient and case</span></div>';
  $('#simTestEvents').innerHTML = '';
  try {
    const response = await fetch(`${API_BASE_URL}${ENDPOINTS.simTestPatient}/${encodeURIComponent(patient)}?horizon=${horizon}&case_index=${caseIndex}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Could not load selected patient context');
    if (requestId !== simContextRequestId) return;
    renderSimTestContext(data.context);
  } catch (error) {
    if (requestId !== simContextRequestId) return;
    $('#simTestContext').innerHTML = `<div class="stat"><small>CGM context</small><b>Unavailable</b><span>${esc(error.message)}</span></div>`;
  }
}

function renderSimTestResults(data) {
  $('#simTestResults').classList.remove('is-loading');
  const horizon = Number($('#simTestHorizon').value || state.horizon);
  const selected = data.predictions[String(horizon)];
  const risk = rangeOf(selected.predicted);
  const cards = [
    ['Predicted', `${mg(selected.predicted)} mg/dL`, `${horizon} min ahead`, risk.cls],
    ['Actual', `${mg(selected.actual)} mg/dL`, `${data.timestamp} · held-out value`, ''],
    ['Case error', `${mg(selected.error)} mg/dL`, `Patient-wide RMSE: ${mg(data.metrics[String(horizon)].rmse)} mg/dL`, ''],
    ['Clarke zone', selected.clarke_zone, `${data.n_rows.toLocaleString()} scored patient readings`, ZONE_CLASS[selected.clarke_zone] || ''],
    ['Risk level', risk.label, 'Based on selected forecast glucose', risk.cls],
  ];
  $('#simTestResults').innerHTML = cards.map(([label, value, detail, cls], i) => stat(label, value, detail, cls, { reveal: true, delay: i * 70 })).join('');
  renderSimTestContext(data.context);
  paintIcons();
}

function renderSimTestContext(context) {
  $('#simTestContext').innerHTML = [
    stat('Current glucose', `${mg(context.current)} mg/dL`, 'At selected forecast time'),
    stat('Rate of change', `${context.roc >= 0 ? '+' : ''}${mg(context.roc)} mg/dL`, 'Recent CGM trend'),
    stat('Recent mean', `${mg(context.mean)} mg/dL`, '3-hour window'),
    stat('Variability', `${mg(context.sd)} mg/dL`, 'Standard deviation'),
    stat('Time in range', `${context.tir}%`, '70–180 mg/dL in window'),
  ].map((card, i) => card.replace('<div class="stat"', `<div class="stat reveal" style="animation-delay:${i * 70}ms"`)).join('');
  $('#simTestEvents').innerHTML = `<li>Insulin delivered in the last 60 minutes: ${context.insulin_60}</li><li>CHO recorded in the last 60 minutes: ${context.cho_60} g</li>`;
  paintIcons();
}

async function runSimTest() {
  const patient = $('#simTestPatient').value;
  if (!patient) return;
  const button = $('#runSimTest');
  button.disabled = true;
  button.setAttribute('aria-busy', 'true');
  const results = $('#simTestResults');
  results.classList.add('is-loading');
  results.innerHTML = `<div class="skeleton-grid">${Array.from({ length: 5 }, () => '<div class="skeleton-card" aria-hidden="true"></div>').join('')}</div><div class="loading-status" role="status">${LOAD_STEPS[0]}</div>`;
  let step = 0;
  const timer = setInterval(() => {
    step = Math.min(step + 1, LOAD_STEPS.length - 1);
    const status = results.querySelector('.loading-status');
    if (status) status.textContent = LOAD_STEPS[step];
  }, 480);
  try {
    state.dataset = 'sim';
    state.simPatient = patient;
    const startedAt = performance.now();
    const delayMs = 1500 + Math.random() * 1000;
    await loadSimDashboard({ startedAt, delayMs });
  } catch (error) {
    $('#simTestResults').innerHTML = `<div class="stat"><small>Evaluation</small><b>Failed</b><span>${esc(error.message)}</span></div>`;
  } finally {
    clearInterval(timer);
    button.disabled = false;
    button.removeAttribute('aria-busy');
    paintIcons();
  }
}

function updateUploadHelp() {
  const multimodal = $('#uploadMode').value === 'multimodal';
  $('#uploadHelp').textContent = multimodal
    ? 'CSV needs timestamp, CGM (mg/dL), and insulin (units per row); CHO (grams per row) is optional and defaults to zero. SimGlucose history columns Time, BG, CGM, CHO, insulin are supported.'
    : 'CSV needs timestamp and glucose (mg/dL), or SimGlucose columns Time and CGM, with at least 25 sequential readings at roughly 5-minute intervals.';
}

async function detectCsvProfile(file) {
  const firstLine = (await file.slice(0, 8192).text()).split(/\r?\n/, 1)[0] || '';
  const headers = (firstLine.match(/("(?:[^"]|"")*"|[^,]*)(?:,|$)/g) || [])
    .map((part) => part.replace(/,$/, '').trim().replace(/^"|"$/g, '').replace(/""/g, '"').toLowerCase());
  const hasCgm = headers.includes('cgm');
  const hasInsulin = headers.includes('insulin');
  const isSimulation = hasInsulin && (hasCgm || headers.includes('bg') || headers.includes('glucose'));
  return { headers, isSimulation, isCgmOnly: !isSimulation && (headers.includes('glucose') || headers.includes('bg') || hasCgm) };
}

async function renderUploadModels() {
  const table = $('#uploadModels');
  if (!table) return;
  table.innerHTML = '<tr><th>Model family</th><th>Inputs</th><th>Horizons</th><th>Status</th></tr>' +
    '<tr><td>Ohio XGBoost</td><td>CGM history</td><td>30 / 60 / 90 / 120 min</td><td>Available</td></tr>' +
    '<tr><td>SimGlucose XGBoost</td><td>CGM + insulin + optional CHO</td><td>30 / 60 / 90 / 120 min</td><td>Checking…</td></tr>';
  try {
    const data = await api.getMultimodalModels();
    const status = data && data.models_loaded ? 'Available' : 'Not trained yet';
    table.innerHTML = '<tr><th>Model family</th><th>Inputs</th><th>Horizons</th><th>Status</th></tr>' +
      `<tr><td>Ohio XGBoost</td><td>CGM history</td><td>30 / 60 / 90 / 120 min</td><td>Available</td></tr><tr><td>SimGlucose XGBoost</td><td>CGM + insulin + optional CHO</td><td>30 / 60 / 90 / 120 min</td><td>${status}</td></tr>`;
  } catch {
    table.innerHTML = '<tr><th>Model family</th><th>Inputs</th><th>Horizons</th><th>Status</th></tr><tr><td>Ohio XGBoost</td><td>CGM history</td><td>30 / 60 / 90 / 120 min</td><td>Available</td></tr><tr><td>SimGlucose XGBoost</td><td>CGM + insulin + optional CHO</td><td>30 / 60 / 90 / 120 min</td><td>Backend unavailable</td></tr>';
  }
}

async function uploadAndPredict() {
  const file = $('#uploadCsv').files[0];
  const multimodal = $('#uploadMode').value === 'multimodal';
  let dataProfile = null;
  const results = $('#uploadResults');
  if (!file) {
    showToast('Choose a CSV file first.', 'warn');
    return;
  }
  const button = $('#uploadRun');
  const started = performance.now();
  const delay = 1500 + Math.random() * 1000;
  button.disabled = true;
  button.setAttribute('aria-busy', 'true');
  button.innerHTML = '<span class="spinner" aria-hidden="true"></span><span>Running…</span>';
  results.innerHTML = '<div class="stat"><span class="stat-label">Upload</span><span class="stat-value">Processing…</span><span class="stat-detail">Validating history and running the selected models</span></div>';
  let data;
  let requestError;
  try {
    dataProfile = await detectCsvProfile(file);
    // Keep all horizon records available to dashboard views; the selected horizon
    // controls which upload cards are shown and which EGA is selected.
    data = await api.predictCsv(file, 'all', multimodal);
  } catch (error) {
    requestError = error;
  }
  const remaining = delay - (performance.now() - started);
  if (remaining > 0) await sleep(remaining);
  try {
    if (requestError) throw requestError;
    const requestedHorizon = $('#uploadHorizon').value;
    if (requestedHorizon !== 'all') state.horizon = Number(requestedHorizon);
    const selectedModel = multimodal ? 'multimodal' : 'cgm';
    const previousUpload = state.lastUploadForecast;
    const newAgentPredictions = requestedHorizon === 'all'
      ? [...(data.predictions || [])]
      : (data.predictions || []).filter((item) => Number(item.horizon) === Number(requestedHorizon));
    const samePatientHistory = previousUpload &&
      previousUpload.row_count === data.row_count &&
      previousUpload.patient_used === data.patient_used &&
      previousUpload.timestamp === data.timestamp &&
      Array.isArray(previousUpload.readings) && Array.isArray(data.readings) &&
      previousUpload.readings.length === data.readings.length &&
      previousUpload.readings.every((row, index) => row.timestamp === data.readings[index]?.timestamp && Number(row.glucose) === Number(data.readings[index]?.glucose));
    const sameForecastFamily = samePatientHistory && previousUpload.selected_model === selectedModel;
    const mergedAgentPredictions = new Map();
    if (sameForecastFamily) {
      (previousUpload.agent_predictions || []).forEach((item) => mergedAgentPredictions.set(Number(item.horizon), item));
    }
    newAgentPredictions.forEach((item) => mergedAgentPredictions.set(Number(item.horizon), item));
    state.lastUploadForecast = {
      ...data,
      requested_horizon: requestedHorizon,
      agent_predictions: [...mergedAgentPredictions.values()],
      dataset_context_key: samePatientHistory
        ? previousUpload.dataset_context_key
        : `patient-data-${Math.random().toString(36).slice(2, 11)}`,
      selected_model: selectedModel,
      data_profile: dataProfile,
      agentContextId: data.agent_dataset_id || '',
    };
    state.lastUploadFileName = file.name;
    $('#askCsvAgent').disabled = !hasUploadedPatientContext();
    getActiveForecastContext();
    document.dispatchEvent(new Event('agent-upload-context-change'));
    const predictions = (data.predictions || []).filter((item) => requestedHorizon === 'all' || Number(item.horizon) === Number(requestedHorizon));
    const disclaimerBanner = `
      <div class="alert-banner disclaimer-banner" style="grid-column: 1 / -1; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; padding: 10px 14px; background: rgba(251, 191, 36, 0.08); border: 1px solid rgba(251, 191, 36, 0.28); border-radius: 8px;">
        <i data-lucide="shield-alert" style="width: 16px; height: 16px; flex-shrink: 0; color: var(--warn);"></i>
        <span style="font-size: 12px; line-height: 1.4; color: var(--text);">
          <strong style="color: var(--warn);">Clinical Research Disclaimer:</strong> Investigational prototype. ${multimodal ? 'The SimGlucose model uses CGM, past insulin, and optional carbohydrate history.' : 'The Ohio model uses CGM history only.'} Predictions are not for clinical or treatment use. Consult your diabetes care team.
        </span>
      </div>
    `;
    const mismatchBanner = !multimodal && dataProfile?.isSimulation
      ? '<div class="alert-banner danger upload-model-warning" role="alert"><strong>MODEL / DATA MISMATCH:</strong> This CSV looks like SimGlucose data (CGM and insulin columns), but the Ohio CGM-only model was selected. Predictions are shown, but this model ignores insulin and may not match the dataset. Select the SimGlucose model and upload again for a matching evaluation.</div>'
      : '';
    results.innerHTML = disclaimerBanner + mismatchBanner + predictions.map((item) => stat(
      `${item.horizon} min forecast`,
      `${mg(item.predicted_glucose)} mg/dL`,
      `Change from latest ${mg(item.change_from_latest)} mg/dL · ${item.zone_label}`,
      item.zone_label === 'low' ? 'bad' : item.zone_label === 'high' ? 'warn' : 'ok'
    )).join('');
    renderAll({ animate: true, draw: true });
    if (multimodal && data.meal_defaulted_to_zero) showToast('No CHO column found; meals were treated as zero.');
  } catch (error) {
    $('#askCsvAgent').disabled = true;
    const mismatch = multimodal && dataProfile?.isCgmOnly;
    results.innerHTML = `<div class="alert-banner danger upload-model-error" role="alert"><strong>${mismatch ? 'WRONG MODEL FOR THIS FILE' : 'UPLOAD PREDICTION FAILED'}</strong><p>${mismatch ? 'This CSV appears to contain glucose readings without the CGM + insulin columns required by the SimGlucose model. Choose the Ohio CGM-only model to predict from this file.' : esc(error.message)}</p></div>`;
    renderAll({ animate: false, draw: true });
  } finally {
    button.disabled = false;
    button.removeAttribute('aria-busy');
    button.innerHTML = '<i data-lucide="upload" aria-hidden="true"></i><span>Upload &amp; Predict</span>';
    paintIcons();
    renderUploadModels();
  }
}

function fillPatients() {
  const options = patients().map((id) => `<option value="${esc(id)}">Patient ${esc(id)}</option>`).join('');
  $all('.sel-patient').forEach((select) => {
    select.innerHTML = options;
  });
  fillSidebarPatients();
}

function fillSidebarPatients() {
  const select = $('#sidebarPatient');
  if (!select) return;
  const ohioOptions = patients().map((id) => `<option value="ohio|${esc(id)}">Patient ${esc(id)}</option>`).join('');
  const simOptions = simTestPatientIds.map((id) => `<option value="sim|${esc(id)}">${esc(id)}</option>`).join('');
  const previous = select.value;
  select.innerHTML = `<optgroup label="OhioT1DM patients">${ohioOptions}</optgroup><optgroup label="SimGlucose held-out patients">${simOptions || '<option value="sim|">Unavailable</option>'}</optgroup>`;
  select.value = previous && Array.from(select.options).some((option) => option.value === previous)
    ? previous
    : `ohio|${state.patient || patients()[0]}`;
}

function renderMarkdown(text) {
  if (!text) return '';
  const lines = text.split('\n');
  const html = [];
  let inList = false;
  let listType = null;
  let inTable = false;
  let tableRows = [];
  let inQuote = false;
  let quoteLines = [];
  let isOverviewSection = false;
  let overviewMetrics = [];

  function flushList() {
    if (inList) {
      html.push(listType === 'ol' ? '</ol>' : '</ul>');
      inList = false;
      listType = null;
    }
  }

  function flushOverviewMetrics() {
    if (overviewMetrics.length > 0) {
      let cardsHtml = '<div class="report-metrics-grid">';
      overviewMetrics.forEach(({ label, rest }) => {
        let value = rest;
        let sub = '';
        const parenMatch = rest.match(/^(.*?)\s*\((.*?)\)$/);
        const sepMatch = rest.match(/^(.*?)\s*[·|]\s*(.*?)$/);
        if (parenMatch) {
          value = parenMatch[1];
          sub = parenMatch[2];
        } else if (sepMatch) {
          value = sepMatch[1];
          sub = sepMatch[2];
        }
        cardsHtml += `
          <div class="report-metric-card">
            <span class="report-metric-label">${inlineFormat(esc(label.trim()))}</span>
            <span class="report-metric-value">${inlineFormat(esc(value.trim()))}</span>
            ${sub ? `<span class="report-metric-sub">${inlineFormat(esc(sub.trim()))}</span>` : ''}
          </div>
        `;
      });
      cardsHtml += '</div>';
      html.push(cardsHtml);
      overviewMetrics = [];
    }
  }

  function flushQuote() {
    if (inQuote) {
      const quoteText = quoteLines.join(' ');
      let calloutClass = 'report-callout';
      const lower = quoteText.toLowerCase();
      if (lower.includes('emergency') || lower.includes('critical') || lower.includes('severe')) {
        calloutClass += ' alert';
      } else if (lower.includes('warn') || lower.includes('caution') || lower.includes('disclaimer')) {
        calloutClass += ' warn';
      }
      html.push(`<blockquote class="${calloutClass}">${inlineFormat(esc(quoteText))}</blockquote>`);
      inQuote = false;
      quoteLines = [];
    }
  }

  function parseTableRow(line) {
    let clean = line.trim();
    if (clean.startsWith('|')) clean = clean.slice(1);
    if (clean.endsWith('|')) clean = clean.slice(0, -1);
    return clean.split('|').map((c) => c.trim());
  }

  function flushTable() {
    if (inTable && tableRows.length > 0) {
      let headerCells = [];
      let bodyRows = [];
      let isFirstRowHeader = false;

      if (tableRows.length >= 2 && /^\|?(\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?$/.test(tableRows[1].trim())) {
        isFirstRowHeader = true;
        headerCells = parseTableRow(tableRows[0]);
        for (let i = 2; i < tableRows.length; i++) {
          bodyRows.push(parseTableRow(tableRows[i]));
        }
      } else {
        for (let i = 0; i < tableRows.length; i++) {
          bodyRows.push(parseTableRow(tableRows[i]));
        }
      }

      let tableHtml = '<div class="report-table-wrap"><table class="report-table">';
      if (isFirstRowHeader && headerCells.length > 0) {
        tableHtml += '<thead><tr>' + headerCells.map((c) => `<th>${inlineFormat(esc(c))}</th>`).join('') + '</tr></thead>';
      }
      if (bodyRows.length > 0) {
        tableHtml += '<tbody>';
        bodyRows.forEach((row) => {
          tableHtml += '<tr>' + row.map((c) => `<td>${inlineFormat(esc(c))}</td>`).join('') + '</tr>';
        });
        tableHtml += '</tbody>';
      }
      tableHtml += '</table></div>';
      html.push(tableHtml);

      inTable = false;
      tableRows = [];
    }
  }

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    const line = rawLine.trim();

    // Table line
    const isTableLine = line.startsWith('|') && line.endsWith('|') && line.length > 2;
    if (isTableLine) {
      flushList();
      flushOverviewMetrics();
      flushQuote();
      inTable = true;
      tableRows.push(line);
      continue;
    } else if (inTable) {
      flushTable();
    }

    // Blockquote
    if (line.startsWith('>')) {
      flushList();
      flushOverviewMetrics();
      inQuote = true;
      quoteLines.push(line.replace(/^>\s*/, ''));
      continue;
    } else if (inQuote) {
      flushQuote();
    }

    if (!line) {
      flushList();
      flushOverviewMetrics();
      continue;
    }

    // Horizontal Rule
    if (/^(\-{3,}|\*{3,}|_{3,})$/.test(line)) {
      flushList();
      flushOverviewMetrics();
      html.push('<hr class="report-hr" />');
      continue;
    }

    // Headings
    if (line.startsWith('#### ')) {
      flushList();
      flushOverviewMetrics();
      const title = line.slice(5).trim();
      isOverviewSection = /overview|metric|summary/i.test(title);
      html.push(`<h4>${inlineFormat(esc(title))}</h4>`);
      continue;
    }
    if (line.startsWith('### ')) {
      flushList();
      flushOverviewMetrics();
      const title = line.slice(4).trim();
      isOverviewSection = /overview|metric|summary/i.test(title);
      html.push(`<h3>${inlineFormat(esc(title))}</h3>`);
      continue;
    }
    if (line.startsWith('## ')) {
      flushList();
      flushOverviewMetrics();
      const title = line.slice(3).trim();
      isOverviewSection = /overview|metric|summary/i.test(title);
      html.push(`<h2>${inlineFormat(esc(title))}</h2>`);
      continue;
    }
    if (line.startsWith('# ')) {
      flushList();
      flushOverviewMetrics();
      const title = line.slice(2).trim();
      isOverviewSection = /overview|metric|summary/i.test(title);
      html.push(`<h2>${inlineFormat(esc(title))}</h2>`);
      continue;
    }

    // Check for overview metric bullet
    if (isOverviewSection) {
      const metricMatch = line.match(/^[-*]\s+\*\*([^*:]+)\*\*:\s*(.*)$/);
      if (metricMatch) {
        overviewMetrics.push({ label: metricMatch[1], rest: metricMatch[2] });
        continue;
      } else {
        flushOverviewMetrics();
      }
    }

    // Normal Bullet list item
    const bulletMatch = line.match(/^[-*]\s+(.*)$/);
    if (bulletMatch) {
      if (!inList || listType !== 'ul') {
        flushList();
        html.push('<ul>');
        inList = true;
        listType = 'ul';
      }
      html.push(`<li>${inlineFormat(esc(bulletMatch[1]))}</li>`);
      continue;
    }

    // Numbered list item
    const numberMatch = line.match(/^\d+\.\s+(.*)$/);
    if (numberMatch) {
      if (!inList || listType !== 'ol') {
        flushList();
        html.push('<ol>');
        inList = true;
        listType = 'ol';
      }
      html.push(`<li>${inlineFormat(esc(numberMatch[1]))}</li>`);
      continue;
    }

    // Normal paragraph
    flushList();
    html.push(`<p>${inlineFormat(esc(line))}</p>`);
  }

  flushList();
  flushOverviewMetrics();
  flushQuote();
  flushTable();

  return html.join('');
}

function inlineFormat(escapedText) {
  return escapedText
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

function bindWellness() {
  const form = $('#wellnessForm');
  const input = $('#wellnessInput');
  const submitBtn = $('#wellnessSubmit');
  const stateCard = $('#wellnessStateCard');
  const stateContent = $('#wellnessStateContent');
  const responseCard = $('#wellnessResponseCard');
  const responseContent = $('#wellnessResponseContent');
  const modelPill = $('#wellnessModelPill');
  const disclaimerText = $('#wellnessDisclaimerText');

  if (!form) return;

  $all('[data-wellness-prompt]').forEach((chip) => {
    chip.addEventListener('click', () => {
      const prompt = chip.dataset.wellnessPrompt;
      input.value = prompt;
      input.focus();
    });
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    submitBtn.disabled = true;
    submitBtn.setAttribute('aria-busy', 'true');
    submitBtn.innerHTML = '<span class="spinner" aria-hidden="true"></span><span>Consulting Groq…</span>';

    responseCard.hidden = true;
    stateCard.hidden = false;
    stateContent.innerHTML = `
      <div class="wellness-loading-box">
        <div class="wellness-loading-spinner" aria-hidden="true"></div>
        <strong>Generating Wellness Suggestions…</strong>
        <span class="muted">Querying Groq with Type 1 Diabetes wellness safety guidelines.</span>
      </div>
    `;

    try {
      const data = await api.getWellness(message);

      stateCard.hidden = true;
      responseCard.hidden = false;
      modelPill.textContent = data.model || 'Groq';
      if (data.disclaimer) {
        disclaimerText.textContent = data.disclaimer;
      }

      let bodyText = data.response || '';
      const disclaimer = data.disclaimer || 'Disclaimer: This is only a suggestion, not medical advice. Please consult your doctor before following any meal or exercise plan.';
      if (bodyText.toLowerCase().startsWith(disclaimer.toLowerCase())) {
        bodyText = bodyText.slice(disclaimer.length).trim();
      }

      responseContent.innerHTML = renderMarkdown(bodyText);
      paintIcons();

      if (mobileQuery.matches) {
        responseCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    } catch (error) {
      responseCard.hidden = true;
      stateCard.hidden = false;

      if (error.status === 503) {
        stateContent.innerHTML = `
          <div class="wellness-missing-key-box" role="alert">
            <i data-lucide="key" aria-hidden="true"></i>
            <h3>Groq API Key Not Configured</h3>
            <p>The wellness suggestion agent requires a Groq API key on the backend. Please configure <code>GROQ_API_KEY</code> in your environment or <code>.env</code> file, then restart the backend service.</p>
            <p class="muted">You can obtain a free Groq API key at <a href="https://console.groq.com" target="_blank" rel="noopener" style="color: var(--accent); text-decoration: underline;">console.groq.com</a>.</p>
          </div>
        `;
      } else {
        stateContent.innerHTML = `
          <div class="wellness-error-box" role="alert">
            <i data-lucide="alert-octagon" aria-hidden="true"></i>
            <h3>Unable to Retrieve Suggestions</h3>
            <p>${esc(error.message || 'An error occurred while connecting to Groq. Please check your backend connection and try again.')}</p>
            <button type="button" class="btn primary" id="wellnessRetryBtn" style="margin-top: 8px;">
              <i data-lucide="refresh-cw" aria-hidden="true"></i><span>Try Again</span>
            </button>
          </div>
        `;
        const retryBtn = $('#wellnessRetryBtn');
        if (retryBtn) {
          retryBtn.addEventListener('click', () => {
            form.requestSubmit();
          });
        }
      }
      paintIcons();
    } finally {
      submitBtn.disabled = false;
      submitBtn.removeAttribute('aria-busy');
      submitBtn.innerHTML = '<i data-lucide="send" aria-hidden="true"></i><span>Get Suggestions</span>';
      paintIcons();
    }
  });
}

let currentAgentSessionId = 'session_' + Math.random().toString(36).substring(2, 9);

function bindAgentChat() {
  const form = $('#agentChatForm');
  const input = $('#agentInput');
  const clearBtn = $('#agentClearBtn');
  const sendBtn = $('#agentSendBtn');
  const thread = $('#agentThread');
  const chatCard = $('.agent-chat-card');
  const chatBackdrop = $('#agentChatBackdrop');
  const expandBtn = $('#agentExpandBtn');
  const suggestionsToggle = $('#agentSuggestionsToggle');
  const suggestionsPanel = $('#agentSuggestions');
  const typing = $('#agentTyping');
  const typingStatus = $('#agentTypingStatus');
  const resetBtn = $('#agentResetBtn');
  const headerModel = $('#agentHeaderModel');
  const onlineDot = $('#agentOnlineDot');
  const stickyDisclaimer = $('#agentStickyDisclaimer');
  let suggestionChosen = false;
  let activeContextId = '';

  if (!form) return;

  const introMessage = () => `
    <div class="agent-msg assistant">
      <div class="msg-avatar"><i data-lucide="bot" aria-hidden="true"></i></div>
      <div class="msg-content"><div class="msg-body">
        <p><strong>👋 Hello! I’m your T1D Forecasting &amp; Wellness Assistant.</strong> I’m here to make your glucose data easier to understand and help you explore your forecast.</p>
        <p>${hasUploadedPatientContext()
          ? `This conversation is using <strong>${esc(state.lastUploadFileName || 'your uploaded CSV')}</strong>. No demo or test-patient data is used.`
          : '<em>No upload has been completed, so patient context cannot be determined. No demo or test-patient data is used.</em>'}</p>
        <p>With uploaded patient data, I can help with:</p>
        <ul><li>📊 <strong>Your readings:</strong> Look up glucose and other recorded measurements at specific times, and spot trends together.</li><li>🔮 <strong>Your forecast:</strong> Explore predictions at 30, 60, 90, and 120 minutes and what they mean.</li><li>🥗 <strong>Meals &amp; nutrition:</strong> Find general meal ideas and simple ways to balance a plate.</li><li>🏃 <strong>Movement:</strong> Get practical, general activity ideas.</li><li>💧 <strong>Low or high readings:</strong> Review general safety information and when to follow your care plan or contact your care team.</li></ul>
        <div class="agent-welcome-disclaimer"><strong>Research prototype:</strong> Patient-specific answers use uploaded patient information only. This is not medical advice, diagnosis, or insulin-dosing guidance. Follow your care plan and consult your clinician.</div>
        <p><em>✨ Patient-specific answers are grounded in available uploaded data or model output.</em></p>
      </div></div>
    </div>`;

  thread?.addEventListener('click', async (event) => {
    const copyButton = event.target.closest('[data-copy-prompt]');
    if (!copyButton) return;
    const promptText = copyButton.closest('.agent-msg')?.querySelector('.msg-body')?.innerText?.trim();
    if (!promptText) return;
    try {
      await navigator.clipboard.writeText(promptText);
      copyButton.setAttribute('aria-label', 'Prompt copied');
      copyButton.innerHTML = '<i data-lucide="check" aria-hidden="true"></i><span>Copied</span>';
      paintIcons();
      window.setTimeout(() => {
        if (!copyButton.isConnected) return;
        copyButton.setAttribute('aria-label', 'Copy prompt');
        copyButton.innerHTML = '<i data-lucide="copy" aria-hidden="true"></i><span>Copy</span>';
        paintIcons();
      }, 1600);
    } catch {
      showToast('Could not copy this prompt. Select its text and copy it manually.', 'warn');
    }
  });

  const syncAgentContextUi = () => {
    const available = hasUploadedPatientContext();
    const upload = state.lastUploadForecast;
    // The backend issues a fresh opaque ID on every upload request. Use the
    // stable content key so re-running another horizon for the same readings
    // keeps the current conversation and its context intact.
    const contextId = available ? (upload.dataset_context_key || upload.agent_dataset_id || '') : '';

    if (contextId !== activeContextId) {
      const previousSessionId = currentAgentSessionId;
      if (activeContextId) void api.agentReset(previousSessionId);
      currentAgentSessionId = 'session_' + Math.random().toString(36).substring(2, 9);
      activeContextId = contextId;
      input.value = '';
      input.style.height = 'auto';
      updateClearButton();
      suggestionChosen = false;
      if (suggestionsPanel) suggestionsPanel.hidden = true;
      suggestionsToggle?.setAttribute('aria-expanded', 'false');
      if (stickyDisclaimer) stickyDisclaimer.hidden = true;
      if (thread) thread.innerHTML = introMessage();
    }

    const status = $('#agentContextStatus');
    if (status) {
      status.classList.toggle('is-ready', available);
      status.innerHTML = available
        ? `<i data-lucide="file-check-2" aria-hidden="true"></i><span><strong>Uploaded patient context active.</strong> ${esc(state.lastUploadFileName || 'CSV upload')} · ${upload.readings.length} CGM readings. Demo and test data are excluded.</span>`
        : '<i data-lucide="info" aria-hidden="true"></i><span><strong>No upload done yet.</strong> Patient context cannot be determined. Upload a patient CSV to enable the agent.</span>';
    }

    input.disabled = !available;
    input.placeholder = available
      ? 'Ask about the uploaded patient’s glucose history and forecast...'
      : 'Upload a patient CSV to determine context and enable the agent.';
    sendBtn.disabled = !available;
    if (suggestionsToggle) suggestionsToggle.disabled = !available;
    $all('[data-agent-prompt]').forEach((chip) => { chip.disabled = !available; });
    if (!available && suggestionsPanel) suggestionsPanel.hidden = true;
    if (status) paintIcons();
  };

  document.addEventListener('agent-upload-context-change', syncAgentContextUi);

  const autoResizeInput = () => {
    if (!input) return;
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  };
  const updateClearButton = () => {
    if (clearBtn) clearBtn.hidden = !input.value;
  };
  syncAgentContextUi();

  const setChatExpanded = (expanded) => {
    chatCard?.classList.toggle('is-expanded', expanded);
    document.body.classList.toggle('agent-chat-expanded', expanded);
    if (chatBackdrop) chatBackdrop.hidden = !expanded;
    if (expandBtn) {
      expandBtn.setAttribute('aria-pressed', String(expanded));
      expandBtn.setAttribute('aria-label', expanded ? 'Exit full screen chat' : 'Expand chat');
      expandBtn.title = expanded ? 'Exit full screen' : 'Expand chat';
      expandBtn.innerHTML = `<i data-lucide="${expanded ? 'minimize-2' : 'maximize-2'}" aria-hidden="true"></i>`;
      paintIcons();
    }
  };

  expandBtn?.addEventListener('click', () => setChatExpanded(!chatCard?.classList.contains('is-expanded')));
  chatBackdrop?.addEventListener('click', () => setChatExpanded(false));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && chatCard?.classList.contains('is-expanded')) setChatExpanded(false);
  });

  input.addEventListener('input', () => {
    suggestionChosen = false;
    autoResizeInput();
    updateClearButton();
  });
  clearBtn?.addEventListener('click', () => {
    input.value = '';
    suggestionChosen = false;
    autoResizeInput();
    updateClearButton();
    input.focus();
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  suggestionsToggle?.addEventListener('click', () => {
    const open = suggestionsToggle.getAttribute('aria-expanded') !== 'true';
    suggestionsToggle.setAttribute('aria-expanded', String(open));
    if (suggestionsPanel) suggestionsPanel.hidden = !open;
  });

  $all('[data-agent-prompt]').forEach((chip) => {
    chip.addEventListener('click', () => {
      suggestionChosen = true;
      input.value = chip.dataset.agentPrompt;
      autoResizeInput();
      updateClearButton();
      input.focus();
    });
  });

  api.agentHealth().then((health) => {
    if (health) {
      if (headerModel) headerModel.textContent = health.model || 'openai/gpt-oss-120b';
      if (onlineDot) {
        onlineDot.className = health.groq_configured ? 'dot ok' : 'dot degraded';
      }
    }
  });

  if (resetBtn) {
    resetBtn.addEventListener('click', async () => {
      resetBtn.disabled = true;
      try {
        await api.agentReset(currentAgentSessionId);
        currentAgentSessionId = 'session_' + Math.random().toString(36).substring(2, 9);
        if (stickyDisclaimer) stickyDisclaimer.hidden = true;
        suggestionChosen = false;
        if (suggestionsPanel) suggestionsPanel.hidden = true;
        suggestionsToggle?.setAttribute('aria-expanded', 'false');
        thread.innerHTML = introMessage();
        syncAgentContextUi();
        paintIcons();
        showToast('Conversation reset', 'info');
      } catch (err) {
        showToast('Failed to reset session', 'error');
      } finally {
        resetBtn.disabled = false;
      }
    });
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const query = input.value.trim();
    if (!query || sendBtn.disabled) return;
    if (!hasUploadedPatientContext()) {
      syncAgentContextUi();
      return;
    }

    if (suggestionChosen) {
      suggestionsPanel.hidden = true;
      suggestionsToggle?.setAttribute('aria-expanded', 'false');
      suggestionChosen = false;
    }

    if (stickyDisclaimer) {
      stickyDisclaimer.hidden = false;
    }

    const userMsgEl = document.createElement('div');
    userMsgEl.className = 'agent-msg user';
    userMsgEl.innerHTML = `<div class="msg-body">${esc(query)}</div><button type="button" class="agent-copy-btn" data-copy-prompt aria-label="Copy prompt"><i data-lucide="copy" aria-hidden="true"></i><span>Copy</span></button>`;
    thread.appendChild(userMsgEl);
    paintIcons();
    input.value = '';
    input.style.height = 'auto';
    updateClearButton();
    thread.scrollTop = thread.scrollHeight;

    if (typing) {
      typing.hidden = false;
      if (typingStatus) typingStatus.textContent = 'Running models & reasoning…';
    }
    requestAnimationFrame(() => { thread.scrollTop = thread.scrollHeight; });
    sendBtn.disabled = true;
    sendBtn.setAttribute('aria-busy', 'true');
    sendBtn.innerHTML = '<span class="spinner" aria-hidden="true"></span>';

    const startTime = performance.now();
    const requestSessionId = currentAgentSessionId;
    try {
      const data = await api.agentChat(query, requestSessionId);
      if (requestSessionId !== currentAgentSessionId) return;

      const elapsed = performance.now() - startTime;
      if (elapsed < 2000) {
        await sleep(2000 - elapsed);
      }

      const assistantMsgEl = document.createElement('div');
      assistantMsgEl.className = 'agent-msg assistant';
      assistantMsgEl.innerHTML = `
        <div class="msg-avatar"><i data-lucide="bot"></i></div>
        <div class="msg-content">
          <div class="msg-body">${renderMarkdown(data.answer)}</div>
        </div>
      `;
      thread.appendChild(assistantMsgEl);
      paintIcons();
      thread.scrollTop = thread.scrollHeight;
    } catch (error) {
      if (requestSessionId !== currentAgentSessionId) return;
      let errorNotice = esc(error.message || 'Unknown network error');
      if (error.status === 429) {
        errorNotice = 'Groq rate limit reached (429). Please wait a few seconds before asking another question.';
      } else if (error.status === 503) {
        errorNotice = 'Groq API key is not configured. Please set GROQ_API_KEY on the backend server.';
      }

      const errorMsgEl = document.createElement('div');
      errorMsgEl.className = 'agent-msg assistant error';
      errorMsgEl.innerHTML = `
        <div class="msg-avatar" style="border-color: rgba(248, 113, 113, 0.4); color: var(--bad);"><i data-lucide="alert-octagon"></i></div>
        <div class="msg-content">
          <div class="msg-body" style="border-color: color-mix(in srgb, var(--bad) 40%, transparent); background: var(--panel-2); color: var(--text);">
            <p><strong style="color: var(--bad);">Could not complete request:</strong> ${errorNotice}</p>
          </div>
        </div>
      `;
      thread.appendChild(errorMsgEl);
      paintIcons();
      thread.scrollTop = thread.scrollHeight;
    } finally {
      if (requestSessionId !== currentAgentSessionId) return;
      if (typing) typing.hidden = true;
      sendBtn.disabled = !hasUploadedPatientContext();
      sendBtn.removeAttribute('aria-busy');
      sendBtn.innerHTML = '<i data-lucide="send" aria-hidden="true"></i><span>Send</span>';
      paintIcons();
      thread.scrollTop = thread.scrollHeight;
    }
  });
}

async function init() {
  initCursorDots();
  configureChartDefaults();
  fillPatients();
  state.patient = patients()[0];
  state.caseId = casesFor(state.patient)[0].id;
  state.horizon = 60;
  syncSelects();
  bindSelects();
  bindNavigation();
  bindChrome();
  bindWellness();
  bindAgentChat();
  renderStatus();
  showView('overview');
  paintIcons();
  setDrawer(false);

  checkHealth();
  loadSimTestPatients();
  loadOverallMetrics();
  loadPatientMetrics(state.patient);
  try {
    await api.getPatients();
    await api.getCases(state.patient);
  } catch {
    /* Patient and case catalogs stay on the built-in held-out samples. */
  }
}

document.addEventListener('DOMContentLoaded', init);
