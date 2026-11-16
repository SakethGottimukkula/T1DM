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
  accent: '#22d3ee',
  blue: '#60a5fa',
  ok: '#34d399',
  warn: '#fbbf24',
  bad: '#f87171',
  mute: '#8fa3b8',
  grid: 'rgba(148, 183, 210, 0.14)',
  text: '#e8eef6',
};

const ZONE_COLORS = {
  A: COLORS.ok,
  B: COLORS.accent,
  C: COLORS.warn,
  D: '#fb923c',
  E: COLORS.bad,
};

const ZONE_RGBA = {
  A: [52, 211, 153, 78],
  B: [34, 211, 238, 42],
  C: [251, 191, 36, 70],
  D: [251, 146, 60, 78],
  E: [248, 113, 113, 88],
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
const patientMetricsLoading = new Set();
const patientMetricsFailed = new Set();
let overallMetrics = null;
let overallMetricsLoading = false;

const state = {
  view: 'overview',
  patient: '',
  caseId: '',
  horizon: 60,
  running: false,
  checked: false,
  connected: false,
  health: 'unknown',
  usingDemo: true,
  // caseId whose forecast has been run and may be shown. null = nothing run yet.
  forecastFor: null,
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
          ctx.globalAlpha = pointer.opacity * fade * fade * 0.42;
          ctx.fillStyle = '#22d3ee';
          ctx.beginPath();
          ctx.arc(x, y, 1.15, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
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
  return livePredictions.get(caseId) || getCase(caseId).preds;
}

function currentSample() {
  return getCase(state.caseId);
}

// Predictions are only visible after Run Forecast completes for the selected case.
// Changing patient, case, or horizon clears them again.
function hasForecast() {
  return state.forecastFor === state.caseId;
}

function currentForecast(horizon = state.horizon) {
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
  } catch (error) {
    patientMetricsFailed.add(id);
    console.warn(`Could not load RMSE for patient ${id}.`, error.message);
  } finally {
    patientMetricsLoading.delete(id);
    if (String(state.patient) === id) renderAll({ animate: false, draw: false });
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
  Chart.defaults.font.family = '"IBM Plex Sans", system-ui, sans-serif';
  Chart.defaults.color = COLORS.mute;
  Chart.defaults.borderColor = COLORS.grid;
  Chart.defaults.plugins.tooltip.backgroundColor = '#101a2c';
  Chart.defaults.plugins.tooltip.borderColor = 'rgba(148, 183, 210, 0.24)';
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
    ctx.fillStyle = 'rgba(52, 211, 153, 0.10)';
    ctx.fillRect(chartArea.left, top, chartArea.right - chartArea.left, bottom - top);
    ctx.strokeStyle = 'rgba(52, 211, 153, 0.35)';
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
    ctx.fillStyle = 'rgba(34, 211, 238, 0.06)';
    ctx.fillRect(xNow, chartArea.top, chartArea.right - xNow, chartArea.bottom - chartArea.top);
    ctx.strokeStyle = COLORS.mute;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(xNow, chartArea.top);
    ctx.lineTo(xNow, chartArea.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = COLORS.mute;
    ctx.font = '11px "IBM Plex Sans", sans-serif';
    ctx.fillText('History', chartArea.left + 6, chartArea.top + 14);
    ctx.fillText('Now', xNow + 6, chartArea.top + 14);
    ctx.fillText('Future', chartArea.right - 48, chartArea.top + 14);
    ctx.restore();
  },
};

function axis(title) {
  return {
    grid: { color: COLORS.grid },
    ticks: { color: COLORS.mute },
    title: {
      display: Boolean(title),
      text: title,
      color: COLORS.mute,
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
          borderColor: COLORS.blue,
          backgroundColor: COLORS.blue,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
          data: [],
        },
        {
          label: 'Actual future (held out)',
          borderColor: COLORS.ok,
          backgroundColor: COLORS.ok,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
          data: [],
        },
        {
          label: 'Forecast',
          borderColor: COLORS.accent,
          backgroundColor: COLORS.accent,
          borderWidth: 2,
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
        legend: { labels: { color: COLORS.mute, boxWidth: 12, usePointStyle: true } },
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
  const sample = currentSample();
  if (!chart || !sample) return;

  const ready = hasForecast();
  const horizons = ready ? horizonsFor(sample.id) : null;
  const visible = ready ? sample.series : sample.series.filter((point) => point.x <= 0);
  const values = visible.map((point) => point.y)
    .concat(ready ? HZ.map((horizon) => horizons[horizon].predicted) : []);
  const yMin = Math.min(50, ...values) - 15;
  const yMax = Math.max(200, ...values) + 15;
  chart.options.scales.y.min = Math.max(30, Math.floor(yMin / 10) * 10);
  chart.options.scales.y.max = Math.min(400, Math.ceil(yMax / 10) * 10);

  chart.data.datasets[0].data = sample.series.filter((point) => point.x <= 0);
  // Held-out actuals and the forecast stay hidden until Run Forecast has finished.
  chart.data.datasets[1].data = ready ? sample.series.filter((point) => point.x >= 0) : [];
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
    if (context.dataIndex === 0) return COLORS.accent;
    return HZ[context.dataIndex - 1] === state.horizon ? COLORS.accent : 'rgba(34, 211, 238, 0.45)';
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
    ctx.font = '600 13px "IBM Plex Sans", sans-serif';
    ctx.fillStyle = 'rgba(232, 238, 246, 0.92)';
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
          pointBackgroundColor: CLARKE_POINTS.map((point) => ZONE_COLORS[point.z]),
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
          backgroundColor: 'rgba(34, 211, 238, 0.55)',
          borderColor: COLORS.accent,
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

function resizeActiveCharts() {
  const map = {
    overview: ['cMain'],
    forecast: ['cDetail'],
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
  const sample = currentSample();
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
    stat('Selected patient', `Patient ${sample.patient}`, 'OhioT1DM 2020', '', reveal(opts, 2)),
    stat('Forecast horizon', `${state.horizon} min`, 'CGM-only forecast', '', reveal(opts, 3)),
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
      `Actual ${mg(item.actual)} · error ${mg(item.error)} · ${assessment.label}`,
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

  if (ready) {
    const assessment = forecastRiskOf(forecast);
    const status = errorStatus(forecast.error);
    const predictionNote = assessment.label === 'Unreliable forecast'
      ? `This prediction should be treated as unreliable (${assessment.detail.toLowerCase()}).`
      : `The estimate indicates ${assessment.detail.toLowerCase()}.`;
    $('#interp').innerHTML = `At ${state.horizon} minutes the model projects <b>${mg(forecast.predicted)} mg/dL</b>. ${predictionNote} The held-out glucose was <b>${mg(forecast.actual)} mg/dL</b>, an absolute error of ${mg(forecast.error)} mg/dL (${status.label.toLowerCase()}, Clarke zone ${forecast.clarke_zone}). This readout uses CGM history only from patient ${esc(sample.patient)} in the OhioT1DM 2020 cohort. It is a research interpretation, not a treatment recommendation.`;
  } else {
    $('#interp').innerHTML = `No forecast has been run for patient ${esc(sample.patient)} (${esc(sample.label)}) yet. Pick a horizon in the <button type="button" class="link-btn" data-go="explorer">Test Explorer</button> and press <b>Run Forecast</b> to see the prediction, the held-out value and the error. Until then only the CGM history is shown.`;
  }

  if (state.view === 'overview') {
    ensureChart('cMain', (canvas) => new Chart(canvas, trajectoryConfig()));
    setTrajectory('cMain', opts);
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
    const zoneClass = ['A', 'B'].includes(item.clarke_zone) ? `zone-${rangeOf(item.predicted).zone}` : 'zone-review';
    return `<div class="risk-chip ${zoneClass}"><span class="hz">${horizon} min</span><span class="val ${assessment.cls}">${mg(item.predicted)}</span><span class="zone ${assessment.cls}">${assessment.label}</span></div>`;
  }).join('');

  const unreliable = HZ.map((horizon) => ({ horizon, item: horizons[horizon] }))
    .find(({ item }) => !['A', 'B'].includes(item.clarke_zone));
  const excursion = HZ.map((horizon) => ({ horizon, item: horizons[horizon] }))
    .filter(({ item }) => ['A', 'B'].includes(item.clarke_zone))
    .map(({ horizon, item }) => ({ horizon, value: item.predicted }))
    .find((item) => item.value < 70 || item.value > 180);
  if (excursion) {
    const message = excursion.value < 70 ? 'Low glucose risk signal' : 'High glucose risk signal';
    const reliabilityNote = unreliable
      ? ` Another forecast at ${unreliable.horizon} min is unreliable (Clarke zone ${unreliable.item.clarke_zone}).`
      : '';
    banner.classList.add(excursion.value < 70 ? 'danger' : 'warn');
    banner.innerHTML = `<strong>${message} (zone A/B).</strong> Earliest horizon outside range: ${excursion.horizon} min, predicted ${mg(excursion.value)} mg/dL.${reliabilityNote}`;
  } else if (unreliable) {
    banner.classList.add('warn');
    banner.innerHTML = `<strong>Forecast needs review.</strong> The ${unreliable.horizon} min prediction is outside Clarke zones A/B (zone ${unreliable.item.clarke_zone}), so it is too unreliable to label as high or low glucose.`;
  } else {
    banner.classList.add('ok');
    banner.innerHTML = '<strong>No threshold excursion predicted.</strong> Forecasts from 30 to 120 minutes stay inside 70–180 mg/dL.';
  }
}

function renderForecast(opts = {}) {
  const sample = currentSample();
  const header = '<tr><th>Horizon</th><th>Predicted</th><th>Actual</th><th>Absolute error</th><th>Clarke</th><th>Status</th></tr>';
  if (hasForecast()) {
    const rows = HZ.map((horizon) => {
      const item = horizonsFor(sample.id)[horizon];
      const status = errorStatus(item.error);
      const selected = horizon === state.horizon ? ' class="is-selected"' : '';
      return `<tr${selected}><td>${horizon} min</td><td>${mg(item.predicted)}</td><td>${mg(item.actual)}</td><td class="${status.cls}">${mg(item.error)} mg/dL</td><td>${item.clarke_zone}</td><td class="${status.cls}">${status.label}</td></tr>`;
    }).join('');
    $('#errTable').innerHTML = header + rows;
  } else {
    $('#errTable').innerHTML = `${header}<tr><td colspan="6" class="muted">No forecast yet. Run one in the <button type="button" class="link-btn" data-go="explorer">Test Explorer</button> to fill this table.</td></tr>`;
  }

  if (state.view === 'forecast') {
    ensureChart('cDetail', (canvas) => new Chart(canvas, trajectoryConfig()));
    setTrajectory('cDetail', opts);
  }
}

function renderExplorer(opts = {}) {
  if (state.running) return;
  const sample = currentSample();
  const forecast = currentForecast();
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
  const now = scale(sample.context.current);
  const predMarker = forecast
    ? `<span class="mk below" style="left:${scale(forecast.predicted)}%;color:${COLORS.accent}">Pred ${mg(forecast.predicted)}</span>`
    : '';
  $('#range').innerHTML = `<div class="rbar" role="img" aria-label="Glucose range from low through in range to high">
      <i style="width:${scale(70)}%;background:${COLORS.bad}"></i>
      <i style="width:${scale(180) - scale(70)}%;background:${COLORS.ok}"></i>
      <i style="flex:1;background:${COLORS.warn}"></i>
      <span class="mk" style="left:${now}%;color:${COLORS.text}">Now ${mg(sample.context.current)}</span>
      ${predMarker}
    </div>`;
  $('#rangeNote').textContent = `Low is under 70 mg/dL, in range is 70–180, and high is above 180. For ${state.horizon} min, held-out Clarke A+B is ${SAFE[state.horizon]}%. This view does not recommend a treatment.`;

  const shares = ZONE_SHARE[state.horizon];
  $('#zones').innerHTML = Object.entries(shares).map(([zone, percent]) =>
    `<div class="zrow"><b>${zone}</b><div><i style="width:${percent}%;background:${ZONE_COLORS[zone]}"></i></div><span>${percent.toFixed(2)}%</span></div>`
  ).join('');

  if (state.view !== 'clinical') return;
  const chart = ensureClarke();
  if (!chart) return;
  const horizons = forecast ? horizonsFor(sample.id) : null;
  chart.data.datasets[1].data = horizons
    ? HZ.map((horizon) => ({
        x: horizons[horizon].actual,
        y: horizons[horizon].predicted,
        horizon,
      }))
    : [];
  chart.data.datasets[1].pointRadius = (context) => (context.raw && context.raw.horizon === state.horizon ? 9 : 6);
  chart.update(reduceMotion ? 'none' : undefined);
}

function renderPerformance() {
  $('#perfCards').innerHTML = HZ.map((horizon) =>
    stat(`${horizon} min · Overall RMSE`, overallRmseText(horizon), `Patient ${state.patient}: ${patientRmseText(state.patient, horizon)} · A+B overall ${SAFE[horizon].toFixed(2)}%`)
  ).join('');

  $('#hTable').innerHTML = `<tr><th>Horizon</th><th>Overall RMSE</th><th>Patient ${esc(state.patient)} RMSE</th><th>Clinically safe A+B (overall)</th></tr>${
    HZ.map((horizon) => `<tr class="${horizon === state.horizon ? 'is-selected' : ''}"><td>${horizon} min</td><td>${overallRmseText(horizon)}</td><td>${patientRmseText(state.patient, horizon)}</td><td>${SAFE[horizon].toFixed(2)}%</td></tr>`).join('')
  }`;

  $('#mTable').innerHTML = `<tr><th>Metric</th><th>30 min</th><th>60 min</th><th>90 min</th><th>120 min</th></tr>
    <tr><td>Overall RMSE · Ohio test</td>${HZ.map((horizon) => `<td>${overallRmseText(horizon)}</td>`).join('')}</tr>
    <tr><td>Overall MAE · Ohio test</td>${HZ.map((horizon) => `<td>${overallMaeText(horizon)}</td>`).join('')}</tr>
    <tr><td>Patient ${esc(state.patient)} RMSE</td>${HZ.map((horizon) => `<td>${patientRmseText(state.patient, horizon)}</td>`).join('')}</tr>
    <tr><td>Patient ${esc(state.patient)} MAE</td>${HZ.map((horizon) => `<td>${patientMaeText(state.patient, horizon)}</td>`).join('')}</tr>
    <tr><td>Clinically safe A+B · overall</td>${HZ.map((horizon) => `<td>${SAFE[horizon].toFixed(2)}%</td>`).join('')}</tr>`;

  if (state.view === 'performance') {
    const chart = ensureErrorChart();
    chart.data.datasets[0].label = `Patient ${state.patient} RMSE (mg/dL)`;
    chart.data.datasets[0].data = HZ.map((horizon) => patientRmse(state.patient, horizon));
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
  renderStatus();
}

async function runForecast() {
  if (state.running) return;
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
      state.patient = select.value;
      state.caseId = casesFor(state.patient)[0].id;
      state.forecastFor = null;
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
      state.caseId = select.value;
      state.patient = getCase(state.caseId).patient;
      state.forecastFor = null;
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
    state.horizon = Number($('#selH').value);
    state.forecastFor = null;
    renderAll({ animate: false, draw: true });
  });
}

function bindChrome() {
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
  $('#run').addEventListener('click', runForecast);
  $('#rand').addEventListener('click', chooseRandomCase);
  $('#uploadMode').addEventListener('change', updateUploadHelp);
  $('#chooseCsv').addEventListener('click', () => $('#uploadCsv').click());
  $('#uploadCsv').addEventListener('change', () => {
    $('#uploadFileName').textContent = $('#uploadCsv').files[0]?.name || 'No file selected';
  });
  $('#uploadRun').addEventListener('click', uploadAndPredict);
  renderUploadModels();
}

function updateUploadHelp() {
  const multimodal = $('#uploadMode').value === 'multimodal';
  $('#uploadHelp').textContent = multimodal
    ? 'CSV needs timestamp, CGM (mg/dL), and insulin (units per row); CHO (grams per row) is optional and defaults to zero. SimGlucose history columns Time, BG, CGM, CHO, insulin are supported.'
    : 'CSV needs timestamp and glucose (mg/dL), or SimGlucose columns Time and CGM, with at least 25 sequential readings at roughly 5-minute intervals.';
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
    data = await api.predictCsv(file, $('#uploadHorizon').value, multimodal);
  } catch (error) {
    requestError = error;
  }
  const remaining = delay - (performance.now() - started);
  if (remaining > 0) await sleep(remaining);
  try {
    if (requestError) throw requestError;
    const predictions = data.predictions || [];
    results.innerHTML = predictions.map((item) => stat(
      `${item.horizon} min forecast`,
      `${mg(item.predicted_glucose)} mg/dL`,
      `Change from latest ${mg(item.change_from_latest)} mg/dL · ${item.zone_label}`,
      item.zone_label === 'low' ? 'bad' : item.zone_label === 'high' ? 'warn' : 'ok'
    )).join('');
    if (multimodal && data.meal_defaulted_to_zero) showToast('No CHO column found; meals were treated as zero.');
  } catch (error) {
    results.innerHTML = `<div class="stat"><span class="stat-label">Could not run prediction</span><span class="stat-value">Check CSV</span><span class="stat-detail">${esc(error.message)}</span></div>`;
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
  renderStatus();
  showView('overview');
  paintIcons();
  setDrawer(false);

  checkHealth();
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
