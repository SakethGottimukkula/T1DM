/* Demo / research prototype using CGM-only OhioT1DM 2020 replay data. */

const API_BASE_URL = 'http://127.0.0.1:8000';

/**
 * Attempt to fetch predictions from the backend API.
 * Returns null if the API is unreachable or returns an error.
 */
async function fetchPredictions(readings, horizon = 'all') {
  try {
    const response = await fetch(`${API_BASE_URL}/v1/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ readings, horizon }),
    });

    if (!response.ok) {
      console.warn(`API returned ${response.status}: ${response.statusText}`);
      return null;
    }

    return await response.json();
  } catch (error) {
    console.warn('Glucose Horizon API unreachable — using local demo data.', error.message);
    return null;
  }
}

const HZ = [30, 60, 90, 120];
const PATIENTS = ['559', '563', '570', '575', '588', '591'];
const RMSE = { 30: 20.94, 60: 33.67, 90: 42.44, 120: 48.08 };
const SAFE = { 30: 98.29, 60: 96.72, 90: 95.14, 120: 93.87 };
const REFERENCE = { 30: 99.6, 60: 99.0, 90: 97.6, 120: 96.4 };

const MODEL_ROWS = [
  { model: 'Ridge/Lasso', note: 'Linear baselines considered', comparison: '?' },
  { model: 'ARIMA', note: 'Classical time-series baseline considered', comparison: '?' },
  { model: 'ARX/ARIMAX', note: 'Ruled out because required insulin/carb inputs are unavailable', comparison: '?' },
  { model: 'XGBoost', note: 'Selected primary model: strong for small-to-mid-sized tabular data, captures nonlinear patterns in engineered CGM features, and provides interpretable feature importance for this dataset scale.', comparison: 'Selected' },
  { model: 'Four horizon-specific models', note: 'One XGBoost model trained per forecast horizon (30/60/90/120 min)', comparison: '?' },
  { model: 'Xiong et al. (2025) ? multimodal reference model', note: 'CGM + insulin + meal composition + other patient/time context; Transformer + BiLSTM', comparison: '30 min 99.6%; 60 min 99.0%; 90 min 97.6%; 120 min 96.4%' }
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

function selectElement(selector, root = document) {
  return root.querySelector(selector);
}

function selectAll(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function createRandomGenerator(seed) {
  return function generateRandomNumber() {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };
}

function buildCases() {
  const times = ['07:35', '11:10', '14:50', '18:25', '21:45', '00:20'];
  const allCases = [];

  for (let caseIndex = 0; caseIndex < PATIENTS.length * 3; caseIndex += 1) {
    const patientId = PATIENTS[caseIndex % PATIENTS.length];
    const randomGenerator = createRandomGenerator(caseIndex * 97 + 13);
    const baseGlucose = 96 + (caseIndex % 6) * 7 + randomGenerator() * 18;
    const amplitude = 22 + randomGenerator() * 42;
    const phase = ((caseIndex + 1) * 19) % 200;
    const curve = [];

    for (let minute = -180; minute <= 120; minute += 5) {
      const waveValue = Math.sin((minute + phase) / 34) * amplitude;
      const secondWave = Math.cos((minute + phase) / 68) * (amplitude * 0.46);
      const drift = (minute / 180) * (randomGenerator() - 0.5) * 20;
      const noise = (randomGenerator() - 0.5) * 9;
      const value = Math.round(clamp(baseGlucose + waveValue + secondWave + drift + noise, 48, 280));
      curve.push({ x: minute, y: value });
    }

    function getValueAtTime(targetTime) {
      const matchingPoint = curve.find(point => point.x === targetTime);

      if (matchingPoint) {
        return matchingPoint.y;
      }

      return curve[0].y;
    }

    const currentGlucose = getValueAtTime(0);
    const historyValues = curve
      .filter(point => point.x <= 0)
      .map(point => point.y);
    const meanValue = historyValues.reduce((sum, value) => sum + value, 0) / historyValues.length;
    const minimumValue = Math.min(...historyValues);
    const maximumValue = Math.max(...historyValues);
    const recentFiveMinuteChange = getValueAtTime(0) - getValueAtTime(-5);
    const recentThirtyMinuteChange = getValueAtTime(0) - getValueAtTime(-30);
    const timestamp = `2020-01-${String((caseIndex % 28) + 1).padStart(2, '0')} ${times[caseIndex % times.length]}`;
    const predictions = {};

    HZ.forEach(hour => {
      const actualValue = getValueAtTime(hour);
      const driftBias = (Math.sin((caseIndex + hour) / 13) + Math.cos((caseIndex * 3 + hour) / 9)) * 3.5;
      const forecastValue = Math.round(clamp(actualValue + driftBias + (randomGenerator() - 0.5) * 12, 35, 330));

      predictions[hour] = {
        pred: forecastValue,
        actual: actualValue,
        err: Math.abs(forecastValue - actualValue)
      };
    });

    allCases.push({
      id: caseIndex,
      patient: patientId,
      ts: timestamp,
      curve,
      cur: currentGlucose,
      preds: predictions,
      ctx: {
        'Current glucose': [currentGlucose, 'mg/dL', 'at forecast time'],
        'Number of CGM samples': [historyValues.length, 'samples', 'in the 3 h window'],
        'Window duration': ['3 h', '', 'CGM history used'],
        'Minimum glucose': [minimumValue, 'mg/dL', 'in window'],
        'Maximum glucose': [maximumValue, 'mg/dL', 'in window'],
        'Mean glucose': [Math.round(meanValue), 'mg/dL', 'in window'],
        'Recent 5-minute change': [recentFiveMinuteChange, 'mg/dL', 'from t-5 to now'],
        'Recent 30-minute change': [recentThirtyMinuteChange, 'mg/dL', 'from t-30 to now'],
        'Forecast timestamp': [timestamp, '', '']
      },
      events: [
        'CGM sampled every 5 minutes',
        'Historical glucose window used for forecasting',
        'Lag features derived from glucose history',
        'Rolling statistics derived from the CGM stream',
        '5-minute and 30-minute rate-of-change features',
        'Hour sine/cosine time encoding'
      ]
    });
  }

  return allCases;
}

const CASES = buildCases();

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

const clarkeRandom = createRandomGenerator(4242);
const CLARKE = [];

for (let pointIndex = 0; pointIndex < 180; pointIndex += 1) {
  const referenceGlucose = Math.round(55 + clarkeRandom() * 300);
  const predictedGlucose = Math.round(
    clamp(referenceGlucose + (clarkeRandom() - 0.5) * 2 * (0.11 * referenceGlucose + 6), 40, 400)
  );

  CLARKE.push({
    x: referenceGlucose,
    y: predictedGlucose,
    z: clarkeZone(referenceGlucose, predictedGlucose)
  });
}

const appState = {
  caseIndex: 0,
  horizon: 60
};

const charts = {};

function getCurrentCase() {
  return CASES[appState.caseIndex];
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
  const glucoseCurve = currentCase.curve;
  const indexOfNow = glucoseCurve.findIndex(point => point.x === 0);
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

  chart.data.datasets[0].data = currentCase.curve.filter(point => point.x <= 0);
  chart.data.datasets[1].data = currentCase.curve.filter(point => point.x >= 0);
  chart.data.datasets[2].data = [{ x: 0, y: currentCase.cur }].concat(
    HZ.map(hour => ({ x: hour, y: currentCase.preds[hour].pred }))
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
  const currentStatus = thresholdState(currentCase.cur);

  const trajectoryItems = HZ.map(hour => {
    const forecastValue = currentCase.preds[hour].pred;
    const state = thresholdState(forecastValue);
    return `<div style="display:flex; align-items:center; justify-content:space-between; gap:10px; padding:8px 0; border-bottom:1px solid var(--line);"><span>${hour} min — ${forecastValue} mg/dL</span><span class="${state.className}" style="font-weight:600;">${state.label}</span></div>`;
  }).join('');

  const earliestPrediction = HZ.map(hour => ({
    h: hour,
    value: currentCase.preds[hour].pred
  })).find(item => item.value < 70 || item.value > 180);

  let alertHtml = 'No threshold excursion predicted within the selected forecast horizon.';

  if (earliestPrediction) {
    const excursionType = earliestPrediction.value < 70
      ? 'Potential hypoglycemic excursion'
      : 'Potential hyperglycemic excursion';

    alertHtml = `<strong>${excursionType}</strong> — earliest horizon: ${earliestPrediction.h} min — predicted glucose: ${earliestPrediction.value} mg/dL`;
  }

  const summaryHtml = stat('Current glucose', `${currentCase.cur} mg/dL`, currentStatus.label, currentStatus.className)
    + stat('Current status', currentStatus.label, 'Threshold band', currentStatus.className)
    + stat('Selected patient', `Patient ${currentCase.patient}`, 'OhioT1DM 2020')
    + stat('Selected horizon', `${appState.horizon} min`, 'Current forecast setting');

  selectElement('#riskSummary').innerHTML = summaryHtml;
  selectElement('#riskTrajectory').innerHTML = trajectoryItems;
  selectElement('#riskAlert').innerHTML = `<div class="stat" style="padding:14px 16px;"><small>Earliest predicted excursion</small><b>${alertHtml}</b></div>`;
}

function renderOverview() {
  const currentCase = getCurrentCase();
  const currentForecast = currentCase.preds[appState.horizon];
  const [currentRangeLabel, currentRangeClass] = rangeOf(currentCase.cur);
  const [trendLabel, trendValue] = trendText();

  const summaryHtml = stat('Current Glucose', `${currentCase.cur} mg/dL`, currentRangeLabel, currentRangeClass)
    + stat('Current Trend', trendLabel, trendValue)
    + stat('Selected Patient', `Patient ${currentCase.patient}`, 'OhioT1DM 2020')
    + stat('Forecast Horizon', `${appState.horizon} min`, 'CGM-only forecast');

  selectElement('#summary').innerHTML = summaryHtml;

  const futureCardsHtml = HZ.map(hour => {
    const forecastItem = currentCase.preds[hour];
    const selectedClass = hour === appState.horizon ? 'sel' : '';
    return `<div class="stat ${selectedClass}"><small>${hour} min forecast</small><b>${forecastItem.pred} mg/dL</b><span>Actual ${forecastItem.actual} · error ${forecastItem.err}</span></div>`;
  }).join('');

  selectElement('#fcards').innerHTML = futureCardsHtml;

  const [predictionRangeLabel] = rangeOf(currentForecast.pred);
  const [forecastErrorText] = errorStatus(currentForecast.err);

  selectElement('#interp').innerHTML = `At ${appState.horizon} min the model projects <b>${currentForecast.pred} mg/dL</b> (${predictionRangeLabel.toLowerCase()}); the held-out value was <b>${currentForecast.actual} mg/dL</b>, an absolute error of ${currentForecast.err} mg/dL (${forecastErrorText.toLowerCase()}). This CGM-only replay uses the OhioT1DM 2020 cohort with 6 patients, 5-minute sampling, and 30/60/90/120 min forecasting. It is a research prototype and not treatment guidance.`;

  renderRiskAlerts();
  setLine('cMain');
}

function renderForecast() {
  const currentCase = getCurrentCase();
  let tableHtml = '<tr><th>Horizon</th><th>Current</th><th>Predicted</th><th>Actual</th><th>Absolute error</th></tr>';

  HZ.forEach(hour => {
    const forecastItem = currentCase.preds[hour];
    const [errorText, errorClass] = errorStatus(forecastItem.err);
    tableHtml += `<tr><td>${hour} min</td><td>${currentCase.cur}</td><td>${forecastItem.pred}</td><td>${forecastItem.actual}</td><td class="${errorClass}">${forecastItem.err} mg/dL</td></tr>`;
  });

  selectElement('#errTable').innerHTML = tableHtml;
  setLine('cDetail');
}

function renderExplorer() {
  const currentCase = getCurrentCase();
  const selectedForecast = currentCase.preds[appState.horizon];
  const [errorText, errorClass] = errorStatus(selectedForecast.err);
  const historyValues = currentCase.curve
    .filter(point => point.x <= 0)
    .map(point => point.y);

  selectElement('#selH').value = appState.horizon;

  const resultsHtml = stat('History Window', '3 h', `${historyValues.length} CGM samples | ${Math.min(...historyValues)}?${Math.max(...historyValues)} mg/dL`)
    + stat('Prediction', `${selectedForecast.pred} mg/dL`, `${appState.horizon} min ahead`)
    + stat('Actual Value', `${selectedForecast.actual} mg/dL`, 'Held-out')
    + stat('Absolute Error', `${selectedForecast.err} mg/dL`, '')
    + stat('Forecast Status', errorText, 'Error ?15 close, ?30 moderate', errorClass);

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
  const selectedForecast = currentCase.preds[appState.horizon];

  function positionOnRangeScale(glucoseValue) {
    return clamp((glucoseValue - 40) / 360 * 100, 0, 100);
  }

  function makeMarker(value, labelText, color, markerTop) {
    return `<span class="mk" style="left:${positionOnRangeScale(value)}%;color:${color};top:${markerTop}px">${labelText} ${value}</span>`;
  }

  const rangeBarHtml = `<div class="rbar"><i style="width:${positionOnRangeScale(70)}%;background:${COLORS.bad}"></i><i style="width:${positionOnRangeScale(180) - positionOnRangeScale(70)}%;background:${COLORS.ok}"></i><i style="flex:1;background:${COLORS.warn}"></i>${makeMarker(currentCase.cur, 'Now', '#fff', -26)}${makeMarker(selectedForecast.pred, 'Pred', COLORS.accent, -26)}</div>`;

  selectElement('#range').innerHTML = rangeBarHtml;
  selectElement('#rangeNote').textContent = `Clarke Error Grid Analysis is used to evaluate clinical acceptability of prediction errors. This research prototype does not provide treatment or insulin recommendations. For the selected horizon, ${appState.horizon} min, clinically safe A+B = ${SAFE[appState.horizon]}%.`;

  const zoneValues = zoneDistribution(appState.horizon);
  const zoneRowsHtml = Object.entries(zoneValues).map(([zoneName, zonePercentage]) => {
    return `<div class="zrow"><b>${zoneName}</b><div><i style="width:${zonePercentage}%;background:${ZONE_COLORS[zoneName]}"></i></div><span>${zonePercentage.toFixed(2)}%</span></div>`;
  }).join('');

  selectElement('#zones').innerHTML = zoneRowsHtml;

  const clarkeChart = charts.cClarke;
  clarkeChart.data.datasets[0].data = CLARKE;
  clarkeChart.data.datasets[0].backgroundColor = CLARKE.map(point => ZONE_COLORS[point.z]);
  clarkeChart.data.datasets[1].data = HZ.map(hour => ({
    x: currentCase.preds[hour].actual,
    y: currentCase.preds[hour].pred
  }));
  clarkeChart.update();
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
  const selectedPatientId = getCurrentCase().patient;

  const optionsHtml = CASES.filter(caseItem => caseItem.patient === selectedPatientId)
    .map(caseItem => `<option value="${caseItem.id}">${caseItem.ts}</option>`)
    .join('');

  caseSelect.innerHTML = optionsHtml;
}

function renderAll() {
  selectAll('.sel-patient').forEach(patientSelect => {
    patientSelect.value = getCurrentCase().patient;
  });

  selectAll('.sel-case').forEach(caseSelect => {
    fillCases(caseSelect);
    caseSelect.value = String(appState.caseIndex);
  });

  renderOverview();
  renderForecast();
  renderExplorer();
  renderClinical();
}

function runForecast() {
  const resultsBox = selectElement('#results');
  resultsBox.classList.add('loading');

  const currentCase = getCurrentCase();

  // Build readings from the history portion of the curve (x <= 0)
  const historyPoints = currentCase.curve.filter(point => point.x <= 0);
  const baseDate = new Date('2020-01-15T12:00:00');
  const readings = historyPoints.map(point => {
    const ts = new Date(baseDate.getTime() + point.x * 60000);
    return {
      timestamp: ts.toISOString(),
      glucose: Math.max(40, Math.min(400, point.y)),
    };
  });

  fetchPredictions(readings, 'all').then(apiResult => {
    if (apiResult && apiResult.predictions) {
      // Update the case predictions with API results
      apiResult.predictions.forEach(pred => {
        const hz = pred.horizon;
        if (currentCase.preds[hz]) {
          currentCase.preds[hz].pred = Math.round(pred.predicted_glucose);
          currentCase.preds[hz].err = Math.abs(
            currentCase.preds[hz].pred - currentCase.preds[hz].actual
          );
        }
      });
    }
    // If apiResult is null the local demo data is already in place (graceful fallback)

    resultsBox.classList.remove('loading');
    renderAll();
  });
}

function init() {
  const patientList = [...new Set(CASES.map(caseItem => caseItem.patient))];

  selectAll('.sel-patient').forEach(patientSelect => {
    patientSelect.innerHTML = patientList.map(patientId => `<option value="${patientId}">Patient ${patientId}</option>`).join('');

    patientSelect.onchange = () => {
      const chosenPatient = patientSelect.value;
      const matchingCase = CASES.find(caseItem => caseItem.patient === chosenPatient);
      appState.caseIndex = matchingCase.id;
      renderAll();
    };
  });

  selectAll('.sel-case').forEach(caseSelect => {
    caseSelect.onchange = () => {
      appState.caseIndex = Number(caseSelect.value);
      renderAll();
    };
  });

  selectElement('#selH').onchange = event => {
    appState.horizon = Number(event.target.value);
    renderAll();
  };

  selectElement('#run').onclick = runForecast;

  selectElement('#rand').onclick = () => {
    let randomIndex = 0;

    do {
      randomIndex = Math.floor(Math.random() * CASES.length);
    } while (randomIndex === appState.caseIndex);

    appState.caseIndex = randomIndex;
    runForecast();
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

  renderAll();
  lucide.createIcons();
}

document.addEventListener('DOMContentLoaded', init);
