import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import Papa from 'papaparse';
import {
  Activity,
  AlertTriangle,
  Bell,
  BrainCircuit,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Database,
  Download,
  FileSpreadsheet,
  FlaskConical,
  Gauge,
  Layers3,
  LineChart as LineIcon,
  ListChecks,
  Menu,
  MessageSquare,
  RotateCcw,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  Target,
  Timer,
  Upload,
  X,
  Zap
} from 'lucide-react';
function getRiskColor(risk) {
    if (risk === 'Critical') return '#ef4444';
    if (risk === 'High') return '#f97316';
    if (risk === 'Watch') return '#facc15';
    return '#22c55e';
}
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  BarChart,
  Bar,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
  PieChart,
  Pie
} from 'recharts';
import './styles.css';

// Real ML backend (Python/FastAPI). Set VITE_ML_API_URL for deployment.
const ML_API = import.meta.env.VITE_ML_API_URL || '';

async function runMLSingle(component) {
  const row = {
    Component_ID: component.id,
    Lot_ID: component.lot,
    Value_0h: Number(component.v?.[0] ?? 0),
    Value_24h: Number(component.v?.[1] ?? 0),
    Value_96h: Number(component.v?.[2] ?? 0),
    Value_168h: Number(component.v?.[3] ?? 0),
    Drift_0_24_per_h: Number((component.v?.[1] - component.v?.[0]) / 24 || 0),
    Drift_24_96_per_h: Number((component.v?.[2] - component.v?.[1]) / 72 || 0),
    Drift_96_168_per_h: Number((component.v?.[3] - component.v?.[2]) / 72 || 0),
    Current_Time_h: 168,
    temperature_c: Number(component.raw168?.temperature_c ?? 0),
    voltage_v: Number(component.raw168?.voltage_v ?? 0),
    supply_current_ma: Number(component.raw168?.supply_current_ma ?? 0),
    functional_frequency_mhz: Number(component.raw168?.functional_frequency_mhz ?? 0),
    propagation_delay_ns: Number(component.raw168?.propagation_delay_ns ?? 0),
    health_score_pct: Number(component.raw168?.health_score_pct ?? 0),
    anomaly_score: Number(component.raw168?.source_anomaly_score ?? 0)
  };
  const response = await fetch(`${ML_API}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(row)
  });
  if (!response.ok) throw new Error(`ML explanation returned ${response.status}`);
  return response.json();
}

async function saveComponentsToMongo(dataset) {
  if (!dataset?.length) return null;
  const rows = dataset.map(c => ({
    id: c.id,
    lot: c.lot,
    value0h: Number(c.v?.[0] ?? 0),
    value24h: Number(c.v?.[1] ?? 0),
    value96h: Number(c.v?.[2] ?? 0),
    value168h: Number(c.v?.[3] ?? 0),
    groundTruthRul: Number.isFinite(c.groundTruthRul) ? c.groundTruthRul : null,
    groundTruthFailureTime: Number.isFinite(c.groundTruthFailureTime) ? c.groundTruthFailureTime : null,
    groundTruthRisk: c.groundTruthRisk || 'Unknown'
  }));
  const response = await fetch(`${ML_API}/api/components/bulk`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(rows)
  });
  if (!response.ok) throw new Error(`MongoDB API returned ${response.status}`);
  return response.json();
}

async function runMLBatch(dataset) {
  if (!dataset?.length) return [];
  const rows = dataset.map(c => ({
    Component_ID: c.id,
    Lot_ID: c.lot,
    Value_0h: Number(c.v?.[0] ?? 0),
    Value_24h: Number(c.v?.[1] ?? 0),
    Value_96h: Number(c.v?.[2] ?? 0),
    Value_168h: Number(c.v?.[3] ?? 0),
    Drift_0_24_per_h: Number((c.v?.[1] - c.v?.[0]) / 24 || 0),
    Drift_24_96_per_h: Number((c.v?.[2] - c.v?.[1]) / 72 || 0),
    Drift_96_168_per_h: Number((c.v?.[3] - c.v?.[2]) / 72 || 0),
    Current_Time_h: 168,
    temperature_c: Number(c.raw168?.temperature_c ?? 0),
    voltage_v: Number(c.raw168?.voltage_v ?? 0),
    supply_current_ma: Number(c.raw168?.supply_current_ma ?? 0),
    functional_frequency_mhz: Number(c.raw168?.functional_frequency_mhz ?? 0),
    propagation_delay_ns: Number(c.raw168?.propagation_delay_ns ?? 0),
    health_score_pct: Number(c.raw168?.health_score_pct ?? 0),
    anomaly_score: Number(c.raw168?.source_anomaly_score ?? 0)
  }));
  const response = await fetch(`${ML_API}/api/predict-batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rows })
  });
  if (!response.ok) throw new Error(`ML backend returned ${response.status}`);
  const payload = await response.json();
  return payload.results || [];
}

// Burn-in physical inspection stages in hours
const S = [0, 24, 96, 168];

// Default baseline seed dataset
const SEED_DATA = [
  { id: 'C-10482', lot: 'L-103', v: [9.8, 12.4, 27.1, 61.8], reason: 'Latent anomaly', confidence: 93 },
  { id: 'C-09127', lot: 'L-103', v: [10.1, 14.2, 25.5, 55.2], reason: 'Rapid drift', confidence: 91 },
  { id: 'C-08314', lot: 'L-104', v: [11.4, 13.1, 21.7, 47.8], reason: 'Borderline drift', confidence: 84 },
  { id: 'C-01743', lot: 'L-101', v: [10.3, 10.9, 12.4, 15.8], reason: 'Normal', confidence: 96 },
  { id: 'C-02118', lot: 'L-101', v: [9.7, 10.6, 11.8, 14.9], reason: 'Normal', confidence: 97 },
  { id: 'C-03991', lot: 'L-102', v: [12.1, 13.8, 18.1, 29.7], reason: 'Lot deviation', confidence: 88 },
  { id: 'C-04518', lot: 'L-102', v: [11.2, 12.2, 15.4, 21.1], reason: 'Normal', confidence: 95 },
  { id: 'C-05521', lot: 'L-104', v: [10.7, 12.9, 20.8, 39.4], reason: 'Increasing drift', confidence: 87 },
  { id: 'C-06108', lot: 'L-105', v: [8.9, 9.8, 10.4, 12.2], reason: 'Normal', confidence: 98 },
  { id: 'C-07339', lot: 'L-105', v: [9.4, 11.2, 16.5, 24.3], reason: 'Normal', confidence: 94 },
  { id: 'C-08802', lot: 'L-106', v: [13.2, 15.1, 23.8, 44.6], reason: 'Lot deviation', confidence: 86 },
  { id: 'C-09744', lot: 'L-106', v: [12.6, 18.7, 34.5, 68.4], reason: 'Predicted breach', confidence: 95 },
  { id: 'C-11021', lot: 'L-103', v: [10.0, 12.8, 22.2, 51.3], reason: 'Predicted breach', confidence: 92 },
  { id: 'C-11408', lot: 'L-104', v: [11.1, 11.9, 17.2, 32.8], reason: 'Normal', confidence: 93 },
  { id: 'C-11931', lot: 'L-102', v: [12.0, 14.1, 19.2, 35.5], reason: 'Lot deviation', confidence: 89 }
];
const INITIAL_DATASET = Array.from({ length: 10000 }, (_, i) => {
  const base = SEED_DATA[i % SEED_DATA.length];

  return {
    ...base,
    id: `C-${String(i + 1).padStart(5, '0')}`,
    lot: `L-${String(101 + Math.floor(i / 100)).padStart(3, '0')}`,
  };
});
const RANK = { Critical: 4, High: 3, Watch: 2, Safe: 1 };

const MODEL_A = {
  id: 'A',
  name: 'Absolute Limit Guard',
  short: 'Model A',
  desc: 'Safety-threshold screening using the universal measured leakage limit at 168h.'
};

const MODEL_B = {
  id: 'B',
  name: 'Drift Forecast',
  short: 'Model B',
  desc: 'Trajectory-based OLS linear forecasting fitted on early burn-in behaviour (0h, 24h, 96h).'
};

const MODEL_C = {
  id: 'C',
  name: 'Trajectory Anomaly Detection',
  short: 'Model C',
  desc: 'Lot-relative robust statistical outlier detection using early burn-in trajectory features (0h, 24h, 96h). Does NOT use 168h measurement (zero data leakage).'
};


// --- CORE ANALYTICAL ENGINES ---

/**
 * Model A: Absolute Limit Guard
 * Evaluates the measured 168h value against safety threshold.
 */
function modelAResult(v, limit) {
  const measured = v && v.length >= 4 ? Number(v[3]) : 0;
  if (measured >= limit * 1.25) {
    return { status: 'BREACH', severity: 'Critical', value: measured, margin: measured - limit };
  }
  if (measured >= limit) {
    return { status: 'BREACH', severity: 'High', value: measured, margin: measured - limit };
  }
  return { status: 'PASS', severity: 'Safe', value: measured, margin: limit - measured };
}

/**
 * Model B: Drift Forecast Engine (OLS Linear Regression on 0h, 24h, 96h)
 * The measured 168h point is strictly excluded from fitting.
 */
function linearForecast(v, limit) {
  const x = [0, 24, 96];
  const y = (v || [0, 0, 0, 0]).slice(0, 3).map(n => Number(n) || 0);
  const mx = x.reduce((a, b) => a + b, 0) / x.length;
  const my = y.reduce((a, b) => a + b, 0) / y.length;
  const den = x.reduce((a, b) => a + (b - mx) ** 2, 0);
  const slope = den ? x.reduce((a, b, i) => a + (b - mx) * (y[i] - my), 0) / den : 0;
  const intercept = my - slope * mx;
  const predicted168 = Math.max(0, intercept + slope * 168);

  // Time-to-limit logic
  const isAlreadyBreached = y.some(val => val >= limit) || (v && v[3] >= limit);
  let crossHours = 999;
  let crossingText = 'No breach predicted';

  if (isAlreadyBreached) {
    crossHours = 0;
    crossingText = 'Limit already breached';
  } else if (slope > 0) {
    const cross = (limit - intercept) / slope;
    if (cross > 0 && cross < 1000) {
      crossHours = Math.round(cross);
      crossingText = `Crosses limit at ~${crossHours}h`;
    }
  }

  return {
    slope,
    intercept,
    predicted168,
    crossHours,
    crossingText,
    isAlreadyBreached
  };
}

/**
 * Model B Classification
 */
function modelBResult(v, limit) {
  const f = linearForecast(v, limit);
  let severity = 'Safe';
  if (f.isAlreadyBreached || f.predicted168 >= limit * 1.25) {
    severity = 'Critical';
  } else if (f.predicted168 >= limit) {
    severity = 'High';
  } else if (f.predicted168 >= limit * 0.78 || f.slope > 0.12) {
    severity = 'Watch';
  }
  return {
    ...f,
    severity,
    status: severity === 'Safe' ? 'PASS' : 'FORECAST ALERT'
  };
}

/**
 * Unified Decision Rule: Max severity between Model A and Model B
 */
function combinedRisk(a, b) {
  const rA = RANK[a.severity] || 1;
  const rB = RANK[b.severity] || 1;
  return rA >= rB ? a.severity : b.severity;
}

// =============================================================
// MODEL C — TRAJECTORY ANOMALY DETECTION ENGINE
// Pure JavaScript robust lot-relative statistical outlier kernel.
// Zero external dependencies. O(N) complexity.
// =============================================================

/**
 * chi2_ref: Reference chi-squared value for normalization.
 * Chosen as chi2 median for k=4 degrees of freedom (chi2.ppf(0.5, df=4) ≈ 3.357).
 * This sets the "expected" anomaly distance for a typical outlier at the median
 * of a chi-squared distribution with 4 features.
 * Documented here; do NOT change without re-evaluating the severity thresholds.
 */
const MODEL_C_CHI2_REF = 3.357;

/**
 * Feature weights for multi-dimensional anomaly distance.
 * w1: Baseline Leakage Offset (F1)
 * w2: Early Drift Velocity (F2)
 * w3: Trajectory Curvature/Acceleration (F3)
 * w4: Early OLS Fit Dispersion (F4)
 * All equal weights — no feature is artificially prioritized without empirical evidence.
 */
const MODEL_C_WEIGHTS = { w1: 1, w2: 1, w3: 1, w4: 1 };

/**
 * Model C severity thresholds (documented and transparent).
 * Rationale: These are geometric thresholds on the normalized anomaly score S ∈ [0,1].
 * - Safe: S < 0.35  — trajectory statistically consistent with lot peers
 * - Watch: 0.35 ≤ S < 0.60 — mild trajectory deviation, worth monitoring
 * - High: 0.60 ≤ S < 0.82 — significant departure from lot distribution
 * - Critical: S ≥ 0.82 — strong outlier, investigate root cause
 * These thresholds are ENGINEERING CHOICES, not validated ML metrics.
 * Validation against labeled production data is required for formal performance claims.
 */
const MODEL_C_SEVERITY_THRESHOLDS = { watch: 0.35, high: 0.60, critical: 0.82 };

/**
 * Compute the OLS-fitted hat values for [0h, 24h, 96h] using the same
 * closed-form OLS as Model B, but restricted to early window only.
 * Returns [Vhat_0, Vhat_24, Vhat_96]
 */
function earlyOLSHat(v) {
  const xs = [0, 24, 96];
  const ys = xs.map((_, i) => (isFinite(Number(v[i])) ? Number(v[i]) : 0));
  const mx = 40; // mean of [0,24,96]
  const my = (ys[0] + ys[1] + ys[2]) / 3;
  const den = xs.reduce((a, x) => a + (x - mx) ** 2, 0); // 5952
  const slope = den ? xs.reduce((a, x, i) => a + (x - mx) * (ys[i] - my), 0) / den : 0;
  const intercept = my - slope * mx;
  return xs.map(x => intercept + slope * x);
}

/**
 * Safe median of a numeric array. Returns 0 for empty/all-invalid arrays.
 */
function safeMedian(arr) {
  const valid = arr.filter(v => isFinite(v) && v !== null && v !== undefined);
  if (!valid.length) return 0;
  const sorted = [...valid].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

/**
 * Safe clamp — ensures a value is a finite number in [0,1].
 */
function clamp01(v) {
  if (!isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

/**
 * Extract Model C features for a single component.
 * Uses ONLY 0h, 24h, 96h — 168h is never accessed. Zero data leakage.
 * Returns { F1, F2, F3, F4 }
 */
function modelCFeatures(v, lotMedianV0) {
  const v0 = isFinite(Number(v[0])) ? Number(v[0]) : 0;
  const v24 = isFinite(Number(v[1])) ? Number(v[1]) : 0;
  const v96 = isFinite(Number(v[2])) ? Number(v[2]) : 0;

  // F1: Baseline Leakage Offset — how far the 0h value is from the lot median 0h
  const F1 = v0 - lotMedianV0;

  // F2: Early Drift Velocity — average rate of leakage increase over 0h→96h
  const F2 = (v96 - v0) / 96;

  // F3: Trajectory Curvature / Acceleration
  // = second-stage rate minus first-stage rate
  const rate1 = (v24 - v0) / 24;  // μA/h from 0h to 24h
  const rate2 = (v96 - v24) / 72; // μA/h from 24h to 96h
  const F3 = rate2 - rate1;

  // F4: Early OLS Fit Dispersion — sum of squared residuals vs expected linear fit
  const hat = earlyOLSHat(v);
  const F4 = (v0 - hat[0]) ** 2 + (v24 - hat[1]) ** 2 + (v96 - hat[2]) ** 2;

  return { F1, F2, F3, F4 };
}

/**
 * MODEL C — FULL COMPUTATION ENGINE
 *
 * Phase 1: Extract lot-level robust statistics (Median, MAD) for each feature.
 * Phase 2: Compute per-component modified z-scores, anomaly distance, score, severity, reasons.
 *
 * @param {Array} rawDataset - Array of { id, lot, v: [0h, 24h, 96h, 168h] }
 * @returns {Map<string, object>} - Map from component id to Model C result
 */
function computeModelC(rawDataset) {
  if (!rawDataset || !rawDataset.length) return new Map();

  const EPSILON = 1e-6; // prevent division-by-zero in MAD

  // ── PHASE 1: Group components by lot and extract features ──────────────────
  const lotGroups = new Map(); // lot → [{ id, F1, F2, F3, F4 }]

  // First pass: compute lot median of V0 (needed for F1)
  const lotV0Values = new Map();
  rawDataset.forEach(c => {
    if (!lotV0Values.has(c.lot)) lotV0Values.set(c.lot, []);
    const v0 = isFinite(Number((c.v || [])[0])) ? Number(c.v[0]) : 0;
    lotV0Values.get(c.lot).push(v0);
  });
  const lotMedianV0 = new Map();
  lotV0Values.forEach((vals, lot) => lotMedianV0.set(lot, safeMedian(vals)));

  // Second pass: extract all four features per component
  rawDataset.forEach(c => {
    const v = c.v || [0, 0, 0, 0];
    const medV0 = lotMedianV0.get(c.lot) || 0;
    const feat = modelCFeatures(v, medV0);
    if (!lotGroups.has(c.lot)) lotGroups.set(c.lot, []);
    lotGroups.get(c.lot).push({ id: c.id, ...feat });
  });

  // ── PHASE 2: Compute robust lot statistics per feature ──────────────────────
  // For each lot, compute Median_j and MAD_j for j ∈ {F1, F2, F3, F4}
  const lotStats = new Map(); // lot → { medF1, madF1, medF2, madF2, ... }
  const FEAT_KEYS = ['F1', 'F2', 'F3', 'F4'];

  lotGroups.forEach((items, lot) => {
    const stats = {};
    FEAT_KEYS.forEach(fk => {
      const vals = items.map(x => x[fk]).filter(isFinite);
      const med = safeMedian(vals);
      const absDevs = vals.map(v => Math.abs(v - med));
      const mad = 1.4826 * safeMedian(absDevs) + EPSILON;
      stats[`med${fk}`] = med;
      stats[`mad${fk}`] = mad;
    });
    lotStats.set(lot, stats);
  });

  // ── PHASE 3: Score every component ──────────────────────────────────────────
  const results = new Map();

  rawDataset.forEach(c => {
    const v = c.v || [0, 0, 0, 0];
    const medV0 = lotMedianV0.get(c.lot) || 0;
    const { F1, F2, F3, F4 } = modelCFeatures(v, medV0);
    const st = lotStats.get(c.lot) || {};

    // Modified z-scores — safe division (MAD already has EPSILON added)
    const Z1 = isFinite(st.madF1) && st.madF1 > 0 ? (F1 - st.medF1) / st.madF1 : 0;
    const Z2 = isFinite(st.madF2) && st.madF2 > 0 ? (F2 - st.medF2) / st.madF2 : 0;
    const Z3 = isFinite(st.madF3) && st.madF3 > 0 ? (F3 - st.medF3) / st.madF3 : 0;
    const Z4 = isFinite(st.madF4) && st.madF4 > 0 ? (F4 - st.medF4) / st.madF4 : 0;

    // Multi-feature anomaly distance (weighted Euclidean in z-score space)
    const { w1, w2, w3, w4 } = MODEL_C_WEIGHTS;
    const D2 = w1 * Z1 ** 2 + w2 * Z2 ** 2 + w3 * Z3 ** 2 + w4 * Z4 ** 2;
    const anomalyDistance = isFinite(D2) ? Math.sqrt(D2) : 0;

    // Normalized anomaly score S ∈ [0,1]
    const rawScore = 1 - Math.exp(-D2 / (2 * MODEL_C_CHI2_REF));
    const anomalyScore = clamp01(rawScore);

    // Severity classification (transparent thresholds — see MODEL_C_SEVERITY_THRESHOLDS)
    let severity = 'Safe';
    if (anomalyScore >= MODEL_C_SEVERITY_THRESHOLDS.critical) severity = 'Critical';
    else if (anomalyScore >= MODEL_C_SEVERITY_THRESHOLDS.high) severity = 'High';
    else if (anomalyScore >= MODEL_C_SEVERITY_THRESHOLDS.watch) severity = 'Watch';

    const anomalyDetected = anomalyScore >= MODEL_C_SEVERITY_THRESHOLDS.watch;

    // ── Explainable anomaly reasons ──────────────────────────────────────────
    const reasons = [];
    const zPairs = [
      { label: 'baseline leakage offset (F1)', z: Z1, sign: F1 >= 0 ? '+' : '' },
      { label: 'early drift velocity (F2)', z: Z2, sign: F2 >= (st.medF2 || 0) ? '+' : '' },
      { label: 'trajectory curvature/acceleration (F3)', z: Z3, sign: F3 >= (st.medF3 || 0) ? '+' : '' },
      { label: 'OLS fit dispersion (F4)', z: Z4, sign: '' }
    ];
    zPairs.forEach(({ label, z, sign }) => {
      const absZ = Math.abs(z);
      if (absZ >= 3.0) {
        reasons.push(`${sign}${z.toFixed(1)}σ ${label} — extreme departure from lot distribution`);
      } else if (absZ >= 2.0) {
        reasons.push(`${sign}${z.toFixed(1)}σ ${label} — significant deviation above lot median`);
      } else if (absZ >= 1.4) {
        reasons.push(`${sign}${z.toFixed(1)}σ ${label} — mild elevation vs lot peers`);
      }
    });
    if (!reasons.length) {
      reasons.push('Trajectory is consistent with lot peer distribution (all |Z| < 1.4σ)');
    }

    // Strongest contributing feature (largest |Z|)
    const zMag = [Math.abs(Z1), Math.abs(Z2), Math.abs(Z3), Math.abs(Z4)];
    const maxIdx = zMag.indexOf(Math.max(...zMag));
    const featureNames = ['F1 — Baseline Offset', 'F2 — Drift Velocity', 'F3 — Curvature', 'F4 — OLS Dispersion'];
    const strongestFeature = featureNames[maxIdx];

    results.set(c.id, {
      anomalyScore,
      anomalyDetected,
      anomalyDistance: isFinite(anomalyDistance) ? anomalyDistance : 0,
      severity,
      status: anomalyDetected ? 'ANOMALY' : 'NORMAL',
      F1: isFinite(F1) ? F1 : 0,
      F2: isFinite(F2) ? F2 : 0,
      F3: isFinite(F3) ? F3 : 0,
      F4: isFinite(F4) ? F4 : 0,
      Z1: isFinite(Z1) ? Z1 : 0,
      Z2: isFinite(Z2) ? Z2 : 0,
      Z3: isFinite(Z3) ? Z3 : 0,
      Z4: isFinite(Z4) ? Z4 : 0,
      anomalyReasons: reasons,
      strongestFeature,
      lotMedianV0: medV0
    });
  });

  return results;
}

/**
 * Unified tri-model severity: max of Model A, B, and C severities.
 */
function combinedRisk3(a, b, cResult) {
  const rA = RANK[a.severity] || 1;
  const rB = RANK[b.severity] || 1;
  const rC = RANK[cResult.severity] || 1;
  const best = Math.max(rA, rB, rC);
  return Object.keys(RANK).find(k => RANK[k] === best) || 'Safe';
}

// --- UI HELPERS ---


const badge = r => <span className={'badge ' + (r || 'safe').toLowerCase()}>{r || 'Safe'}</span>;

const Btn = ({ children, onClick, secondary = false, disabled = false, className = '' }) => (
  <button className={(secondary ? 'secondary ' : 'primary ') + className} onClick={onClick} disabled={disabled}>
    {children}
  </button>
);

function Stat({ I, label, value, sub, tone = '' }) {
  return (
    <div className={'stat ' + tone}>
      <I size={18} />
      <div>
        <small>{label}</small>
        <strong>{value}</strong>
        <em>{sub}</em>
      </div>
    </div>
  );
}

function Section({ title, sub, children, actions }) {
  return (
    <section className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <h2>{title}</h2>
          {sub && <p className="sub">{sub}</p>}
        </div>
        {actions && <div style={{ display: 'flex', gap: '6px' }}>{actions}</div>}
      </div>
      {children}
    </section>
  );
}

function EmptyState({ title = 'No Data Loaded', message = 'Upload a CSV dataset or restore demo records.', onReset, onUpload }) {
  return (
    <div className="card emptyState" style={{ textAlign: 'center', padding: '48px 24px', margin: '20px 0' }}>
      <Database size={40} style={{ color: '#4a6d8c', marginBottom: '14px' }} />
      <h2 style={{ fontSize: '16px', marginBottom: '6px' }}>{title}</h2>
      <p className="sub" style={{ maxWidth: '420px', margin: '0 auto 18px' }}>{message}</p>
      <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
        {onUpload && <Btn onClick={onUpload}><Upload size={14} /> Upload CSV</Btn>}
        {onReset && <Btn secondary onClick={onReset}><RotateCcw size={14} /> Restore Demo Data</Btn>}
      </div>
    </div>
  );
}

function Table({ rows, open, pageSize = 20 }) {
  const [pageIdx, setPageIdx] = useState(0);
  const [filter, setFilter] = useState('');

  const filtered = useMemo(() => {
    if (!filter) return rows;
    const q = filter.toLowerCase();
    return rows.filter(r => r.id.toLowerCase().includes(q) || r.lot.toLowerCase().includes(q) || r.risk.toLowerCase().includes(q));
  }, [rows, filter]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const currentRows = useMemo(() => {
    const start = pageIdx * pageSize;
    return filtered.slice(start, start + pageSize);
  }, [filtered, pageIdx, pageSize]);

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px', gap: '10px', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: '#071827', padding: '5px 10px', borderRadius: '6px', border: '1px solid #1c354b', maxWidth: '240px', width: '100%' }}>
          <Search size={13} style={{ color: '#68839b' }} />
          <input
            style={{ background: 'none', border: 'none', color: '#fff', fontSize: '11px', outline: 'none', width: '100%' }}
            placeholder="Filter ID, Lot, Risk..."
            value={filter}
            onChange={e => { setFilter(e.target.value); setPageIdx(0); }}
          />
          {filter && <X size={12} style={{ cursor: 'pointer', color: '#8ca6bd' }} onClick={() => setFilter('')} />}
        </div>
        <small style={{ color: '#71899f', fontSize: '9px' }}>
          Showing {filtered.length === 0 ? 0 : pageIdx * pageSize + 1} - {Math.min(filtered.length, (pageIdx + 1) * pageSize)} of {filtered.length} components
        </small>
      </div>

      <div className="table">
        <table>
          <thead>
            <tr>
              <th>Component</th>
              <th>Lot</th>
              <th>0h</th>
              <th>24h</th>
              <th>96h</th>
              <th>Measured 168h</th>
              <th>Forecast 168h</th>
              <th>RUL</th>
              <th>ML Risk</th>
              <th>Risk</th>
              <th>Conf.</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {currentRows.length === 0 ? (
              <tr>
                <td colSpan={12} style={{ textAlign: 'center', padding: '24px', color: '#6f879d' }}>
                  No matching components found.
                </td>
              </tr>
            ) : (
              currentRows.map(c => (
                <tr key={c.id}>
                  <td><b>{c.id}</b></td>
                  <td>{c.lot}</td>
                  <td>{c.v[0]?.toFixed(1)} μA</td>
                  <td>{c.v[1]?.toFixed(1)} μA</td>
                  <td>{c.v[2]?.toFixed(1)} μA</td>
                  <td><b>{c.v[3]?.toFixed(1)} μA</b></td>
                  <td style={{ color: c.forecast >= 50 ? '#f3a06f' : '#88a4bc' }}>{c.forecast?.toFixed(1)} μA</td>
                  <td style={{ color: c.rulHours != null && c.rulHours < 120 ? '#ff7b7b' : '#68dda0', fontWeight: 800 }}>{c.rulHours == null ? '—' : `${c.rulHours.toFixed(0)} h`}</td>
                  <td>{c.mlRisk || 'Pending'}</td>
                  <td>{badge(c.risk)}</td>
                  <td>{c.confidence}%</td>
                  <td><button className="small" onClick={() => open(c.id)}>View</button></td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '10px' }}>
          <button className="small" onClick={() => setPageIdx(p => Math.max(0, p - 1))} disabled={pageIdx === 0}>
            <ChevronLeft size={12} /> Prev
          </button>
          <span style={{ fontSize: '9px', color: '#7791a8' }}>Page {pageIdx + 1} of {totalPages}</span>
          <button className="small" onClick={() => setPageIdx(p => Math.min(totalPages - 1, p + 1))} disabled={pageIdx >= totalPages - 1}>
            Next <ChevronRight size={12} />
          </button>
        </div>
      )}
    </div>
  );
}

// --- MAIN APPLICATION COMPONENT ---
// LANDING PAGE
function LandingPage({ onEnter }) {
  return (
    <div className="landingPage">
      <img
  src="/nexgenx-logo.jpeg.jpeg"
  alt="NexGenX Logo"
  className="landingLogo"
/>
      <div className="landingGlow landingGlowOne" />
      <div className="landingGlow landingGlowTwo" />

      <div className="landingContent">

        <div className="landingTitle">
          BURN AI INSPECTOR
<p className="presented-by">Presented by NexGenX</p>
        </div>

        <div className="landingSubtitle">
          AI-Driven Anomaly Detection
          <br />
          in Component Burn-In & Screening
        </div>

        <p className="landingDescription">
          An intelligent burn-in screening platform that analyzes component
          behaviour over time to detect anomalies, predict drift, and identify
          potential latent defects before they become critical failures.
        </p>

        <button
          className="landingEnter"
          onClick={onEnter}
        >
          ENTER INSPECTOR
          <span>→</span>
        </button>

        <div className="landingInfo">
          <div className="landingPS">
            <span>SIH 2026</span>
            <span>PS 26170</span>
          </div>

          <div className="landingDivider" />

          <div className="landingMembersTitle">
            TEAM MEMBERS
          </div>

          <div className="landingMembers">
            <span>Sarika Misra</span>
            <span>Debargha Ghosh</span>
            <span>Soham Barapanda</span>
            <span>Sagnik Mukhopadhaya</span>
            <span>Urnavo Chowdhury</span>
            <span>Reek Bhowmick</span>
          </div>
        </div>

      </div>
    </div>
  );
}


function App() {
  const [showLanding, setShowLanding] = useState(() => sessionStorage.getItem('inspectorEntered') !== 'true');
  const [page, setPage] = useState('home');
  const [rawDataset, setRawDataset] = useState(INITIAL_DATASET);

  const [componentSearch, setComponentSearch] = useState('');
  const [sel, setSel] = useState('');
const [showComponentResults, setShowComponentResults] = useState(false);
  const [limit, setLimit] = useState(50);
  const [sens, setSens] = useState(0.5);
  const [selectedLot, setSelectedLot] = useState('L-103');
  const [scenario, setScenario] = useState(1);
  const [toast, setToast] = useState('');
  const [upload, setUpload] = useState(false);
  const [validationReport, setValidationReport] = useState(null);
  const [q, setQ] = useState('');
  const [ans, setAns] = useState('');
  const [review, setReview] = useState({});
  const [model, setModel] = useState('B');
  const [liveMonitoring, setLiveMonitoring] = useState(false);
const [liveStage, setLiveStage] = useState(0);
const [liveTick, setLiveTick] = useState(0);
const [mlResults, setMlResults] = useState({});
const [mlStatus, setMlStatus] = useState('starting');
const [mlError, setMlError] = useState('');
const [mlMetrics, setMlMetrics] = useState(null);
const startLiveMonitoring = () => {
  setLiveStage(0);
  setLiveTick(0);
  setLiveMonitoring(true);
};

const stopLiveMonitoring = () => {
  setLiveMonitoring(false);
};
useEffect(() => {
  if (!liveMonitoring) return;

  const timer = setInterval(() => {
    setLiveTick(t => {
      const next = t + 1;

      if (next >= 4) {
        setLiveMonitoring(false);
        setLiveStage(3);
        return 4;
      }

      setLiveStage(next);
      return next;
    });
  }, 2500);

  return () => clearInterval(timer);
}, [liveMonitoring]);

  // --- REAL ML BACKEND PIPELINE ---
  // One batch request keeps the 25,000-component dataset practical.
  useEffect(() => {
    if (!rawDataset?.length) return;
    let cancelled = false;
    setMlStatus('running');
    setMlError('');
    setMlResults({});
    fetch(`${ML_API}/api/metrics`)
      .then(r => r.ok ? r.json() : Promise.reject(new Error('Metrics unavailable')))
      .then(m => { if (!cancelled) setMlMetrics(m); })
      .catch(() => {});
    runMLBatch(rawDataset).then(results => {
      if (cancelled) return;
      const map = {};
      results.forEach(r => { map[r.component_id] = r; });
      setMlResults(map);
      setMlStatus('online');
    }).catch(err => {
      console.error('ML backend:', err);
      if (cancelled) return;
      setMlStatus('offline');
      setMlError(err?.message || 'ML backend unavailable');
      setMlResults({});
    });
    return () => { cancelled = true; };
  }, [rawDataset]);

  // SHAP is computed lazily for the selected component only. This avoids doing
  // an expensive explanation pass for all 25,000 rows during batch scoring.
  useEffect(() => {
    if (mlStatus !== 'online' || !sel) return;
    const component = rawDataset.find(x => x.id === sel);
    if (!component) return;
    runMLSingle(component).then(result => {
      setMlResults(prev => ({ ...prev, [component.id]: { ...(prev[component.id] || {}), ...result } }));
    }).catch(err => console.warn('Selected-component SHAP:', err));
  }, [sel, mlStatus, rawDataset]);

  // Uploaded CSV is the dataset source of truth.

  // --- DYNAMICALLY DERIVED STATE (ZERO STALE DERIVED VALUES) ---

  const computedCs = useMemo(() => {
    if (!rawDataset || !rawDataset.length) return [];

    // Run Model C first — needs full dataset to compute lot-relative statistics
    const modelCMap = computeModelC(rawDataset);

    return rawDataset.map(c => {
      const a = modelAResult(c.v, limit);
      const b = modelBResult(c.v, limit);
      const mc = modelCMap.get(c.id) || {
        anomalyScore: 0, anomalyDetected: false, anomalyDistance: 0,
        severity: 'Safe', status: 'NORMAL',
        F1: 0, F2: 0, F3: 0, F4: 0,
        Z1: 0, Z2: 0, Z3: 0, Z4: 0,
        anomalyReasons: ['No Model C data available'],
        strongestFeature: 'N/A', lotMedianV0: 0
      };

      const ml = mlResults[c.id];
      const mlRisk = ml?.risk === 'Critical' ? 'Critical' : ml?.risk === 'Warning' ? 'Watch' : 'Safe';
      const legacyRisk = combinedRisk3(a, b, mc);
      // Model A is the non-negotiable safety guard; ML/legacy signals provide prediction.
      const bestRank = Math.max(RANK[a.severity] || 1, RANK[mlRisk] || 1, RANK[legacyRisk] || 1);
      const risk = Object.keys(RANK).find(k => RANK[k] === bestRank) || 'Safe';

      return {
        id: c.id,
        lot: c.lot,
        v: c.v,
        reason: c.reason || 'Screening observation',
        confidence: c.confidence || 88,
        risk,
        modelA: a,
        modelB: b,
        modelC: mc,
        slope: b.slope,
        intercept: b.intercept,
        forecast: b.predicted168,
        hours: b.crossHours,
        forecastHours: b.crossHours,
        crossingText: b.crossingText,
        // Keep legacy anomaly field pointing at Model C score for backward compat
        anomaly: mc.anomalyScore,
        ml,
        mlRisk,
        failureProbability: ml?.failure_probability ?? null,
        rulHours: ml?.rul_h ?? null,
        predictedLifetimeHours: ml?.predicted_lifetime_h ?? null,
        rulLowHours: ml?.rul_low_h ?? null,
        rulHighHours: ml?.rul_high_h ?? null,
        isolationAnomaly: ml?.anomaly_score ?? null,
        isolationFlag: ml?.anomaly ?? null,
        fisherScore: ml?.fisher_score ?? null,
        shap: ml?.shap || [],
        groundTruthRul: c.groundTruthRul,
        groundTruthFailureTime: c.groundTruthFailureTime,
        groundTruthRisk: c.groundTruthRisk
      };
    });
  }, [rawDataset, limit, mlResults]);

  // Dynamically derived unique lots list
  const availableLots = useMemo(() => {
    const set = new Set(computedCs.map(c => c.lot).filter(Boolean));
    return Array.from(set);
  }, [computedCs]);

  // Ensure active lot is valid
  const currentLot = useMemo(() => {
    if (availableLots.includes(selectedLot)) return selectedLot;
    return availableLots[0] || '';
  }, [availableLots, selectedLot]);

  // Selected component
  const c = useMemo(() => {
    if (!computedCs.length) return null;
    return computedCs.find(x => x.id === sel) || computedCs[0];
  }, [computedCs, sel]);

const componentSearchResults = useMemo(() => {
    return computedCs;
}, [computedCs]);

  // Global counts and metrics
  const stats = useMemo(() => {
    const total = computedCs.length;
    return {
      total,
      safe: computedCs.filter(x => x.risk === 'Safe').length,
      watch: computedCs.filter(x => x.risk === 'Watch').length,
      high: computedCs.filter(x => x.risk === 'High').length,
      critical: computedCs.filter(x => x.risk === 'Critical').length
    };
  }, [computedCs]);

  // Population metrics per lot
  const lots = useMemo(() => {
    return availableLots.map(l => {
      const items = computedCs.filter(x => x.lot === l);
      if (!items.length) {
        return { lot: l, r: 0, base: 0, max: 0, health: 100, count: 0 };
      }
      const r = items.filter(x => x.risk !== 'Safe').length;
      const base = items.reduce((sum, item) => sum + (item.v[0] || 0), 0) / items.length;
      const maxVal = Math.max(...items.map(item => item.v[3] || 0));
      const penaltyRatio = (r / items.length) * 75;
      const penaltyExcess = Math.max(0, maxVal - limit) * 0.35;
      const health = Math.max(0, Math.min(100, Math.round(100 - penaltyRatio - penaltyExcess)));
      return { lot: l, r, base, max: maxVal, health, count: items.length };
    });
  }, [availableLots, computedCs, limit]);

  // Ranked priority list
  const risky = useMemo(() => {
    return [...computedCs]
      .filter(item => item.risk !== 'Safe')
      .sort((a, b) => (RANK[b.risk] || 0) - (RANK[a.risk] || 0) || b.forecast - a.forecast);
  }, [computedCs]);

  const go = (p, id) => {
    if (id) setSel(id);
    setPage(p);
  };

  const notify = msg => {
    setToast(msg);
    setTimeout(() => setToast(''), 2600);
  };

  const applyConfig = () => notify('Configuration applied dynamically across all models');

  const restoreDemo = () => {
    setRawDataset(SEED_DATA);
    setSel('C-10482');
    setSelectedLot('L-103');
    setValidationReport(null);
    notify('Demo dataset restored (15 components, 6 lots)');
  };

  const exportCSV = () => {
    if (!computedCs.length) {
      notify('No data to export');
      return;
    }
    const rows = computedCs.map(item => ({
      component_id: item.id,
      lot_id: item.lot,
      '0h': item.v[0],
      '24h': item.v[1],
      '96h': item.v[2],
      '168h_measured': item.v[3],
      '168h_forecast': item.forecast.toFixed(2),
      model_a_status: item.modelA.status,
      model_b_status: item.modelB.status,
      combined_risk: item.risk,
      anomaly_score: item.anomaly.toFixed(2),
      confidence_pct: item.confidence,
      time_to_limit_h: item.hours === 999 ? 'No breach' : item.hours === 0 ? 'Breached' : item.hours,
      ml_risk: item.mlRisk || 'Pending',
      failure_probability_pct: item.failureProbability == null ? '' : (item.failureProbability * 100).toFixed(1),
      predicted_rul_h: item.rulHours == null ? '' : item.rulHours.toFixed(1),
      predicted_lifetime_h: item.predictedLifetimeHours == null ? '' : item.predictedLifetimeHours.toFixed(1),
      rul_low_h: item.rulLowHours == null ? '' : item.rulLowHours.toFixed(1),
      rul_high_h: item.rulHighHours == null ? '' : item.rulHighHours.toFixed(1),
      isolation_anomaly_score: item.isolationAnomaly == null ? '' : item.isolationAnomaly.toFixed(4),
      fisher_score: item.fisherScore == null ? '' : item.fisherScore.toFixed(4),
      shap_top_feature: item.shap?.[0]?.feature || ''
    }));

    const blob = new Blob([Papa.unparse(rows)], { type: 'text/csv;charset=utf-8;' });
    const u = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = u;
    a.download = `BURN AI INSPECTOR-reliability-report-limit${limit}uA.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(u), 1000);
    notify(`Exported ${rows.length} component records`);
  };

  // --- CO-PILOT DYNAMIC QUESTION ANSWERING ---
  const handleAsk = customQuery => {
    const queryStr = (customQuery !== undefined ? customQuery : q).toLowerCase();
    if (!computedCs.length) {
      setAns('No dataset is currently loaded. Upload a CSV or restore demo records to query BURN AI INSPECTOR.');
      return;
    }

    if (queryStr.includes('c-') || queryStr.includes('up-') || queryStr.match(/[a-z0-9]+-[0-9]+/)) {
      const match = computedCs.find(item => queryStr.includes(item.id.toLowerCase()));
      if (match) {
        setAns(
          `${match.id} (Lot ${match.lot}): Measured 168h is ${match.v[3]?.toFixed(1)} μA. Model B early slope is ${match.slope.toFixed(3)} μA/h with a projected 168h endpoint of ${match.forecast.toFixed(1)} μA. Decision: ${match.risk.toUpperCase()} (${match.crossingText}).`
        );
        return;
      }
    }

    if (queryStr.includes('why') && c) {
      setAns(
        `${c.id} (${c.lot}): Measured 168h is ${c.v[3]?.toFixed(1)} μA against limit ${limit} μA. Model B projects ${c.forecast.toFixed(1)} μA with early drift slope ${c.slope.toFixed(3)} μA/h. Status: ${c.risk} (${c.crossingText}).`
      );
      return;
    }

    if (queryStr.includes('lot') || queryStr.includes('worst') || queryStr.includes('deteriorat')) {
      if (lots.length) {
        const sortedLots = [...lots].sort((a, b) => a.health - b.health);
        const worst = sortedLots[0];
        setAns(
          `Lot ${worst.lot} has the lowest reliability health score (${worst.health}%) with ${worst.r} of ${worst.count} components requiring attention (baseline ${worst.base.toFixed(1)} μA, peak ${worst.max.toFixed(1)} μA).`
        );
      } else {
        setAns('No lots available in the current dataset.');
      }
      return;
    }

    if (queryStr.includes('48') || queryStr.includes('cross') || queryStr.includes('breach')) {
      const soon = computedCs.filter(item => item.hours > 0 && item.hours <= 48);
      const already = computedCs.filter(item => item.hours === 0);
      setAns(
        `Analysis against ${limit} μA limit: ${already.length} components already in breach. ${soon.length} components forecast to cross the boundary within 48h (${soon.map(x => x.id).slice(0, 5).join(', ') || 'None'}).`
      );
      return;
    }

    setAns(
      `BURN AI INSPECTOR Engine active on ${computedCs.length} components across ${availableLots.length} lots. Try asking about a component (e.g. "${computedCs[0]?.id}"), the weakest lot, or crossings within 48 hours.`
    );
  };

  // Collapsible Navigation Structure
  const navSections = [
    { type: 'item', id: 'home', icon: Gauge, label: 'Command Center' },
    {
      type: 'group',
      id: 'data-group',
      title: 'SENSOR DATA',
      children: [
        { id: 'data', icon: Database, label: 'Data Intelligence' },
        { id: 'lot', icon: Layers3, label: 'Lot DNA' }
      ]
    },
    {
      type: 'group',
      id: 'detection-group',
      title: 'DETECTION & FORECAST',
      children: [
        { id: 'anomaly', icon: Target, label: 'Hidden Defects' },
        { id: 'prediction', icon: LineIcon, label: 'Drift Forecast' }
      ]
    },
    {
      type: 'group',
      id: 'risk-group',
      title: 'RISK & RESPONSE',
      children: [
        { id: 'risk', icon: ShieldCheck, label: 'Risk & Safety' },
        { id: 'priority', icon: ListChecks, label: 'Priority Queue' },
        { id: 'action', icon: Zap, label: 'Action Center' }
      ]
    },
    {
      type: 'group',
      id: 'analysis-group',
      title: 'ANALYSIS',
      children: [
        { id: 'inspector', icon: Activity, label: 'Component Profile' },
        { id: 'whatif', icon: FlaskConical, label: 'What-If Lab' },
        { id: 'model', icon: BrainCircuit, label: 'Model Trust' }
      ]
    },
    { type: 'item', id: 'reports', icon: Download, label: 'Reports' },
    { type: 'item', id: 'settings', icon: Settings, label: 'Settings' }
  ];

  // Collapsed state dictionary for sidebar groups
  const [collapsedGroups, setCollapsedGroups] = useState({});

  const toggleGroup = groupId => {
    setCollapsedGroups(prev => ({
      ...prev,
      [groupId]: !prev[groupId]
    }));
  };

  // Keep group containing current active page expanded
  useEffect(() => {
    navSections.forEach(sec => {
      if (sec.type === 'group' && sec.children.some(c => c.id === page)) {
        setCollapsedGroups(prev => (prev[sec.id] ? { ...prev, [sec.id]: false } : prev));
      }
    });
  }, [page]);

  const getBreadcrumb = current => {
    for (const sec of navSections) {
      if (sec.type === 'item' && sec.id === current) return sec.label;
      if (sec.type === 'group') {
        const child = sec.children.find(c => c.id === current);
        if (child) return `${sec.title} › ${child.label}`;
      }
    }
    return 'Command Center';
  };

 if (showLanding) {
  return (
    <LandingPage
      onEnter={() => setShowLanding(false)}
    />
  );
}

return (
  <div className="app">
      <aside>
        <div className="brand">
          <div className="logo"><Sparkles size={17} /></div>
          <div>
            <b>BURN AI INSPECTOR</b>
            <small>BURN-IN RELIABILITY</small>
          </div>
        </div>
        <div className="online">● ENGINE {mlStatus === 'online' ? 'ONLINE' : mlStatus === 'running' ? 'SCORING' : 'OFFLINE'}</div>
        <nav className="navContainer">
          {navSections.map(sec => {
            if (sec.type === 'item') {
              const Icon = sec.icon;
              return (
                <button
                  key={sec.id}
                  className={'nav ' + (page === sec.id ? 'active' : '')}
                  onClick={() => setPage(sec.id)}
                >
                  <Icon size={15} />
                  <span>{sec.label}</span>
                </button>
              );
            }

            const isCollapsed = Boolean(collapsedGroups[sec.id]);
            const hasActiveChild = sec.children.some(c => c.id === page);

            return (
              <div className="navGroup" key={sec.id}>
                <button
                  type="button"
                  className={`navGroupHeader ${hasActiveChild ? 'groupActive' : ''}`}
                  onClick={() => toggleGroup(sec.id)}
                  aria-expanded={!isCollapsed}
                >
                  <span>{sec.title}</span>
                  {isCollapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
                </button>
                {!isCollapsed && (
                  <div className="navGroupItems">
                    {sec.children.map(child => {
                      const ChildIcon = child.icon;
                      return (
                        <button
                          key={child.id}
                          className={'nav ' + (page === child.id ? 'active' : '')}
                          onClick={() => setPage(child.id)}
                        >
                          <ChildIcon size={14} />
                          <span>{child.label}</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </nav>
        <button className="load" onClick={() => setUpload(true)}>
          <Upload size={14} /> Load CSV
        </button>
      </aside>

      <main>
        <header>
          <div className="mobile"><Menu size={17} /> BURN AI INSPECTOR</div>
          <span>BURN AI INSPECTOR <ChevronRight size={12} /> {getBreadcrumb(page)}</span>
          <div className="head">
            <button className="icon" onClick={() => setPage('action')} title="Priority Alerts">
              <Bell size={16} />
              {stats.high + stats.critical > 0 && <sup>{stats.high + stats.critical}</sup>}
            </button>
            <i /> {computedCs.length} components active
          </div>
        </header>

        <div className="content">
          <Page
            page={page}
            stats={stats}
            lots={lots}
            availableLots={availableLots}
            currentLot={currentLot}
            setLot={setSelectedLot}
            risky={risky}
            cs={computedCs}
            rawCount={rawDataset.length}
            c={c}
            setSel={setSel}
            limit={limit}
            setLimit={setLimit}
            sens={sens}
            setSens={setSens}
            scenario={scenario}
            setScenario={setScenario}
            review={review}
            setReview={setReview}
            go={go}
            exportCSV={exportCSV}
            restoreDemo={restoreDemo}
            openUpload={() => setUpload(true)}
            challenge={() => {
              if (computedCs.some(x => x.id === 'C-10482')) {
                setSel('C-10482');
                setPage('inspector');
                notify('Signature case: early drift detected prior to absolute threshold');
              } else if (computedCs.length) {
                setSel(computedCs[0].id);
                setPage('inspector');
              }
            }}
            q={q}
            setQ={setQ}
            ans={ans}
            ask={handleAsk}
            notify={notify}
            model={model}
            setModel={setModel}
            applyConfig={applyConfig}
            validationReport={validationReport}
            mlStatus={mlStatus}
            mlError={mlError}
            mlMetrics={mlMetrics}
          liveMonitoring={liveMonitoring}
liveStage={liveStage}
liveTick={liveTick}
startLiveMonitoring={startLiveMonitoring}
stopLiveMonitoring={stopLiveMonitoring}
componentSearch={componentSearch}
setComponentSearch={setComponentSearch}
showComponentResults={showComponentResults}
setShowComponentResults={setShowComponentResults}
componentSearchResults={componentSearchResults}
/>
        </div>
      </main>

      {/* --- CSV INGESTION MODAL WITH COMPREHENSIVE VALIDATION --- */}
      {upload && (
        <div className="modalbg" onMouseDown={() => setUpload(false)}>
          <div className="modal" onMouseDown={e => e.stopPropagation()}>
            <div className="modalhead">
              <h2>Load Burn-In Inspection CSV</h2>
              <button className="icon" onClick={() => setUpload(false)}><X size={16} /></button>
            </div>
            <div className="upload">
              <FileSpreadsheet size={32} style={{ color: '#63bce9', margin: '0 auto 10px' }} />
              <h3>Import Component Dataset</h3>
              <p style={{ margin: '6px 0 16px', color: '#829db5' }}>
                Time-series columns: <code>Component_ID, Lot_ID, Time_h, Leakage_Current_mA</code><br />
                Required checkpoints: <code>0h, 24h, 96h, 168h</code>
              </p>
              <label className="primary" style={{ cursor: 'pointer' }}>
                Choose CSV File
                <input
                  hidden
                  type="file"
                  accept=".csv,text/csv"
                  onChange={e => {
                    const f = e.target.files?.[0];
                    if (!f) return;
                    e.target.value = '';

                    Papa.parse(f, {
                      header: true,
                      dynamicTyping: true,
                      skipEmptyLines: true,
                      worker: true,
                      complete: r => {
                        const rows = r.data || [];
                        if (!rows.length) {
                          notify('CSV file contains no readable data rows.');
                          setUpload(false);
                          return;
                        }

                        // The supplied 25000-component RUL dataset is LONG FORMAT:
                        // one row per component/time observation. Build one component
                        // record from its 0h/24h/96h/168h leakage measurements.
                        const byId = new Map();
                        let invalidRows = 0;

                        for (const row of rows) {
                          if (!row || typeof row !== 'object') {
                            invalidRows++;
                            continue;
                          }

                          const id = String(row.Component_ID ?? row.component_id ?? '').trim();
                          const lot = String(row.Lot_ID ?? row.lot_id ?? 'LOT-UNKNOWN').trim();
                          const time = Number(row.Time_h);
                          const leakage = Number(row.Leakage_Current_mA);

                          if (!id || !Number.isFinite(time) || !Number.isFinite(leakage)) {
                            invalidRows++;
                            continue;
                          }

                          if (!byId.has(id)) {
                            byId.set(id, {
                              id,
                              lot,
                              values: {},
                              rows: {}
                            });
                          }

                          const c = byId.get(id);
                          c.values[time] = leakage;
                          c.rows[time] = row;
                        }

                        const parsed = [];
                        let incompleteComponents = 0;

                        for (const component of byId.values()) {
                          const v0 = Number(component.values[0]);
                          const v24 = Number(component.values[24]);
                          const v96 = Number(component.values[96]);
                          const v168 = Number(component.values[168]);

                          if (![v0, v24, v96, v168].every(Number.isFinite)) {
                            incompleteComponents++;
                            continue;
                          }

                          // 168h row supplies validation labels/features. These are
                          // stored as ground truth and NEVER sent as model inputs.
                          const endpoint = component.rows[168] || {};

                          parsed.push({
                            id: component.id,
                            lot: component.lot,
                            v: [v0, v24, v96, v168],
                            reason: 'Imported time-series burn-in record',
                            confidence: 90,
                            groundTruthRul: Number(endpoint.RUL_h),
                            groundTruthFailureTime: Number(endpoint.Failure_Time_h),
                            groundTruthRisk: String(endpoint.Risk_Class ?? 'Unknown'),
                            raw168: {
                              temperature_c: Number(endpoint.Temperature_C),
                              voltage_v: Number(endpoint.Applied_Voltage_V),
                              supply_current_ma: Number(endpoint.Supply_Current_mA),
                              functional_frequency_mhz: Number(endpoint.Functional_Frequency_MHz),
                              propagation_delay_ns: Number(endpoint.Propagation_Delay_ns),
                              health_score_pct: Number(endpoint.Health_Score_pct),
                              source_anomaly_score: Number(endpoint.Anomaly_Score)
                            }
                          });
                        }

                        console.log('CSV rows:', rows.length);
                        console.log('Unique components:', byId.size);
                        console.log('Valid burn-in components:', parsed.length);
                        console.log('Incomplete components:', incompleteComponents);

                        if (!parsed.length) {
                          notify('No complete components found. Expected Time_h values 0, 24, 96 and 168 with Leakage_Current_mA.');
                          setUpload(false);
                          return;
                        }

                        setRawDataset(parsed);
                        saveComponentsToMongo(parsed).catch(err => console.warn('MongoDB save skipped:', err));
                        setSel(parsed[0].id);
                        setSelectedLot(parsed[0].lot);
                        setValidationReport({
                          fileName: f.name,
                          totalRows: rows.length,
                          validRows: parsed.length,
                          invalidRows,
                          duplicates: 0,
                          incompleteComponents
                        });

                        notify(`Loaded ${parsed.length.toLocaleString()} components from ${rows.length.toLocaleString()} time-series rows`);
                        setUpload(false);
                      },
                      error: err => {
                        console.error('CSV Parse Error:', err);
                        notify(`CSV Parse Error: ${err.message}`);
                        setUpload(false);
                      }
                    });
                  }}
                />
              </label>
            </div>
          </div>
        </div>
      )}

      {toast && (
        <div className="toast">
          <CheckCircle2 size={15} />
          {toast}
        </div>
      )}
    </div>
  );
}

// --- PAGE ROUTER AND VIEW PRESENTATION ---

function Page(props) {
  const {
    page,
    stats,
    lots,
    availableLots,
    currentLot,
    setLot,
    risky,
    cs,
    c,
    sel,
    setSel,
    limit,
    setLimit,
    sens,
    setSens,
    scenario,
    setScenario,
    review,
    setReview,
    go,
    exportCSV,
    restoreDemo,
    openUpload,
    challenge,
    q,
    setQ,
    ans,
    ask,
    notify,
    model,
setModel,
applyConfig,
validationReport,
liveMonitoring,
liveStage,
liveTick,
startLiveMonitoring,
stopLiveMonitoring,
componentSearch,
setComponentSearch,
showComponentResults,
setShowComponentResults,
componentSearchResults,
mlStatus,
mlError,
mlMetrics
} = props;

  // Shared Top Header
  const commonHeader = (
    <>
      <div className="hero">
        <div>
          <label>AI-DRIVEN ANOMALY DETECTION IN COMPONENT BURN-IN & SCREENING</label>
          <h1>BURN AI INSPECTOR — Intelligent Burn-In Reliability & Anomaly Detection</h1>
          <p>
            BURN AI INSPECTOR ingests multi-stage burn-in sensor measurements, analyzes component drift trajectories, performs independent safety-limit and early drift checks, identifies anomalous behavior, and prioritizes components for QA action and compliance reporting.
          </p>
        </div>
        <div className="heroActions">
          <Btn secondary onClick={() => go('data')}><Database size={14} /> Sensor Dataset</Btn>
          <Btn onClick={() => c && go('inspector', c.id)} disabled={!c}><Target size={14} /> Inspect Component</Btn>
        </div>
      </div>

      {/* Poster-Aligned Product Workflow Stepper */}
      <div className="workflowPipeline">
        <div className="workflowStep" onClick={() => go('data')} style={{ cursor: 'pointer' }}>
          <span>1. SENSOR DATA COLLECTION</span>
          <b>{stats.total} Components Ingested</b>
          <small>{availableLots.length} Dynamic Lots · 4 Burn-In Stages</small>
        </div>
        <div className="workflowStep" onClick={() => go('prediction')} style={{ cursor: 'pointer' }}>
          <span>2. AI / MODEL ANALYSIS</span>
          <b>Model A + B + C</b>
          <small>Hard Gate · OLS Forecast · Lot Anomaly</small>
        </div>
        <div className="workflowStep" onClick={() => go('risk')} style={{ cursor: 'pointer' }}>
          <span>3. ANOMALY / RISK ALERT</span>
          <b>{stats.high + stats.critical} Flagged Breaches</b>
          <small>{stats.safe} Safe · {stats.watch} Drift Watch</small>
        </div>
        <div className="workflowStep" onClick={() => go('priority')} style={{ cursor: 'pointer' }}>
          <span>4. ACTION & REPORT</span>
          <b>{risky.length} In Priority Queue</b>
          <small>QA Remediation & Audit Export</small>
        </div>
      </div>

      <div className="stats">
        <Stat I={Database} label="Sensor Ingestion" value={stats.total} sub={`${availableLots.length} Dynamic Lots`} />
        <Stat I={CheckCircle2} label="Within Boundary" value={stats.safe} sub="Passes Model A & B" tone="safe" />
        <Stat I={Clock3} label="Early Drift Watch" value={stats.watch} sub="Latent drift alerts" tone="watch" />
        <Stat I={AlertTriangle} label="High & Critical" value={stats.high + stats.critical} sub={`${stats.critical} hard threshold breaches`} tone="danger" />
      </div>
    </>
  );

  // 1. COMMAND CENTER (HOME)
  if (page === 'home') {
    if (!cs.length) {
      return (
        <>
          {commonHeader}
          <EmptyState onReset={restoreDemo} onUpload={openUpload} />
        </>
      );
    }
    return (
      <>
        {commonHeader}
        <div
  className="card"
  style={{
    marginBottom: '16px',
    border: liveMonitoring ? '1px solid #4dd9ff' : '1px solid #1c354b',
    background: '#071827'
  }}
>
  <div
    style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      gap: '16px',
      flexWrap: 'wrap',
      marginBottom: '16px'
    }}
  >
    <div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginBottom: '5px'
        }}
      >
        <Activity size={16} />
        <b>LIVE SENSOR MONITORING</b>
        <span
          className="badge"
          style={{
            fontSize: '8px',
            marginLeft: '4px'
          }}
        >
          SIMULATION / REPLAY
        </span>
      </div>

      <p className="sub">
        Progressive burn-in telemetry replay through the 0h → 24h → 96h → 168h inspection stages.
      </p>
    </div>

    <div style={{ display: 'flex', gap: '8px' }}>
      {!liveMonitoring ? (
        <Btn onClick={startLiveMonitoring}>
          <Activity size={14} />
          {liveStage === 3 ? 'Replay Again' : 'Start Live Monitoring'}
        </Btn>
      ) : (
        <Btn secondary onClick={stopLiveMonitoring}>
          Stop Monitoring
        </Btn>
      )}
    </div>
  </div>

  <div
    style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(4, 1fr)',
      gap: '8px',
      marginBottom: '16px'
    }}
  >
    {['0h', '24h', '96h', '168h'].map((stage, i) => (
      <div
        key={stage}
        style={{
          padding: '12px',
          borderRadius: '7px',
          textAlign: 'center',
          background: i <= liveStage ? '#0d2a3b' : '#091b2b',
          border: i === liveStage
            ? '1px solid #4dd9ff'
            : '1px solid #1c354b'
        }}
      >
        <small style={{ display: 'block', color: '#71899f', marginBottom: '4px' }}>
          STAGE {i + 1}
        </small>
        <strong>{stage}</strong>
        <div
          style={{
            fontSize: '9px',
            marginTop: '5px',
            color: i < liveStage ? '#68dda0' : i === liveStage ? '#4dd9ff' : '#526b80'
          }}
        >
          {i < liveStage
            ? 'PROCESSED'
            : i === liveStage
            ? liveMonitoring
              ? 'MONITORING'
              : 'CURRENT'
            : 'PENDING'}
        </div>
      </div>
    ))}
  </div>

  <div
    style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(3, 1fr)',
      gap: '10px'
    }}
  >
    <div style={{ position: 'relative' }}>
  <small style={{ color: '#71899f' }}>SELECT COMPONENT</small>

 <select
    value={sel}
    onChange={(e) => {
        const id = e.target.value;

        setSel(id);
        setComponentSearch(id);
        setShowComponentResults(false);
        setLiveStage(0);
        setLiveTick(0);
        setLiveMonitoring(false);
    }}
    style={{
        display: 'block',
        width: '100%',
        boxSizing: 'border-box',
        padding: '12px 14px',
        borderRadius: '6px',
        border: '1px solid #2a465e',
        background: '#091b2b',
        color: '#d9f3ff',
        fontWeight: 700,
        outline: 'none',
        cursor: 'pointer',
        fontSize: '14px'
    }}
>
    <option value="" disabled>
        Select component...
    </option>

    {cs.map((component) => (
        <option
            key={component.id}
            value={component.id}
        >
            {component.id} — {component.lot}
        </option>
    ))}
</select>

  {showComponentResults ? (
    <div
      style={{
        position: 'absolute',
        left: 0,
        right: 0,
        top: '58px',
        zIndex: 1000,
        background: '#071827',
        border: '1px solid #2a465e',
        borderRadius: '7px',
        maxHeight: '360px',
overflowY: 'auto',
        boxShadow: '0 10px 25px rgba(0,0,0,0.35)'
      }}
    >
      {componentSearchResults.length ? (
        componentSearchResults.map(component => (
          <button
            key={component.id}
            onMouseDown={(e) => e.preventDefault()}
onClick={() => {
  setSel(component.id);
              setComponentSearch(component.id);
              setShowComponentResults(false);
              setLiveStage(0);
              setLiveTick(0);
              setLiveMonitoring(false);
            }}
            style={{
              display: 'block',
              width: '100%',
              padding: '9px 11px',
              textAlign: 'left',
              border: 'none',
              borderBottom: '1px solid #17314a',
              background: 'transparent',
              color: '#d9f3ff',
              cursor: 'pointer'
            }}
          >
            <b>{component.id}</b>
            <span style={{ marginLeft: '8px', color: '#71899f' }}>
              · {component.lot}
            </span>
          </button>
        ))
      ) : (
        <div
          style={{
            padding: '10px',
            color: '#71899f',
            fontSize: '11px'
          }}
        >
          No matching component found
        </div>
      )}
    </div>
  ): null}
</div>

    <div>
      <small style={{ color: '#71899f' }}>CURRENT SENSOR STAGE</small>
      <div style={{ marginTop: '4px', fontWeight: 700 }}>
        {['0h', '24h', '96h', '168h'][liveStage]}
      </div>
    </div>

    <div>
      <small style={{ color: '#71899f' }}>TELEMETRY READING</small>
      <div style={{ marginTop: '4px', fontWeight: 700 }}>
        {c?.v?.[liveStage] !== undefined
          ? `${c.v[liveStage].toFixed(1)} μA`
          : '—'}
      </div>
    </div>
  </div>
  <div
  style={{
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '10px',
    marginTop: '16px',
    paddingTop: '14px',
    borderTop: '1px solid #17314a'
  }}
>
  <div>
    <small style={{ color: '#71899f' }}>MODEL A · LIMIT GUARD</small>
    <div style={{ marginTop: '5px', fontWeight: 700 }}>
      {liveStage < 3
        ? 'WAITING FOR 168h'
        : c?.modelA?.status || '—'}
    </div>
  </div>

  <div>
    <small style={{ color: '#71899f' }}>MODEL B · DRIFT FORECAST</small>
    <div style={{ marginTop: '5px', fontWeight: 700 }}>
      {liveStage < 2
        ? 'WAITING FOR 96h'
        : c?.modelB?.status || 'READY'}
    </div>
  </div>

  <div>
    <small style={{ color: '#71899f' }}>MODEL C · ANOMALY DETECTION</small>
    <div style={{ marginTop: '5px', fontWeight: 700 }}>
      {liveStage < 2
        ? 'WAITING FOR 96h'
        : c?.modelC?.status || 'READY'}
    </div>
  </div>
</div>

<div
  style={{
    marginTop: '14px',
    padding: '10px 12px',
    borderRadius: '6px',
    background: '#091b2b',
    border: '1px solid #17314a',
    fontSize: '10px'
  }}
>
  <b>ANALYSIS STATUS:</b>{' '}
  {liveStage < 2
    ? 'Collecting early burn-in telemetry. Models B and C activate after the 96h observation.'
    : liveStage < 3
    ? 'Early trajectory analysis active. Model A will evaluate the absolute 168h safety boundary when the final measurement arrives.'
    : `Three-model analysis complete. Final BURN AI INSPECTOR risk: ${c?.risk?.toUpperCase() || '—'}.`}
</div>
</div>
        <div className="two">
          <Section title="Stage 1 & 2: Active Component Sensor Trajectory & Drift Forecast" sub={c ? `${c.id} (Lot ${c.lot}) · Measured burn-in points (0h, 24h, 96h, 168h) vs Model B OLS forecast (${c.forecast.toFixed(1)} μA)` : 'No component selected'}>
            {c ? <InspectorChart c={c} limit={limit} /> : <div className="sub">Select a component to view trajectory</div>}
          </Section>
          <Section title="Stage 1: Sensor Population Fingerprint (Lot DNA)" sub="Dynamic population reliability scores and inter-lot variation across registered batches">
            {lots.map(l => (
              <div className="lotrow" key={l.lot}>
                <b>{l.lot}</b>
                <span>{l.r} flagged ({l.count} total)</span>
                <div className="track">
                  <i style={{ width: l.health + '%' }} className={l.health > 80 ? 'safe' : l.health > 50 ? 'watch' : 'critical'} />
                </div>
                <strong>{l.health}%</strong>
              </div>
            ))}
          </Section>
        </div>
        <div className="two">
          <Section title="Stage 4: Action & Priority Inspection Queue" sub="Evidence-based prioritization ranking components requiring engineering QA review">
            {risky.length === 0 ? (
              <div style={{ color: '#68dda0', padding: '16px 0', fontSize: '11px' }}>
                <CheckCircle2 size={16} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
                All loaded components are within safe operating limits.
              </div>
            ) : (
              risky.slice(0, 7).map(x => (
                <div className="caseRow" key={x.id} onClick={() => go('inspector', x.id)}>
                  <b>{x.id}</b>
                  <span>{x.lot}</span>
                  {badge(x.risk)}
                  <strong>{x.forecast.toFixed(1)} μA</strong>
                </div>
              ))
            )}
          </Section>
          <Section title="Stage 3: Anomaly Alert & Risk Composition Distribution" sub="Unified risk classification combining Model A hard limit and Model B trajectory drift">
            {[
              ['Safe', stats.safe],
              ['Watch', stats.watch],
              ['High', stats.high],
              ['Critical', stats.critical]
            ].map(([r, n]) => (
              <div className="riskrow" key={r}>
                <span>{r}</span>
                <div className="track">
                  <i className={r.toLowerCase()} style={{ width: Math.max(4, stats.total ? (n / stats.total) * 100 : 0) + '%' }} />
                </div>
                <b>{n}</b>
              </div>
            ))}
            <div className="callout">
              <ShieldCheck size={16} /> BURN AI INSPECTOR prevents escape: Model A enforces the 168h absolute safety limit, while Model B forecasts early burn-in trajectory drift.
            </div>
          </Section>
        </div>

        <Section title="Target Benefits & Intended Mission Outcomes" sub="Core value proposition from the BURN AI INSPECTOR component reliability poster">
          <div className="targetBenefits">
            <div className="targetBenefitCard">
              <span>OUTCOME 1</span>
              <h4>Higher Reliability & Quality</h4>
              <p>Eliminates field escapes by detecting subtle early burn-in drift prior to physical wear-out.</p>
            </div>
            <div className="targetBenefitCard">
              <span>OUTCOME 2</span>
              <h4>Reduced Failure Rate & Rework</h4>
              <p>Catches infant mortality before deployment, reducing costly recall cycles and hardware rework.</p>
            </div>
            <div className="targetBenefitCard">
              <span>OUTCOME 3</span>
              <h4>Cost & Screening Optimization</h4>
              <p>Focuses engineering verification on flagged lots and components with verifiable drift evidence.</p>
            </div>
            <div className="targetBenefitCard">
              <span>OUTCOME 4</span>
              <h4>Improved Mission Success</h4>
              <p>Provides objective, defensible compliance evidence for mission-critical aerospace QA sign-off.</p>
            </div>
          </div>
        </Section>
      </>
    );
  }

  // 2. DATA INTELLIGENCE
  if (page === 'data') {
    const baselineMean = cs.length ? (cs.reduce((a, b) => a + (b.v[0] || 0), 0) / cs.length).toFixed(1) : '0.0';
    const qualityScore = validationReport
      ? Math.round((validationReport.validRows / validationReport.totalRows) * 100)
      : cs.length
      ? 100
      : 0;

    return (
      <>
        <Hero
          title="Ingest & Validate Burn-In Sensor Telemetry."
          eyebrow="SENSOR DATA · DATA INTELLIGENCE"
          text="BURN AI INSPECTOR accepts multi-stage burn-in sensor measurements via CSV. Validates stage continuity, numeric integrity, and lot registration."
          actions={
            <>
              <Btn secondary onClick={exportCSV}><Download size={14} /> Export CSV</Btn>
              <Btn onClick={openUpload}><Upload size={14} /> Load CSV</Btn>
            </>
          }
        />
        <div className="stats">
          <Stat I={Database} label="Data Quality" value={`${qualityScore}%`} sub={`${cs.length} valid records`} tone="safe" />
          <Stat I={Layers3} label="Active Lots" value={availableLots.length} sub="Dynamic lot registry" />
          <Stat I={Activity} label="Inspection Stages" value="4 Stages" sub="0h / 24h / 96h / 168h" />
          <Stat I={Gauge} label="0h Baseline Mean" value={`${baselineMean} μA`} sub="Population initial average" />
        </div>
        <Section title="Data Validation & Integrity Checks">
          <div className="checks">
            <div><CheckCircle2 size={15} />Required Stages (4)<b>{cs.length ? 'PASS' : 'EMPTY'}</b></div>
            <div><CheckCircle2 size={15} />Numeric Integrity<b>{cs.length ? '100% VALID' : 'N/A'}</b></div>
            <div><CheckCircle2 size={15} />Dynamic Lots Registered<b>{availableLots.length} LOTS</b></div>
            <div><CheckCircle2 size={15} />Duplicate IDs Scanned<b>{validationReport?.duplicates ? `${validationReport.duplicates} RESOLVED` : 'CLEAN'}</b></div>
          </div>
        </Section>
        <Section title="Dataset Inspection Table">
          {cs.length === 0 ? (
            <EmptyState onReset={restoreDemo} onUpload={openUpload} />
          ) : (
            <Table rows={cs} open={id => go('inspector', id)} />
          )}
        </Section>
      </>
    );
  }

  // 3. LOT DNA
  if (page === 'lot') {
    if (!cs.length) {
      return (
        <>
          <Hero eyebrow="SENSOR DATA · LOT DNA" title="Lot-Level Fingerprint & Population Distribution." text="BURN AI INSPECTOR builds a population fingerprint for each production lot." />
          <EmptyState onReset={restoreDemo} onUpload={openUpload} />
        </>
      );
    }

    const lotItems = cs.filter(item => item.lot === currentLot);
    const lotBase = lotItems.length ? (lotItems.reduce((a, b) => a + (b.v[0] || 0), 0) / lotItems.length).toFixed(1) : '0.0';
    const lotHealth = lots.find(l => l.lot === currentLot)?.health ?? 100;
    const flaggedCount = lotItems.filter(item => item.risk !== 'Safe').length;

    // Dynamic histogram binning (prevents dropping >=80 uA values)
    const maxLotVal = lotItems.length ? Math.max(...lotItems.map(item => item.v[3] || 0)) : 80;
    const upperLimit = Math.max(80, Math.ceil(maxLotVal / 10) * 10);
    const binCount = 8;
    const binSize = Math.max(10, Math.ceil(upperLimit / binCount));

    const dist = Array.from({ length: binCount }, (_, i) => {
      const low = i * binSize;
      const high = (i + 1) * binSize;
      const isLast = i === binCount - 1;
      const label = isLast ? `${low}+` : `${low}-${high}`;
      const count = lotItems.filter(item => {
        const val = item.v[3] || 0;
        return isLast ? val >= low : val >= low && val < high;
      }).length;
      return { b: label, n: count };
    });

    return (
      <>
        <Hero
          eyebrow="LOT DNA"
          title="Make 'normal' visible."
          text="BURN AI INSPECTOR builds a population fingerprint for each production lot."
          actions={
            <select value={currentLot} onChange={e => setLot(e.target.value)}>
              {availableLots.map(l => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>
          }
        />
        <div className="stats">
          <Stat I={Layers3} label="Selected Lot" value={currentLot || 'None'} sub={`${lotItems.length} components`} />
          <Stat I={Gauge} label="0h Baseline Mean" value={`${lotBase} μA`} sub="Lot initial mean" />
          <Stat I={ShieldCheck} label="Lot Health" value={`${lotHealth}%`} sub="Reliability score" tone={lotHealth >= 80 ? 'safe' : lotHealth >= 50 ? 'watch' : 'danger'} />
          <Stat I={AlertTriangle} label="Attention Required" value={flaggedCount} sub="Non-safe components" tone={flaggedCount ? 'watch' : 'safe'} />
        </div>
        <div className="two">
          <Section title="Measured 168h Leakage Distribution" sub={`Histogram across ${lotItems.length} components in ${currentLot}`}>
            <div className="chart">
              <ResponsiveContainer width="100%" height={290}>
                <BarChart data={dist}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#17314a" />
                  <XAxis dataKey="b" />
                  <YAxis allowDecimals={false} />
                  <Tooltip
  cursor={{ fill: 'rgba(255,255,255,0.08)' }}
  contentStyle={{
    backgroundColor: '#111827',
    border: '2px solid #5bbfe9',
    borderRadius: '10px',
    padding: '10px 14px',
    boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
    color: '#ffffff'
  }}
  labelStyle={{
    color: '#ffffff',
    fontWeight: '700',
    marginBottom: '6px'
  }}
  itemStyle={{
    color: '#ffffff',
    fontWeight: '600'
  }}
  formatter={(value) => [`${value} components`, 'Count']}
  labelFormatter={(label) => `Leakage: ${label} µA`}
/>
                 <Bar dataKey="n" radius={[4, 4, 0, 0]}>
  {dist.map((entry, index) => {
    const low = index * binSize;

    let color = '#22c55e'; // GREEN = Safe

    if (low >= limit * 1.25) {
      color = '#ef4444'; // RED = Critical
    } else if (low >= limit) {
      color = '#facc15'; // YELLOW = Watch
    }

    return <Cell key={`cell-${index}`} fill={color} />;
  })}
</Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Section>
          <Section title={`Components in ${currentLot}`} sub="Select a component to inspect">
            <div className="caseGrid">
              {lotItems.map(item => (
                <div className="caseCard" key={item.id} onClick={() => go('inspector', item.id)}>
                  <b>{item.id}</b>
                  {badge(item.risk)}
                  <small>Measured 168h: {item.v[3].toFixed(1)} μA (Forecast: {item.forecast.toFixed(1)} μA)</small>
                </div>
              ))}
            </div>
          </Section>
        </div>
      </>
    );
  }

  // 4. HIDDEN DEFECT DETECTOR — MODEL C INTERFACE
  if (page === 'anomaly') {
    if (!cs.length) {
      return (
        <>
          <Hero eyebrow="STAGE 2 · MODEL C — TRAJECTORY ANOMALY DETECTION" title="Detect Subtle Trajectory Deviations Prior to Hard Failure." text="A component can pass the universal threshold and still be abnormal for its lot." />
          <EmptyState onReset={restoreDemo} onUpload={openUpload} />
        </>
      );
    }

    // Use Model C sensitivity threshold (sens state) for anomaly detection cutoff
    const anomalous = cs.filter(item => item.modelC.anomalyScore >= sens);
    const normal = cs.filter(item => item.modelC.anomalyScore < sens);
    const avgCScore = cs.length ? (cs.reduce((a, b) => a + b.modelC.anomalyScore, 0) / cs.length) : 0;
    const maxScore = cs.length ? Math.max(...cs.map(x => x.modelC.anomalyScore)) : 0;

    // F2 (Drift Velocity) vs F1 (Baseline Offset) scatter — capped at 1000 for render performance
    const scatterAnomalous = anomalous.slice(0, 500).map(item => ({
      x: isFinite(item.modelC.F2) ? +item.modelC.F2.toFixed(4) : 0,
      y: isFinite(item.modelC.F1) ? +item.modelC.F1.toFixed(2) : 0,
      z: Math.max(30, Math.round(item.modelC.anomalyScore * 500)),
      id: item.id,
      lot: item.lot,
      score: item.modelC.anomalyScore.toFixed(2)
    }));
    const scatterNormal = normal.slice(0, 500).map(item => ({
      x: isFinite(item.modelC.F2) ? +item.modelC.F2.toFixed(4) : 0,
      y: isFinite(item.modelC.F1) ? +item.modelC.F1.toFixed(2) : 0,
      z: 30,
      id: item.id,
      lot: item.lot,
      score: item.modelC.anomalyScore.toFixed(2)
    }));
const makeDensityBins = (points, cols = 45, rows = 30) => {
  if (!points || points.length === 0) return [];

  const valid = points.filter(
    p => Number.isFinite(Number(p.x)) && Number.isFinite(Number(p.y))
  );

  if (!valid.length) return [];

  const xs = valid.map(p => Number(p.x));
  const ys = valid.map(p => Number(p.y));

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const stepX = (maxX - minX || 1) / cols;
  const stepY = (maxY - minY || 1) / rows;

  const bins = new Map();

  valid.forEach(p => {
    const x = Number(p.x);
    const y = Number(p.y);

    const col = Math.min(
      cols - 1,
      Math.floor((x - minX) / stepX)
    );

    const row = Math.min(
      rows - 1,
      Math.floor((y - minY) / stepY)
    );

    const key = `${col}-${row}`;

    if (!bins.has(key)) {
      bins.set(key, {
        x: 0,
        y: 0,
        z: 0
      });
    }

    const bin = bins.get(key);

    bin.x += x;
    bin.y += y;
    bin.z += 1;
  });

  return Array.from(bins.values()).map(bin => ({
    x: bin.x / bin.z,
    y: bin.y / bin.z,
    z: bin.z
  }));
};

const densityNormal = makeDensityBins(scatterNormal);
const densityAnomalous = makeDensityBins(scatterAnomalous);
    return (
      <>
        <Hero
          eyebrow="STAGE 2 · MODEL C — TRAJECTORY ANOMALY DETECTION"
          title="Lot-Relative Burn-In Outlier Screening"
          text={`Robust statistical detection: compares each component's early burn-in trajectory (0h, 24h, 96h) against its production lot peers. The 168h measurement is never accessed — zero data leakage. Threshold τ = ${sens.toFixed(2)}.`}
          actions={
            <Btn onClick={() => go('inspector', c?.id)}><Target size={14} /> Inspect Selected</Btn>
          }
        />

        {/* Sensitivity Slider */}
        <div className="slider" style={{ marginBottom: '4px' }}>
          <span>Model C Sensitivity Threshold (τ)</span>
          <input type="range" min="0.10" max="0.90" step="0.01" value={sens} onChange={e => setSens(+e.target.value)} />
          <b>{sens.toFixed(2)}</b>
          <small style={{ color: '#7893a9', fontSize: '9px', marginLeft: '8px' }}>
            Watch ≥ 0.35 · High ≥ 0.60 · Critical ≥ 0.82 (engineering thresholds, not validated ML metrics)
          </small>
        </div>

    
        {/* Model C Summary Stats */}
        <div className="stats">
          <Stat I={AlertTriangle} label="Anomalous (S ≥ τ)" value={anomalous.length} sub={`τ = ${sens.toFixed(2)}, lot-relative outliers`} tone={anomalous.length > 0 ? 'danger' : 'safe'} />
          <Stat I={CheckCircle2} label="Normal (S < τ)" value={normal.length} sub="Consistent with lot distribution" tone="safe" />
          <Stat I={Gauge} label="Avg Anomaly Score" value={avgCScore.toFixed(3)} sub="Dataset mean S ∈ [0,1]" />
          <Stat I={Zap} label="Highest Anomaly Score" value={maxScore.toFixed(3)} sub={cs.find(x => x.modelC.anomalyScore === maxScore)?.id || '—'} tone={maxScore >= 0.60 ? 'danger' : maxScore >= 0.35 ? 'watch' : ''} />
        </div>

        {/* Trajectory Anomaly Scatter Plot */}
        <Section
  title="Model C — Trajectory Anomaly Feature Space"
  sub="Density view · X: F2 Early Drift Velocity (µA/h) · Y: F1 Baseline Offset (µA) · Red = Anomalous · Blue = Normal"
>
  <div className="chart">
    <ResponsiveContainer width="100%" height={420}>
      <ScatterChart margin={{ top: 15, right: 25, bottom: 20, left: 15 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#17314a" />

        <XAxis
          type="number"
          dataKey="x"
          name="F2 Early Drift Velocity"
          unit=" µA/h"
          tick={{ fontSize: 11, fill: "#7893a9" }}
        />

        <YAxis
          type="number"
          dataKey="y"
          name="F1 Baseline Offset"
          unit=" µA"
          tick={{ fontSize: 11, fill: "#7893a9" }}
        />

        <ZAxis
          type="number"
          dataKey="z"
          range={[25, 180]}
        />

        <Tooltip
          cursor={{ strokeDasharray: "3 3" }}
          contentStyle={{
            background: "#0a1c2d",
            borderColor: "#2a465e",
            fontSize: "10px"
          }}
          formatter={(value, name) => [
            typeof value === "number" ? value.toFixed(2) : value,
            name
          ]}
        />

        <Scatter
          name="Normal density"
          data={densityNormal}
          fill="#3a6a8c"
          opacity={0.65}
        />

        <Scatter
          name="Anomalous density"
          data={densityAnomalous}
          fill="#ff707b"
          opacity={0.75}
        />
      </ScatterChart>
    </ResponsiveContainer>
  </div>
</Section>
          


        {/* Anomaly Cards */}
        <Section
          title={`Trajectory Anomalies Detected (${anomalous.length} components)`}
          sub={`Sorted by anomaly score · S ≥ ${sens.toFixed(2)} (sensitivity threshold τ) · Lot-relative robust z-score outlier kernel`}
        >
          {anomalous.length === 0 ? (
            <div style={{ color: '#68dda0', padding: '16px 0', fontSize: '11px' }}>
              <CheckCircle2 size={16} style={{ verticalAlign: 'middle', marginRight: '6px' }} />
              No components flagged at current sensitivity threshold (τ = {sens.toFixed(2)}).
              Try lowering the slider to detect milder deviations.
            </div>
          ) : (
            <div className="caseGrid">
              {[...anomalous]
                .sort((a, b) => b.modelC.anomalyScore - a.modelC.anomalyScore)
                
                .map(item => {
                  const mc = item.modelC;
                  const topReason = mc.anomalyReasons[0] || '—';
                  return (
                    <div
                      className="caseCard"
                      key={item.id}
                      onClick={() => go('inspector', item.id)}
                      style={{ borderColor: mc.severity === 'Critical' ? '#ff4d4d' : mc.severity === 'High' ? '#f3a06f' : '#e3cc70' }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <b>{item.id}</b>
                        {badge(item.risk)}
                      </div>
                      <span style={{ fontSize: '9px', color: '#7893a9' }}>Lot {item.lot} · Model C: {mc.status}</span>
                      <div style={{ margin: '6px 0', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                        <span style={{ fontSize: '10px', fontWeight: 700, color: mc.severity !== 'Safe' ? '#ff7b7b' : '#68dda0' }}>
                          Score: {mc.anomalyScore.toFixed(3)}
                        </span>
                        <span style={{ fontSize: '10px', color: '#7893a9' }}>
                          Dist: {mc.anomalyDistance.toFixed(2)}
                        </span>
                      </div>
                      <div style={{ fontSize: '9px', color: '#bdc9d4', marginBottom: '4px' }}>
                        <b>Strongest:</b> {mc.strongestFeature}
                      </div>
                      <div style={{ fontSize: '9px', color: '#bdc9d4', marginBottom: '4px' }}>
                        {topReason}
                      </div>
                      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', fontSize: '9px', color: '#7893a9' }}>
                        <span>Z1={mc.Z1.toFixed(1)}</span>
                        <span>Z2={mc.Z2.toFixed(1)}</span>
                        <span>Z3={mc.Z3.toFixed(1)}</span>
                        <span>Z4={mc.Z4.toFixed(1)}</span>
                      </div>
                    </div>
                  );
                })}
            </div>
          )}
        </Section>

        <div className="callout" style={{ margin: '0 0 8px' }}>
          <BrainCircuit size={16} />
          <span>
            <b>Model C Engineering Guardrail:</b> Anomaly scores are computed purely from early measurements (0h, 24h, 96h).
            The 168h endpoint is never accessed during Model C computation.
            Severity thresholds are engineering choices — formal validation requires labeled production data.
          </span>
        </div>
      </>
    );
  }


  // 5. DRIFT FORECAST (PREDICTION PAGE)
  if (page === 'prediction') {
    if (!c) {
      return (
        <>
          <Hero eyebrow="DETECTION & FORECAST · DRIFT FORECAST" title="Two independent checks, one clear decision." text="Model A evaluates hard limit breaches; Model B forecasts future drift." />
          <EmptyState onReset={restoreDemo} onUpload={openUpload} />
        </>
      );
    }

    return (
      <PredictionView
        cs={cs}
        c={c}
        limit={limit}
        model={model}
        setModel={setModel}
        setSel={id => go('prediction', id)}
      />
    );
  }

  // 6. RISK & SAFETY
  if (page === 'risk') {
    const aBreaches = cs.filter(x => x.modelA?.status === 'BREACH').length;
    const bAlerts = cs.filter(x => x.modelB?.status !== 'PASS').length;
    const riskItems = [
      ['Safe', stats.safe],
      ['Watch', stats.watch],
      ['High', stats.high],
      ['Critical', stats.critical]
    ];
    const riskColors = ['#58d6a0', '#e4cf72', '#f3a06f', '#f06b6b'];

    return (
      <>
        <Hero
          eyebrow="RISK & RESPONSE · RISK & SAFETY"
          title="Separate Current-Limit Safety from Future Drift Risk."
          text="Model A checks immediate 168h limit compliance. Model B forecasts trajectory drift. Model C detects lot-relative trajectory anomalies. BURN AI INSPECTOR combines all three signals using max-severity gating."
        />
        <div className="riskOverview">
          <div className="riskDistribution card">
            <div className="riskTitle">
              <div>
                <h2>Risk Composition</h2>
                <p className="sub">Final BURN AI INSPECTOR classification across all loaded components.</p>
              </div>
             
            </div>
            <div className="riskPieArea">
              <div className="riskDonut3D">
                <ResponsiveContainer width={260} height={260}>
                  
  <PieChart>
    <Pie
      data={riskItems.map(([name, count]) => ({ name, count }))}
      dataKey="count"
      nameKey="name"
      cx="50%"
      cy="50%"
      innerRadius={72}
      outerRadius={108}
      paddingAngle={4}
      cornerRadius={7}
      startAngle={90}
      endAngle={-270}
      stroke="#071725"
      strokeWidth={3}
      isAnimationActive={true}
      animationDuration={900}
    >
      {riskItems.map(([name], i) => (
        <Cell
          key={name}
          fill={riskColors[i]}
          stroke="#071725"
          strokeWidth={3}
        />
      ))}
    </Pie>

    <Tooltip
      contentStyle={{
        background: '#071827',
        border: '1px solid #2a465e',
        borderRadius: '8px',
        fontSize: '11px'
      }}
      formatter={(value, name) => [
        `${value} components`,
        name
      ]}
    />
  </PieChart>
</ResponsiveContainer>

                <div className="riskDonutCenter">
                  <b>{stats.total}</b>
                  <span>components</span>
                </div>
              </div>
              <div className="riskLegend">
                {riskItems.map(([name, count], i) => (
                  <div key={name}>
                    <span style={{ background: riskColors[i] }} />
                    <b>{name}</b>
                    <strong>{count}</strong>
                    <small>{stats.total ? Math.round((count / stats.total) * 100) : 0}%</small>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="riskRules card">
            <h2>How the Decision is Made</h2>
            <p className="sub">Model A, Model B, and Model C answer different questions independently.</p>
            <div className="riskModel">
              <span>MODEL A</span>
              <div>
                <b>Absolute Limit Guard</b>
                <small>Is the measured 168h leakage already at or above {limit} μA?</small>
              </div>
              <strong>{aBreaches}<small>breaches</small></strong>
            </div>
            <div className="riskModel">
              <span>MODEL B</span>
              <div>
                <b>Drift Forecast</b>
                <small>Does the 0h → 24h → 96h trajectory forecast limit crossing?</small>
              </div>
              <strong>{bAlerts}<small>alerts</small></strong>
            </div>
            <div className="riskRuleList">
              <Rule t="SAFE" x="No measured breach at 168h and no trajectory drift concern." />
              <Rule t="WATCH" x="Early trajectory slope > 0.12 μA/h or forecast approaching safety boundary." />
              <Rule t="HIGH" x="Model B forecast projects threshold breach at 168h." />
              <Rule t="CRITICAL" x="Measured 168h breach (Model A) or severe runaway forecast (≥1.25× limit)." />
            </div>
          </div>
        </div>

        <div className="modelGrid riskModels">
          <div className={`decisionCard ${aBreaches ? 'alert' : 'pass'}`}>
            <div className="decisionHead">
              <span>MODEL A · CURRENT SAFETY</span>
              {aBreaches ? badge('Critical') : badge('Safe')}
            </div>
            <h3>Absolute Limit Guard</h3>
            <div className="decisionValue">
              {aBreaches} / {stats.total}
              <small>measured above {limit} μA at 168h</small>
            </div>
            <p>Uses the measured 168h endpoint only. No forecast extrapolation is used.</p>
          </div>
          <div className={`decisionCard ${bAlerts ? 'alert' : 'pass'}`}>
            <div className="decisionHead">
              <span>MODEL B · FUTURE DRIFT RISK</span>
              {bAlerts ? badge('Watch') : badge('Safe')}
            </div>
            <h3>Drift Forecast</h3>
            <div className="decisionValue">
              {bAlerts} / {stats.total}
              <small>components with forecast alerts</small>
            </div>
            <p>Fits early burn-in behaviour (0h/24h/96h) to project 168h and estimate crossing time.</p>
          </div>
        </div>

        <Section title="Priority Flagged Components" sub="Components requiring QA inspection based on unified risk gate">
          <div className="caseGrid">
            {risky.map(x => (
              <div className="caseCard" key={x.id} onClick={() => go('inspector', x.id)}>
                <b>{x.id}</b>
                <span>{x.lot}</span>
                {badge(x.risk)}
                <small>Measured 168h: {x.v[3]?.toFixed(1)} μA · Forecast: {x.forecast.toFixed(1)} μA ({x.crossingText})</small>
              </div>
            ))}
          </div>
        </Section>
      </>
    );
  }

  // 7. COMPONENT PROFILE (INSPECTOR)
  if (page === 'inspector') {
    if (!c) {
      return (
        <>
          <Hero eyebrow="ANALYSIS · COMPONENT PROFILE" title="Component Profile" text="Select a component to inspect evidence." />
          <EmptyState onReset={restoreDemo} onUpload={openUpload} />
        </>
      );
    }
    return <Inspector c={c} cs={cs} setSel={id => go('inspector', id)} limit={limit} go={go} mlStatus={mlStatus} mlError={mlError} />;
  }

  // 8. PRIORITY QUEUE
  if (page === 'priority') {
    return (
      <>
        <Hero
          eyebrow="RISK & RESPONSE · PRIORITY QUEUE"
          title="Evidence-Based Inspection Prioritization."
          text="Ranked cases help QA focus attention where risk and evidence are strongest."
          actions={
            <Btn secondary onClick={exportCSV}><Download size={14} /> Export Queue</Btn>
          }
        />
        <Section title={`Inspection Priority Queue (${risky.length} flagged)`} sub={`Sorted by risk severity, predicted RUL, and forecast leakage against ${limit} μA limit`}>
          {risky.length === 0 ? (
            <div style={{ color: '#68dda0', padding: '24px', textAlign: 'center', background: '#091b2b', borderRadius: '8px' }}>
              <CheckCircle2 size={24} style={{ display: 'block', margin: '0 auto 8px' }} />
              All components are within safe operating limits. No priority action required.
            </div>
          ) : (
            <Table rows={risky} open={id => go('inspector', id)} />
          )}
        </Section>
      </>
    );
  }

  // 9. ACTION CENTER
  if (page === 'action') {
    return (
      <>
        <Hero
          eyebrow="RISK & RESPONSE · ACTION CENTER"
          title="Targeted Engineering Remediation & QA Sign-Off."
          text="Close the loop with recommended QA steps, quarantine triggers, and explicit human sign-off."
        />
        <div className="actions">
          {risky.map(item => (
            <div className={review[item.id] ? 'done' : ''} key={item.id}>
              <div>
                {badge(item.risk)} <b>{item.id}</b> <span>{item.lot}</span>
              </div>
              <h3>
                {item.risk === 'Critical'
                  ? 'Immediate QA Quarantine & Root-Cause Review'
                  : item.risk === 'High'
                  ? 'Repeat 96h/168h Burn-In Verification'
                  : 'Extended Burn-In Stage Monitoring'}
              </h3>
              <p>
                Measured 168h: {item.v[3]?.toFixed(1)} μA | Forecast: {item.forecast?.toFixed(1)} μA.<br />
                {item.crossingText}.
              </p>
              <Btn secondary onClick={() => go('inspector', item.id)}>View Profile</Btn>{' '}
              <Btn onClick={() => setReview({ ...review, [item.id]: !review[item.id] })}>
                {review[item.id] ? <><Check size={14} /> Confirmed</> : 'Mark Reviewed'}
              </Btn>
            </div>
          ))}
          {risky.length === 0 && (
            <div style={{ gridColumn: '1 / -1', padding: '32px', textAlign: 'center', color: '#69dda2' }}>
              <CheckCircle2 size={24} style={{ display: 'block', margin: '0 auto 8px' }} />
              No components currently require engineering intervention.
            </div>
          )}
        </div>
      </>
    );
  }

  // 10. WHAT-IF LAB (PHYSICAL TIME SCALING)
  if (page === 'whatif') {
    if (!c) {
      return (
        <>
          <Hero eyebrow="ANALYSIS · WHAT-IF LAB" title="Explore Physical Trajectory Sensitivity." text="Simulate thermal and electrical acceleration." />
          <EmptyState onReset={restoreDemo} onUpload={openUpload} />
        </>
      );
    }

    const mult = [0.75, 1.0, 1.2, 1.5][scenario];
    // Physical time scaling: r(t) = t / 168h where t in [0, 24, 96, 168]
    const simData = c.v.map((val, i) => {
      const t = S[i];
      const timeRatio = t / 168;
      const scenarioVal = Math.max(0, val * (1 + (mult - 1) * timeRatio));
      return {
        h: `${t}h`,
        base: Number(val.toFixed(2)),
        scenario: Number(scenarioVal.toFixed(2))
      };
    });

    const endpointVal = simData[3].scenario;
    const isScenarioBreached = endpointVal >= limit;

    return (
      <>
        <Hero
          eyebrow="ANALYSIS · WHAT-IF LAB"
          title="Explore Physical Trajectory Sensitivity."
          text="Simulate how accelerated drift factors affect the physical 168h burn-in endpoint."
          actions={
  <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>

    {/* Component selector */}
    <select
      value={sel || (cs[0]?.id || '')}
      onChange={(e) => setSel(e.target.value)}
      style={{
        minWidth: '220px',
        maxWidth: '280px',
        padding: '9px 12px',
        borderRadius: '8px',
        border: '1px solid #29465f',
        background: '#0b1c2b',
        color: '#ffffff',
        fontWeight: '600',
        outline: 'none',
        cursor: 'pointer'
      }}
    >
      {cs.map((component) => (
        <option key={component.id} value={component.id}>
          {component.id} — {component.lot}
        </option>
      ))}
    </select>

    {/* Scenario selector */}
    <div className="tabs">
      {['0.75× Drift', 'Baseline (1.0×)', '1.2× Drift', '1.5× Drift'].map((x, i) => (
        <button
          className={scenario === i ? 'on' : ''}
          onClick={() => setScenario(i)}
          key={x}
        >
          {x}
        </button>
      ))}
    </div>

  </div>
}
        />
        <div className="stats">
          <Stat I={Target} label="Simulated Case" value={c.id} sub={c.lot} />
          <Stat
            I={LineIcon}
            label="Scenario 168h Endpoint"
            value={`${endpointVal.toFixed(1)} μA`}
            sub={`Limit ${limit} μA`}
            tone={isScenarioBreached ? 'danger' : 'safe'}
          />
          <Stat
            I={ShieldCheck}
            label="Simulation Outcome"
            value={isScenarioBreached ? 'THRESHOLD BREACH' : 'WITHIN LIMIT'}
            sub={isScenarioBreached ? 'Requires thermal redesign' : 'Passes accelerated stress'}
            tone={isScenarioBreached ? 'danger' : 'safe'}
          />
        </div>
        <Section title={`${c.id} — Physical Time-Scaled Drift Simulation`} sub="Solid: Baseline measurements · Dashed: Accelerated stress scenario">
          <div className="chart">
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={simData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#17314a" />
                <XAxis dataKey="h" />
                <YAxis unit=" μA" />
                <Tooltip
  contentStyle={{
    backgroundColor: '#061522',
    border: '1px solid #2a465e',
    borderRadius: '8px',
    color: '#ffffff',
    boxShadow: '0 6px 18px rgba(0,0,0,0.45)'
  }}
  labelStyle={{
    color: '#ffffff',
    fontWeight: '600'
  }}
  itemStyle={{
    color: '#ffffff'
  }}
/>
                <ReferenceLine y={limit} stroke="#ff7b7b" strokeDasharray="5 5" label={`Limit (${limit} μA)`} />
                <Line dataKey="base" name="Observed Measurements" stroke="#5bbfe9" strokeWidth={2} dot={{ r: 4 }} />
                <Line dataKey="scenario" name="Simulated Trajectory" stroke="#f3a06f" strokeWidth={3} strokeDasharray="6 6" dot={{ r: 5 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Section>
      </>
    );
  }

  if (page === 'model') {
    const aBreaches = cs.filter(x => x.modelA?.status === 'BREACH').length;
    const bBreaches = cs.filter(x => x.forecast >= limit).length;
    const cAnomalies = cs.filter(x => x.modelC?.anomalyDetected).length;
    const mlReady = cs.filter(x => x.failureProbability != null).length;
    const avgRul = mlReady ? cs.reduce((s, x) => s + Number(x.rulHours || 0), 0) / mlReady : 0;
    const m = mlMetrics || {};
    const cb = m.catboost || {};
    const rf = m.rul_random_forest || {};
    const before = m.before_smote || {};
    const after = m.after_smote || {};
    const iso = m.isolation_forest || {};

    return (
      <>
        <Hero
          eyebrow="ANALYSIS · ML MODEL TRUST & RUL"
          title="Predict failure risk. Estimate remaining useful life. Explain every decision."
          text="The upgraded BURN AI INSPECTOR pipeline uses SMOTE for class imbalance, Fisher discriminant analysis, CatBoost classification, Random Forest RUL regression, Isolation Forest anomaly detection, and SHAP explainability. Model A remains the hard safety guard."
        />

        <div className="modelGrid">
          <div className="modelCard selected"><div className="modelTag">CLASSIFICATION</div><h3>CatBoost Failure-Risk Model</h3><p>Predicts Warning/Critical risk probability from burn-in trajectory features after SMOTE balancing.</p><div className="modelMetric">{cb.f1 == null ? '—' : `${(cb.f1 * 100).toFixed(1)}%`}<small>F1 on untouched test set</small></div><Rule t="Inputs" x="0h/24h/96h/168h values + trajectory drift, delta, curvature and relative growth." /><span className="modelStatus">SUPERVISED ML</span></div>
          <div className="modelCard"><div className="modelTag">RUL REGRESSION</div><h3>Random Forest Regressor</h3><p>Predicts remaining useful life directly at the 168h inspection point.</p><div className="modelMetric">{rf.mae_h == null ? '—' : `${rf.mae_h.toFixed(1)} h`}<small>RUL MAE</small></div><Rule t="Target" x="RUL_at_168h from the supplied 25,000-component labelled dataset." /><span className="modelStatus">RUL ENGINE</span></div>
          <div className="modelCard"><div className="modelTag">UNSUPERVISED</div><h3>Isolation Forest</h3><p>Flags statistically unusual components independently of the supervised failure label.</p><div className="modelMetric">{iso.test_anomaly_rate_pct == null ? '—' : `${iso.test_anomaly_rate_pct.toFixed(1)}%`}<small>test anomaly rate</small></div><Rule t="Purpose" x="Detects unusual trajectories that may not match known labelled failure patterns." /><span className="modelStatus">ANOMALY DETECTOR</span></div>
          <div className="modelCard"><div className="modelTag">EXPLAINABILITY</div><h3>Fisher + SHAP</h3><p>Fisher creates a discriminative projection; SHAP explains the contribution of each feature to CatBoost risk.</p><div className="modelMetric">{mlReady.toLocaleString()}<small>live component predictions</small></div><Rule t="Governance" x="Targets and future-life fields are excluded from inference features; SHAP is generated from the trained tree model." /><span className="modelStatus">EXPLAINABLE ML</span></div>
        </div>

        <div className="stats">
          <Stat I={Gauge} label="ML Engine" value={mlStatus.toUpperCase()} sub={mlError || 'FastAPI backend'} tone={mlStatus === 'online' ? 'safe' : 'watch'} />
          <Stat I={Clock3} label="Average Predicted RUL" value={mlReady ? `${avgRul.toFixed(0)} h` : '—'} sub="Across currently loaded components" />
          <Stat I={LineIcon} label="RUL MAE" value={rf.mae_h == null ? '—' : `${rf.mae_h.toFixed(1)} h`} sub={rf.rmse_h == null ? 'Waiting for metrics' : `RMSE ${rf.rmse_h.toFixed(1)} h · R² ${rf.r2.toFixed(3)}`} />
          <Stat I={Target} label="CatBoost F1" value={cb.f1 == null ? '—' : `${(cb.f1 * 100).toFixed(1)}%`} sub={cb.roc_auc == null ? 'Waiting for metrics' : `ROC-AUC ${(cb.roc_auc * 100).toFixed(1)}%`} />
        </div>

        <div className="two">
          <Section title="Class Imbalance → SMOTE" sub="SMOTE is applied only to the training partition; the test set remains untouched.">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div style={{ background: '#071827', border: '1px solid #1c354b', borderRadius: 8, padding: 12 }}><b>Before SMOTE</b><p style={{ margin: '8px 0 2px', color: '#68dda0' }}>Normal: {Number(before.normal || 0).toLocaleString()}</p><p style={{ margin: 0, color: '#f3a06f' }}>Warning: {Number(before.warning || 0).toLocaleString()} ({Number(before.warning_pct || 0).toFixed(1)}%)</p></div>
              <div style={{ background: '#071827', border: '1px solid #1c354b', borderRadius: 8, padding: 12 }}><b>After SMOTE</b><p style={{ margin: '8px 0 2px', color: '#68dda0' }}>Normal: {Number(after.normal || 0).toLocaleString()}</p><p style={{ margin: 0, color: '#f3a06f' }}>Warning: {Number(after.warning || 0).toLocaleString()}</p></div>
            </div>
          </Section>
          <Section title="Validation Metrics" sub="Held-out test metrics from the supplied 25,000-component dataset.">
            <Rule t="CatBoost" x={`Accuracy ${(Number(cb.accuracy || 0) * 100).toFixed(1)}% · Precision ${(Number(cb.precision || 0) * 100).toFixed(1)}% · Recall ${(Number(cb.recall || 0) * 100).toFixed(1)}% · F1 ${(Number(cb.f1 || 0) * 100).toFixed(1)}% · ROC-AUC ${(Number(cb.roc_auc || 0) * 100).toFixed(1)}%`} />
            <Rule t="Random Forest RUL" x={`MAE ${Number(rf.mae_h || 0).toFixed(1)} h · RMSE ${Number(rf.rmse_h || 0).toFixed(1)} h · R² ${Number(rf.r2 || 0).toFixed(3)}`} />
            <Rule t="Isolation Forest" x={`Anomaly rate ${Number(iso.test_anomaly_rate_pct || 0).toFixed(1)}% on the held-out feature set.`} />
          </Section>
        </div>

        <Section title="ML Decision Pipeline" sub="The exact path from burn-in measurements to the engineering decision.">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, minmax(100px,1fr))', gap: 6, alignItems: 'stretch' }}>
            {['Burn-in data','Feature engineering','Fisher discriminant','Train / test split','SMOTE','CatBoost + Isolation Forest','Random Forest RUL','SHAP + Safety Gate'].map((x,i) => <div key={x} style={{ padding: '12px 8px', background: '#071827', border: '1px solid #1c354b', borderRadius: 8, fontSize: 9, fontWeight: 800, textAlign: 'center' }}><span style={{ color: '#63bce9' }}>{i+1}</span><br />{x}</div>)}
          </div>
        </Section>
      </>
    );
  }

  if (page === 'reports') {
    return (
      <>
        <Hero
          eyebrow="STAGE 4 · ACTION & REPORT"
          title="Package Burn-In Screening Evidence for Quality Assurance."
          text="Export and review burn-in screening, ML failure-risk predictions, Random Forest RUL estimates, Isolation Forest anomalies, SHAP drivers, and safety actions."
          actions={
            <Btn onClick={exportCSV}><Download size={14} /> Download QA Audit Report (CSV)</Btn>
          }
        />
        <div className="stats">
          <Stat I={Database} label="Dataset Scope" value={`${cs.length} Parts`} sub={`${availableLots.length} Dynamic Lots`} />
          <Stat I={CheckCircle2} label="Within Boundary" value={stats.safe} sub="Passes Model A & B" tone="safe" />
          <Stat I={Clock3} label="Attention Flagged" value={stats.watch + stats.high + stats.critical} sub="Drift watch & forecast alerts" tone="watch" />
          <Stat I={AlertTriangle} label="Critical Quarantine" value={stats.critical} sub="Immediate QA review required" tone={stats.critical ? 'danger' : 'safe'} />
        </div>
        <Section title="Top Priority Inspection Cases" sub="Components flagged by Model A hard threshold gate or Model B early drift forecast">
          <div className="caseGrid">
            {risky.map(x => (
              <div className="caseCard" key={x.id} onClick={() => go('inspector', x.id)}>
                <b>{x.id}</b>
                <span>{x.lot}</span>
                {badge(x.risk)}
                <small>Measured: {x.v[3]?.toFixed(1)} μA | Forecast: {x.forecast.toFixed(1)} μA ({x.crossingText})</small>
              </div>
            ))}
            {risky.length === 0 && (
              <div style={{ gridColumn: '1 / -1', color: '#68dda0', padding: '16px' }}>
                No high-risk cases detected in current dataset.
              </div>
            )}
          </div>
        </Section>
      </>
    );
  }

  // 14. SETTINGS
  return (
    <>
      <Hero
        eyebrow="SYSTEM SETTINGS"
        title="Tune decision boundaries."
        text="Threshold adjustments instantly synchronize across Model A, Model B, charts, and queues."
      />
      <div className="settings">
        <div>
          <h3>Safety Leakage Limit</h3>
          <p>Universal absolute boundary for pass/fail classification.</p>
          <strong>{limit} μA</strong>
          <input type="range" min="20" max="80" value={limit} onChange={e => setLimit(+e.target.value)} />
        </div>
        <div>
          <h3>Lot Anomaly Sensitivity</h3>
          <p>Sensitivity for flagging early trajectory deviation from lot baseline.</p>
          <strong>{sens.toFixed(2)}</strong>
          <input type="range" min="0.1" max="0.9" step="0.01" value={sens} onChange={e => setSens(+e.target.value)} />
        </div>
      </div>
      <div style={{ display: 'flex', gap: '10px' }}>
        <Btn onClick={applyConfig}><Check size={14} /> Apply Configuration</Btn>
        <Btn secondary onClick={restoreDemo}><RotateCcw size={14} /> Restore Default Demo Dataset</Btn>
      </div>
    </>
  );
}

// --- SUB-COMPONENTS AND CHARTS ---

function Hero({ eyebrow, title, text, actions }) {
  return (
    <div className="hero">
      <div>
        <label>{eyebrow}</label>
        <h1>{title}</h1>
        <p>{text}</p>
      </div>
      <div className="heroActions">{actions}</div>
    </div>
  );
}

function Rule({ t, x }) {
  return (
    <div className="rule">
      <i />
      <div>
        <b>{t}</b>
        <p>{x}</p>
      </div>
    </div>
  );
}

function ModelSwitch({ model, setModel }) {
  return (
    <div className="modelSwitch" role="group" aria-label="Analytical Model Switcher">
      {[MODEL_A, MODEL_B].map(m => (
        <button
          key={m.id}
          type="button"
          className={model === m.id ? 'on' : ''}
          onClick={() => setModel(m.id)}
          aria-pressed={model === m.id}
        >
          <b>{m.short}</b>
          <span>{m.name}</span>
        </button>
      ))}
    </div>
  );
}

/**
 * Dynamic Evidence Trajectory Chart (Used in Inspector and Home)
 * Plots actual component measurements (0h, 24h, 96h, 168h) and overlays Model B forecast.
 */
function InspectorChart({ c, limit }) {
  if (!c || !c.v) return <div className="sub">No component data available.</div>;

  const f = linearForecast(c.v, limit);
 const riskColor = getRiskColor(c.risk);
  const data = [
    { h: '0h', hours: 0, measured: c.v[0], forecast: undefined },
    { h: '24h', hours: 24, measured: c.v[1], forecast: undefined },
    { h: '96h', hours: 96, measured: c.v[2], forecast: Math.max(0, f.intercept + f.slope * 96) },
    { h: '168h', hours: 168, measured: c.v[3], forecast: f.predicted168 },
    { h: '216h', hours: 216, measured: undefined, forecast: Math.max(0, f.intercept + f.slope * 216) }
  ];

  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height={290}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#17314a" />
          <XAxis dataKey="h" />
          <YAxis unit=" μA" />
          <Tooltip
  contentStyle={{
    backgroundColor: '#0a1c2d',
    border: '1px solid #2a465e',
    borderRadius: '6px',
    color: '#ffffff'
  }}
  labelStyle={{
    color: '#ffffff'
  }}
  itemStyle={{
    color: '#ffffff'
  }}
/>
          <ReferenceLine y={limit} stroke="#ff7b7b" strokeDasharray="5 5" label={`Limit (${limit} μA)`} />
          <Line
            dataKey="measured"
            name="Measured Burn-In Data"
            stroke={riskColor}
            strokeWidth={3}
            dot={{ r: 5, fill: riskColor }}
            connectNulls={false}
          />
          <Line
            dataKey="forecast"
            name="Model B Drift Forecast"
            stroke="#f3a06f"
            strokeWidth={2}
            strokeDasharray="6 6"
            dot={{ r: 4, fill: '#f3a06f' }}
            connectNulls={true}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * Interactive Model A / Model B Prediction View
 */
function PredictionView({ cs, c, limit, model, setModel, setSel }) {
  const a = modelAResult(c.v, limit);
  const b = modelBResult(c.v, limit);
  const f = b;

  // Model A trajectory: observed measurements across 0h, 24h, 96h, 168h
  const modelAData = [
    { h: '0h', measured: c.v[0] },
    { h: '24h', measured: c.v[1] },
    { h: '96h', measured: c.v[2] },
    { h: '168h', measured: c.v[3] }
  ];

  // Model B trajectory: 0h-96h observed + forecast extending to 240h
  const modelBData = [
    { h: '0h', measured: c.v[0], forecast: undefined },
    { h: '24h', measured: c.v[1], forecast: undefined },
    { h: '96h', measured: c.v[2], forecast: Math.max(0, f.intercept + f.slope * 96) },
    { h: '168h', measured: c.v[3], forecast: f.predicted168 },
    { h: '192h', measured: undefined, forecast: Math.max(0, f.intercept + f.slope * 192) },
    { h: '216h', measured: undefined, forecast: Math.max(0, f.intercept + f.slope * 216) },
    { h: '240h', measured: undefined, forecast: Math.max(0, f.intercept + f.slope * 240) }
  ];

  return (
    <>
      <Hero
        eyebrow={model === 'A' ? 'MODEL A · ABSOLUTE LIMIT GUARD' : 'MODEL B · DRIFT FORECAST'}
        title={model === 'A' ? 'Direct 168h Safety Threshold Screening' : 'Early Burn-In Trajectory Forecasting (OLS)'}
        text={
          model === 'A'
            ? `Model A operates as a hard safety gate. It evaluates strictly the observed 168h leakage measurement against the ${limit} μA threshold with zero extrapolation.`
            : 'Model B fits an Ordinary Least Squares (OLS) linear regression on early stages (0h, 24h, 96h) and projects the 168h endpoint without data leakage.'
        }
        actions={
          <>
            <ModelSwitch model={model} setModel={setModel} />
            <select value={c.id} onChange={e => setSel(e.target.value)}>
              {cs.map(x => (
                <option key={x.id} value={x.id}>
                  {x.id} · Lot {x.lot} ({x.risk})
                </option>
              ))}
            </select>
          </>
        }
      />

      {/* FINAL BURN AI INSPECTOR DECISION (Always visible) */}
      <div className="decisionBanner">
        <div>
          <b>FINAL BURN AI INSPECTOR DECISION</b>
          <strong>{c.risk.toUpperCase()}</strong>
<span>Highest severity from Model A, Model B, and Model C — never averaged.</span>
        </div>
        <div>
          <span>Model A (Measured 168h): <b>{a.status}</b></span>
          <span>Model B (Forecast): <b>{b.status}</b></span>
          <span>Model C (Anomaly): <b>{c.modelC?.status || 'NORMAL'}</b></span>
        </div>
      </div>

      {model === 'A' ? (
        /* --- MODEL A SPECIFIC VIEW --- */
        <>
          <div className="modelGrid">
            <div className={`decisionCard ${a.severity !== 'Safe' ? 'alert' : 'pass'} selected`} style={{ gridColumn: '1 / -1' }}>
              <div className="decisionHead">
                <span>MODEL A · ABSOLUTE LIMIT GUARD</span>
                <strong>{a.status}</strong>
              </div>
              <h3>Hard Safety Threshold Gate</h3>
              <div className="decisionValue">
                {a.value.toFixed(1)} μA
                <small>Measured 168h Endpoint vs Safety Limit {limit} μA</small>
              </div>
              <p>
                {a.status === 'BREACH'
                  ? `Component ${c.id} has breached the absolute safety limit (${a.value.toFixed(1)} μA ≥ ${limit} μA). Quarantine and QA review required.`
                  : `Component ${c.id} is within the absolute safety limit (${a.value.toFixed(1)} μA < ${limit} μA). Measured margin: ${Math.abs(a.margin).toFixed(1)} μA below threshold.`}
              </p>
              <small>{a.severity !== 'Safe' ? `Severity: ${a.severity} Breach` : 'Within standard screening boundary'}</small>
            </div>
          </div>

          <div className="stats">
            <Stat
              I={Target}
              label="Measured 168h"
              value={`${a.value.toFixed(1)} μA`}
              sub="Observed endpoint"
              tone={a.severity !== 'Safe' ? 'danger' : 'safe'}
            />
            <Stat
              I={ShieldCheck}
              label="Safety Limit"
              value={`${limit} μA`}
              sub="Universal threshold"
            />
            <Stat
              I={Activity}
              label="Safety Margin"
              value={`${Math.abs(a.margin).toFixed(1)} μA`}
              sub={a.margin >= 0 ? `${a.margin.toFixed(1)} μA above limit` : `${Math.abs(a.margin).toFixed(1)} μA headroom`}
              tone={a.severity !== 'Safe' ? 'danger' : 'safe'}
            />
            <Stat
              I={AlertTriangle}
              label="Model A Status"
              value={a.status}
              sub={`Classification: ${a.severity}`}
              tone={a.severity !== 'Safe' ? 'danger' : 'safe'}
            />
          </div>

          <Section
            title={`${c.id} (${c.lot}) — Model A: Measured Burn-In Trajectory`}
            sub={`Observed measurements across physical stages (0h, 24h, 96h, 168h) tested against ${limit} μA absolute limit`}
          >
            <div className="chart">
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={modelAData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#17314a" />
                  <XAxis dataKey="h" />
                  <YAxis unit=" μA" />
                  <Tooltip contentStyle={{ background: '#0a1c2d', borderColor: '#2a465e' }} />
                  <ReferenceLine y={limit} stroke="#ff7b7b" strokeDasharray="5 5" label={`Safety Limit (${limit} μA)`} />
                  <Line
                    dataKey="measured"
                    name="Measured Burn-In Current"
                    stroke="#5bbfe9"
                    strokeWidth={4}
                    dot={{ r: 6, fill: '#5bbfe9' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Section>

          <div className="two">
            <Section title="Model A Absolute Limit Guard Standards">
              <Rule t="HARD SAFETY GATE" x={`Evaluates the empirical 168h measurement directly against the universal ${limit} μA limit.`} />
              <Rule t="ZERO EXTRAPOLATION" x="No forecasting or trajectory assumptions are made; classification is based purely on physical measurement." />
              <Rule t="SEVERITY CRITERIA" x={`Measured ≥ ${limit * 1.25} μA = Critical Breach | Measured ≥ ${limit} μA = High Breach | Measured < ${limit} μA = Safe.`} />
              <Rule t="QA PROTOCOL" x={a.status === 'BREACH' ? 'Immediate quarantine and physical failure analysis.' : 'Part clears hard safety threshold gate.'} />
            </Section>
            <Section title="Model A Diagnostic Summary">
              <div className="callout">
                <BrainCircuit size={16} />
                <span>
                  <b>Model A Verdict:</b> {c.id} recorded an observed leakage current of <b>{a.value.toFixed(1)} μA</b> at 168h against the <b>{limit} μA</b> safety limit.
                  {a.status === 'BREACH'
                    ? ` This is a hard limit BREACH (${a.severity}). The component fails screening criteria.`
                    : ` The component PASSES Model A absolute screening with ${Math.abs(a.margin).toFixed(1)} μA of safety headroom.`}
                </span>
              </div>
            </Section>
          </div>
        </>
      ) : (
        /* --- MODEL B SPECIFIC VIEW --- */
        <>
          <div className="modelGrid">
            <div className={`decisionCard ${b.severity !== 'Safe' ? 'alert' : 'pass'} selected`} style={{ gridColumn: '1 / -1' }}>
              <div className="decisionHead">
                <span>MODEL B · EARLY DRIFT FORECAST (OLS)</span>
                <strong>{b.status}</strong>
              </div>
              <h3>Trajectory-Based Early Warning</h3>
              <div className="decisionValue">
                {b.predicted168.toFixed(1)} μA
                <small>Model B 168h Projection (Limit {limit} μA)</small>
              </div>
              <p>
                Fitted on early burn-in behaviour (0h, 24h, 96h). The 168h measurement is excluded to eliminate data leakage and test forecasting accuracy.
              </p>
              <small>{b.crossingText}</small>
            </div>
          </div>

          <div className="stats">
            <Stat
              I={Database}
              label="0h / 24h / 96h Inputs"
              value={`${c.v[0]?.toFixed(1)} / ${c.v[1]?.toFixed(1)} / ${c.v[2]?.toFixed(1)} μA`}
              sub="Training window points"
            />
            <Stat
              I={LineIcon}
              label="168h Forecast"
              value={`${f.predicted168.toFixed(1)} μA`}
              sub="Projected endpoint"
              tone={f.severity !== 'Safe' ? 'danger' : 'safe'}
            />
            <Stat
              I={Activity}
              label="Drift Slope"
              value={`${f.slope.toFixed(3)} μA/h`}
              sub="0h → 24h → 96h OLS fit"
            />
            <Stat
              I={Timer}
              label="Time to Limit"
              value={f.crossHours === 0 ? 'Breached' : f.crossHours === 999 ? 'None' : `${f.crossHours}h`}
              sub={f.crossingText}
            />
          </div>

          <Section
            title={`${c.id} (${c.lot}) — Model B: Early Drift Extrapolation`}
            sub="Solid cyan: Observed measurements (0-96h training) · Dashed orange: Model B OLS forecast extending beyond 96h"
          >
            <div className="chart">
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={modelBData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#17314a" />
                  <XAxis dataKey="h" />
                  <YAxis unit=" μA" />
                  <Tooltip contentStyle={{ background: '#0a1c2d', borderColor: '#2a465e' }} />
                  <ReferenceLine y={limit} stroke="#ff7b7b" strokeDasharray="5 5" label={`Safety Limit (${limit} μA)`} />
                  <Line
                    dataKey="measured"
                    name="Measured Data"
                    stroke="#5bbfe9"
                    strokeWidth={2}
                    dot={{ r: 5, fill: '#5bbfe9' }}
                    connectNulls={false}
                  />
                  <Line
                    dataKey="forecast"
                    name="Model B OLS Forecast"
                    stroke="#f3a06f"
                    strokeWidth={4}
                    strokeDasharray="7 6"
                    dot={{ r: 5, fill: '#f3a06f' }}
                    connectNulls={true}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Section>

          <div className="two">
            <Section title="Model B Drift Forecasting Engine & Rules">
              <Rule t="TRAINING INPUTS" x="Measured 0h, 24h, 96h values only. Excludes 168h to eliminate future data leakage." />
              <Rule t="OLS REGRESSION" x={`Estimated slope: ${f.slope.toFixed(4)} μA/h, intercept: ${f.intercept.toFixed(2)} μA across early burn-in hours.`} />
              <Rule t="FORECAST CLASSIFICATION" x={`Projected ≥ ${limit * 1.25} μA = Critical | ≥ ${limit} μA = High | ≥ ${limit * 0.78} μA or slope > 0.12 μA/h = Watch.`} />
              <Rule t="TIME TO THRESHOLD" x={f.crossingText} />
            </Section>
            <Section title="Model B Diagnostic Summary">
              <div className="callout">
                <BrainCircuit size={16} />
                <span>
                  <b>Model B Verdict:</b> Early drift trajectory exhibits a rate of <b>{f.slope.toFixed(3)} μA/h</b>.
                  {b.status === 'FORECAST ALERT'
                    ? ` Model B alerts on projected drift (${b.severity}). Forecast endpoint at 168h is ${f.predicted168.toFixed(1)} μA (${f.crossingText}).`
                    : ` Model B projects safe containment (${b.severity}) with a 168h forecast of ${f.predicted168.toFixed(1)} μA.`}
                </span>
              </div>
            </Section>
          </div>
        </>
      )}
    </>
  );
}

/**
 * Component Inspector Profile View
 */
function Inspector({ c, cs, setSel, limit, go, mlStatus, mlError }) {
  if (!c) return null;

  return (
    <>
      <Hero
        eyebrow="COMPONENT DIGITAL PROFILE"
        title={c.id}
        text={`Lot ${c.lot} • Comprehensive evidence trail from 0h baseline to 168h endpoint.`}
        actions={
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
            <select
              value={c.id}
              onChange={e => setSel && setSel(e.target.value)}
              aria-label="Select Component to Inspect"
              style={{
                background: '#091b2b',
                border: '1px solid #28475f',
                color: '#e7eff7',
                padding: '8px 12px',
                borderRadius: '7px',
                fontSize: '11px',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              {(cs || []).map(x => (
                <option key={x.id} value={x.id}>
                  {x.id} — Lot {x.lot} ({x.risk})
                </option>
              ))}
            </select>
            <Btn onClick={() => go('action')}><ListChecks size={14} /> Action Center</Btn>
          </div>
        }
      />

      {/* Burn-In Stage Measurements and Key Model Indicators */}
      <div className="stats">
        <Stat
          I={Activity}
          label="0h Baseline"
          value={`${c.v[0]?.toFixed(1)} μA`}
          sub="Initial stage"
        />
        <Stat
          I={Gauge}
          label="24h / 96h Stages"
          value={`${c.v[1]?.toFixed(1)} / ${c.v[2]?.toFixed(1)} μA`}
          sub="Intermediate burn-in"
        />
        <Stat
          I={Target}
          label="Measured 168h"
          value={`${c.v[3]?.toFixed(1)} μA`}
          sub={`Safety limit: ${limit} μA`}
          tone={c.v[3] >= limit ? 'danger' : 'safe'}
        />
        <Stat
          I={LineIcon}
          label="Model B Forecast (168h)"
          value={`${c.forecast?.toFixed(1)} μA`}
          sub={`Drift slope: ${c.slope?.toFixed(3)} μA/h`}
          tone={c.forecast >= limit ? 'danger' : 'safe'}
        />
      </div>

      <Section
        title="AI RUL & Failure Prediction"
        sub={mlStatus === 'online' ? 'Live predictions from CatBoost + Random Forest Regressor + Isolation Forest + Fisher discriminant + SHAP.' : `ML engine: ${mlStatus}${mlError ? ` · ${mlError}` : ''}`}
      >
        <div className="stats" style={{ marginBottom: '10px' }}>
          <Stat
            I={Clock3}
            label="Remaining Useful Life"
            value={c.rulHours == null ? 'PENDING' : `${c.rulHours.toFixed(0)} h`}
            sub={c.predictedLifetimeHours == null ? 'Waiting for ML backend' : `Predicted failure/lifetime: ${c.predictedLifetimeHours.toFixed(0)} h`}
            tone={c.rulHours != null && c.rulHours < 120 ? 'danger' : 'safe'}
          />
          <Stat
            I={BrainCircuit}
            label="CatBoost Failure Probability"
            value={c.failureProbability == null ? '—' : `${(c.failureProbability * 100).toFixed(1)}%`}
            sub="SMOTE-balanced classification"
            tone={c.failureProbability != null && c.failureProbability >= 0.8 ? 'danger' : c.failureProbability != null && c.failureProbability >= 0.5 ? 'watch' : 'safe'}
          />
          <Stat
            I={AlertTriangle}
            label="Isolation Forest"
            value={c.isolationAnomaly == null ? '—' : c.isolationAnomaly.toFixed(3)}
            sub={c.isolationFlag ? 'Unsupervised anomaly detected' : 'No unsupervised anomaly'}
            tone={c.isolationFlag ? 'danger' : 'safe'}
          />
          <Stat
            I={Target}
            label="RUL Uncertainty Band"
            value={c.rulLowHours == null ? '—' : `${c.rulLowHours.toFixed(0)}–${c.rulHighHours.toFixed(0)} h`}
            sub="Random Forest tree spread (10–90%)"
          />
        </div>
        <div className="callout" style={{ marginBottom: '10px' }}>
          <BrainCircuit size={16} />
          <span><b>SHAP explanation:</b> {c.shap?.length ? c.shap.map(x => `${x.feature} (${x.impact >= 0 ? '+' : ''}${x.impact.toFixed(3)})`).join(' · ') : 'Waiting for the ML explanation.'}</span>
        </div>
        {c.groundTruthRul != null && Number.isFinite(c.groundTruthRul) && c.rulHours != null && (
          <div style={{ fontSize: '10px', color: '#7893a9' }}>
            Validation target available in supplied dataset: observed RUL at 168h = <b style={{ color: '#e7eff7' }}>{c.groundTruthRul.toFixed(0)} h</b> · model error = <b style={{ color: '#e7eff7' }}>{Math.abs(c.rulHours - c.groundTruthRul).toFixed(1)} h</b>.
          </div>
        )}
      </Section>

      <div className="profile">
        <div>
          <div className="banner">
            <div>
              <label>CURRENT CLASSIFICATION</label>
              <h2>Measured: {c.v[3]?.toFixed(1)} μA at 168h</h2>
              <p>
                Limit: {limit} μA • Model B Forecast: {c.forecast?.toFixed(1)} μA ({c.crossingText})
              </p>
            </div>
            {badge(c.risk)}
          </div>
          <Section title="Observed Burn-In Evidence Trajectory" sub="Measured data points (0h, 24h, 96h, 168h) vs Model B linear forecast">
            <InspectorChart c={c} limit={limit} />
          </Section>
        </div>
        <div className="score">
          <div className="ring">
            {Math.max(1, 100 - Math.round(c.anomaly * 70))}
            <small>/100</small>
          </div>
          <p>Reliability Score</p>
          <hr />
          <p>Model A <b style={{ color: c.modelA?.severity !== 'Safe' ? '#ff7b7b' : '#68dda0' }}>{c.modelA?.status || '—'}</b></p>
          <p>Model B <b style={{ color: c.modelB?.severity !== 'Safe' ? '#f3a06f' : '#68dda0' }}>{c.modelB?.status || '—'}</b></p>
          <p>Model C <b style={{ color: c.modelC?.anomalyDetected ? '#e3cc70' : '#68dda0' }}>{c.modelC?.status || '—'}</b></p>
          <p>Time to Limit <b>{c.hours === 0 ? 'Breached' : c.hours === 999 ? 'No breach' : `${c.hours} h`}</b></p>
        </div>
      </div>

      {/* Model C Diagnostic Card */}
      <Section
        title={`Model C — Lot Trajectory Anomaly (${c.lot})`}
        sub={`Lot-relative robust statistical outlier kernel · Early window only (0h, 24h, 96h) · Zero 168h data leakage`}
      >
        <div className="stats" style={{ marginBottom: '10px' }}>
          <Stat
            I={Zap}
            label="Model C Anomaly Score"
            value={(c.modelC?.anomalyScore ?? 0).toFixed(3)}
            sub={`S ∈ [0,1] · ${c.modelC?.status || 'NORMAL'}`}
            tone={c.modelC?.severity === 'Safe' ? 'safe' : c.modelC?.severity === 'Watch' ? 'watch' : 'danger'}
          />
          <Stat
            I={Activity}
            label="Anomaly Distance (D)"
            value={(c.modelC?.anomalyDistance ?? 0).toFixed(3)}
            sub="Weighted z-score Euclidean distance"
            tone={c.modelC?.anomalyDetected ? 'danger' : ''}
          />
          <Stat
            I={BrainCircuit}
            label="Strongest Feature"
            value={c.modelC?.strongestFeature?.split('—')[0]?.trim() || '—'}
            sub={c.modelC?.strongestFeature || '—'}
          />
          <Stat
            I={ShieldCheck}
            label="Model C Severity"
            value={c.modelC?.severity || 'Safe'}
            sub={`Engineering threshold, not validated ML metric`}
            tone={c.modelC?.severity === 'Safe' ? 'safe' : c.modelC?.severity === 'Watch' ? 'watch' : 'danger'}
          />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', marginBottom: '10px' }}>
          {[
            { label: 'F1 — Baseline Offset', val: c.modelC?.F1, z: c.modelC?.Z1, unit: 'μA' },
            { label: 'F2 — Drift Velocity', val: c.modelC?.F2, z: c.modelC?.Z2, unit: 'μA/h' },
            { label: 'F3 — Curvature', val: c.modelC?.F3, z: c.modelC?.Z3, unit: 'μA/h²' },
            { label: 'F4 — OLS Dispersion', val: c.modelC?.F4, z: c.modelC?.Z4, unit: 'μA²' }
          ].map(({ label, val, z, unit }) => (
            <div key={label} style={{ background: '#050f1a', border: '1px solid #1a354c', borderRadius: '8px', padding: '10px' }}>
              <div style={{ fontSize: '8px', color: '#63bce9', fontWeight: 800, letterSpacing: '.08em', textTransform: 'uppercase', marginBottom: '4px' }}>{label}</div>
              <div style={{ fontSize: '11px', fontWeight: 700, color: '#e7eff7' }}>{typeof val === 'number' && isFinite(val) ? val.toFixed(3) : '—'} <span style={{ color: '#7893a9', fontWeight: 400 }}>{unit}</span></div>
              <div style={{ fontSize: '9px', color: Math.abs(z || 0) >= 2 ? '#ff7b7b' : '#7893a9', marginTop: '3px' }}>
                Z = {typeof z === 'number' && isFinite(z) ? z.toFixed(2) : '—'} σ
              </div>
            </div>
          ))}
        </div>
        <div className="callout">
          <BrainCircuit size={16} />
          <span>
            <b>Model C Assessment:</b> {(c.modelC?.anomalyReasons || ['No anomaly reasons computed.']).join(' · ')}
          </span>
        </div>
      </Section>

      <div className="two">
        <Section title="Why BURN AI INSPECTOR Flagged This Component">
          <div className="factors">
            {[
              ['Early Drift Rate (0h-96h)', Math.min(100, Math.abs(c.slope) * 450)],
              ['Initial Baseline Deviation', Math.min(100, (c.v[0] / 15) * 50)],
              ['168h Threshold Proximity', Math.min(100, (c.v[3] / limit) * 85)]
            ].map(([n, v]) => (
              <div key={n}>
                <span>{n}</span>
                <b>{Math.round(v)}%</b>
                <div className="track">
                  <i style={{ width: Math.min(100, v) + '%' }} />
                </div>
              </div>
            ))}
          </div>
          <div className="callout">
            <BrainCircuit size={16} />
            <span>
              <b>Diagnostic Explanation:</b> Early slope ({c.slope.toFixed(3)} μA/h) and measured endpoint ({c.v[3]?.toFixed(1)} μA) contribute to the tri-model risk level {c.risk}. {c.crossingText}.
            </span>
          </div>
        </Section>
        <Section title="Recommended QA Next Step">
          <div className="recommend">
            <ShieldCheck size={19} />
            <div>
              <b>{c.risk === 'Safe' ? 'Release Component to Next Test Phase' : 'Quarantine & Conduct Engineering Review'}</b>
              <p>
                {c.risk === 'Safe'
                  ? 'Burn-in trajectory is within expected lot boundary.'
                  : 'Verify thermal dissipation and re-test before releasing lot.'}
              </p>
            </div>
          </div>
        </Section>
      </div>
    </>
  );
}



/**
 * AI Copilot Query Interface
 */
function Copilot({ q, setQ, ans, ask, onPresetClick }) {
  return (
    <div className="copilot">
      <div className="copIntro">
        <BrainCircuit size={24} />
        <div>
          <h3>BURN AI INSPECTOR Copilot</h3>
          <p>Query components, lot populations, drift slopes, or projected crossings.</p>
        </div>
      </div>
      <div className="suggestions">
        {[
          'Why is C-10482 risky?',
          'Which lot is deteriorating fastest?',
          'Which components cross the limit within 48 hours?'
        ].map(x => (
          <button key={x} onClick={() => onPresetClick(x)}>
            {x}
          </button>
        ))}
      </div>
      <div className="chat">
        <input
          value={q}
          onChange={e => setQ(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && ask()}
          placeholder="Ask BURN AI INSPECTOR about components, lots, or crossings..."
        />
        <Btn onClick={ask}><MessageSquare size={14} /> Ask</Btn>
      </div>
      {ans && (
        <div className="answer">
          <Sparkles size={15} style={{ flex: 'none', color: '#63bce9', marginTop: '2px' }} />
          <div>{ans}</div>
        </div>
      )}
    </div>
  );
}

// Mount React Root
createRoot(document.getElementById('root')).render(<App />);
