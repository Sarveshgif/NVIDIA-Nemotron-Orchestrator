// NDVI colour ramp shared with scripts/fetch-sentinel.py, plus the legacy synthetic scene renderer
// (no longer used for evidence: before/after images are real Sentinel-2 chips).
// The renderer paints a field and its neighbours from the same values that drive the charts,
// so the preview, the time series and the model summary always agree.

import type { Field, FieldEvidence } from '../types';
import { CELL_LAT, CELL_LNG, ringBounds } from '../data/geo';

export type ImageMode = 'natural' | 'ndvi';

type RGB = [number, number, number];
const lerp = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const rgb = (c: RGB, k = 1) => `rgb(${Math.round(c[0] * k)},${Math.round(c[1] * k)},${Math.round(c[2] * k)})`;

export function hash2(i: number, j: number) {
  let h = (i * 374761393 + j * 668265263) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

const SOIL: RGB = [158, 132, 104];
const DRY: RGB = [196, 170, 112];
const GREEN: RGB = [58, 104, 50];
const LUSH: RGB = [44, 92, 44];

export function naturalColor(ndvi: number, lswi: number): RGB {
  const g = clamp01((ndvi - 0.2) / 0.6);
  const m = clamp01((lswi + 0.05) / 0.37);
  const veg = lerp(DRY, lerp(GREEN, LUSH, m), Math.pow(m, 0.8));
  return lerp(SOIL, veg, Math.pow(g, 0.85));
}

const NDVI_STOPS: [number, RGB][] = [
  [0.0, [120, 78, 44]],
  [0.2, [168, 120, 62]],
  [0.35, [222, 196, 96]],
  [0.55, [150, 196, 86]],
  [0.75, [58, 148, 64]],
  [0.9, [20, 96, 44]],
];
export function ndviColor(v: number): RGB {
  if (v <= NDVI_STOPS[0][0]) return NDVI_STOPS[0][1];
  for (let i = 1; i < NDVI_STOPS.length; i++) {
    const [x1, c1] = NDVI_STOPS[i];
    const [x0, c0] = NDVI_STOPS[i - 1];
    if (v <= x1) return lerp(c0, c1, (v - x0) / (x1 - x0));
  }
  return NDVI_STOPS[NDVI_STOPS.length - 1][1];
}
export const NDVI_GRADIENT = NDVI_STOPS.map(([x, c]) => `${rgb(c)} ${Math.round((x / 0.9) * 100)}%`).join(', ');

interface Scene {
  field: Field;
  evidence: FieldEvidence;
  date: string;
  mode: ImageMode;
  width: number;
  height: number;
  dpr: number;
}

/** Neighbouring parcels have a seasonal "crop" too, so they also change between dates. */
function neighbourValues(i: number, j: number, day: number) {
  const r = hash2(i, j);
  const kind = r < 0.35 ? 'orchard' : r < 0.55 ? 'alfalfa' : r < 0.72 ? 'fallow' : r < 0.86 ? 'cotton' : 'grain';
  const t = day / 160;
  switch (kind) {
    case 'orchard': return { ndvi: 0.62 + 0.06 * hash2(j, i), lswi: 0.18, rows: 'dots' as const };
    case 'alfalfa': return { ndvi: 0.45 + 0.3 * Math.abs(Math.sin(day / 9 + r * 6)), lswi: 0.14 + 0.1 * Math.abs(Math.sin(day / 9 + r * 6)), rows: 'none' as const };
    case 'fallow': return { ndvi: 0.17 + 0.04 * hash2(i + 3, j), lswi: -0.1, rows: 'none' as const };
    case 'cotton': return { ndvi: 0.35 + 0.4 * Math.sin(Math.PI * clamp01(t * 1.1)), lswi: 0.05 + 0.15 * Math.sin(Math.PI * clamp01(t)), rows: 'lines' as const };
    default: return { ndvi: 0.22 + 0.1 * hash2(i, j + 9), lswi: -0.06, rows: 'lines' as const };
  }
}

export function renderScene(canvas: HTMLCanvasElement, s: Scene) {
  const { field, evidence, date, mode, width, height, dpr } = s;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const b = ringBounds(field.polygon);
  const latC = (b.minLat + b.maxLat) / 2;
  const kx = Math.cos((latC * Math.PI) / 180);
  const spanLat = b.maxLat - b.minLat;
  const spanLng = (b.maxLng - b.minLng) * kx;
  const pad = 0.55;
  const scale = Math.min(width / (spanLng * (1 + 2 * pad)), height / (spanLat * (1 + 2 * pad)));
  const cx = (b.minLng + b.maxLng) / 2;
  const project = (lat: number, lng: number): [number, number] => [
    width / 2 + (lng - cx) * kx * scale,
    height / 2 - (lat - latC) * scale,
  ];
  const day = Math.round((Date.parse(date + 'T00:00:00Z') - Date.UTC(2026, 5, 1)) / 864e5);
  const colorFor = (ndvi: number, lswi: number) => (mode === 'ndvi' ? ndviColor(ndvi) : naturalColor(ndvi, lswi));

  // ground
  ctx.fillStyle = mode === 'ndvi' ? '#6b5a44' : '#b9ab90';
  ctx.fillRect(0, 0, width, height);

  // neighbouring parcels on the same quarter-section grid as the basemap
  const LNG0 = -120, LAT0 = 37;
  const viewLatMin = latC - height / 2 / scale, viewLatMax = latC + height / 2 / scale;
  const viewLngMin = cx - width / 2 / (kx * scale), viewLngMax = cx + width / 2 / (kx * scale);
  const i0 = Math.floor((viewLngMin - LNG0) / CELL_LNG) - 1, i1 = Math.ceil((viewLngMax - LNG0) / CELL_LNG) + 1;
  const j0 = Math.floor((LAT0 - viewLatMax) / CELL_LAT) - 1, j1 = Math.ceil((LAT0 - viewLatMin) / CELL_LAT) + 1;
  for (let i = i0; i <= i1; i++) {
    for (let j = j0; j <= j1; j++) {
      const west = LNG0 + i * CELL_LNG, north = LAT0 - j * CELL_LAT;
      const [x0, y0] = project(north - 0.00016, west + 0.0002);
      const [x1, y1] = project(north - CELL_LAT + 0.00016, west + CELL_LNG - 0.0002);
      const nv = neighbourValues(i, j, day);
      const c = colorFor(nv.ndvi, nv.lswi);
      ctx.fillStyle = rgb(c);
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      if (nv.rows === 'dots') {
        ctx.fillStyle = rgb(c, 0.78);
        const step = 5;
        for (let x = x0 + 2; x < x1; x += step) for (let y = y0 + 2; y < y1; y += step) ctx.fillRect(x, y, 2.2, 2.2);
      } else if (nv.rows === 'lines') {
        ctx.strokeStyle = rgb(c, 0.9);
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let y = y0 + 2; y < y1; y += 3.2) { ctx.moveTo(x0, y); ctx.lineTo(x1, y); }
        ctx.stroke();
      }
    }
  }

  // field interior: 3 × 2 patches clipped to the polygon
  const ring = field.polygon.map(([la, ln]) => project(la, ln));
  ctx.save();
  ctx.beginPath();
  ring.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.clip();
  const [fx0, fy0] = project(b.maxLat, b.minLng);
  const [fx1, fy1] = project(b.minLat, b.maxLng);
  const pw = (fx1 - fx0) / 3, ph = (fy1 - fy0) / 2;
  const opt = evidence.optical.find((o) => o.date === date);
  const fieldLswi = opt?.lswi ?? 0.1;
  let rngState = field.seed;
  const rnd = () => ((rngState = (rngState * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  for (const p of evidence.patches) {
    const v = p.ndvi.find((n) => n.date === date)?.value ?? opt?.ndvi ?? 0.5;
    const fieldNdvi = opt?.ndvi ?? v;
    const lswi = fieldLswi + (v - fieldNdvi) * 0.6;
    const c = colorFor(v, lswi);
    const px = fx0 + p.col * pw, py = fy0 + p.row * ph;
    ctx.fillStyle = rgb(c);
    ctx.fillRect(px - 0.5, py - 0.5, pw + 1, ph + 1);
    // furrows run north–south
    ctx.strokeStyle = rgb(c, v > 0.5 ? 0.86 : 1.08);
    ctx.lineWidth = 0.9;
    ctx.beginPath();
    for (let x = px + 1.5; x < px + pw; x += 2.6) { ctx.moveTo(x, py); ctx.lineTo(x, py + ph); }
    ctx.stroke();
    // speckle
    for (let k = 0; k < (pw * ph) / 14; k++) {
      ctx.fillStyle = rgb(c, 0.85 + rnd() * 0.3);
      ctx.fillRect(px + rnd() * pw, py + rnd() * ph, 1.4, 1.4);
    }
  }
  ctx.restore();

  // outline
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ring.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.strokeStyle = 'rgba(0,0,0,0.55)';
  ctx.lineWidth = 3.5;
  ctx.stroke();
  ctx.strokeStyle = '#fffdf6';
  ctx.lineWidth = 1.6;
  ctx.setLineDash([6, 3]);
  ctx.stroke();
  ctx.setLineDash([]);
}

/** Small patch-grid swatch for the within-field variation view. */
export function patchColorAt(evidence: FieldEvidence, patchIndex: number, date: string): string {
  const p = evidence.patches[patchIndex];
  const v = p.ndvi.find((n) => n.date === date)?.value;
  return v === undefined ? 'var(--unres-bg)' : rgb(ndviColor(v));
}
