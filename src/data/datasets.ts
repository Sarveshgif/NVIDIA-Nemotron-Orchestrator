// Prepared datasets. Two exist and are never mixed:
//  * ca2023: Tulare County, CA — DWR 2023 outlines + Sentinel-2/1 2023. No ground truth; rule-based only.
//  * nl2025: South-east Netherlands — PDOK BRP 2025 parcels + Sentinel-2/1 2025. Farmer-declared silage/grain
//    labels exist, so a Random Forest was trained and evaluated on held-out regions (pipeline/).
// The NL reference labels are kept in a separate file and are only read by the evaluation panel.

import type { Field, FieldEvidence, PatchSeries, ReviewReason } from '../types';
import { CA_SEASON, setSeason, type SeasonConfig } from './season';
import { SENTINEL_ATTRIBUTION as CA_ATTR, chipDates as caChipDates } from './generate';

export type DatasetKey = 'ca2023' | 'nl2025' | 'barc';

export interface EvalRow {
  model: string; label: string; n: number; coverage: number; accuracy: number; balanced_accuracy: number;
  precision_silage: number; recall_silage: number; fpr_silage: number; precision_grain: number; recall_grain: number;
  f1_grain: number; roc_auc?: number; pr_auc_grain?: number; unresolved_area_frac?: number;
}

export interface NlMeta {
  title: string;
  year: number;
  season: SeasonConfig;
  provenance: Record<string, string>;
  model: { name: string; version: string; threshold: number; review_band: [number, number]; features: number; note: string };
  evaluation: {
    split: string; positive: string; test_fields: number; test_silage: number; test_grain: number;
    comparison: EvalRow[]; confusion: Record<string, { tp: number; fn: number; fp: number; tn: number }>;
    ci: Record<string, Record<string, [number, number]>>; latency: Record<string, number | string>;
    findings: string[]; deep?: Record<string, unknown>; gap?: Record<string, unknown>;
    years?: { experiment: string; train_years: string; test_year: number; n_train: number; grain_train: number; model: string; n_test: number;
      balanced_accuracy: number; roc_auc: number | null; recall_grain: number; precision_grain: number; gap_balanced_accuracy: number | null }[];
  };
  outline: [number, number][];
  chipPx: number;
}

export interface DatasetInfo {
  key: DatasetKey;
  regionId: string;
  cropYear: number;
  validated: boolean;
  outlineSource: string;
  obsSource: string;
  attribution: string;
  chipBase: string;
  mlAvailable: boolean;
  mlNote: string;
  rulesNote: string;
  scopeNote: string;
  season: SeasonConfig;
  chipDates: (fieldId: string) => string[];
  nl?: NlMeta;
}

export const CA_DATASET: DatasetInfo = {
  key: 'ca2023', regionId: 'tulare-ca', cropYear: 2023, validated: false,
  outlineSource: 'DWR 2023 Statewide Crop Mapping (F16 corn/sorghum/sudan)',
  obsSource: 'Copernicus Sentinel-2 L2A / Sentinel-1 RTC, 2023',
  attribution: CA_ATTR,
  chipBase: 's2/',
  mlAvailable: false,
  mlNote: 'No trained model for Tulare: there are no silage/grain labels here yet. See the Netherlands 2025 dataset for a trained, evaluated model.',
  rulesNote: 'Thresholds on the NDVI dry-down shape. Weak labels only — not validated in California.',
  scopeNote: '48 large DWR corn-group fields in four 10 km windows of Tulare County (demo subset).',
  season: CA_SEASON,
  chipDates: caChipDates,
};

let active: DatasetInfo = CA_DATASET;
export const activeDataset = () => active;
export function setActiveDataset(d: DatasetInfo) {
  active = d;
  setSeason(d.season);
}

// ---------------- Netherlands 2025 ----------------

interface NlField {
  id: string; brp: string; ring: [number, number][]; ha: number; ac: number; block: string; region: string;
  p: number; pred: 'silage' | 'grain'; review: boolean; reasons: ReviewReason[]; summary: string;
  tr: [string, string, number] | null; maxGap: number; nObs: number;
  opt: [string, number, number][]; rad: [string, number, number][]; zones: [string, ...(number | null)[]][];
  chips: string[]; showcase?: string;
}
interface NlFile { meta: NlMeta; fields: NlField[] }

/** Largest NDVI fall (> 0.15) after the zone's peak, between two clear looks (same logic as the Tulare patches). */
function dropWindow(pts: { date: string; value: number }[]) {
  if (pts.length < 3) return null;
  let pk = 0;
  pts.forEach((q, i) => { if (q.value > pts[pk].value) pk = i; });
  let best = -1, fall = 0.15;
  for (let i = pk + 1; i < pts.length; i++) {
    const d = pts[i - 1].value - pts[i].value;
    if (d > fall) { fall = d; best = i; }
  }
  if (best < 0) return null;
  const a = pts[best - 1].date, b = pts[best].date;
  return { start: a, end: b, gapDays: Math.round((Date.parse(b) - Date.parse(a)) / 864e5) };
}

let nlPromise: Promise<{ info: DatasetInfo; fields: Field[]; evidence: Record<string, FieldEvidence> }> | null = null;
const base = () => import.meta.env.BASE_URL;

export function loadNl() {
  if (nlPromise) return nlPromise;
  nlPromise = (async () => {
    const r = await fetch(`${base()}nl/nl2025.json`);
    if (!r.ok) throw new Error(`Netherlands dataset not found (HTTP ${r.status}). Run pipeline/07_export_console.py.`);
    const data = (await r.json()) as NlFile;
    const m = data.meta;
    const chips = new Map(data.fields.map((f) => [f.id, f.chips]));
    const info: DatasetInfo = {
      key: 'nl2025', regionId: 'nl-se', cropYear: m.year, validated: true,
      outlineSource: m.provenance.outlines, obsSource: m.provenance.observations, attribution: m.provenance.attribution,
      chipBase: 'nl/s2/', mlAvailable: true,
      mlNote: m.model.note,
      rulesNote: 'The original rule set (rules-v1, tuned for Tulare) run unchanged on Dutch data, for comparison. It is not validated here.',
      scopeNote: m.provenance.scope,
      season: m.season, chipDates: (id) => chips.get(id) ?? [], nl: m,
    };
    const fields: Field[] = [];
    const evidence: Record<string, FieldEvidence> = {};
    data.fields.forEach((f, i) => {
      const conf = Math.max(f.p, 1 - f.p);
      fields.push({
        id: f.id, polygon: f.ring, sourceId: f.brp,
        centroid: [f.ring.reduce((s, q) => s + q[0], 0) / f.ring.length, f.ring.reduce((s, q) => s + q[1], 0) / f.ring.length],
        acres: f.ac, cluster: f.region,
        prediction: { source: 'ml', predicted: f.pred, confidence: Math.round(conf * 100) / 100, pSilage: Math.round(f.p * 1000) / 1000, modelVersion: m.model.version },
        reasons: f.reasons,
        harvestWindow: f.tr ? { start: f.tr[0], end: f.tr[1], gapDays: f.tr[2] } : null,
        coverage: f.nObs < 14 ? 'insufficient' : f.maxGap > 15 ? 'gap' : 'sufficient',
        maxGapNearHarvest: f.maxGap,
        evidenceSummary: f.summary,
        showcase: f.showcase,
        seed: i,
      });
      const optical = f.opt.map(([date, ndvi, lswi]) => ({ date, ndvi, lswi }));
      const gaps: FieldEvidence['gaps'] = [];
      for (let k = 1; k < optical.length; k++) {
        const d = (Date.parse(optical[k].date) - Date.parse(optical[k - 1].date)) / 864e5;
        if (d > 10) gaps.push({ start: optical[k - 1].date, end: optical[k].date, days: Math.round(d) });
      }
      const patches: PatchSeries[] = [0, 1, 2, 3].map((z) => {
        const ndvi = f.zones.flatMap((row) => (row[z + 1] === null || row[z + 1] === undefined ? [] : [{ date: row[0] as string, value: row[z + 1] as number }]));
        return { id: `Z${z + 1}`, label: `${z < 2 ? 'North' : 'South'}-${z % 2 ? 'east' : 'west'}`, col: z % 2, row: Math.floor(z / 2), changeWindow: dropWindow(ndvi), ndvi };
      });
      evidence[f.id] = {
        fieldId: f.id, optical, radar: f.rad.map(([date, vv, vh]) => ({ date, vv, vh })), gaps, patches,
        before: f.tr ? optical.find((o) => o.date === f.tr![0]) ?? null : null,
        after: f.tr ? optical.find((o) => o.date === f.tr![1]) ?? null : null,
        pattern: f.pred === 'grain' ? 'grain' : 'silage',
      };
    });
    return { info, fields, evidence };
  })();
  nlPromise.catch(() => { nlPromise = null; });
  return nlPromise;
}

/** Reference labels (BRP farmer declarations) — evaluation panel only, never mixed into predictions. */
export async function loadNlReference(): Promise<Record<string, 'silage' | 'grain'>> {
  const r = await fetch(`${base()}nl/nl2025_reference.json`);
  if (!r.ok) throw new Error('Reference labels not found.');
  return (await r.json()).labels;
}
