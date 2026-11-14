// Dataset for Tulare County, 2023 season, built from real inputs only:
//  * field outlines: DWR 2023 Statewide Crop Mapping (dwrFields.ts)
//  * time series: Copernicus Sentinel-2 L2A and Sentinel-1 RTC (sentinel2023.json, from scripts/fetch-sentinel.py)
// Nothing here is invented. There is no trained ML model yet and no ground truth: predictions come from
// the rule set in lib/rules.ts, and the review history starts empty.

import type { Decision, Field, FieldEvidence, HarvestWindow, PatchSeries, ReviewReason } from '../types';
import { ringAcres, ringCentroid } from './geo';
import { DWR_FIELDS } from './dwrFields';
import { isoToDay } from './season';
import { applyRules, DEFAULT_THRESHOLDS, extractFeatures, type RuleResult } from '../lib/rules';
import REAL from './sentinel2023.json';

interface RealOptical { date: string; ndvi: number; lswi: number; clear: number; patches: (number | null)[] }
interface RealRadar { date: string; vv: number; vh: number }
const SERIES = (REAL as unknown as { fields: Record<string, { optical: RealOptical[]; radar: RealRadar[]; chips: string[] }> }).fields;
/** Dates (in strip order) of the Sentinel-2 chips in public/s2/<field>_{rgb,ndvi}.jpg. */
export const chipDates = (fieldId: string): string[] => SERIES[fieldId]?.chips ?? [];
export const SENTINEL_ATTRIBUTION = (REAL as { attribution: string }).attribution;

const round = (v: number, d = 1) => Math.round(v * 10 ** d) / 10 ** d;
const PATCH_LABEL = (col: number, row: number) => `${row === 0 ? 'North' : 'South'}-${['west', 'centre', 'east'][col]}`;

/** Largest NDVI fall (> 0.15) after the series peak, as a window between two clear looks. */
function dropWindow(points: { date: string; value: number }[]): HarvestWindow | null {
  if (points.length < 3) return null;
  let pk = 0;
  points.forEach((p, i) => { if (p.value > points[pk].value) pk = i; });
  let best = -1, fall = 0.15;
  for (let i = pk + 1; i < points.length; i++) {
    const f = points[i - 1].value - points[i].value;
    if (f > fall) { fall = f; best = i; }
  }
  if (best < 0) return null;
  const a = points[best - 1].date, b = points[best].date;
  return { start: a, end: b, gapDays: isoToDay(b) - isoToDay(a) };
}

function buildEvidence(id: string): FieldEvidence {
  const s = SERIES[id] ?? { optical: [], radar: [] };
  const optical = s.optical.map((o) => ({ date: o.date, ndvi: o.ndvi, lswi: o.lswi }));
  const gaps: FieldEvidence['gaps'] = [];
  for (let i = 1; i < optical.length; i++) {
    const d = isoToDay(optical[i].date) - isoToDay(optical[i - 1].date);
    if (d > 10) gaps.push({ start: optical[i - 1].date, end: optical[i].date, days: d });
  }
  const patches: PatchSeries[] = Array.from({ length: 6 }, (_, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const ndvi = s.optical.flatMap((o) => (o.patches[i] === null || o.patches[i] === undefined ? [] : [{ date: o.date, value: o.patches[i] as number }]));
    return { id: `P${i + 1}`, label: PATCH_LABEL(col, row), col, row, ndvi, changeWindow: dropWindow(ndvi) };
  });
  return { fieldId: id, optical, radar: s.radar.map((r) => ({ date: r.date, vv: r.vv, vh: r.vh })), gaps, patches, before: null, after: null, pattern: 'silage' };
}

export interface Dataset {
  fields: Field[];
  evidence: Map<string, FieldEvidence>;
  seedDecisions: Decision[];
}

function build(): Dataset {
  const counters = new Map<string, number>();
  const evidence = new Map<string, FieldEvidence>();
  const drafts = DWR_FIELDS.map((f) => {
    const n = (counters.get(f.cluster) ?? 0) + 1;
    counters.set(f.cluster, n);
    const id = `TUL-${f.cluster}${String(n).padStart(2, '0')}`;
    const ev = buildEvidence(id);
    const rule: RuleResult = applyRules(id, extractFeatures(ev), DEFAULT_THRESHOLDS);
    const feat = rule.features;
    if (feat.dropStart && feat.dropEnd) {
      ev.before = ev.optical.find((o) => o.date === feat.dropStart) ?? null;
      ev.after = ev.optical.find((o) => o.date === feat.dropEnd) ?? null;
    }
    ev.pattern = feat.clearObs < DEFAULT_THRESHOLDS.minClearObs ? 'sparse' : rule.label === 'grain' ? 'grain' : 'silage';
    evidence.set(id, ev);
    return { f, id, ev, rule, acres: round(ringAcres(f.ring)) };
  });

  // Fixed (prediction-independent) review reasons, all measured from the real series.
  const fields: Field[] = drafts.map(({ f, id, ev, rule, acres }, i) => {
    const feat = rule.features;
    const mid = feat.dropStart && feat.dropEnd ? (isoToDay(feat.dropStart) + isoToDay(feat.dropEnd)) / 2 : null;
    let maxGapNear = 0;
    if (mid !== null) {
      for (let k = 1; k < ev.optical.length; k++) {
        const a = isoToDay(ev.optical[k - 1].date), b = isoToDay(ev.optical[k].date);
        if (b >= mid - 20 && a <= mid + 20) maxGapNear = Math.max(maxGapNear, b - a);
      }
    }
    const windows = ev.patches.map((p) => p.changeWindow).filter((w): w is HarvestWindow => !!w);
    const mids = windows.map((w) => (isoToDay(w.start) + isoToDay(w.end)) / 2);
    const spread = mids.length > 1 ? Math.max(...mids) - Math.min(...mids) : 0;

    const reasons: ReviewReason[] = [];
    if (maxGapNear > DEFAULT_THRESHOLDS.maxGapAtDrop) reasons.push('observation_gap');
    if (feat.radarDrop !== null && feat.radarDrop < DEFAULT_THRESHOLDS.minRadarDrop) reasons.push('sensor_disagreement');
    if (spread > 14) reasons.push('within_field_variation');

    const coverage: Field['coverage'] =
      feat.clearObs < DEFAULT_THRESHOLDS.minClearObs || mid === null ? 'insufficient' : maxGapNear > DEFAULT_THRESHOLDS.maxGapAtDrop ? 'gap' : 'sufficient';

    return {
      id,
      polygon: f.ring,
      sourceId: f.dwrId,
      centroid: ringCentroid(f.ring),
      acres,
      cluster: f.cluster,
      // No trained model yet. The ML slot stays empty until one is plugged in.
      prediction: { source: 'ml', predicted: 'unclassified', confidence: null, pSilage: null, modelVersion: 'none' },
      reasons,
      harvestWindow: feat.dropStart && feat.dropEnd ? { start: feat.dropStart, end: feat.dropEnd, gapDays: feat.gapAtDrop ?? 0 } : null,
      coverage,
      maxGapNearHarvest: maxGapNear,
      evidenceSummary: rule.explanation,
      seed: 1000 + i * 17,
    };
  });

  // Audit sample: every 6th confidently rule-labelled field, so confident calls also get checked.
  drafts
    .filter((d) => d.rule.label !== 'abstain' && (d.rule.prediction.confidence ?? 0) >= 0.9)
    .filter((_, i) => i % 6 === 2)
    .forEach((d) => fields.find((f) => f.id === d.id)!.reasons.push('audit_sample'));

  // Showcase: real examples of each situation, picked from the data rather than staged.
  const byId = new Map(fields.map((f) => [f.id, f]));
  const tag = (id: string | undefined, label: string) => {
    const f = id ? byId.get(id) : undefined;
    if (f && !f.showcase) f.showcase = label;
  };
  const conf = (d: (typeof drafts)[number]) => d.rule.prediction.confidence ?? 0;
  const of = (lbl: string) => drafts.filter((d) => d.rule.label === lbl).sort((a, b) => conf(b) - conf(a));
  tag(of('silage')[0]?.id, 'Clear silage transition');
  tag(of('grain')[0]?.id, 'Clear grain dry-down');
  tag(drafts.find((d) => d.rule.abstain === 'gap')?.id, 'Long cloud gap at harvest');
  tag(drafts.find((d) => d.rule.abstain === 'sensor_conflict')?.id, 'Radar does not confirm harvest');
  tag(drafts.find((d) => d.rule.abstain === 'in_between')?.id, 'Between silage and grain');
  tag(fields.find((f) => f.reasons.includes('within_field_variation'))?.id, 'Harvested at different dates');
  tag([...drafts].sort((a, b) => b.acres - a.acres).find((d) => conf(d) < 0.9)?.id, 'Large uncertain field');

  return { fields, evidence, seedDecisions: [] };
}

export const DATASET: Dataset = build();
export const LATEST_OBSERVATION =
  [...DATASET.evidence.values()].flatMap((e) => e.optical.map((o) => o.date)).sort().pop() ?? null;
