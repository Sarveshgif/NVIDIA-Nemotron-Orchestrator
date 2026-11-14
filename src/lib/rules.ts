// Rule-based harvest-type labelling from a field's satellite time series.
//
// Used two ways:
//  1. Weak labels: rules only label clear cases and abstain otherwise. Abstentions are the
//     point — a small set of confident labels is better training data than a full noisy one.
//  2. Predictions: until a trained ML model is ready, the same features give every field a
//     rule-based call. In-between fields get a low-confidence lean so they land in review.
//
// Physical basis: silage is chopped while still green (NDVI and moisture high right up to an
// abrupt drop); grain corn dries down for weeks before harvest (NDVI already low at the drop).

import type { Field, FieldEvidence, Prediction, ReviewReason } from '../types';
import { isoToDay, season } from '../data/season';

export const RULESET_VERSION = 'rules-v1';

export interface RuleThresholds {
  /** Silage rule: NDVI at the last clear look before the drop is at least this. */
  silageMinNdvi: number;
  /** Silage rule: dry-down before the drop is at most this many days. */
  silageMaxDryDown: number;
  /** Grain rule: dry-down before the drop is at least this many days. */
  grainMinDryDown: number;
  /** Grain rule: NDVI before the drop is at most this. */
  grainMaxNdvi: number;
  /** Abstain: fewer clear optical observations than this in the season. */
  minClearObs: number;
  /** Abstain: the clear-sky gap spanning the drop is longer than this. */
  maxGapAtDrop: number;
  /** Abstain: radar VH falls by less than this (dB) across the optical harvest window. */
  minRadarDrop: number;
}

export const DEFAULT_THRESHOLDS: RuleThresholds = {
  silageMinNdvi: 0.6,
  silageMaxDryDown: 14,
  grainMinDryDown: 25,
  grainMaxNdvi: 0.55,
  minClearObs: 14,
  maxGapAtDrop: 15,
  minRadarDrop: 1.5,
};

export interface RuleFeatures {
  clearObs: number;
  peakNdvi: number | null;
  /** Date of the last clear look before the harvest drop. */
  dropStart: string | null;
  dropEnd: string | null;
  /** Days between the two clear looks spanning the drop. */
  gapAtDrop: number | null;
  ndviBeforeDrop: number | null;
  lswiBeforeDrop: number | null;
  /** Days of continuous decline (over a third of the way from summer peak to bare soil) before the drop. 0 = green up to harvest. */
  dryDownDays: number | null;
  /** Mean VH of the two passes before the drop minus the two after (dB). Positive = canopy removed. */
  radarDrop: number | null;
}

export type WeakLabel = 'silage' | 'grain' | 'abstain';

export type AbstainReason = 'few_obs' | 'gap' | 'sensor_conflict' | 'in_between' | 'no_drop';

export const ABSTAIN_LABEL: Record<AbstainReason, string> = {
  few_obs: 'Too few clear observations',
  gap: 'Cloud gap spans the harvest drop',
  sensor_conflict: 'Radar does not confirm the harvest',
  in_between: 'Between the silage and grain rules',
  no_drop: 'No harvest drop found',
};

export interface RuleResult {
  fieldId: string;
  features: RuleFeatures;
  label: WeakLabel;
  abstain?: AbstainReason;
  /** Rule that produced the label, e.g. "Silage: green cut". */
  rule: string | null;
  prediction: Prediction;
  explanation: string;
}

// Dry-down = NDVI has lost more than this share of the way from the summer peak to bare soil (the first
// look after the drop). Relative, so dense canopies (peak ~0.98) and thinner ones (~0.85) are treated alike;
// normal late-season leaf ageing in a green-cut field stays well under a third.
const DECLINE_SHARE = 1 / 3;
// Summer (corn) cycle only. Many Tulare fields are double-cropped: a winter cereal is green in spring and
// cut in May, and the next winter crop greens up from November. Those cycles must not be read as corn.
const r2 = (v: number) => Math.round(v * 100) / 100;
const fmtD = (iso: string) =>
  new Date(iso + 'T00:00:00Z').toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });

export function extractFeatures(ev: FieldEvidence): RuleFeatures {
  const opt = [...ev.optical].sort((a, b) => a.date.localeCompare(b.date));
  const f: RuleFeatures = {
    clearObs: opt.length, peakNdvi: null, dropStart: null, dropEnd: null, gapAtDrop: null,
    ndviBeforeDrop: null, lswiBeforeDrop: null, dryDownDays: null, radarDrop: null,
  };
  if (opt.length < 3) return f;
  const S = season();
  const PEAK_FROM = `${S.year}-${S.peakFrom}`, PEAK_TO = `${S.year}-${S.peakTo}`, DROP_TO = `${S.year}-${S.dropTo}`;
  let pk = -1;
  opt.forEach((o, i) => {
    if (o.date >= PEAK_FROM && o.date <= PEAK_TO && (pk < 0 || o.ndvi > opt[pk].ndvi)) pk = i;
  });
  if (pk < 0) return f;
  f.peakNdvi = r2(opt[pk].ndvi);

  // Harvest drop = largest NDVI fall between consecutive clear looks after the summer peak.
  let best = -1, bestFall = 0.1;
  for (let i = pk + 1; i < opt.length && opt[i].date <= DROP_TO; i++) {
    const fall = opt[i - 1].ndvi - opt[i].ndvi;
    if (fall > bestFall) { bestFall = fall; best = i; }
  }
  if (best < 0) return f;
  const before = opt[best - 1], after = opt[best];
  f.dropStart = before.date;
  f.dropEnd = after.date;
  f.gapAtDrop = isoToDay(after.date) - isoToDay(before.date);
  f.ndviBeforeDrop = r2(before.ndvi);
  f.lswiBeforeDrop = r2(before.lswi);
  // Continuous dry-down: walk back from the drop while NDVI stays below the dry-down line.
  const line = opt[pk].ndvi - (opt[pk].ndvi - after.ndvi) * DECLINE_SHARE;
  let k = best - 1;
  while (k - 1 > pk && opt[k - 1].ndvi < line) k--;
  f.dryDownDays = before.ndvi >= line ? 0 : isoToDay(before.date) - isoToDay(opt[k].date);

  // Radar check across the same window: up to two passes within 30 days on each side.
  const rad = [...ev.radar].sort((a, b) => a.date.localeCompare(b.date));
  const s0 = isoToDay(before.date), s1 = isoToDay(after.date);
  const pre = rad.filter((r) => r.date <= before.date && s0 - isoToDay(r.date) <= 30).slice(-2);
  const post = rad.filter((r) => r.date >= after.date && isoToDay(r.date) - s1 <= 30).slice(0, 2);
  if (pre.length && post.length) {
    const mean = (xs: typeof rad) => xs.reduce((a, r) => a + r.vh, 0) / xs.length;
    f.radarDrop = Math.round((mean(pre) - mean(post)) * 10) / 10;
  }
  return f;
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

export function applyRules(fieldId: string, f: RuleFeatures, th: RuleThresholds): RuleResult {
  const abstain = (reason: AbstainReason, lean?: { cls: 'silage' | 'grain'; text: string }): RuleResult => {
    const pred: Prediction = lean
      ? { source: 'rules', predicted: lean.cls, confidence: 0.55, pSilage: lean.cls === 'silage' ? 0.55 : 0.45, modelVersion: RULESET_VERSION }
      : { source: 'rules', predicted: 'unclassified', confidence: null, pSilage: null, modelVersion: RULESET_VERSION };
    return {
      fieldId, features: f, label: 'abstain', abstain: reason, rule: null, prediction: pred,
      explanation: `No weak label: ${ABSTAIN_LABEL[reason].toLowerCase()}.${lean ? ` ${lean.text}` : ' Needs an analyst.'}`,
    };
  };

  if (f.clearObs < th.minClearObs) return abstain('few_obs');
  if (f.ndviBeforeDrop === null || f.dryDownDays === null || !f.dropStart || !f.dropEnd) return abstain('no_drop');
  const when = `between ${fmtD(f.dropStart)} and ${fmtD(f.dropEnd)}`;
  if ((f.gapAtDrop ?? 0) > th.maxGapAtDrop) return abstain('gap');
  if (f.radarDrop !== null && f.radarDrop < th.minRadarDrop) return abstain('sensor_conflict');

  const ndvi = f.ndviBeforeDrop, dd = f.dryDownDays;
  if (ndvi >= th.silageMinNdvi && dd <= th.silageMaxDryDown) {
    const m = Math.min(clamp01((ndvi - th.silageMinNdvi) / 0.2), clamp01((th.silageMaxDryDown - dd) / Math.max(1, th.silageMaxDryDown)));
    const conf = r2(0.7 + 0.27 * m);
    return {
      fieldId, features: f, label: 'silage', rule: 'Silage: green cut',
      prediction: { source: 'rules', predicted: 'silage', confidence: conf, pSilage: conf, modelVersion: RULESET_VERSION },
      explanation: `Still green (NDVI ${ndvi.toFixed(2)}) at the last clear look, then an abrupt drop ${when} after ${dd} days of decline. Matches the silage rule (NDVI ≥ ${th.silageMinNdvi.toFixed(2)}, dry-down ≤ ${th.silageMaxDryDown} d).`,
    };
  }
  if (dd >= th.grainMinDryDown && ndvi <= th.grainMaxNdvi) {
    const m = Math.min(clamp01((dd - th.grainMinDryDown) / 15), clamp01((th.grainMaxNdvi - ndvi) / 0.15));
    const conf = r2(0.7 + 0.27 * m);
    return {
      fieldId, features: f, label: 'grain', rule: 'Grain: long dry-down',
      prediction: { source: 'rules', predicted: 'grain', confidence: conf, pSilage: r2(1 - conf), modelVersion: RULESET_VERSION },
      explanation: `Dried down for ${dd} days (NDVI ${ndvi.toFixed(2)} before the final drop ${when}). Matches the grain rule (dry-down ≥ ${th.grainMinDryDown} d, NDVI ≤ ${th.grainMaxNdvi.toFixed(2)}).`,
    };
  }
  const mid = (th.silageMaxDryDown + th.grainMinDryDown) / 2;
  const cls = dd < mid ? 'silage' : 'grain';
  return abstain('in_between', {
    cls,
    text: `Dry-down of ${dd} days and NDVI ${ndvi.toFixed(2)} before the drop sit between the rules; the prediction leans ${cls} with low confidence.`,
  });
}

const REASON_ORDER: ReviewReason[] = [
  'low_confidence', 'observation_gap', 'sensor_disagreement', 'within_field_variation', 'large_acreage', 'audit_sample',
];

/** Field as seen under a prediction source. Confidence-driven review reasons are recomputed. */
export function withPrediction(base: Field, rule: RuleResult | undefined): Field {
  if (!rule) return base;
  const p = rule.prediction;
  const set = new Set<ReviewReason>(base.reasons.filter((r) => r !== 'low_confidence' && r !== 'large_acreage'));
  if (p.confidence === null || p.confidence < 0.7) set.add('low_confidence');
  if (base.acres >= season().largeAcres && (p.confidence === null || p.confidence < 0.9)) set.add('large_acreage');
  return {
    ...base,
    prediction: p,
    reasons: REASON_ORDER.filter((r) => set.has(r)),
    evidenceSummary: rule.explanation,
  };
}

export function runRules(evidence: Record<string, FieldEvidence>, th: RuleThresholds): Record<string, RuleResult> {
  const out: Record<string, RuleResult> = {};
  for (const [id, ev] of Object.entries(evidence)) out[id] = applyRules(id, extractFeatures(ev), th);
  return out;
}
