import { activeDataset } from '../data/datasets';
import type { DecisionClass, PredictedClass, ReviewReason, ReviewStatus, PredictionSource } from '../types';

export const fmtAcres = (v: number, digits = 1) =>
  v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });

export const fmtPct = (v: number | null) => (v === null ? '—' : `${Math.round(v * 100)}%`);

export const fmtDate = (iso: string, withYear = false) =>
  new Date(iso.length === 10 ? iso + 'T00:00:00Z' : iso).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', ...(withYear ? { year: 'numeric' } : {}), timeZone: 'UTC',
  });

export const fmtDateTime = (iso: string) =>
  new Date(iso).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });

export const daysBetween = (a: string, b: string) =>
  Math.round((Date.parse(b + 'T00:00:00Z') - Date.parse(a + 'T00:00:00Z')) / 864e5);

export const CLASS_LABEL: Record<PredictedClass | DecisionClass | 'unresolved', string> = {
  silage: 'Silage',
  grain: 'Grain',
  mixed: 'Mixed',
  unclassified: 'Unclassified',
  unresolved: 'Unresolved',
};

export const STATUS_LABEL: Record<ReviewStatus, string> = {
  pending: 'Not reviewed',
  accepted: 'Accepted',
  changed: 'Changed',
  insufficient: 'Insufficient evidence',
};

/** True when the active dataset serves trained-model predictions (Netherlands 2025). */
export const mlAvailable = () => activeDataset().mlAvailable;

export const SOURCE_LABEL: Record<PredictionSource, string> = { rules: 'Rule-based', ml: 'ML model' };
export const sourceNote = (src: PredictionSource) => (src === 'rules' ? activeDataset().rulesNote : activeDataset().mlNote);

export const REASON_META: Record<ReviewReason, { label: string; short: string; explain: string }> = {
  low_confidence: {
    label: 'Low prediction confidence',
    short: 'Low confidence',
    explain: 'The score is below 70%, or the field could not be classified (rules abstained or the model returned no class).',
  },
  observation_gap: {
    label: 'Long observation gap near harvest',
    short: 'Observation gap',
    explain: 'More than 15 days passed between clear optical images around the likely harvest.',
  },
  sensor_disagreement: {
    label: 'Radar does not confirm the harvest',
    short: 'Radar unconfirmed',
    explain: 'Sentinel-1 VH backscatter did not fall across the optical harvest window, so the two sensors do not agree that the canopy was removed.',
  },
  within_field_variation: {
    label: 'Strong variation within the field',
    short: 'Within-field variation',
    explain: 'Different parts of the field changed at clearly different times.',
  },
  large_acreage: {
    label: 'Large acreage impact',
    short: 'Large acreage',
    explain: 'At least 100 acres with a demo score below 90%. An error here moves totals the most.',
  },
  audit_sample: {
    label: 'High-confidence audit sample',
    short: 'Audit sample',
    explain: 'Randomly drawn from confident predictions. High confidence does not guarantee correctness.',
  },
};

export const REASON_ORDER: ReviewReason[] = [
  'low_confidence', 'observation_gap', 'sensor_disagreement', 'within_field_variation', 'large_acreage', 'audit_sample',
];

export function confidenceBand(c: number | null): 'none' | 'low' | 'medium' | 'high' {
  if (c === null) return 'none';
  if (c < 0.7) return 'low';
  if (c < 0.85) return 'medium';
  return 'high';
}
