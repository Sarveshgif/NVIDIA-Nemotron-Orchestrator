import type { Field } from '../types';

export const PRIORITY_EXPLAINER =
  'Ranked by uncertainty (50%), acres at stake (30%) and evidence problems such as gaps, sensor disagreement or variation (20%). Audit samples get a small boost so they are not buried.';

export interface PriorityBreakdown {
  score: number;
  uncertainty: number;
  acreage: number;
  evidence: number;
  audit: number;
}

const MAX_ACRES = 300;

export function priority(f: Field): PriorityBreakdown {
  const c = f.prediction.confidence;
  const u = c === null ? 1 : Math.min(1, (1 - c) / 0.45);
  const a = Math.min(1, f.acres / MAX_ACRES);
  let e = 0;
  if (f.reasons.includes('observation_gap')) e += 0.5;
  if (f.reasons.includes('sensor_disagreement')) e += 0.6;
  if (f.reasons.includes('within_field_variation')) e += 0.5;
  e = Math.min(1, e);
  const audit = f.reasons.includes('audit_sample') ? 12 : 0;
  const uncertainty = 50 * u;
  const acreage = 30 * a * Math.max(u, 0.25);
  const evidence = 20 * e;
  return {
    score: Math.round(uncertainty + acreage + evidence + audit),
    uncertainty: Math.round(uncertainty),
    acreage: Math.round(acreage),
    evidence: Math.round(evidence),
    audit,
  };
}

export function priorityBand(score: number): 'high' | 'medium' | 'low' {
  if (score >= 45) return 'high';
  if (score >= 25) return 'medium';
  return 'low';
}
