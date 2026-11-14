import type { Decision, Field, ReviewStatus } from '../types';

/** The class shown for a field after applying any analyst decision. */
export type EffectiveClass = 'silage' | 'grain' | 'mixed' | 'unresolved';

export interface FieldAllocation {
  silage: number;
  grain: number;
  unresolved: number;
  source: 'model' | 'analyst';
  effective: EffectiveClass;
  status: ReviewStatus;
}

/**
 * Splits one field's acres into silage / grain / unresolved exactly once.
 * Every summary, map style and export derives from this function so totals reconcile.
 */
export function allocate(field: Field, decision?: Decision): FieldAllocation {
  const a = field.acres;
  if (!decision) {
    const p = field.prediction.predicted;
    return {
      silage: p === 'silage' ? a : 0,
      grain: p === 'grain' ? a : 0,
      unresolved: p === 'unclassified' ? a : 0,
      source: 'model',
      effective: p === 'unclassified' ? 'unresolved' : p,
      status: 'pending',
    };
  }
  if (decision.status === 'insufficient' || decision.decided === null) {
    return { silage: 0, grain: 0, unresolved: a, source: 'analyst', effective: 'unresolved', status: decision.status };
  }
  if (decision.decided === 'mixed' && decision.allocation) {
    const { silage, grain } = decision.allocation;
    return {
      silage, grain, unresolved: round1(a - silage - grain),
      source: 'analyst', effective: 'mixed', status: decision.status,
    };
  }
  if (decision.decided === 'mixed') {
    return { silage: 0, grain: 0, unresolved: a, source: 'analyst', effective: 'mixed', status: decision.status };
  }
  return {
    silage: decision.decided === 'silage' ? a : 0,
    grain: decision.decided === 'grain' ? a : 0,
    unresolved: 0,
    source: 'analyst',
    effective: decision.decided,
    status: decision.status,
  };
}

const round1 = (v: number) => Math.round(v * 10) / 10;

export interface AcreageSummary {
  fieldCount: number;
  total: number;
  silage: { model: number; reviewed: number };
  grain: { model: number; reviewed: number };
  unresolved: { model: number; insufficient: number; mixedRemainder: number };
  reviewedFields: number;
  reviewedAcres: number;
  queueFields: number;
  queueRemaining: number;
  changedFields: number;
}

export function summarize(fields: Field[], decisions: Record<string, Decision>): AcreageSummary {
  const s: AcreageSummary = {
    fieldCount: fields.length,
    total: 0,
    silage: { model: 0, reviewed: 0 },
    grain: { model: 0, reviewed: 0 },
    unresolved: { model: 0, insufficient: 0, mixedRemainder: 0 },
    reviewedFields: 0,
    reviewedAcres: 0,
    queueFields: 0,
    queueRemaining: 0,
    changedFields: 0,
  };
  for (const f of fields) {
    const d = decisions[f.id];
    const al = allocate(f, d);
    s.total += f.acres;
    const bucket = al.source === 'model' ? 'model' : 'reviewed';
    s.silage[bucket] += al.silage;
    s.grain[bucket] += al.grain;
    if (al.source === 'model') s.unresolved.model += al.unresolved;
    else if (al.status === 'insufficient') s.unresolved.insufficient += al.unresolved;
    else s.unresolved.mixedRemainder += al.unresolved;
    if (d) {
      s.reviewedFields++;
      s.reviewedAcres += f.acres;
      if (d.status === 'changed') s.changedFields++;
    }
    if (f.reasons.length) {
      s.queueFields++;
      if (!d) s.queueRemaining++;
    }
  }
  return s;
}

export const silageTotal = (s: AcreageSummary) => s.silage.model + s.silage.reviewed;
export const grainTotal = (s: AcreageSummary) => s.grain.model + s.grain.reviewed;
export const unresolvedTotal = (s: AcreageSummary) =>
  s.unresolved.model + s.unresolved.insufficient + s.unresolved.mixedRemainder;

/** True when the three classes add back to the field total (within rounding). */
export const reconciles = (s: AcreageSummary) =>
  Math.abs(silageTotal(s) + grainTotal(s) + unresolvedTotal(s) - s.total) < 0.05;
