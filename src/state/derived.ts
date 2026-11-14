import { useMemo } from 'react';
import type { Decision, Field } from '../types';
import { allocate, summarize } from '../lib/acreage';
import { confidenceBand } from '../lib/format';
import { priority, type PriorityBreakdown } from '../lib/priority';
import { EMPTY_FILTERS, useStore, type Filters, type QueueTab } from './store';

export function matchesFilters(f: Field, d: Decision | undefined, flt: Filters): boolean {
  if (flt.search) {
    const q = flt.search.trim().toLowerCase();
    if (q && !f.id.toLowerCase().includes(q) && !(f.showcase ?? '').toLowerCase().includes(q)) return false;
  }
  const al = allocate(f, d);
  if (flt.types.length && !flt.types.includes(al.effective)) return false;
  if (flt.statuses.length && !flt.statuses.includes(al.status)) return false;
  if (flt.confidence.length && !flt.confidence.includes(confidenceBand(f.prediction.confidence))) return false;
  if (flt.reasons.length && !flt.reasons.some((r) => f.reasons.includes(r))) return false;
  const min = parseFloat(flt.minAcres);
  const max = parseFloat(flt.maxAcres);
  if (!Number.isNaN(min) && f.acres < min) return false;
  if (!Number.isNaN(max) && f.acres > max) return false;
  return true;
}

export const filtersActive = (f: Filters) => JSON.stringify(f) !== JSON.stringify(EMPTY_FILTERS);

export function inTab(f: Field, d: Decision | undefined, tab: QueueTab) {
  if (tab === 'needs') return !d && f.reasons.length > 0;
  if (tab === 'reviewed') return !!d;
  return true;
}

export function useDerived() {
  const fields = useStore((s) => s.fields);
  const decisions = useStore((s) => s.decisions);
  const filters = useStore((s) => s.filters);
  const queueTab = useStore((s) => s.queueTab);
  const sort = useStore((s) => s.sort);

  return useMemo(() => {
    const byId = new Map(fields.map((f) => [f.id, f]));
    const pri = new Map<string, PriorityBreakdown>(fields.map((f) => [f.id, priority(f)]));
    const filtered = fields.filter((f) => matchesFilters(f, decisions[f.id], filters));
    const filteredIds = new Set(filtered.map((f) => f.id));
    const dir = sort.dir === 'asc' ? 1 : -1;
    const queue = filtered
      .filter((f) => inTab(f, decisions[f.id], queueTab))
      .sort((a, b) => {
        let v = 0;
        if (sort.key === 'priority') v = pri.get(a.id)!.score - pri.get(b.id)!.score;
        else if (sort.key === 'acres') v = a.acres - b.acres;
        else if (sort.key === 'confidence') v = (a.prediction.confidence ?? 0) - (b.prediction.confidence ?? 0);
        else v = a.id.localeCompare(b.id);
        return v * dir || a.id.localeCompare(b.id);
      });
    const tabCounts = {
      needs: filtered.filter((f) => inTab(f, decisions[f.id], 'needs')).length,
      all: filtered.length,
      reviewed: filtered.filter((f) => inTab(f, decisions[f.id], 'reviewed')).length,
    };
    return {
      byId,
      pri,
      filtered,
      filteredIds,
      queue,
      tabCounts,
      summaryAll: summarize(fields, decisions),
      summaryFiltered: summarize(filtered, decisions),
      active: filtersActive(filters),
    };
  }, [fields, decisions, filters, queueTab, sort]);
}

export type Derived = ReturnType<typeof useDerived>;
