import { useEffect, useRef } from 'react';
import { ArrowDownWideNarrow, ArrowUpNarrowWide, ChevronLeft, ListChecks, Search, SlidersHorizontal, AlertTriangle } from 'lucide-react';
import type { ReviewReason, ReviewStatus } from '../types';
import { allocate, type EffectiveClass } from '../lib/acreage';
import { REASON_META, REASON_ORDER, fmtAcres } from '../lib/format';
import { PRIORITY_EXPLAINER, priorityBand } from '../lib/priority';
import { prefetchEvidence } from '../services/analysisService';
import { useStore, type ConfBand, type QueueTab, type SortKey } from '../state/store';
import type { Derived } from '../state/derived';
import { Chip, ConfidenceMeter, InfoTip, StatusPill, TypePill } from './ui';

const toggle = <T,>(arr: T[], v: T) => (arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v]);

function FilterPanel({ derived }: { derived: Derived }) {
  const f = useStore((s) => s.filters);
  const set = useStore((s) => s.setFilters);
  const clear = useStore((s) => s.clearFilters);
  return (
    <div className="filter-panel" id="queue-filters">
      <fieldset>
        <legend>Harvest type (current)</legend>
        <div className="chips">
          {(['silage', 'grain', 'mixed', 'unresolved'] as EffectiveClass[]).map((t) => (
            <Chip key={t} on={f.types.includes(t)} onClick={() => set({ types: toggle(f.types, t) })}>
              {t[0].toUpperCase() + t.slice(1)}
            </Chip>
          ))}
        </div>
      </fieldset>
      <fieldset>
        <legend>Review status</legend>
        <div className="chips">
          {([['pending', 'Not reviewed'], ['accepted', 'Accepted'], ['changed', 'Changed'], ['insufficient', 'Insufficient']] as [ReviewStatus, string][]).map(([v, l]) => (
            <Chip key={v} on={f.statuses.includes(v)} onClick={() => set({ statuses: toggle(f.statuses, v) })}>{l}</Chip>
          ))}
        </div>
      </fieldset>
      <fieldset>
        <legend>Prediction confidence</legend>
        <div className="chips">
          {([['low', 'Under 70%'], ['medium', '70–84%'], ['high', '85%+'], ['none', 'No score']] as [ConfBand, string][]).map(([v, l]) => (
            <Chip key={v} on={f.confidence.includes(v)} onClick={() => set({ confidence: toggle(f.confidence, v) })}>{l}</Chip>
          ))}
        </div>
      </fieldset>
      <fieldset>
        <legend>Review reason</legend>
        <div className="chips">
          {REASON_ORDER.map((r: ReviewReason) => (
            <Chip key={r} on={f.reasons.includes(r)} onClick={() => set({ reasons: toggle(f.reasons, r) })}>{REASON_META[r].short}</Chip>
          ))}
        </div>
      </fieldset>
      <fieldset>
        <legend>Acreage</legend>
        <div className="range-row">
          <label>
            <span className="sr-only">Minimum acres</span>
            <input id="flt-min" type="number" inputMode="decimal" min={0} placeholder="Min" value={f.minAcres} onChange={(e) => set({ minAcres: e.target.value })} />
          </label>
          <span aria-hidden>–</span>
          <label>
            <span className="sr-only">Maximum acres</span>
            <input id="flt-max" type="number" inputMode="decimal" min={0} placeholder="Max" value={f.maxAcres} onChange={(e) => set({ maxAcres: e.target.value })} />
          </label>
          <span className="muted">ac</span>
        </div>
      </fieldset>
      <div className="filter-foot">
        <span className="muted">{derived.filtered.length} of {derived.byId.size} fields match</span>
        <button type="button" className="link-btn" onClick={clear} disabled={!derived.active}>Clear filters</button>
      </div>
    </div>
  );
}

export function ReviewQueue({ derived }: { derived: Derived }) {
  const queueOpen = useStore((s) => s.queueOpen);
  const toggleQueue = useStore((s) => s.toggleQueue);
  const tab = useStore((s) => s.queueTab);
  const setTab = useStore((s) => s.setQueueTab);
  const filters = useStore((s) => s.filters);
  const setFilters = useStore((s) => s.setFilters);
  const filtersOpen = useStore((s) => s.filtersOpen);
  const toggleFiltersOpen = useStore((s) => s.toggleFiltersOpen);
  const sort = useStore((s) => s.sort);
  const setSort = useStore((s) => s.setSort);
  const selectedId = useStore((s) => s.selectedId);
  const hoveredId = useStore((s) => s.hoveredId);
  const decisions = useStore((s) => s.decisions);
  const select = useStore((s) => s.select);
  const hover = useStore((s) => s.hover);
  const listRef = useRef<HTMLOListElement>(null);

  // keep the selected row visible when selection comes from the map
  useEffect(() => {
    if (!selectedId || !listRef.current) return;
    const row = listRef.current.querySelector<HTMLElement>(`[data-id="${selectedId}"]`);
    row?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [selectedId]);

  // warm the evidence cache for the top of the queue
  useEffect(() => {
    prefetchEvidence(derived.queue.slice(0, 8).map((f) => f.id));
  }, [derived.queue]);

  const activeCount =
    filters.types.length + filters.statuses.length + filters.confidence.length + filters.reasons.length +
    (filters.minAcres ? 1 : 0) + (filters.maxAcres ? 1 : 0);

  if (!queueOpen) {
    return (
      <aside className="queue queue-collapsed" aria-label="Review queue (collapsed)">
        <button type="button" className="queue-rail-btn" onClick={() => toggleQueue(true)} aria-label="Open review queue">
          <ListChecks size={18} aria-hidden />
          <span className="queue-rail-count">{derived.tabCounts.needs}</span>
          <span className="queue-rail-label">Queue</span>
        </button>
      </aside>
    );
  }

  const onKeyNav = (e: React.KeyboardEvent<HTMLOListElement>) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    const btns = Array.from(listRef.current?.querySelectorAll<HTMLButtonElement>('button.q-row') ?? []);
    const i = btns.indexOf(document.activeElement as HTMLButtonElement);
    const next = btns[e.key === 'ArrowDown' ? Math.min(btns.length - 1, i + 1) : Math.max(0, i - 1)];
    if (next) {
      e.preventDefault();
      next.focus();
    }
  };

  return (
    <aside className="queue" aria-label="Review queue">
      <header className="queue-head">
        <div className="queue-title">
          <h2>Review queue</h2>
          <InfoTip label="How the queue is ranked">{PRIORITY_EXPLAINER}</InfoTip>
        </div>
        <button type="button" className="icon-btn" onClick={() => toggleQueue(false)} aria-label="Collapse review queue">
          <ChevronLeft size={16} />
        </button>
      </header>

      <div className="queue-tabs" role="tablist" aria-label="Queue view">
        {([['needs', 'Needs review'], ['all', 'All fields'], ['reviewed', 'Reviewed']] as [QueueTab, string][]).map(([v, l]) => (
          <button key={v} role="tab" type="button" aria-selected={tab === v} className={tab === v ? 'on' : ''} onClick={() => setTab(v)}>
            {l} <span className="count">{derived.tabCounts[v]}</span>
          </button>
        ))}
      </div>

      <div className="queue-tools">
        <label className="search">
          <Search size={14} aria-hidden />
          <span className="sr-only">Search fields</span>
          <input
            id="queue-search"
            type="search"
            placeholder="Search field ID"
            value={filters.search}
            onChange={(e) => setFilters({ search: e.target.value })}
          />
        </label>
        <button
          type="button"
          className={`tool-btn ${filtersOpen || activeCount ? 'on' : ''}`}
          onClick={toggleFiltersOpen}
          aria-expanded={filtersOpen}
          aria-controls="queue-filters"
        >
          <SlidersHorizontal size={14} aria-hidden /> Filters{activeCount ? ` · ${activeCount}` : ''}
        </button>
      </div>
      <div className="queue-sort">
        <label>
          Sort
          <select id="queue-sort" value={sort.key} onChange={(e) => setSort(e.target.value as SortKey)}>
            <option value="priority">Priority</option>
            <option value="acres">Acreage</option>
            <option value="confidence">Confidence</option>
            <option value="id">Field ID</option>
          </select>
        </label>
        <button type="button" className="icon-btn icon-btn-sm" onClick={() => setSort(sort.key)} aria-label={`Sort ${sort.dir === 'asc' ? 'ascending' : 'descending'}; click to reverse`} title="Reverse order">
          {sort.dir === 'asc' ? <ArrowUpNarrowWide size={15} /> : <ArrowDownWideNarrow size={15} />}
        </button>
        <span className="muted queue-count">{derived.queue.length} shown</span>
      </div>

      {filtersOpen && <FilterPanel derived={derived} />}

      <ol className="q-list" ref={listRef} onKeyDown={onKeyNav}>
        {derived.queue.map((f) => {
          const d = decisions[f.id];
          const al = allocate(f, d);
          const pri = derived.pri.get(f.id)!;
          const band = d ? 'done' : f.reasons.length ? priorityBand(pri.score) : 'none';
          return (
            <li key={f.id}>
              <button
                type="button"
                data-id={f.id}
                className={`q-row ${selectedId === f.id ? 'sel' : ''} ${hoveredId === f.id ? 'hov' : ''}`}
                aria-current={selectedId === f.id}
                onClick={() => select(f.id)}
                onMouseEnter={() => hover(f.id)}
                onMouseLeave={() => hover(null)}
                onFocus={() => prefetchEvidence([f.id])}
              >
                <span className={`pri pri-${band}`} title={d ? 'Reviewed' : `Priority score ${pri.score}`}>
                  {d ? '✓' : f.reasons.length ? pri.score : '–'}
                </span>
                <span className="q-main">
                  <span className="q-line1">
                    <span className="q-id">{f.id}</span>
                    <span className="q-acres">{fmtAcres(f.acres)} ac</span>
                  </span>
                  <span className="q-line2">
                    <TypePill cls={f.prediction.predicted} source="model" />
                    <ConfidenceMeter value={f.prediction.confidence} />
                    {d && d.status === 'changed' && <TypePill cls={al.effective} source="analyst" compact />}
                  </span>
                  <span className="q-line3">
                    {f.reasons.length ? (
                      <span className="q-reason">
                        <AlertTriangle size={12} aria-hidden />
                        {REASON_META[f.reasons[0]].short}
                        {f.reasons.length > 1 && <span className="muted"> +{f.reasons.length - 1}</span>}
                      </span>
                    ) : (
                      <span className="muted">No review flags</span>
                    )}
                    <StatusPill status={al.status} />
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
      {derived.queue.length === 0 && (
        <div className="empty">
          {derived.active ? (
            <>
              <p>No fields match these filters.</p>
              <button type="button" className="btn btn-quiet" onClick={() => useStore.getState().clearFilters()}>Clear filters</button>
            </>
          ) : tab === 'needs' ? (
            <>
              <p><strong>Queue cleared.</strong> Every flagged field has a decision.</p>
              <button type="button" className="btn btn-quiet" onClick={() => setTab('reviewed')}>See reviewed fields</button>
            </>
          ) : (
            <p>No fields in this view yet.</p>
          )}
        </div>
      )}
    </aside>
  );
}
