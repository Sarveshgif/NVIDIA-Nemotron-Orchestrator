import { useEffect, useState } from 'react';
import { AlertTriangle, ChevronDown, ChevronLeft, ChevronRight, Loader2, X } from 'lucide-react';
import type { Field, FieldEvidence } from '../types';
import { allocate } from '../lib/acreage';
import { CLASS_LABEL, REASON_META, SOURCE_LABEL, fmtAcres, fmtPct } from '../lib/format';
import { PRIORITY_EXPLAINER } from '../lib/priority';
import { getFieldEvidence, peekEvidence, prefetchEvidence } from '../services/analysisService';
import { useStore } from '../state/store';
import type { Derived } from '../state/derived';
import { ConfidenceMeter, InfoTip, StatusPill, TypePill } from './ui';
import { EvidenceImagery } from './EvidenceImagery';
import { EvidenceCharts } from './EvidenceCharts';
import { PatchView } from './PatchView';
import { BarcInvestigation } from './BarcInvestigation';
import { DecisionPanel } from './DecisionPanel';

function useEvidence(fieldId: string) {
  const [state, setState] = useState<{ ev: FieldEvidence | null; error: string | null }>(() => ({ ev: peekEvidence(fieldId) ?? null, error: null }));
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let live = true;
    const cached = peekEvidence(fieldId);
    setState({ ev: cached ?? null, error: null });
    if (!cached) {
      getFieldEvidence(fieldId)
        .then((ev) => live && setState({ ev, error: null }))
        .catch((e) => live && setState({ ev: null, error: e instanceof Error ? e.message : 'Could not load evidence.' }));
    }
    return () => {
      live = false;
    };
  }, [fieldId, attempt]);
  return { ...state, retry: () => setAttempt((a) => a + 1) };
}

export function neighbours(derived: Derived, id: string) {
  const q = derived.queue;
  const i = q.findIndex((f) => f.id === id);
  if (i < 0) return { prev: null, next: q[0] && q[0].id !== id ? q[0] : null };
  return { prev: q[i - 1] ?? null, next: q[i + 1] ?? null };
}

export function FieldDrawer({ field, derived }: { field: Field; derived: Derived }) {
  const select = useStore((s) => s.select);
  const decision = useStore((s) => s.decisions[field.id]);
  const queueTab = useStore((s) => s.queueTab);
  const { ev, error, retry } = useEvidence(field.id);
  const [patchOpen, setPatchOpen] = useState(false);
  const al = allocate(field, decision);
  const pri = derived.pri.get(field.id)!;
  const { prev, next } = neighbours(derived, field.id);
  const [frozenNext, setFrozenNext] = useState<string | null>(null);

  // Remember the next field at open time; saving can remove the current field from the "needs review" list.
  useEffect(() => {
    setFrozenNext(next?.id ?? null);
    if (next) prefetchEvidence([next.id]);
    setPatchOpen(field.reasons.includes('within_field_variation') || !!field.subregions);
  }, [field.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const goNext = () => {
    const target = next?.id ?? frozenNext;
    if (target && target !== field.id) select(target);
  };
  const p = field.prediction;

  return (
    <aside className="drawer" aria-label={`Field ${field.id} details`}>
      <header className="drawer-head">
        <div className="drawer-title">
          <h2>{field.id}</h2>
          <span className="drawer-acres">{fmtAcres(field.acres)} ac</span>
          {field.showcase && <span className="tag">{field.showcase}</span>}
        </div>
        <div className="drawer-nav">
          <button type="button" className="icon-btn" disabled={!prev} onClick={() => prev && select(prev.id)} aria-label="Previous field in queue" title="Previous field">
            <ChevronLeft size={17} />
          </button>
          <button type="button" className="btn btn-sm" disabled={!next && !frozenNext} onClick={goNext} title="Next field in queue (N)">
            Next field <ChevronRight size={15} />
          </button>
          <button type="button" className="icon-btn" onClick={() => select(null)} aria-label="Close field details (Esc)" title="Close (Esc)">
            <X size={17} />
          </button>
        </div>
      </header>

      <div className="drawer-scroll">
        <section className="pvd" aria-label="Prediction and decision">
          <div className="pvd-col">
            <div className="pvd-label">{SOURCE_LABEL[p.source]} prediction</div>
            <div className="pvd-value">
              <TypePill cls={p.predicted} source="model" />
              <ConfidenceMeter value={p.confidence} />
            </div>
            <div className="fine">
              {p.source === 'rules'
                ? `Score from rule set ${p.modelVersion}: how far the field sits inside the rule thresholds.`
                : `Score from model ${p.modelVersion}.`} High scores can still be wrong.
            </div>
          </div>
          <div className="pvd-arrow" aria-hidden>→</div>
          <div className="pvd-col">
            <div className="pvd-label">Analyst decision</div>
            <div className="pvd-value">
              {decision ? (
                decision.decided ? <TypePill cls={decision.decided} source="analyst" /> : <span className="pill pill-unresolved">Insufficient</span>
              ) : (
                <span className="muted">None yet</span>
              )}
              <StatusPill status={al.status} />
            </div>
            <div className="fine">
              Counts as {al.effective === 'unresolved' ? 'unresolved' : CLASS_LABEL[al.effective].toLowerCase()} in totals
              {al.source === 'model' ? ` (${SOURCE_LABEL[p.source].toLowerCase()})` : ' (reviewed)'}.
            </div>
          </div>
        </section>

        {field.reasons.length > 0 && (
          <section className="why" aria-labelledby="why-h">
            <div className="why-head">
              <h3 id="why-h"><AlertTriangle size={15} aria-hidden /> Why this field needs review</h3>
              {!decision && (
                <span className="why-score">
                  Priority {pri.score}
                  <InfoTip label="How priority is calculated">
                    {PRIORITY_EXPLAINER}
                    <span className="pri-break">
                      Uncertainty {pri.uncertainty} · Acres at stake {pri.acreage} · Evidence issues {pri.evidence}
                      {pri.audit ? ` · Audit ${pri.audit}` : ''}
                    </span>
                  </InfoTip>
                </span>
              )}
            </div>
            <ul>
              {field.reasons.map((r) => (
                <li key={r}>
                  <b>{REASON_META[r].label}.</b> <span className="muted">{REASON_META[r].explain}</span>
                </li>
              ))}
            </ul>
          </section>
        )}

        {field.barc && <BarcInvestigation field={field} />}
        <section className="summary" aria-label="Evidence summary">
          <h3>Evidence summary</h3>
          <p>{field.evidenceSummary}</p>
          <div className="summary-facts">
            <span>Est. harvest window <b>{field.harvestWindow ? `${new Date(field.harvestWindow.start + 'T00:00:00Z').toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' })} – ${new Date(field.harvestWindow.end + 'T00:00:00Z').toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' })}` : 'unknown'}</b></span>
            <span>Coverage <b className={`cov-${field.coverage}`}>{field.coverage === 'sufficient' ? 'Sufficient' : field.coverage === 'gap' ? `Gap of ${field.maxGapNearHarvest} days` : 'Insufficient'}</b></span>
            <span>P(silage) <b>{fmtPct(p.pSilage)}</b></span>
          </div>
        </section>

        {error ? (
          <div className="state-box" role="alert">
            <p>Evidence for {field.id} could not be loaded. {error}</p>
            <button type="button" className="btn" onClick={retry}>Try again</button>
          </div>
        ) : !ev ? (
          <div className="state-box loading" aria-busy="true">
            <Loader2 size={18} className="spin" aria-hidden /> Loading evidence…
          </div>
        ) : (
          <>
            {!field.barc && <EvidenceImagery field={field} evidence={ev} />}
            {!field.barc && <EvidenceCharts field={field} evidence={ev} />}
            <section className="ev-block">
              <button type="button" className="disclosure" aria-expanded={patchOpen} onClick={() => setPatchOpen((o) => !o)}>
                <ChevronDown size={16} className={patchOpen ? 'rot' : ''} aria-hidden />
                <h3>Within-field variation</h3>
                {field.reasons.includes('within_field_variation') && <span className="tag tag-warn">Flagged</span>}
                {field.subregions && <span className="tag">Reviewed subregions</span>}
              </button>
              {patchOpen && !field.barc && <PatchView field={field} evidence={ev} />}
            </section>
          </>
        )}
      </div>

      <DecisionPanel field={field} onSaveNext={goNext} hasNext={!!(next || frozenNext)} />
      {queueTab === 'needs' && !decision && <span className="sr-only">Saving removes this field from the Needs review list.</span>}
    </aside>
  );
}
