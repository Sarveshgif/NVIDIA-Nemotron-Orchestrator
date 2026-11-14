import { useEffect, useState } from 'react';
import { Ban, Check, Leaf, Pencil, RotateCcw, Split, Wheat } from 'lucide-react';
import type { Allocation, DecisionClass, Field } from '../types';
import { CLASS_LABEL, STATUS_LABEL, fmtAcres, fmtDateTime } from '../lib/format';
import { useStore } from '../state/store';
import { TypePill } from './ui';

type Choice = 'accept' | 'silage' | 'grain' | 'mixed' | 'insufficient';

export function DecisionPanel({ field, onSaveNext, hasNext }: { field: Field; onSaveNext: () => void; hasNext: boolean }) {
  const decision = useStore((s) => s.decisions[field.id]);
  const record = useStore((s) => s.record);
  const clearDecision = useStore((s) => s.clearDecision);
  const [editing, setEditing] = useState(!decision);
  const [choice, setChoice] = useState<Choice | null>(null);
  const [note, setNote] = useState(decision?.note ?? '');
  const [alloc, setAlloc] = useState({ silage: '', grain: '', unresolved: '' });
  const [touched, setTouched] = useState(false);

  useEffect(() => {
    setEditing(!decision);
    setChoice(null);
    setNote(decision?.note ?? '');
    setAlloc(
      decision?.allocation
        ? { silage: String(decision.allocation.silage), grain: String(decision.allocation.grain), unresolved: String(decision.allocation.unresolved) }
        : { silage: '', grain: '', unresolved: '' },
    );
    setTouched(false);
  }, [field.id, decision]);

  const pred = field.prediction.predicted;
  const n = (v: string) => (v.trim() === '' ? 0 : Number(v));
  const sum = n(alloc.silage) + n(alloc.grain) + n(alloc.unresolved);
  const allocValid =
    [alloc.silage, alloc.grain, alloc.unresolved].every((v) => v.trim() === '' || (!Number.isNaN(Number(v)) && Number(v) >= 0)) &&
    Math.abs(sum - field.acres) < 0.05 && n(alloc.silage) + n(alloc.grain) > 0;
  const canSave = choice !== null && (choice !== 'mixed' || allocValid);

  const save = (next: boolean) => {
    setTouched(true);
    if (!canSave || !choice) return;
    let decided: DecisionClass | null;
    let status: 'accepted' | 'changed' | 'insufficient';
    let allocation: Allocation | undefined;
    if (choice === 'accept') {
      decided = pred as DecisionClass;
      status = 'accepted';
    } else if (choice === 'insufficient') {
      decided = null;
      status = 'insufficient';
    } else {
      decided = choice;
      status = choice === pred ? 'accepted' : 'changed';
      if (choice === 'mixed') {
        allocation = {
          silage: n(alloc.silage), grain: n(alloc.grain), unresolved: Math.round((field.acres - n(alloc.silage) - n(alloc.grain)) * 10) / 10,
          source: field.subregions && n(alloc.silage) === field.subregions.find((s) => s.type === 'silage')?.acres ? 'subregions' : 'analyst',
        };
      }
    }
    record(field.id, { status, decided, allocation, note });
    if (next) onSaveNext();
  };

  if (decision && !editing) {
    return (
      <section className="decision decision-done" aria-label="Analyst decision">
        <div className="dec-current">
          <div>
            <div className="dec-label">Analyst decision</div>
            <div className="dec-value">
              {decision.decided ? <TypePill cls={decision.decided} source="analyst" /> : <span className="pill pill-unresolved"><Ban size={12} /> Insufficient evidence</span>}
              <span className="muted small">{STATUS_LABEL[decision.status]} by {decision.analyst} · {fmtDateTime(decision.at)}</span>
            </div>
            {decision.allocation && (
              <div className="small dec-alloc">
                Silage {fmtAcres(decision.allocation.silage)} ac · Grain {fmtAcres(decision.allocation.grain)} ac
                {decision.allocation.unresolved > 0 && ` · Unresolved ${fmtAcres(decision.allocation.unresolved)} ac`}
                <span className="muted"> ({decision.allocation.source === 'subregions' ? 'from reviewed subregions' : 'analyst-entered proportions'})</span>
              </div>
            )}
            {decision.note && <blockquote className="dec-note">{decision.note}</blockquote>}
          </div>
        </div>
        <div className="dec-actions">
          <button type="button" className="btn btn-quiet" onClick={() => setEditing(true)}><Pencil size={14} /> Edit decision</button>
          <button type="button" className="btn btn-quiet" onClick={() => clearDecision(field.id)}><RotateCcw size={14} /> Return to queue</button>
          {hasNext && <button type="button" className="btn btn-primary" onClick={onSaveNext}>Next field</button>}
        </div>
        <p className="fine">Reviewed means an analyst checked the evidence. It is not independently verified ground truth.</p>
      </section>
    );
  }

  const options: { v: Choice; label: string; sub: string; icon: JSX.Element }[] = [];
  if (pred !== 'unclassified') {
    options.push({ v: 'accept', label: `Accept ${CLASS_LABEL[pred].toLowerCase()}`, sub: 'Agree with the prediction', icon: <Check size={15} /> });
    const other = pred === 'silage' ? 'grain' : 'silage';
    options.push({ v: other, label: `Change to ${other}`, sub: 'Override the prediction', icon: other === 'silage' ? <Leaf size={15} /> : <Wheat size={15} /> });
  } else {
    options.push({ v: 'silage', label: 'Silage', sub: 'Classify manually', icon: <Leaf size={15} /> });
    options.push({ v: 'grain', label: 'Grain', sub: 'Classify manually', icon: <Wheat size={15} /> });
  }
  options.push({ v: 'mixed', label: 'Mixed', sub: 'Split acreage', icon: <Split size={15} /> });
  options.push({ v: 'insufficient', label: 'Insufficient evidence', sub: 'Keep acres unresolved', icon: <Ban size={15} /> });

  return (
    <section className="decision" aria-label="Record a decision">
      <div className="dec-label">Your decision</div>
      <div className="dec-options" role="radiogroup" aria-label="Decision">
        {options.map((o) => (
          <button key={o.v} type="button" role="radio" aria-checked={choice === o.v} className={`dec-opt ${choice === o.v ? 'on' : ''} dec-${o.v}`} onClick={() => setChoice(o.v)}>
            <span className="dec-opt-icon" aria-hidden>{o.icon}</span>
            <span>
              <span className="dec-opt-label">{o.label}</span>
              <span className="dec-opt-sub">{o.sub}</span>
            </span>
          </button>
        ))}
      </div>

      {choice === 'mixed' && (
        <div className="alloc">
          <div className="alloc-head">
            <span>Allocate {fmtAcres(field.acres)} ac</span>
            {field.subregions && (
              <button
                type="button"
                className="link-btn"
                onClick={() => {
                  const s = field.subregions!.find((x) => x.type === 'silage')?.acres ?? 0;
                  const g = field.subregions!.find((x) => x.type === 'grain')?.acres ?? 0;
                  setAlloc({ silage: String(s), grain: String(g), unresolved: String(Math.round((field.acres - s - g) * 10) / 10) });
                }}
              >
                Use reviewed subregions
              </button>
            )}
          </div>
          <div className="alloc-grid">
            {(['silage', 'grain', 'unresolved'] as const).map((k) => (
              <label key={k}>
                <span>{CLASS_LABEL[k]}</span>
                <input id={`alloc-${k}`} type="number" min={0} step={0.1} inputMode="decimal" value={alloc[k]}
                  onChange={(e) => setAlloc({ ...alloc, [k]: e.target.value })} aria-invalid={touched && !allocValid} />
              </label>
            ))}
          </div>
          <div className={`alloc-sum ${allocValid ? 'ok' : touched || sum > 0 ? 'bad' : ''}`} role="status">
            {allocValid
              ? `Allocations add up to ${fmtAcres(field.acres)} ac.`
              : `Must add up to ${fmtAcres(field.acres)} ac. Currently ${fmtAcres(sum)} ac${sum < field.acres ? `, ${fmtAcres(field.acres - sum)} ac left` : ''}.`}
            {!allocValid && sum < field.acres && sum > 0 && (
              <button type="button" className="link-btn" onClick={() => setAlloc({ ...alloc, unresolved: String(Math.round((field.acres - n(alloc.silage) - n(alloc.grain)) * 10) / 10) })}>
                Put remainder in unresolved
              </button>
            )}
          </div>
          {!field.subregions && <p className="fine">No reviewed subregions exist for this field. Enter proportions only if you have evidence for them.</p>}
        </div>
      )}

      <label className="dec-note-input">
        <span>Note <span className="muted">(optional, saved with the decision)</span></span>
        <textarea id="dec-note" rows={1} value={note} onChange={(e) => setNote(e.target.value)} placeholder="What did you see in the evidence?" />
      </label>
      {touched && !canSave && <p className="err" role="alert">{choice === null ? 'Choose a decision first.' : 'Fix the acreage allocation before saving.'}</p>}
      <div className="dec-actions">
        {decision && <button type="button" className="btn btn-quiet" onClick={() => setEditing(false)}>Cancel</button>}
        <button type="button" className={`btn ${hasNext ? '' : 'btn-primary'}`} onClick={() => save(false)} disabled={choice === null}>Save</button>
        {hasNext && (
          <button type="button" className="btn btn-primary" onClick={() => save(true)} disabled={choice === null}>
            Save and next
          </button>
        )}
      </div>
    </section>
  );
}
