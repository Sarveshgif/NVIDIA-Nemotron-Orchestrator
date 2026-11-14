import { useState } from 'react';
import { RotateCcw, Undo2 } from 'lucide-react';
import type { HistoryEntry } from '../types';
import { useStore } from '../state/store';
import { CLASS_LABEL, fmtAcres, fmtDateTime, fmtPct } from '../lib/format';

const ACTION_LABEL: Record<HistoryEntry['action'], string> = {
  accepted: 'Accepted',
  changed: 'Changed',
  insufficient: 'Insufficient evidence',
  note: 'Note updated',
  reset: 'Returned to queue',
};

const outcome = (d: HistoryEntry['after']) => {
  if (!d) return 'Not reviewed';
  if (!d.decided) return 'Insufficient';
  if (d.decided === 'mixed' && d.allocation) return `Mixed (${fmtAcres(d.allocation.silage)} silage / ${fmtAcres(d.allocation.grain)} grain)`;
  return CLASS_LABEL[d.decided];
};

export function HistoryView() {
  const history = useStore((s) => s.history);
  const fields = useStore((s) => s.fields);
  const undoStack = useStore((s) => s.undoStack);
  const undo = useStore((s) => s.undo);
  const resetDemo = useStore((s) => s.resetDemo);
  const setView = useStore((s) => s.setView);
  const select = useStore((s) => s.select);
  const storageOk = useStore((s) => s.storageOk);
  const [filter, setFilter] = useState<'all' | HistoryEntry['action']>('all');
  const [confirmReset, setConfirmReset] = useState(false);
  const rows = history.filter((h) => filter === 'all' || h.action === filter);
  const byId = new Map(fields.map((f) => [f.id, f]));

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <div className="eyebrow">Review history</div>
          <h1>Decisions and changes</h1>
          <p className="muted">
            Every decision keeps the prediction the analyst saw (rule-based or ML) beside the analyst outcome. Stored in this browser only for the
            prototype{storageOk ? '' : ' (browser storage is unavailable, so changes last only until you reload)'}; there is no shared backend.
          </p>
        </div>
        <div className="page-actions">
          <button type="button" className="btn" disabled={!undoStack.length} onClick={undo}><Undo2 size={15} /> Undo last action</button>
          {!confirmReset ? (
            <button type="button" className="btn btn-quiet" onClick={() => setConfirmReset(true)}><RotateCcw size={15} /> Reset demo reviews</button>
          ) : (
            <span className="confirm" role="alert">
              Discard your demo decisions?
              <button type="button" className="btn btn-danger btn-sm" onClick={() => { void resetDemo(); setConfirmReset(false); }}>Reset</button>
              <button type="button" className="btn btn-quiet btn-sm" onClick={() => setConfirmReset(false)}>Keep</button>
            </span>
          )}
        </div>
      </header>

      <div className="hist-tools">
        <label>
          Show
          <select id="hist-filter" value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)}>
            <option value="all">All actions</option>
            <option value="accepted">Accepted</option>
            <option value="changed">Changed</option>
            <option value="insufficient">Insufficient evidence</option>
            <option value="note">Note updated</option>
            <option value="reset">Returned to queue</option>
          </select>
        </label>
        <span className="muted">{rows.length} entries</span>
      </div>

      {rows.length === 0 ? (
        <div className="state-box"><p>No history entries match this filter.</p></div>
      ) : (
        <div className="tbl-wrap card">
          <table className="tbl">
            <thead>
              <tr>
                <th scope="col">When</th><th scope="col">Field</th><th scope="col" className="num">Acres</th><th scope="col">Action</th>
                <th scope="col">Prediction shown</th><th scope="col">Before</th><th scope="col">After</th><th scope="col">Analyst</th><th scope="col">Note</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((h) => (
                <tr key={h.id}>
                  <td className="nowrap">{fmtDateTime(h.at)}</td>
                  <th scope="row">
                    <button type="button" className="link-btn mono" onClick={() => { setView('workspace'); select(h.fieldId); }}>{h.fieldId}</button>
                  </th>
                  <td className="num">{byId.get(h.fieldId) ? fmtAcres(byId.get(h.fieldId)!.acres) : '–'}</td>
                  <td><span className={`act act-${h.action}`}>{ACTION_LABEL[h.action]}</span></td>
                  <td>{CLASS_LABEL[h.original]} <span className="muted">{fmtPct(h.originalConfidence)} · {h.originalSource === 'rules' ? 'rules' : 'ML'}</span></td>
                  <td className="muted">{outcome(h.before)}</td>
                  <td><b>{outcome(h.after)}</b></td>
                  <td className="nowrap">{h.analyst}</td>
                  <td className="note-cell">{h.after?.note || h.before?.note || <span className="muted">–</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
