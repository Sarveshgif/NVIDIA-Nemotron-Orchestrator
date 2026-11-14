import { activeDataset } from '../data/datasets';
import { useMemo } from 'react';
import { CheckCircle2, Download, Printer, AlertTriangle } from 'lucide-react';
import { useStore } from '../state/store';
import type { Derived } from '../state/derived';
import { allocate, grainTotal, reconciles, silageTotal, summarize, unresolvedTotal } from '../lib/acreage';
import { CLASS_LABEL, SOURCE_LABEL, STATUS_LABEL, fmtAcres, fmtDate, fmtPct } from '../lib/format';
import { buildCsv, isFramed, saveFile } from '../lib/export';
import { regionName, seasonLabel } from '../services/analysisService';
import { CompositionBar } from './AcreageSummary';
import { TypePill } from './ui';
import { EvaluationPanel } from './EvaluationPanel';
import { REGIONS } from '../data/geo';

const pct = (v: number, t: number) => (t ? `${((v / t) * 100).toFixed(1)}%` : '—');
const nf = (n: number) => `${n} field${n === 1 ? '' : 's'}`;
const signed = (v: number) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${fmtAcres(Math.abs(v))}`;

export function ReportsView({ derived }: { derived: Derived }) {
  const fields = useStore((s) => s.fields);
  const decisions = useStore((s) => s.decisions);
  const selection = useStore((s) => s.selection);
  const status = useStore((s) => s.status);
  const showToast = useStore((s) => s.showToast);
  const setView = useStore((s) => s.setView);
  const select = useStore((s) => s.select);
  const source = useStore((s) => s.predictionSource);
  const s = derived.summaryAll;
  const baseline = useMemo(() => summarize(fields, {}), [fields]);

  if (!fields.length) {
    return (
      <div className="page">
        <div className="state-box">
          <p>No report is available because no analysis is loaded for this selection.</p>
          <button type="button" className="btn" onClick={() => setView('workspace')}>Back to workspace</button>
        </div>
      </div>
    );
  }

  const gapFields = fields.filter((f) => f.coverage === 'gap');
  const insuffFields = fields.filter((f) => f.coverage === 'insufficient');
  const changed = fields.filter((f) => decisions[f.id]?.status === 'changed');
  const accepted = fields.filter((f) => decisions[f.id]?.status === 'accepted');
  const insufficient = fields.filter((f) => decisions[f.id]?.status === 'insufficient');
  const flaggedOpen = fields.filter((f) => f.reasons.length && !decisions[f.id]);
  const acres = (list: typeof fields) => list.reduce((a, f) => a + f.acres, 0);

  const exportAll = async () => {
    const csv = buildCsv(fields, decisions, {
      region: regionName(selection.regionId), cropYear: selection.cropYear, season: seasonLabel(selection.seasonId),
      latestObservation: status?.latestObservation ?? null, scope: 'all', allFields: fields,
    });
    const name = `harvest-console_${selection.regionId}_${selection.cropYear}_all-fields_${new Date().toISOString().slice(0, 10)}.csv`;
    const r = await saveFile(name, csv);
    showToast(r === 'saved' ? `Exported ${fields.length} fields to ${name}` : r === 'declined' ? 'Export cancelled' : 'Export is not available in this view.');
  };
  const open = (id: string) => {
    setView('workspace');
    select(id);
  };

  return (
    <div className="page report">
      <header className="page-head">
        <div>
          <div className="eyebrow">Report · preliminary estimates · {SOURCE_LABEL[source].toLowerCase()} predictions · {activeDataset().obsSource}</div>
          <h1>{regionName(selection.regionId)}, {REGIONS.find((r) => r.id === selection.regionId)?.state} · crop year {selection.cropYear}</h1>
          <p className="muted">
            {seasonLabel(selection.seasonId)} · latest observation {status?.latestObservation ? fmtDate(status.latestObservation, true) : 'n/a'} · predictions {source === 'rules' ? fields[0]?.prediction.modelVersion : status?.modelVersion} · generated {new Date().toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' })}
          </p>
        </div>
        <div className="page-actions no-print">
          <button type="button" className="btn btn-primary" onClick={exportAll}><Download size={15} /> Download CSV</button>
          {!isFramed() && <button type="button" className="btn" onClick={() => window.print()}><Printer size={15} /> Print</button>}
        </div>
      </header>
      {activeDataset().nl && <EvaluationPanel />}

      <div className="scope-box">
        <strong>Scope.</strong> All {s.fieldCount} loaded fields, {fmtAcres(s.total)} ac. {activeDataset().scopeNote} Totals are preliminary estimates for loaded fields only. {fmtAcres(unresolvedTotal(s))} ac remain unresolved and
        {' '}{flaggedOpen.length} flagged fields ({fmtAcres(acres(flaggedOpen))} ac) are not yet reviewed.
        {reconciles(s) && <span className="reconcile"><CheckCircle2 size={14} aria-hidden /> Silage + grain + unresolved = total</span>}
      </div>

      <div className="report-grid">
        <section className="card">
          <h2>Acreage by harvest type</h2>
          <CompositionBar s={s} tall />
          <table className="tbl">
            <thead>
              <tr><th scope="col">Class</th><th scope="col" className="num">Predicted</th><th scope="col" className="num">Reviewed</th><th scope="col" className="num">Total ac</th><th scope="col" className="num">Share</th><th scope="col" className="num">vs prediction only</th></tr>
            </thead>
            <tbody>
              <tr><th scope="row"><TypePill cls="silage" /></th><td className="num">{fmtAcres(s.silage.model)}</td><td className="num">{fmtAcres(s.silage.reviewed)}</td><td className="num"><b>{fmtAcres(silageTotal(s))}</b></td><td className="num">{pct(silageTotal(s), s.total)}</td><td className="num">{signed(silageTotal(s) - silageTotal(baseline))}</td></tr>
              <tr><th scope="row"><TypePill cls="grain" /></th><td className="num">{fmtAcres(s.grain.model)}</td><td className="num">{fmtAcres(s.grain.reviewed)}</td><td className="num"><b>{fmtAcres(grainTotal(s))}</b></td><td className="num">{pct(grainTotal(s), s.total)}</td><td className="num">{signed(grainTotal(s) - grainTotal(baseline))}</td></tr>
              <tr><th scope="row"><TypePill cls="unclassified" /> <span className="sr-only">Unresolved</span></th><td className="num">{fmtAcres(s.unresolved.model)}</td><td className="num">{fmtAcres(s.unresolved.insufficient + s.unresolved.mixedRemainder)}</td><td className="num"><b>{fmtAcres(unresolvedTotal(s))}</b></td><td className="num">{pct(unresolvedTotal(s), s.total)}</td><td className="num">{signed(unresolvedTotal(s) - unresolvedTotal(baseline))}</td></tr>
            </tbody>
            <tfoot>
              <tr><th scope="row">Total</th><td className="num">{fmtAcres(s.silage.model + s.grain.model + s.unresolved.model)}</td><td className="num">{fmtAcres(s.silage.reviewed + s.grain.reviewed + s.unresolved.insufficient + s.unresolved.mixedRemainder)}</td><td className="num"><b>{fmtAcres(s.total)}</b></td><td className="num">100%</td><td /></tr>
            </tfoot>
          </table>
          <p className="fine">Mixed fields are split by reviewed subregions or analyst-entered proportions, so each acre is counted once.</p>
        </section>

        <section className="card">
          <h2>Review coverage</h2>
          <dl className="kv">
            <div><dt>Fields reviewed</dt><dd>{s.reviewedFields} of {s.fieldCount} ({pct(s.reviewedFields, s.fieldCount)})</dd></div>
            <div><dt>Acres reviewed</dt><dd>{fmtAcres(s.reviewedAcres)} of {fmtAcres(s.total)} ({pct(s.reviewedAcres, s.total)})</dd></div>
            <div><dt>Flagged fields done</dt><dd>{s.queueFields - s.queueRemaining} of {s.queueFields}</dd></div>
            <div><dt>Flagged, not reviewed</dt><dd>{nf(flaggedOpen.length)} · {fmtAcres(acres(flaggedOpen))} ac</dd></div>
          </dl>
          <div className="progress progress-lg" aria-hidden><span style={{ width: `${(s.reviewedAcres / s.total) * 100}%` }} /></div>
          <h3>Unresolved acreage</h3>
          <dl className="kv">
            <div><dt>Unclassified by prediction</dt><dd>{fmtAcres(s.unresolved.model)} ac</dd></div>
            <div><dt>Marked insufficient by analyst</dt><dd>{fmtAcres(s.unresolved.insufficient)} ac</dd></div>
            <div><dt>Unallocated part of mixed fields</dt><dd>{fmtAcres(s.unresolved.mixedRemainder)} ac</dd></div>
          </dl>
        </section>

        <section className="card">
          <h2>Observation coverage limits</h2>
          <dl className="kv">
            <div><dt>Gap over 15 days near harvest</dt><dd>{nf(gapFields.length)} · {fmtAcres(acres(gapFields))} ac</dd></div>
            <div><dt>Insufficient observations</dt><dd>{nf(insuffFields.length)} · {fmtAcres(acres(insuffFields))} ac</dd></div>
          </dl>
          {[...gapFields, ...insuffFields].length > 0 && (
            <p className="small">
              <AlertTriangle size={13} aria-hidden /> Affected:{' '}
              {[...gapFields, ...insuffFields].map((f, i) => (
                <span key={f.id}>{i > 0 && ', '}<button type="button" className="link-btn mono" onClick={() => open(f.id)}>{f.id}</button></span>
              ))}
            </p>
          )}
          <p className="fine">Optical observations come from Sentinel-2 (5-day revisit, cloud-limited). Radar from Sentinel-1 every 12 days. Harvest dates are windows between observations, not exact dates.</p>
        </section>

        <section className="card">
          <h2>Analyst changes</h2>
          <dl className="kv">
            <div><dt>Accepted</dt><dd>{nf(accepted.length)} · {fmtAcres(acres(accepted))} ac</dd></div>
            <div><dt>Changed from prediction</dt><dd>{nf(changed.length)} · {fmtAcres(acres(changed))} ac</dd></div>
            <div><dt>Insufficient evidence</dt><dd>{nf(insufficient.length)} · {fmtAcres(acres(insufficient))} ac</dd></div>
          </dl>
          {changed.length > 0 && (
            <ul className="changes">
              {changed.map((f) => {
                const d = decisions[f.id];
                return (
                  <li key={f.id}>
                    <button type="button" className="link-btn mono" onClick={() => open(f.id)}>{f.id}</button>{' '}
                    {CLASS_LABEL[f.prediction.predicted]} ({fmtPct(f.prediction.confidence)}) → <b>{d.decided ? CLASS_LABEL[d.decided] : 'Insufficient'}</b>
                    <span className="muted"> · {fmtAcres(f.acres)} ac · {d.analyst}</span>
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      </div>

      <section className="card">
        <h2>All loaded fields</h2>
        <div className="tbl-wrap">
          <table className="tbl tbl-fields">
            <thead>
              <tr>
                <th scope="col">Field</th><th scope="col" className="num">Acres</th><th scope="col">{SOURCE_LABEL[source]}</th><th scope="col" className="num">Score</th>
                <th scope="col">Status</th><th scope="col">Final</th><th scope="col" className="num">Silage</th><th scope="col" className="num">Grain</th><th scope="col" className="num">Unresolved</th>
              </tr>
            </thead>
            <tbody>
              {fields.map((f) => {
                const al = allocate(f, decisions[f.id]);
                return (
                  <tr key={f.id}>
                    <th scope="row"><button type="button" className="link-btn mono" onClick={() => open(f.id)}>{f.id}</button></th>
                    <td className="num">{fmtAcres(f.acres)}</td>
                    <td>{CLASS_LABEL[f.prediction.predicted]}</td>
                    <td className="num">{fmtPct(f.prediction.confidence)}</td>
                    <td>{STATUS_LABEL[al.status]}</td>
                    <td>{CLASS_LABEL[al.effective]}{al.source === 'analyst' ? ' ✓' : ''}</td>
                    <td className="num">{al.silage ? fmtAcres(al.silage) : '–'}</td>
                    <td className="num">{al.grain ? fmtAcres(al.grain) : '–'}</td>
                    <td className="num">{al.unresolved ? fmtAcres(al.unresolved) : '–'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
      <p className="fine">Field outlines: {activeDataset().outlineSource}. Time series and images: {activeDataset().obsSource}. Predictions are estimates, not ground truth. Analyst review means reviewed, not independently verified. {activeDataset().attribution}.</p>
    </div>
  );
}
