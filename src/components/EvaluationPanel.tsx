// Held-out evaluation for the Netherlands 2025 dataset. Reads the evaluation summary written by the pipeline and,
// separately, the BRP reference labels (farmer declarations). Reference labels are only used here, never in
// predictions, explanations or the review queue.
import { useEffect, useMemo, useState } from 'react';
import { activeDataset, loadNlReference, type EvalRow } from '../data/datasets';
import { useStore } from '../state/store';
import { SOURCE_LABEL } from '../lib/format';

const p = (v: number | undefined | null, d = 0) => (v === undefined || v === null || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(d)}%`);
const f2 = (v: number | undefined | null) => (v === undefined || v === null || Number.isNaN(v) ? '—' : v.toFixed(2));

function Confusion({ title, c }: { title: string; c: { tp: number; fn: number; fp: number; tn: number } }) {
  return (
    <figure className="cm">
      <figcaption>{title}</figcaption>
      <table className="tbl cm-tbl">
        <thead><tr><th /><th scope="col">pred. silage</th><th scope="col">pred. grain</th></tr></thead>
        <tbody>
          <tr><th scope="row">declared silage</th><td className="cm-ok">{c.tp}</td><td className="cm-bad">{c.fn}</td></tr>
          <tr><th scope="row">declared grain</th><td className="cm-bad">{c.fp}</td><td className="cm-ok">{c.tn}</td></tr>
        </tbody>
      </table>
    </figure>
  );
}

export function EvaluationPanel() {
  const ds = activeDataset();
  const meta = ds.nl;
  const fields = useStore((s) => s.fields);
  const source = useStore((s) => s.predictionSource);
  const [ref, setRef] = useState<Record<string, 'silage' | 'grain'> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { loadNlReference().then(setRef, (e) => setErr(String(e))); }, []);

  const live = useMemo(() => {
    if (!ref) return null;
    const c = { tp: 0, fn: 0, fp: 0, tn: 0, unres: 0 };
    for (const f of fields) {
      const y = ref[f.id]; const pr = f.prediction.predicted;
      if (!y) continue;
      if (pr === 'unclassified') { c.unres++; continue; }
      if (y === 'silage') pr === 'silage' ? c.tp++ : c.fn++;
      else pr === 'grain' ? c.tn++ : c.fp++;
    }
    return c;
  }, [ref, fields]);

  if (!meta) return null;
  const ev = meta.evaluation;
  const rows: EvalRow[] = ev.comparison;
  const best = meta.model.name;
  return (
    <section className="card eval">
      <h2>Held-out evaluation · Netherlands {meta.year}</h2>
      <p className="muted">
        {ev.split}. Labels: {meta.provenance.labels} Positive class: <b>silage</b>. Test set: {ev.test_fields} fields
        ({ev.test_silage} silage, {ev.test_grain} grain). Thresholds, hyper-parameters, calibration and the review band
        were chosen on train/validation regions only.
      </p>
      <div className="tbl-wrap">
        <table className="tbl tbl-eval">
          <thead>
            <tr>
              <th scope="col">Model (same test fields)</th><th scope="col" className="num">Coverage</th><th scope="col" className="num">Bal. acc.</th>
              <th scope="col" className="num">Silage P / R</th><th scope="col" className="num">Grain P / R</th><th scope="col" className="num">FPR (grain→silage)</th>
              <th scope="col" className="num">ROC-AUC</th><th scope="col" className="num">PR-AUC grain</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.model} className={r.model === best ? 'row-best' : ''}>
                <th scope="row">{r.label}{r.model === best ? ' ★ console model' : ''}</th>
                <td className="num">{p(r.coverage)}</td>
                <td className="num">{f2(r.balanced_accuracy)}</td>
                <td className="num">{p(r.precision_silage)} / {p(r.recall_silage)}</td>
                <td className="num">{p(r.precision_grain)} / {p(r.recall_grain)}</td>
                <td className="num">{p(r.fpr_silage)}</td>
                <td className="num">{f2(r.roc_auc)}</td>
                <td className="num">{f2(r.pr_auc_grain)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="fine">
        Rules: metrics on the fields they label (coverage = share labelled). Grain prevalence in test ≈ {p(ev.test_grain / ev.test_fields)}, so
        grain precision is the hard number. 95% intervals for the console model (block bootstrap over test blocks):
        balanced accuracy {ev.ci[best]?.balanced_accuracy ? `${f2(ev.ci[best].balanced_accuracy[0])}–${f2(ev.ci[best].balanced_accuracy[1])}` : '—'},
        grain recall {ev.ci[best]?.recall_grain ? `${p(ev.ci[best].recall_grain[0])}–${p(ev.ci[best].recall_grain[1])}` : '—'}.
      </p>
      <div className="cm-row">
        {Object.entries(ev.confusion).map(([k, c]) => (
          <Confusion key={k} title={rows.find((r) => r.model === k)?.label ?? k} c={c} />
        ))}
      </div>
      <p className="fine">
        Latency: {typeof ev.latency.rf_per_field_ms === 'number' ? ev.latency.rf_per_field_ms.toFixed(1) : ev.latency.rf_per_field_ms} ms per field,
        {' '}{typeof ev.latency.rf_batch_seconds === 'number' ? ev.latency.rf_batch_seconds.toFixed(2) : ev.latency.rf_batch_seconds} s for {ev.latency.rf_batch_fields} fields. {ev.latency.note} The console itself reads cached outputs.
      </p>
      {ev.years && ev.years.length > 0 && (
        <>
          <h3>Does more training data help? Same regions, different training years</h3>
          <div className="tbl-wrap">
            <table className="tbl tbl-eval">
              <thead>
                <tr>
                  <th scope="col">Train years → test year</th><th scope="col">Model</th><th scope="col" className="num">Train field-seasons (grain)</th>
                  <th scope="col" className="num">Test fields</th><th scope="col" className="num">Bal. acc.</th><th scope="col" className="num">ROC-AUC</th>
                  <th scope="col" className="num">Grain P / R</th><th scope="col" className="num">Bal. acc., simulated gap</th>
                </tr>
              </thead>
              <tbody>
                {ev.years.map((r) => (
                  <tr key={r.experiment + r.model}>
                    <th scope="row">{r.train_years} → {r.test_year}</th>
                    <td>{r.model === 'rf_s2_s1' ? 'RF · S2 + zones + S1' : 'RF · S2 + zones'}</td>
                    <td className="num">{r.n_train} ({r.grain_train})</td>
                    <td className="num">{r.n_test}</td>
                    <td className="num">{f2(r.balanced_accuracy)}</td>
                    <td className="num">{f2(r.roc_auc)}</td>
                    <td className="num">{p(r.precision_grain)} / {p(r.recall_grain)}</td>
                    <td className="num">{f2(r.gap_balanced_accuracy)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="fine">Regions and splits are identical every year, so a parcel seen in both years stays in one split. Test regions are never used for training or tuning in any row.</p>
        </>
      )}
      <h3>Check: current console predictions vs reference labels</h3>
      {err && <p className="callout callout-warn">{err}</p>}
      {live && (
        <p>
          {SOURCE_LABEL[source]} predictions on the {fields.length} loaded fields: {live.tp + live.tn} agree with the declaration,
          {' '}{live.fp} declared-grain fields called silage, {live.fn} declared-silage fields called grain{live.unres ? `, ${live.unres} unclassified` : ''}.
          {' '}Analyst decisions are not counted here.
        </p>
      )}
      <p className="fine">Declared crop use is what the farmer registered in spring; a field declared korrelmais can still be chopped for silage. Treat disagreement as "needs a look", not proof of a model error.</p>
    </section>
  );
}
