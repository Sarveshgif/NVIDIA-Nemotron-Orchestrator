import { useEffect, useState } from 'react';
import { Download } from 'lucide-react';
import { useStore } from '../state/store';
import type { Derived } from '../state/derived';
import { EvaluationPanel } from './EvaluationPanel';
interface Evaluation {
  records: number; rf_correct: number; combined_correct: number; high_score_skipped: number; low_score_investigated: number;
  elapsed_seconds: number; mean_backend_seconds: number;
  classification_metrics: Record<string, { confusion_matrix: Record<string, Record<string, number>>; per_class: Record<string, { precision: number; recall: number; false_positive_rate: number }> }>;
}
const pct=(v:number)=>`${(v*100).toFixed(1)}%`;
function BarcReportsView({ derived }: { derived: Derived }) {
  const fields=useStore(s=>s.fields); const year=useStore(s=>s.selection.cropYear);
  const setView=useStore(s=>s.setView); const select=useStore(s=>s.select);
  const [evaluation,setEvaluation]=useState<Evaluation|null>(null);const [error,setError]=useState<string|null>(null);
  useEffect(()=>{let active=true;fetch(`${import.meta.env.BASE_URL}barc/evaluation.json`).then(r=>{if(!r.ok)throw new Error('Saved BARC evaluation is unavailable.');return r.json();}).then(d=>{if(active)setEvaluation(d);}).catch(e=>{if(active)setError(e instanceof Error?e.message:'Could not load report.');});return()=>{active=false;};},[]);
  return <div className="page report"><header className="page-head"><div><div className="eyebrow">BARC only · Maryland · 2021–2023</div><h1>BARC classification report</h1><p className="muted">44 field-years across 34 distinct fields. Reference labels were used in RF recalibration; results are development agreement.</p></div><a className="btn btn-primary" href={`${import.meta.env.BASE_URL}barc/evaluation.json`} download="BARC_evaluation.json"><Download size={15}/> Download metrics</a></header>
    <div className="scope-box"><strong>Current view:</strong> {fields.length} fields from {year}, totaling {derived.summaryAll.total.toFixed(1)} acres. Live satellite/Nemotron analysis is started from an individual field. The saved aggregate metrics below come from the earlier v2.5 weather/context run, not a full spatial-data reevaluation.</div>
    {error&&<p role="alert" className="callout callout-warn">{error}</p>}
    {evaluation&&<><div className="report-grid"><section className="card"><h2>Saved 44-record comparison</h2><table className="tbl"><thead><tr><th>Method</th><th>Correct</th><th>Accuracy</th></tr></thead><tbody><tr><td>RF only</td><td>{evaluation.rf_correct}/{evaluation.records}</td><td>{pct(evaluation.rf_correct/evaluation.records)}</td></tr><tr><td>RF + Nemotron + weather/context</td><td>{evaluation.combined_correct}/{evaluation.records}</td><td>{pct(evaluation.combined_correct/evaluation.records)}</td></tr></tbody></table><p className="fine">35 high-score fields skipped Nemotron; 9 low-score fields used it. No final REVIEW or UNRESOLVED predictions in that saved run. A new 44-record satellite evaluation has not been performed.</p></section><section className="card"><h2>Saved precision and recall</h2><table className="tbl"><thead><tr><th>Method / class</th><th>Precision</th><th>Recall</th><th>False positive rate</th></tr></thead><tbody>{Object.entries(evaluation.classification_metrics).flatMap(([method,m])=>Object.entries(m.per_class).map(([label,c])=><tr key={method+label}><td>{method} · {label.toLowerCase()}</td><td>{pct(c.precision)}</td><td>{pct(c.recall)}</td><td>{pct(c.false_positive_rate)}</td></tr>))}</tbody></table></section></div>
      <div className="report-grid">{Object.entries(evaluation.classification_metrics).map(([method,m])=><section className="card" key={method}><h2>{method} confusion matrix</h2><p className="fine">Rows: actual labels. Columns: predictions.</p><table className="tbl cm-tbl"><thead><tr><th>Actual / predicted</th><th>Grain</th><th>Silage</th></tr></thead><tbody>{['GRAIN','SILAGE'].map(label=><tr key={label}><th>{label.toLowerCase()}</th><td>{m.confusion_matrix[label].GRAIN}</td><td>{m.confusion_matrix[label].SILAGE}</td></tr>)}</tbody></table></section>)}</div></>}
    <section className="card"><h2>BARC fields · {year}</h2><div className="tbl-wrap"><table className="tbl"><thead><tr><th>Field</th><th>Acres</th><th>Original RF</th><th>RF score</th><th>Current displayed label</th><th>Source</th></tr></thead><tbody>{fields.map(f=><tr key={f.id}><td><button className="link-btn mono" type="button" onClick={()=>{setView('workspace');select(f.id);}}>{f.id}</button></td><td>{f.acres.toFixed(2)}</td><td>{f.barc?.rfLabel}</td><td>{pct(f.barc?.rfConfidence??0)}</td><td>{f.prediction.predicted}</td><td>{f.prediction.modelVersion}</td></tr>)}</tbody></table></div></section>
  </div>;
}

function NetherlandsReportsView({ derived }: { derived: Derived }) {
  const fields = useStore(s => s.fields);
  const year = useStore(s => s.selection.cropYear);
  const setView = useStore(s => s.setView);
  const select = useStore(s => s.select);
  return <div className="page report">
    <header className="page-head"><div><div className="eyebrow">South-east Netherlands · {year}</div><h1>Netherlands classification report</h1><p className="muted">Held-out evaluation and current field predictions for the selected area.</p></div></header>
    <div className="scope-box"><strong>Current view:</strong> {fields.length} fields from {year}, totaling {derived.summaryAll.total.toFixed(1)} acres.</div>
    <EvaluationPanel />
    <section className="card"><h2>Netherlands fields · {year}</h2><div className="tbl-wrap"><table className="tbl"><thead><tr><th>Field</th><th>Acres</th><th>Prediction</th><th>Model</th></tr></thead><tbody>{fields.map(f=><tr key={f.id}><td><button className="link-btn mono" type="button" onClick={()=>{setView('workspace');select(f.id);}}>{f.id}</button></td><td>{f.acres.toFixed(2)}</td><td>{f.prediction.predicted}</td><td>{f.prediction.modelVersion}</td></tr>)}</tbody></table></div></section>
  </div>;
}

export function ReportsView({ derived }: { derived: Derived }) {
  const regionId = useStore(s => s.selection.regionId);
  return regionId === 'nl-se' ? <NetherlandsReportsView derived={derived} /> : <BarcReportsView derived={derived} />;
}
