import { useEffect, useState } from 'react';
import { Loader2, Satellite, Download } from 'lucide-react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { Field, FieldEvidence } from '../types';
import { useStore } from '../state/store';
import { savedInvestigation, investigateBarc, type InvestigationReport } from '../services/investigationService';

const title = (v: string) => v === 'GRAIN' ? 'Grain' : v === 'SILAGE' ? 'Silage' : v === 'UNRESOLVED' ? 'Unresolved' : v;
const spatialResult = (r: InvestigationReport | null) => r?.evidence.find(e => e.tool === 'get_sentinel_spatial_evidence');
const hasCurrentSatellite = (r: InvestigationReport | null) => !!spatialResult(r)?.data.rulebook_v4;
const hasImagePreviews = (r: InvestigationReport | null) => { const d=spatialResult(r)?.data.s2?.image_previews; return !!d && (d.items.length > 0 || (d.missing_positions?.length ?? 0) > 0); };
export function BarcInvestigation({ field }: { field: Field }) {
  const [report, setReport] = useState<InvestigationReport | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reference, setReference] = useState<string | null>(null);
  const apply = useStore(s => s.applyInvestigation);
  useEffect(() => {
    let active = true;
    setReport(null); setBusy(true); setError(null); setReference(null);
    void (async () => {
      const saved = await savedInvestigation(field.id).catch(() => null);
      if (!active) return;
      if (hasCurrentSatellite(saved)) setReport(saved);
      if (hasCurrentSatellite(saved) && hasImagePreviews(saved)) { setBusy(false); return; }
      const r = await investigateBarc(field.id);
      if (!active) return;
      setReport(r);
      const source = r.evidence.find(e => e.tool === 'get_sentinel_spatial_evidence');
      const s2 = source?.data.s2?.observations?.filter(o => o.usable) ?? [];
      const s1 = source?.data.s1?.observations?.filter(o => o.usable) ?? [];
      const optical = s2.flatMap(o => o.indices?.ndvi && o.indices?.lswi ? [{ date: o.date, ndvi: o.indices.ndvi.mean, lswi: o.indices.lswi.mean }] : []);
      const evidence: FieldEvidence = { fieldId: field.id, optical, radar: s1.flatMap(o => o.vv_db !== undefined && o.vh_db !== undefined ? [{ date: o.date, vv: o.vv_db, vh: o.vh_db }] : []), patches: [], gaps: [], before: optical[0] ?? null, after: optical[optical.length-1] ?? null, pattern: 'sparse' };
      if (r.final_label === 'GRAIN' || r.final_label === 'SILAGE') apply(field.id, r.final_label.toLowerCase() as 'grain' | 'silage', r.nemotron_used_for_prediction ? 'Nemotron + Sentinel evidence' : r.decision_source === 'rf_fallback' ? 'RF fallback' : 'RF', evidence);
    })().catch(e => { if (active) setError(e instanceof Error ? e.message : 'Satellite analysis failed.'); })
      .finally(() => { if (active) setBusy(false); });
    return () => { active = false; };
  }, [field.id, apply]);
  const spatial = report?.evidence.find(e => e.tool === 'get_sentinel_spatial_evidence');
  const current = hasCurrentSatellite(report);
  const optical = current ? spatial?.data.s2?.observations?.filter(o => o.usable) ?? [] : [];
  const radar = current ? spatial?.data.s1?.observations?.filter(o => o.usable) ?? [] : [];
  const rule = current ? spatial?.data.rulebook_v4 : undefined;
  const imagePreviews = current ? spatial?.data.s2?.image_previews : undefined;
  const charts = optical.map(o => ({ date: o.date, ...Object.fromEntries(Object.entries(o.indices ?? {}).map(([k,v]) => [k,v?.mean ?? null])) }));
  const low = (field.barc?.rfConfidence ?? 1) < .8;
  const download = () => {
    if (!report) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    const a = document.createElement('a'); a.href=url; a.download=`${field.id}_evidence.json`; a.click(); URL.revokeObjectURL(url);
  };
  const showReference = async () => {
    try { const response=await fetch(`${import.meta.env.BASE_URL}barc/reference.json`); if(!response.ok) throw new Error(); const d=await response.json(); setReference(d.labels[field.id] ?? 'Unavailable'); }
    catch { setReference('Unavailable'); }
  };
  const chartTooltip = { background: 'var(--panel)', color: 'var(--ink)', border: '1px solid var(--line)' };
  return <section className="barc-panel" aria-label="BARC satellite and Nemotron analysis">
    <div className="barc-heading"><Satellite size={18}/><h3>BARC · RF → Sentinel evidence → Nemotron</h3></div>
    <p className="fine">{field.barc?.year} · Development data. RF was recalibrated with BARC labels; these are not independent validation results.</p>
    <div className="barc-results">
      <div><span>Original RF</span><strong>{field.barc?.rfLabel === 'grain' ? 'Grain' : 'Silage'}</strong><small>{((field.barc?.rfConfidence ?? 0)*100).toFixed(1)}% RF score</small></div>
      <div><span>{report?.decision_source === 'rf_fallback' ? 'RF fallback' : low ? 'Nemotron result' : 'RF retained'}</span><strong>{report ? title(report.final_label) : 'Pending'}</strong><small>{low ? 'Nemotron runs below 80% RF score' : 'Nemotron skipped at 80% or above'}</small></div>
    </div>
    <p className="barc-status" role="status">{busy ? <><Loader2 size={15} className="spin"/> Loading Sentinel-2 and Sentinel-1 observations automatically…</> : current ? 'Sentinel evidence loaded for this field.' : 'Sentinel evidence is unavailable for this field.'}</p>
    {error && <p className="callout callout-warn" role="alert">{error}</p>}
    <p className="fine">Field bounds are sent to Earth Search and Microsoft Planetary Computer. Derived measurements go to NVIDIA only when RF is below 80%.</p>
    {rule && <div className={`barc-rule barc-rule-${rule.label.toLowerCase()}`}><h4>Sentinel v4 rule · {title(rule.label)}{rule.tier ? ` (${rule.tier})` : ''}</h4><p>{rule.reason}</p><small>Step {rule.step} · fixed thresholds · advisory evidence</small></div>}
    {current && <section className="barc-imagery" aria-label="Sentinel before and after images"><h4>Sentinel-2 · Before and after</h4><p className="fine">Clear satellite pixels clipped to this field, independent of Nemotron. The white outline marks the field area used for measurements.</p><div className="barc-image-pair">{(['before','after'] as const).map(position=>{const image=imagePreviews?.items.find(item=>item.position===position);return <article className="barc-image-card" key={position}><header><strong>{position==='before'?'Before':'After'}</strong>{image?.date&&<time dateTime={image.date}>{image.date}</time>}</header>{image?.true_colour_png&&image?.ndvi_png?<div className="barc-image-modes"><figure><img src={image.true_colour_png} alt={`Sentinel-2 true-colour field image ${position} harvest on ${image.date}`}/><figcaption>True colour</figcaption></figure><figure><img src={image.ndvi_png} alt={`Sentinel-2 NDVI image ${position} harvest on ${image.date}`}/><figcaption>NDVI{image.ndvi_mean!=null?` · ${image.ndvi_mean.toFixed(2)}`:''}</figcaption></figure></div>:<p className="barc-image-missing">{image?.unavailable?'The clear scene was found, but image pixels could not be loaded.':`No clear Sentinel image available ${position} the harvest window.`}</p>}</article>})}</div><p className="fine">{imagePreviews?.source ?? 'Images are generated from observed Sentinel-2 L2A surface reflectance.'}</p></section>}
    {report && <><p>{report.explanation}</p><p className="fine">No calibrated combined confidence is available. Field-evidence strength: {report.evidence_strength}. {low ? (report.spatial_evidence_cited ? 'Nemotron cited satellite evidence.' : 'Nemotron did not cite satellite evidence.') : 'Nemotron was not called for this RF score.'}</p></>}
    {current && <>
      <div className="barc-source-counts"><span>Sentinel-2: <b>{optical.length} usable dates</b></span><span>Sentinel-1: <b>{radar.length} usable dates</b></span></div>
      {optical.length > 0 ? <div className="barc-chart"><h4>Sentinel-2 field indices</h4><ResponsiveContainer width="100%" height={190}><LineChart data={charts}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="date" tick={{fontSize:10,fill:'var(--ink-2)'}} tickFormatter={d=>String(d).slice(5)}/><YAxis domain={['auto','auto']} tick={{fill:'var(--ink-2)'}}/><Tooltip contentStyle={chartTooltip}/>{['ndvi','ndre','lswi','ndti','evi'].map((k,i)=><Line key={k} dataKey={k} name={k.toUpperCase()} type="linear" dot={{r:2}} connectNulls={false} stroke={['#22734d','#c58a24','#327eaa','#8c5ca1','#d45f42'][i]}/>)}</LineChart></ResponsiveContainer><p className="fine">NDVI uses clear field pixels at 20 m. NDRE, LSWI, NDTI and EVI are shown when available.</p></div> : <p className="fine">No usable clear optical observations. Missing images are not treated as grain.</p>}
      {radar.length > 0 ? <div className="barc-chart"><h4>Sentinel-1 radar backscatter</h4><ResponsiveContainer width="100%" height={170}><LineChart data={radar}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="date" tick={{fontSize:10,fill:'var(--ink-2)'}} tickFormatter={d=>String(d).slice(5)}/><YAxis tick={{fill:'var(--ink-2)'}}/><Tooltip contentStyle={chartTooltip}/><Line dataKey="vv_db" name="VV dB" stroke="#327eaa"/><Line dataKey="vh_db" name="VH dB" stroke="#d45f42"/></LineChart></ResponsiveContainer><p className="fine">Relative orbit {radar[0].relative_orbit ?? 'unknown'} · {radar[0].orbit_pass ?? 'pass unknown'}. Mean linear power converted to dB.</p></div> : <p className="fine">No usable radar observations for the selected orbit.</p>}
      {!!optical[optical.length-1]?.ndvi_zones_2x2?.length && <><h4>Latest observed within-field NDVI</h4><div className="barc-zones">{optical[optical.length-1]!.ndvi_zones_2x2!.map(z=><div key={`${z.row}-${z.column}`} style={{background:z.ndvi_mean === null ? '#eee' : `hsl(110 30% ${90-z.ndvi_mean*45}%)`}}><b>{z.ndvi_mean?.toFixed(2) ?? 'No data'}</b><small>{z.clear_pixels} clear pixels</small></div>)}</div></>}
      {spatial?.errors?.length ? <p className="fine">Some scenes were unavailable or exceeded the extraction budget. See the report for coverage limitations.</p> : null}
    </>}
    {report && <><details><summary>Evidence citations and missing information</summary><p className="fine">{report.nemotron_hypothesis?.evidence_ids.join(' · ') ?? 'Nemotron was not used for this RF score'}</p><ul>{report.missing_evidence.map((m,i)=><li key={i}>{m}</li>)}</ul></details><button type="button" className="btn btn-sm" onClick={download}><Download size={14}/> Download evidence report</button></>}
    <div className="barc-reference"><button type="button" className="btn btn-sm" onClick={() => void showReference()}>Show reference label</button>{reference && <span>Reference: <b>{reference}</b> · evaluation display only, never sent to inference</span>}</div>
  </section>;
}
