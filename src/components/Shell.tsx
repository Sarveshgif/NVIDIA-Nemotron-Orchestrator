import { useEffect, useRef, useState } from 'react';
import { Check, CircleHelp, Download, FileBarChart, FlaskConical, Loader2, Map as MapIcon, Undo2, X, CircleDot, CircleSlash } from 'lucide-react';
import { CROP_YEARS, REGIONS, SEASONS } from '../data/geo';
import { NL_SELECTION, PREP_STEPS, regionName, seasonLabel, simulatePreparation } from '../services/analysisService';
import { buildCsv, saveFile } from '../lib/export';
import { mlAvailable, SOURCE_LABEL, sourceNote, fmtDate } from '../lib/format';
import { activeDataset } from '../data/datasets';
import { useStore, type View } from '../state/store';
import type { Derived } from '../state/derived';
import type { PredictionSource } from '../types';
import { InfoTip, Segmented } from './ui';

export function Brand() {
  return (
    <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
      <rect width="32" height="32" rx="8" fill="#e9c46a" />
      <path d="M16 26V12" stroke="#173b2c" strokeWidth="2.2" strokeLinecap="round" />
      <path d="M16 17c-4 0-6.5-2.6-7-6.5 4 0 6.5 2.4 7 6.5zM16 14c3.6 0 6-2.3 6.6-5.8-3.7 0-6 2.2-6.6 5.8z" fill="#173b2c" />
      <path d="M9 26h14" stroke="#173b2c" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function NavRail() {
  const view = useStore((s) => s.view);
  const setView = useStore((s) => s.setView);
  const dismissHint = useStore((s) => s.dismissHint);
  const remaining = useStore((s) => s.fields.filter((f) => f.reasons.length && !s.decisions[f.id]).length);
  const items: { v: View; label: string; icon: JSX.Element }[] = [
    { v: 'workspace', label: 'Workspace', icon: <MapIcon size={19} /> },
    { v: 'reports', label: 'Reports', icon: <FileBarChart size={19} /> },
  ];
  return (
    <nav className="rail" aria-label="Main">
      <div className="rail-brand" title="Harvest Console">
        <Brand />
      </div>
      <ul>
        {items.map((it) => (
          <li key={it.v}>
            <button type="button" className={`rail-btn ${view === it.v ? 'on' : ''}`} aria-current={view === it.v ? 'page' : undefined} onClick={() => setView(it.v)}>
              {it.icon}
              <span>{it.label}</span>
              {it.v === 'workspace' && remaining > 0 && <em className="rail-badge" aria-label={`${remaining} fields need review`}>{remaining}</em>}
            </button>
          </li>
        ))}
      </ul>
      <button type="button" className="rail-btn rail-help" onClick={() => { setView('workspace'); dismissHint(false); }}>
        <CircleHelp size={19} />
        <span>Start here</span>
      </button>
    </nav>
  );
}

export function ContextBar({ derived }: { derived: Derived }) {
  const selection = useStore((s) => s.selection);
  const setSelection = useStore((s) => s.setSelection);
  const status = useStore((s) => s.status);
  const loading = useStore((s) => s.loading);
  const fields = useStore((s) => s.fields);
  const decisions = useStore((s) => s.decisions);
  const showToast = useStore((s) => s.showToast);
  const [exportOpen, setExportOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!exportOpen) return;
    const close = (e: MouseEvent) => menuRef.current && !menuRef.current.contains(e.target as Node) && setExportOpen(false);
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && setExportOpen(false);
    document.addEventListener('mousedown', close);
    document.addEventListener('keydown', esc);
    return () => {
      document.removeEventListener('mousedown', close);
      document.removeEventListener('keydown', esc);
    };
  }, [exportOpen]);

  const doExport = async (scope: 'all' | 'filtered') => {
    setExportOpen(false);
    const list = scope === 'all' ? fields : derived.filtered;
    const csv = buildCsv(list, decisions, {
      region: regionName(selection.regionId), cropYear: selection.cropYear, season: seasonLabel(selection.seasonId),
      latestObservation: status?.latestObservation ?? null, scope, allFields: fields,
    });
    const name = `harvest-console_barc_${selection.cropYear}_${scope === 'all' ? 'all-fields' : 'filtered'}_${new Date().toISOString().slice(0, 10)}.csv`;
    const r = await saveFile(name, csv);
    showToast(r === 'saved' ? `Exported ${list.length} fields to ${name}` : r === 'declined' ? 'Export cancelled' : 'Export is not available in this view. Try opening the page directly.');
  };

  const ready = status?.state === 'ready';
  return (
    <header className="ctx">
      <div className="ctx-brand">
        <span className="ctx-name">Harvest Console</span>
        <span className="ctx-sub">Silage vs grain review</span>
      </div>
      <div className="ctx-selects">
        <label>
          <span>Area</span>
          <select id="sel-region" value={selection.regionId} onChange={(e) => {
            const r = REGIONS.find((x) => x.id === e.target.value);
            setSelection({ ...selection, regionId: e.target.value, cropYear: r?.defaultYear ?? (selection.cropYear === 2025 ? 2023 : selection.cropYear) });
          }}>
            {REGIONS.map((r) => <option key={r.id} value={r.id}>{r.name}, {r.state}</option>)}
          </select>
        </label>
        <label>
          <span>Crop year</span>
          <select id="sel-year" value={selection.cropYear} onChange={(e) => setSelection({ ...selection, cropYear: Number(e.target.value) })}>
            {(selection.regionId === 'nl-se' ? [2025] : CROP_YEARS.filter((y) => y !== 2025)).map((y) => <option key={y} value={y}>{y}</option>)}
          </select>
        </label>
        <label>
          <span>Period</span>
          <select id="sel-season" value={selection.seasonId} onChange={(e) => setSelection({ ...selection, seasonId: e.target.value })}>
            {SEASONS.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
          </select>
        </label>
      </div>
      <div className={`ctx-status st-${loading ? 'loading' : status?.state ?? 'none'}`} role="status">
        {loading ? <Loader2 size={14} className="spin" aria-hidden /> : status?.state === 'ready' ? <Check size={14} aria-hidden /> : status?.state === 'processing' ? <CircleDot size={14} aria-hidden /> : <CircleSlash size={14} aria-hidden />}
        <span>
          {loading ? 'Loading analysis…' : status?.state === 'ready' ? 'Analysis ready' : status?.state === 'processing' ? 'Processing' : 'Not prepared'}
          {status?.latestObservation && !loading && <span className="muted"> · latest observation {fmtDate(status.latestObservation, true)}</span>}
        </span>
      </div>
      {ready && <PredictionToggle />}
      <div className="ctx-right">
        <span className="demo-badge" title={ready ? `${activeDataset().outlineSource}; ${activeDataset().obsSource}. ${activeDataset().validated ? 'Model evaluated on held-out regions against farmer-declared labels.' : 'Rule-based weak labels; not validated.'}` : 'Prototype'}>
          <FlaskConical size={13} aria-hidden /> {ready && activeDataset().key==='barc'?'BARC · development':ready && activeDataset().validated ? 'Validated (NL)' : 'Prototype'}
        </span>
        <div className="menu-wrap" ref={menuRef}>
          <button type="button" className="btn btn-primary btn-sm" disabled={!ready} onClick={() => setExportOpen((o) => !o)} aria-expanded={exportOpen} aria-haspopup="menu">
            <Download size={14} aria-hidden /> Export CSV
          </button>
          {exportOpen && (
            <div className="menu" role="menu">
              <button type="button" role="menuitem" onClick={() => doExport('all')}>
                <b>All loaded fields</b>
                <span>{fields.length} fields · all four sample areas</span>
              </button>
              <button type="button" role="menuitem" onClick={() => doExport('filtered')} disabled={!derived.active}>
                <b>Filtered fields only</b>
                <span>{derived.active ? `${derived.filtered.length} fields · file states what was excluded` : 'No filters applied'}</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

export function PredictionToggle() {
  const source = useStore((s) => s.predictionSource);
  const setSource = useStore((s) => s.setPredictionSource);
  const showToast = useStore((s) => s.showToast);
  const change = (v: PredictionSource) => {
    if (v === source) return;
    setSource(v);
    showToast(v === 'rules' ? 'Showing rule-based predictions' : 'Showing ML model predictions');
  };
  return (
    <div className="ctx-source">
      <span className="ctx-source-label">Prediction</span>
      <Segmented<PredictionSource>
        size="sm"
        label="Prediction source"
        value={source}
        onChange={change}
        options={[
          { value: 'rules', disabled: activeDataset().key==='barc', label: SOURCE_LABEL.rules, title: sourceNote('rules') },
          { value: 'ml', label: <>{SOURCE_LABEL.ml}{!mlAvailable() && <em className="soon">not trained</em>}</>, title: sourceNote('ml'), disabled: !mlAvailable() },
        ]}
      />
      <InfoTip label="About prediction sources">
        <b>Rule-based</b>: {sourceNote('rules')} Fields between the rules get a low-confidence lean and go to review.
        <br /><br />
        <b>ML model</b>: {sourceNote('ml')}
        <br /><br />
        Saved decisions stay as they are when you switch.
      </InfoTip>
    </div>
  );
}

export function Toast() {
  const toast = useStore((s) => s.toast);
  const close = useStore((s) => s.closeToast);
  const undo = useStore((s) => s.undo);
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(close, toast.undoable ? 8000 : 4500);
    return () => clearTimeout(t);
  }, [toast, close]);
  if (!toast) return null;
  return (
    <div className="toast" role="status" aria-live="polite" key={toast.id}>
      <Check size={15} aria-hidden />
      <span>{toast.message}</span>
      {toast.undoable && (
        <button type="button" className="toast-undo" onClick={undo}>
          <Undo2 size={14} aria-hidden /> Undo
        </button>
      )}
      <button type="button" className="icon-btn icon-btn-sm" onClick={close} aria-label="Dismiss notification">
        <X size={14} />
      </button>
    </div>
  );
}

export function FirstUseHint({ derived }: { derived: Derived }) {
  const dismissed = useStore((s) => s.hintDismissed);
  const dismiss = useStore((s) => s.dismissHint);
  const select = useStore((s) => s.select);
  if (dismissed || !derived.byId.size) return null;
  const picks = ['Confident silage', 'Strongest grain call', 'Uncertain: in review band', 'Clear silage transition', 'Clear grain dry-down', 'Long cloud gap at harvest', 'Between silage and grain', 'Radar does not confirm harvest', 'Large uncertain field']
    .map((label) => [...derived.byId.values()].find((f) => f.showcase === label))
    .filter(Boolean)
    .slice(0, 4);
  return (
    <div className="hint" role="dialog" aria-label="Getting started">
      <button type="button" className="icon-btn icon-btn-sm hint-x" onClick={() => dismiss(true)} aria-label="Dismiss">
        <X size={14} />
      </button>
      <p className="hint-lead">Select a field or start with the review queue.</p>
      <p className="muted small">The queue puts uncertain, high-acreage fields first. Try one of these:</p>
      <ul>
        {picks.map((f) => (
          <li key={f!.id}>
            <button type="button" className="hint-pick" onClick={() => select(f!.id)}>
              <span className="mono">{f!.id}</span> {f!.showcase}
            </button>
          </li>
        ))}
      </ul>
      <p className="fine">Shortcuts: <kbd>N</kbd> next field · <kbd>Esc</kbd> close details</p>
    </div>
  );
}

export function NotPrepared() {
  const status = useStore((s) => s.status);
  const selection = useStore((s) => s.selection);
  const setSelection = useStore((s) => s.setSelection);
  const [step, setStep] = useState<number | null>(null);
  const [done, setDone] = useState(false);
  const cancel = useRef<(() => void) | null>(null);
  useEffect(() => {
    setStep(null);
    setDone(false);
    return () => cancel.current?.();
  }, [selection]);
  if (!status || status.state === 'ready') return null;
  const area = `${regionName(selection.regionId)}, ${selection.cropYear} · ${seasonLabel(selection.seasonId)}`;
  const start = () => {
    setDone(false);
    cancel.current?.();
    cancel.current = simulatePreparation(setStep, () => setDone(true));
  };
  const processing = status.state === 'processing';
  return (
    <div className="np-wrap">
      <div className="np-card" role="region" aria-label="Analysis status">
        <div className={`np-badge ${processing ? 'np-proc' : ''}`}>{processing ? 'Processing' : 'Analysis not prepared'}</div>
        <h2>{area}</h2>
        <p>{status.message} This prototype contains BARC field-year data and South-east Netherlands 2025 parcels with Sentinel-1/2 observations and a trained model. It never shows one area's data under another area's name.</p>
        {processing && (
          <div className="np-progress">
            <div className="progress"><span style={{ width: `${(status.progress ?? 0) * 100}%` }} /></div>
            <span className="muted small">About {Math.round((status.progress ?? 0) * 100)}% complete (simulated). Latest observation {status.latestObservation ? fmtDate(status.latestObservation, true) : 'n/a'}.</span>
          </div>
        )}
        {!processing && step === null && (
          <button type="button" className="btn" onClick={start}>Simulate preparation</button>
        )}
        {step !== null && (
          <ol className="np-steps" aria-live="polite">
            {PREP_STEPS.map((s, i) => (
              <li key={s} className={done || i < step ? 'done' : i === step ? 'now' : ''}>
                {done || i < step ? <Check size={13} aria-hidden /> : i === step ? <Loader2 size={13} className="spin" aria-hidden /> : <span className="dot" aria-hidden />}
                {s}
              </li>
            ))}
          </ol>
        )}
        {done && (
          <p className="callout callout-info">
            Simulation finished. No real processing ran and no data exists for this selection, so nothing is shown. In
            production, preliminary estimates would appear here when the job completes.
          </p>
        )}
        <p className="fine">The preparation flow is simulated for demonstration. Nothing is sent or computed.</p>
        <div className="np-actions">
          <button type="button" className="btn btn-primary" onClick={() => setSelection(NL_SELECTION)}>Open South-east Netherlands 2025</button>
        </div>
      </div>
    </div>
  );
}
