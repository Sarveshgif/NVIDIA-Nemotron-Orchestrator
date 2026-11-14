import { activeDataset } from '../data/datasets';
import { useMemo, useState } from 'react';
import { AlertTriangle, Download, RotateCcw } from 'lucide-react';
import { useStore } from '../state/store';
import type { Decision, Field } from '../types';
import {
  ABSTAIN_LABEL, DEFAULT_THRESHOLDS, RULESET_VERSION, type AbstainReason, type RuleResult, type RuleThresholds, type WeakLabel,
} from '../lib/rules';
import { fmtAcres, fmtDate } from '../lib/format';
import { saveFile } from '../lib/export';
import { regionName } from '../services/analysisService';
import { InfoTip, Segmented, TypePill } from './ui';

type TrainLabel = 'silage' | 'grain' | null;
type LabelSource = 'analyst' | 'weak' | 'none';
type Filter = 'all' | 'labelled' | 'abstain' | 'disagree';

interface Row {
  field: Field;
  rule: RuleResult;
  analyst: Decision | undefined;
  analystLabel: TrainLabel;
  train: TrainLabel;
  source: LabelSource;
  disagrees: boolean;
}

/** Analyst silage/grain decisions override weak labels. Mixed or insufficient never become a training label. */
function buildRows(fields: Field[], rules: Record<string, RuleResult>, decisions: Record<string, Decision>): Row[] {
  return fields.flatMap((field) => {
    const rule = rules[field.id];
    if (!rule) return [];
    const analyst = decisions[field.id];
    const analystLabel: TrainLabel = analyst?.decided === 'silage' || analyst?.decided === 'grain' ? analyst.decided : null;
    const weak: TrainLabel = rule.label === 'abstain' ? null : rule.label;
    const train = analystLabel ?? (analyst ? null : weak);
    const source: LabelSource = analystLabel ? 'analyst' : train ? 'weak' : 'none';
    return [{ field, rule, analyst, analystLabel, train, source, disagrees: !!analystLabel && !!weak && analystLabel !== weak }];
  });
}

const n = (v: number | null | undefined, d = 2) => (v === null || v === undefined ? '–' : v.toFixed(d));
const pct = (a: number, b: number) => (b ? `${Math.round((a / b) * 100)}%` : '–');

function Threshold({
  label, k, min, max, step, unit, help,
}: { label: string; k: keyof RuleThresholds; min: number; max: number; step: number; unit?: string; help: string }) {
  const value = useStore((s) => s.ruleThresholds[k]);
  const set = useStore((s) => s.setRuleThresholds);
  const changed = value !== DEFAULT_THRESHOLDS[k];
  const id = `th-${k}`;
  return (
    <div className={`th ${changed ? 'th-changed' : ''}`}>
      <label htmlFor={id} className="th-label">
        {label}
        <InfoTip label={`About ${label}`}>{help}</InfoTip>
      </label>
      <input id={id} type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => set({ [k]: Number(e.target.value) })} />
      <span className="th-val mono">{step < 1 ? value.toFixed(2) : value}{unit ? ` ${unit}` : ''}</span>
    </div>
  );
}

function WeakPill({ rule }: { rule: RuleResult }) {
  if (rule.label === 'abstain') return <span className="pill pill-abstain">Abstain</span>;
  return <TypePill cls={rule.label} />;
}

export function WeakLabelsView() {
  const fields = useStore((s) => s.baseFields);
  const rules = useStore((s) => s.ruleResults);
  const decisions = useStore((s) => s.decisions);
  const thresholds = useStore((s) => s.ruleThresholds);
  const reset = useStore((s) => s.resetRuleThresholds);
  const selection = useStore((s) => s.selection);
  const setView = useStore((s) => s.setView);
  const select = useStore((s) => s.select);
  const showToast = useStore((s) => s.showToast);
  const [filter, setFilter] = useState<Filter>('all');

  const rows = useMemo(() => buildRows(fields, rules, decisions), [fields, rules, decisions]);

  if (!fields.length) {
    return (
      <div className="page">
        <div className="state-box">
          <p>No fields are loaded for this selection, so there is nothing to label.</p>
          <button type="button" className="btn" onClick={() => setView('workspace')}>Back to workspace</button>
        </div>
      </div>
    );
  }

  const count = (lbl: WeakLabel) => rows.filter((r) => r.rule.label === lbl);
  const acres = (list: Row[]) => list.reduce((a, r) => a + r.field.acres, 0);
  const silage = count('silage'), grain = count('grain'), abstain = count('abstain');
  const byReason = (Object.keys(ABSTAIN_LABEL) as AbstainReason[])
    .map((k) => ({ k, rows: abstain.filter((r) => r.rule.abstain === k) }))
    .filter((x) => x.rows.length);
  const checked = rows.filter((r) => r.analystLabel && r.rule.label !== 'abstain');
  const agree = checked.filter((r) => !r.disagrees);
  const train = rows.filter((r) => r.train);
  const trainSilage = train.filter((r) => r.train === 'silage').length;
  const trainGrain = train.length - trainSilage;
  const minority = Math.min(trainSilage, trainGrain) / Math.max(1, train.length);
  const changedRules = (Object.keys(DEFAULT_THRESHOLDS) as (keyof RuleThresholds)[]).filter((k) => thresholds[k] !== DEFAULT_THRESHOLDS[k]).length;

  const shown = rows.filter((r) =>
    filter === 'labelled' ? r.rule.label !== 'abstain' : filter === 'abstain' ? r.rule.label === 'abstain' : filter === 'disagree' ? r.disagrees : true,
  );

  const open = (id: string) => {
    setView('workspace');
    select(id);
  };

  const exportCsv = async () => {
    const esc = (v: unknown) => {
      const s = v === null || v === undefined ? '' : String(v);
      return /[",\n#]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const lines = [
      `# Harvest Console weak labels: ${regionName(selection.regionId)}, crop year ${selection.cropYear}`,
      '# WEAK LABELS ARE RULE OUTPUTS, NOT GROUND TRUTH. training_label = analyst silage/grain decision if any, else the weak label; abstained, mixed and insufficient fields have none.',
      `# Field outlines: ${activeDataset().outlineSource}. Features: ${activeDataset().obsSource} (${activeDataset().attribution}).`,
      `# Rule set ${RULESET_VERSION}; thresholds ${JSON.stringify(thresholds)}. Exported ${new Date().toISOString()}.`,
      [
        'field_id', 'dwr_unique_id', 'cluster', 'area_acres', 'clear_obs', 'peak_ndvi', 'ndvi_before_drop', 'lswi_before_drop',
        'dry_down_days', 'drop_start', 'drop_end', 'gap_at_drop_days', 'radar_vh_drop_db',
        'weak_label', 'weak_rule', 'abstain_reason', 'analyst_decision', 'training_label', 'label_source',
      ].join(','),
    ];
    for (const r of rows) {
      const f = r.rule.features;
      lines.push([
        r.field.id, r.field.sourceId ?? '', r.field.cluster, r.field.acres.toFixed(1), f.clearObs, f.peakNdvi, f.ndviBeforeDrop, f.lswiBeforeDrop,
        f.dryDownDays, f.dropStart, f.dropEnd, f.gapAtDrop, f.radarDrop,
        r.rule.label, r.rule.rule, r.rule.abstain ? ABSTAIN_LABEL[r.rule.abstain] : '',
        r.analyst ? r.analyst.decided ?? 'insufficient_evidence' : '', r.train ?? '', r.source,
      ].map(esc).join(','));
    }
    const name = `harvest-console_weak-labels_tulare_${selection.cropYear}_${new Date().toISOString().slice(0, 10)}.csv`;
    const res = await saveFile(name, lines.join('\n') + '\n');
    showToast(res === 'saved' ? `Exported ${rows.length} fields (${train.length} with training labels) to ${name}` : res === 'declined' ? 'Export cancelled' : 'Export is not available in this view.');
  };

  return (
    <div className="page weak">
      <header className="page-head">
        <div>
          <div className="eyebrow">Weak labels · rule set {RULESET_VERSION} · Sentinel-2/1 2023</div>
          <h1>Weak label builder</h1>
          <p className="muted">
            Rules turn each field's satellite time series into a provisional silage or grain label. Only clear cases get a
            label; everything else abstains and goes to an analyst. Analyst decisions override weak labels. The export is the
            starting training set for the first ML model.
          </p>
        </div>
        <div className="page-actions">
          <button type="button" className="btn btn-primary" onClick={exportCsv}><Download size={15} /> Export training CSV</button>
        </div>
      </header>

      <p className="callout callout-warn">
        <AlertTriangle size={15} aria-hidden />
        <span>
          Weak labels are guesses from rules, not ground truth. Use analyst-reviewed fields as the test set, and never test a
          model on the weak labels it was trained on.
        </span>
      </p>

      <div className="report-grid">
        <section className="card">
          <div className="card-head">
            <h2>Rules</h2>
            <button type="button" className="btn btn-sm btn-quiet" onClick={reset} disabled={!changedRules}>
              <RotateCcw size={14} aria-hidden /> Reset{changedRules ? ` (${changedRules} changed)` : ''}
            </button>
          </div>
          <div className="rule-block rule-silage">
            <h3>Silage: green cut</h3>
            <p className="fine">Still green right before an abrupt drop.</p>
            <Threshold label="NDVI before drop at least" k="silageMinNdvi" min={0.4} max={0.85} step={0.01}
              help="NDVI at the last clear look before the harvest drop. Silage is chopped green, so this stays high." />
            <Threshold label="Dry-down at most" k="silageMaxDryDown" min={0} max={30} step={1} unit="d"
              help="Days the canopy spent more than a third of the way from its summer peak to bare soil before the drop. Zero for a green cut." />
          </div>
          <div className="rule-block rule-grain">
            <h3>Grain: long dry-down</h3>
            <p className="fine">Weeks of senescence before the final drop.</p>
            <Threshold label="Dry-down at least" k="grainMinDryDown" min={10} max={50} step={1} unit="d"
              help="Grain corn is left to dry in the field for several weeks before harvest." />
            <Threshold label="NDVI before drop at most" k="grainMaxNdvi" min={0.3} max={0.7} step={0.01}
              help="By grain harvest the canopy is already brown, so NDVI is low before the last drop." />
          </div>
          <div className="rule-block">
            <h3>Abstain when</h3>
            <Threshold label="Clear observations fewer than" k="minClearObs" min={5} max={30} step={1}
              help="Too few cloud-free Sentinel-2 looks in the season to judge the curve." />
            <Threshold label="Gap across drop longer than" k="maxGapAtDrop" min={5} max={40} step={1} unit="d"
              help="If cloud hides the change, its speed (abrupt or gradual) cannot be seen." />
            <Threshold label="Radar VH drop less than" k="minRadarDrop" min={0} max={5} step={0.1} unit="dB"
              help="Sentinel-1 VH backscatter normally falls 3–4 dB when the canopy is removed. Compares the two passes before the optical drop with the two after." />
          </div>
        </section>

        <section className="card">
          <h2>Result</h2>
          <div className="wl-tiles">
            <div className="wl-tile"><TypePill cls="silage" /><b>{silage.length}</b><span>{fmtAcres(acres(silage))} ac</span></div>
            <div className="wl-tile"><TypePill cls="grain" /><b>{grain.length}</b><span>{fmtAcres(acres(grain))} ac</span></div>
            <div className="wl-tile"><span className="pill pill-abstain">Abstain</span><b>{abstain.length}</b><span>{fmtAcres(acres(abstain))} ac</span></div>
          </div>
          <dl className="kv">
            <div><dt>Coverage</dt><dd>{silage.length + grain.length} of {rows.length} fields labelled ({pct(silage.length + grain.length, rows.length)})</dd></div>
            {byReason.map((x) => (
              <div key={x.k}><dt className="kv-sub">Abstain: {ABSTAIN_LABEL[x.k].toLowerCase()}</dt><dd>{x.rows.length}</dd></div>
            ))}
            <div>
              <dt>Agreement with analysts</dt>
              <dd>{checked.length ? `${agree.length} of ${checked.length} (${pct(agree.length, checked.length)})` : 'No reviewed fields yet'}</dd>
            </div>
          </dl>
          <h3>Training set</h3>
          <dl className="kv">
            <div><dt>Fields with a training label</dt><dd>{train.length} of {rows.length}</dd></div>
            <div><dt className="kv-sub">From analyst decisions</dt><dd>{train.filter((r) => r.source === 'analyst').length}</dd></div>
            <div><dt className="kv-sub">From weak labels</dt><dd>{train.filter((r) => r.source === 'weak').length}</dd></div>
            <div><dt>Class balance</dt><dd>{trainSilage} silage · {trainGrain} grain</dd></div>
          </dl>
          {train.length > 0 && minority < 0.2 && (
            <p className="callout callout-warn small"><AlertTriangle size={14} aria-hidden /> One class has under 20% of labels. Loosen its rule or review more of those fields.</p>
          )}
          <p className="fine">
            {rows.length} fields is enough to test the rules, not to train a reliable model. Aim for a few hundred labelled fields
            across several areas and years, split by area and year for testing.
          </p>
        </section>
      </div>

      <section className="card">
        <div className="card-head">
          <h2>Fields</h2>
          <Segmented<Filter>
            size="sm"
            label="Show fields"
            value={filter}
            onChange={setFilter}
            options={[
              { value: 'all', label: `All (${rows.length})` },
              { value: 'labelled', label: `Labelled (${silage.length + grain.length})` },
              { value: 'abstain', label: `Abstained (${abstain.length})` },
              { value: 'disagree', label: `Analyst disagrees (${rows.filter((r) => r.disagrees).length})` },
            ]}
          />
        </div>
        <div className="tbl-wrap">
          <table className="tbl tbl-weak">
            <thead>
              <tr>
                <th scope="col">Field</th><th scope="col" className="num">Acres</th><th scope="col" className="num">Clear obs</th>
                <th scope="col" className="num">NDVI before drop</th><th scope="col" className="num">Dry-down</th><th scope="col">Drop window</th>
                <th scope="col" className="num">Gap</th><th scope="col" className="num">Radar VH drop</th>
                <th scope="col">Weak label</th><th scope="col">Analyst</th><th scope="col">Training label</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => {
                const f = r.rule.features;
                return (
                  <tr key={r.field.id} className={r.disagrees ? 'row-warn' : undefined}>
                    <th scope="row"><button type="button" className="link-btn mono" onClick={() => open(r.field.id)}>{r.field.id}</button></th>
                    <td className="num">{fmtAcres(r.field.acres)}</td>
                    <td className={`num ${f.clearObs < thresholds.minClearObs ? 'cell-bad' : ''}`}>{f.clearObs}</td>
                    <td className="num">{n(f.ndviBeforeDrop)}</td>
                    <td className="num">{f.dryDownDays === null ? '–' : `${f.dryDownDays} d`}</td>
                    <td className="nowrap">{f.dropStart && f.dropEnd ? `${fmtDate(f.dropStart)} – ${fmtDate(f.dropEnd)}` : '–'}</td>
                    <td className={`num ${(f.gapAtDrop ?? 0) > thresholds.maxGapAtDrop ? 'cell-bad' : ''}`}>{f.gapAtDrop === null ? '–' : `${f.gapAtDrop} d`}</td>
                    <td className={`num ${f.radarDrop !== null && f.radarDrop < thresholds.minRadarDrop ? 'cell-bad' : ''}`}>{f.radarDrop === null ? '–' : `${f.radarDrop.toFixed(1)} dB`}</td>
                    <td>
                      <WeakPill rule={r.rule} />
                      <div className="fine">{r.rule.rule ?? (r.rule.abstain ? ABSTAIN_LABEL[r.rule.abstain] : '')}</div>
                    </td>
                    <td>
                      {r.analyst ? (r.analyst.decided ? <TypePill cls={r.analyst.decided} source="analyst" /> : <span className="pill pill-unresolved">Insufficient</span>) : <span className="muted">–</span>}
                      {r.disagrees && <div className="fine warn-ink">Disagrees with rules</div>}
                    </td>
                    <td>
                      {r.train ? <TypePill cls={r.train} /> : <span className="muted">None</span>}
                      <div className="fine">{r.source === 'analyst' ? 'analyst' : r.source === 'weak' ? 'weak label' : ''}</div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {!shown.length && <p className="muted small">No fields match this view.</p>}
        </div>
      </section>
      <p className="fine">Field outlines: {activeDataset().outlineSource}. The rules read real {activeDataset().obsSource} series. {activeDataset().attribution}.</p>
    </div>
  );
}
