import { activeDataset } from '../data/datasets';
import { useStore } from '../state/store';
import type { Derived } from '../state/derived';
import { grainTotal, silageTotal, unresolvedTotal, type AcreageSummary as S } from '../lib/acreage';
import { SOURCE_LABEL, fmtAcres } from '../lib/format';
import { InfoTip, Segmented, TypeIcon } from './ui';

export function CompositionBar({ s, tall }: { s: S; tall?: boolean }) {
  const total = s.total || 1;
  const segs = [
    { k: 'silage-r', v: s.silage.reviewed, label: 'Silage, reviewed' },
    { k: 'silage-m', v: s.silage.model, label: 'Silage, predicted' },
    { k: 'grain-r', v: s.grain.reviewed, label: 'Grain, reviewed' },
    { k: 'grain-m', v: s.grain.model, label: 'Grain, predicted' },
    { k: 'unres', v: unresolvedTotal(s), label: 'Unresolved' },
  ];
  return (
    <div className={`comp-bar ${tall ? 'comp-bar-tall' : ''}`} role="img"
      aria-label={segs.map((x) => `${x.label} ${fmtAcres(x.v)} acres`).join(', ')}>
      {segs.map((x) =>
        x.v > 0 ? (
          <span key={x.k} className={`seg-${x.k}`} style={{ width: `${(x.v / total) * 100}%` }} title={`${x.label}: ${fmtAcres(x.v)} ac`} />
        ) : null,
      )}
    </div>
  );
}

export function AcreageSummary({ derived }: { derived: Derived }) {
  const scope = useStore((s) => s.scope);
  const setScope = useStore((s) => s.setScope);
  const source = useStore((st) => st.predictionSource);
  const s = scope === 'filtered' ? derived.summaryFiltered : derived.summaryAll;
  const all = derived.summaryAll;
  const pctReviewed = all.queueFields ? (all.queueFields - all.queueRemaining) / all.queueFields : 0;
  const fields = useStore((st) => st.fields);
  const decisions = useStore((st) => st.decisions);
  // Calibrated expectation for totals: reviewed fields count as decided; the rest contribute p(silage) x acres.
  // Hard calls at the balanced-accuracy threshold over-count the minority (grain) class; on held-out NL fields the
  // expectation was within ~25% of declared grain area while hard calls were ~3x (see Reports).
  const expected = source === 'ml' && activeDataset().validated
    ? fields.reduce((a, f) => {
        const d = decisions[f.id];
        if (d && (d.decided === 'silage' || d.decided === 'grain')) { a[d.decided] += f.acres; return a; }
        const p = f.prediction.pSilage;
        if (p === null) return a;
        a.silage += p * f.acres; a.grain += (1 - p) * f.acres; return a;
      }, { silage: 0, grain: 0 })
    : null;

  return (
    <section className="acreage" aria-label="Acreage summary">
      <div className="acreage-top">
        <Segmented<'all' | 'filtered'>
          size="sm"
          label="Summary scope"
          value={scope}
          onChange={setScope}
          options={[
            { value: 'all', label: `All loaded fields (${all.fieldCount})` },
            { value: 'filtered', label: `Filtered (${derived.summaryFiltered.fieldCount})`, title: derived.active ? 'Fields matching current filters' : 'No filters applied; same as all fields' },
          ]}
        />
        <span className="prelim">
          Preliminary estimates · {SOURCE_LABEL[source].toLowerCase()} predictions
          <InfoTip label="About these totals">
            Acres come from {activeDataset().outlineSource} outlines; predictions from {activeDataset().obsSource} time series. Reviewed acres use the analyst decision; the rest use the active
            prediction source ({SOURCE_LABEL[source]}). Each field is counted once. Mixed fields are split only by defined subregions or entered
            proportions. Scope: {activeDataset().scopeNote}
          </InfoTip>
        </span>
      </div>
      <div className="acreage-stats">
        <div className="stat">
          <span className="stat-label">Total corn</span>
          <span className="stat-num">{fmtAcres(s.total)}<small> ac</small></span>
          <span className="stat-sub">{s.fieldCount} fields</span>
        </div>
        <div className="stat stat-silage">
          <span className="stat-label"><TypeIcon cls="silage" size={13} /> Silage</span>
          <span className="stat-num">{fmtAcres(silageTotal(s))}<small> ac</small></span>
          <span className="stat-sub">{fmtAcres(s.silage.reviewed)} reviewed</span>
        </div>
        <div className="stat stat-grain">
          <span className="stat-label"><TypeIcon cls="grain" size={13} /> Grain</span>
          <span className="stat-num">{fmtAcres(grainTotal(s))}<small> ac</small></span>
          <span className="stat-sub">{fmtAcres(s.grain.reviewed)} reviewed</span>
        </div>
        <div className="stat stat-unres">
          <span className="stat-label"><TypeIcon cls="unclassified" size={13} /> Unresolved</span>
          <span className="stat-num">{fmtAcres(unresolvedTotal(s))}<small> ac</small></span>
          <span className="stat-sub">
            {s.unresolved.insufficient > 0 ? `${fmtAcres(s.unresolved.insufficient)} marked insufficient` : 'unclassified by model'}
          </span>
        </div>
        <div className="stat stat-progress">
          <span className="stat-label">Review queue</span>
          <span className="stat-num">
            {all.queueFields - all.queueRemaining}<small> / {all.queueFields} done</small>
          </span>
          <span className="progress" aria-hidden>
            <span style={{ width: `${pctReviewed * 100}%` }} />
          </span>
          <span className="stat-sub">{all.reviewedFields} fields reviewed in total</span>
        </div>
      </div>
      <div className="acreage-bar-row">
        <CompositionBar s={s} />
        <span className="bar-key">
          <i className="k-solid" /> Reviewed <i className="k-pattern" /> Model <i className="k-unres" /> Unresolved
        </span>
      </div>
      {expected && scope === 'all' && (
        <p className="expected-line">
          Calibrated estimate for totals: <b>silage {fmtAcres(expected.silage)} ac</b> · <b>grain {fmtAcres(expected.grain)} ac</b>
          <InfoTip label="Why two sets of totals">
            The tiles above count each field by its hard call (threshold tuned for balanced accuracy), which over-counts grain.
            This line sums calibrated scores (p × acres) and uses analyst decisions where they exist. On held-out Dutch fields it
            came within about 25% of the declared grain area; the hard calls were about three times too high. Use this line for totals and the hard calls for field review.
          </InfoTip>
        </p>
      )}
    </section>
  );
}
