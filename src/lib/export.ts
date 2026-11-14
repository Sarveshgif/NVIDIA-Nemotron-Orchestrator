import { activeDataset } from '../data/datasets';
import type { Decision, Field } from '../types';
import { allocate, summarize, silageTotal, grainTotal, unresolvedTotal } from './acreage';
import { REASON_META } from './format';

declare global {
  interface Window {
    claude?: { use: (name: string) => Promise<any> };
  }
}

const esc = (v: unknown) => {
  const s = v === null || v === undefined ? '' : String(v);
  return /[",\n#]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export interface ExportMeta {
  region: string;
  cropYear: number;
  season: string;
  latestObservation: string | null;
  scope: 'all' | 'filtered';
  allFields: Field[];
}

export function buildCsv(fields: Field[], decisions: Record<string, Decision>, meta: ExportMeta): string {
  const s = summarize(fields, decisions);
  const all = summarize(meta.allFields, decisions);
  const excluded = all.total - s.total;
  const lines: string[] = [];
  lines.push(`# Harvest Console export: ${meta.region}, crop year ${meta.cropYear}, ${meta.season}`);
  lines.push(`# Field outlines: ${activeDataset().outlineSource}. Observations: ${activeDataset().obsSource}. Not ownership records.`);
  const src = fields[0]?.prediction.source ?? 'ml';
  lines.push(`# Prediction source: ${src === 'rules' ? 'rule-based (' + (fields[0]?.prediction.modelVersion ?? '') + ')' : 'ML model (' + (fields[0]?.prediction.modelVersion ?? '') + ')'}.`);
  lines.push('# Totals are preliminary estimates. Analyst review means reviewed, not independently verified.');
  lines.push(
    meta.scope === 'all'
      ? `# Scope: all ${all.fieldCount} loaded fields (${all.total.toFixed(1)} ac). This is four sample areas, not complete county corn coverage.`
      : `# Scope: FILTERED subset of ${s.fieldCount} of ${all.fieldCount} loaded fields (${s.total.toFixed(1)} of ${all.total.toFixed(1)} ac). Excluded: ${all.fieldCount - s.fieldCount} fields, ${excluded.toFixed(1)} ac.`,
  );
  lines.push(
    `# Silage ${silageTotal(s).toFixed(1)} ac; grain ${grainTotal(s).toFixed(1)} ac; unresolved ${unresolvedTotal(s).toFixed(1)} ac; reviewed fields ${s.reviewedFields}/${s.fieldCount}.`,
  );
  lines.push(`# Latest observation: ${meta.latestObservation ?? 'n/a'}. Exported ${new Date().toISOString()}.`);
  const header = [
    'field_id', 'dwr_unique_id', 'area_acres', 'prediction_source', 'prediction', 'prediction_score', 'review_status', 'analyst_decision',
    'final_class', 'decision_source', 'silage_acres', 'grain_acres', 'unresolved_acres', 'review_reasons',
    'est_harvest_window_start', 'est_harvest_window_end', 'observation_coverage', 'analyst', 'decided_at', 'analyst_note',
  ];
  lines.push(header.join(','));
  for (const f of fields) {
    const d = decisions[f.id];
    const al = allocate(f, d);
    lines.push(
      [
        f.id, f.sourceId ?? '', f.acres.toFixed(1), f.prediction.source, f.prediction.predicted,
        f.prediction.confidence === null ? '' : f.prediction.confidence.toFixed(2),
        al.status, d ? (d.decided ?? 'insufficient_evidence') : '', al.effective, al.source,
        al.silage.toFixed(1), al.grain.toFixed(1), al.unresolved.toFixed(1),
        f.reasons.map((r) => REASON_META[r].short).join('; '),
        f.harvestWindow?.start ?? '', f.harvestWindow?.end ?? '', f.coverage,
        d?.analyst ?? '', d?.at ?? '', d?.note ?? '',
      ].map(esc).join(','),
    );
  }
  return lines.join('\n') + '\n';
}

/** Saves a file through the artifact host when available, else a normal browser download. */
export async function saveFile(filename: string, data: string): Promise<'saved' | 'declined' | 'failed'> {
  if (window.claude?.use) {
    // Inside the hosted viewer, page-initiated downloads are blocked; use the host's save flow.
    try {
      const dl = await window.claude.use('downloads');
      if (!dl) return 'failed';
      await dl.save({ filename, data });
      return 'saved';
    } catch (e: any) {
      return e && e.code === 'declined' ? 'declined' : 'failed';
    }
  }
  try {
    const url = URL.createObjectURL(new Blob([data], { type: 'text/csv;charset=utf-8' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
    return 'saved';
  } catch {
    return 'failed';
  }
}

export const isFramed = () => {
  try {
    return window.self !== window.top;
  } catch {
    return true;
  }
};
