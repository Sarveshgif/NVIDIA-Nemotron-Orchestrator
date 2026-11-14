import { useEffect, useMemo, useState } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import type { Field, FieldEvidence } from '../types';
import { patchColorAt } from '../lib/imagery';
import { seasonTicks } from '../data/season';
import { daysBetween, fmtAcres, fmtDate } from '../lib/format';
import { C } from '../lib/palette';
import { TypePill } from './ui';

const t = (iso: string) => Date.parse(iso + 'T00:00:00Z');
const monthFmt = (v: number) => new Date(v).toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' });

export function PatchView({ field, evidence }: { field: Field; evidence: FieldEvidence }) {
  const dates = evidence.optical.map((o) => o.date);
  const defaultIdx = Math.max(0, dates.indexOf(evidence.after?.date ?? dates[0]));
  const [idx, setIdx] = useState(defaultIdx);
  useEffect(() => setIdx(defaultIdx), [evidence]); // eslint-disable-line react-hooks/exhaustive-deps
  const date = dates[idx] ?? dates[0];

  const starts = evidence.patches.map((p) => p.changeWindow?.start).filter(Boolean) as string[];
  const sorted = [...starts].sort();
  const spread = sorted.length > 1 ? daysBetween(sorted[0], sorted[sorted.length - 1]) : 0;
  const earliest = sorted[0];

  const chartData = useMemo(
    () => dates.map((d) => ({ t: t(d), ...Object.fromEntries(evidence.patches.map((p) => [p.id, p.ndvi.find((n) => n.date === d)?.value])) })),
    [evidence], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const early = (pid: string) => {
    const p = evidence.patches.find((x) => x.id === pid);
    return !!p?.changeWindow && daysBetween(earliest, p.changeWindow.start) <= 10;
  };

  return (
    <div className="patch">
      <p className={`patch-verdict ${spread > 14 ? 'is-var' : ''}`}>
        {spread > 14
          ? `Strong variation: parts of the field changed up to ${spread} days apart.`
          : `Uniform change: all parts of the field changed within ${Math.max(spread, 5)} days.`}
        <span className="muted"> This shows variation in timing. It does not by itself mean the field had two harvest types.</span>
      </p>
      <div className="patch-row">
        <div className="patch-grid-wrap">
          <div className="patch-grid" style={{ gridTemplateColumns: `repeat(${Math.max(...evidence.patches.map((q) => q.col)) + 1}, 1fr)` }} role="img" aria-label={`NDVI by part of field on ${fmtDate(date, true)}`}>
            {evidence.patches.map((p, i) => (
              <div key={p.id} className="patch-cell" style={{ background: patchColorAt(evidence, i, date) }}>
                <span>{p.label}</span>
                <b>{p.changeWindow ? `${fmtDate(p.changeWindow.start)}–${fmtDate(p.changeWindow.end)}` : 'no change'}</b>
              </div>
            ))}
          </div>
          <label className="patch-scrub">
            <span>NDVI on <b>{fmtDate(date, true)}</b></span>
            <input id="patch-date" type="range" min={0} max={dates.length - 1} value={idx} onChange={(e) => setIdx(Number(e.target.value))} aria-label="Observation date" />
          </label>
        </div>
        <div className="patch-chart">
          <ResponsiveContainer width="100%" height={150}>
            <LineChart data={chartData} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid vertical={false} stroke="var(--grid)" />
              <XAxis dataKey="t" type="number" scale="time" domain={['dataMin', 'dataMax']} tickFormatter={monthFmt} tick={{ fontSize: 10, fill: 'var(--muted)' }} stroke="var(--line)" ticks={seasonTicks()} />
              <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} width={30} tick={{ fontSize: 10, fill: 'var(--muted)' }} stroke="var(--line)" />
              <Tooltip
                labelFormatter={(v) => fmtDate(new Date(Number(v)).toISOString().slice(0, 10), true)}
                formatter={(v: number, name: string) => [v?.toFixed(2), evidence.patches.find((p) => p.id === name)?.label ?? name]}
                contentStyle={{ fontSize: 12 }}
              />
              {evidence.patches.map((p) => (
                <Line key={p.id} dataKey={p.id} stroke={early(p.id) ? C.silage : C.grain} strokeWidth={1.3} dot={{ r: 1.6 }} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
          <div className="chart-key">
            <span><i style={{ background: C.silage }} /> Changed in earliest window</span>
            <span><i style={{ background: C.grain }} /> Changed later</span>
          </div>
        </div>
      </div>

      {field.subregions && (
        <div className="subregions">
          <div className="sub-head">
            <strong>Reviewed subregions</strong>
            <span className="tag tag-demo">Analyst-defined demo case</span>
          </div>
          <p className="muted small">
            These subregions come from analyst review, not from the model. The model output for this field stays a single
            field-level prediction.
          </p>
          <div className="sub-body">
            <svg viewBox="0 0 120 60" className="sub-diagram" role="img" aria-label="Field split into subregions">
              {field.subregions.map((s) => (
                <polygon key={s.id} points={s.shape.map(([x, y]) => `${x * 120},${y * 60}`).join(' ')}
                  fill={s.type === 'silage' ? C.silage : C.grain} fillOpacity={0.85} stroke="#fff" strokeWidth={1.5} />
              ))}
              {field.subregions.map((s) => {
                const cx = (s.shape.reduce((a, p) => a + p[0], 0) / s.shape.length) * 120;
                return <text key={s.id} x={cx} y={34} textAnchor="middle" fontSize="9" fill="#fff" fontWeight="600">{s.type === 'silage' ? 'Silage' : 'Grain'}</text>;
              })}
            </svg>
            <ul>
              {field.subregions.map((s) => (
                <li key={s.id}>
                  <TypePill cls={s.type} source="analyst" /> <b>{s.label}</b> · {fmtAcres(s.acres)} ac
                  <div className="muted small">{s.note} Reviewed by {s.reviewedBy}.</div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
