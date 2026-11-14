import { useMemo, useState } from 'react';
import {
  CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { AlertTriangle } from 'lucide-react';
import type { Field, FieldEvidence } from '../types';
import { C } from '../lib/palette';
import { fmtDate } from '../lib/format';
import { season, seasonDomain, seasonTicks } from '../data/season';

const t = (iso: string) => Date.parse(iso + 'T00:00:00Z');
const monthFmt = (v: number) => new Date(v).toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' });

interface Series {
  key: string;
  label: string;
  color: string;
}

interface PanelProps {
  id: string;
  title: string;
  unit: string;
  data: Record<string, number>[];
  series: Series[];
  domain: [number, number];
  ticks: number[];
  fmt: (v: number) => string;
  field: Field;
  evidence: FieldEvidence;
  hoverT: number | null;
  setHoverT: (v: number | null) => void;
  showX: boolean;
  showGaps: boolean;
}

function ChartTip({ active, payload, unit, fmt }: any) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="chart-tip">
      <div className="chart-tip-date">{fmtDate(new Date(row.t).toISOString().slice(0, 10), true)} · observed</div>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="chart-tip-row">
          <i style={{ background: p.color }} /> {p.name}: <b>{fmt(p.value)}</b> {unit}
        </div>
      ))}
    </div>
  );
}

function Panel(p: PanelProps) {
  const win = p.field.harvestWindow;
  return (
    <div className="chart-panel">
      <div className="chart-label">
        <span className="chart-title">{p.title}</span>
        <span className="chart-legend">
          {p.series.map((s) => (
            <span key={s.key}><i style={{ background: s.color }} />{s.label}</span>
          ))}
          {p.unit && <span className="muted">{p.unit}</span>}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={p.showX ? 132 : 112}>
        <ComposedChart
          data={p.data}
          margin={{ top: 6, right: 10, bottom: 0, left: 0 }}
          onMouseMove={(s: any) => (s && s.activeLabel !== undefined ? p.setHoverT(Number(s.activeLabel)) : undefined)}
          onMouseLeave={() => p.setHoverT(null)}
        >
          <defs>
            <pattern id={`gap-${p.id}`} patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">
              <rect width="6" height="6" fill="var(--gap-bg)" />
              <rect width="1.4" height="6" fill="var(--gap-line)" />
            </pattern>
          </defs>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis
            dataKey="t" type="number" scale="time" domain={seasonDomain()} ticks={seasonTicks()} tickFormatter={monthFmt}
            hide={!p.showX} tick={{ fontSize: 11, fill: 'var(--muted)' }} stroke="var(--line)" height={20}
          />
          <YAxis domain={p.domain} ticks={p.ticks} tickFormatter={p.fmt} width={40} tick={{ fontSize: 11, fill: 'var(--muted)' }} stroke="var(--line)" />
          {p.showGaps &&
            p.evidence.gaps.map((g) => (
              <ReferenceArea key={g.start} x1={t(g.start)} x2={t(g.end)} fill={`url(#gap-${p.id})`} fillOpacity={1} ifOverflow="hidden" />
            ))}
          {win && (
            <ReferenceArea x1={t(win.start)} x2={t(win.end)} fill={C.forest} fillOpacity={0.14} stroke={C.forest} strokeOpacity={0.5} strokeDasharray="3 3" ifOverflow="hidden" />
          )}
          {p.hoverT !== null && <ReferenceLine x={p.hoverT} stroke="var(--ink)" strokeOpacity={0.55} />}
          <Tooltip content={<ChartTip unit={p.unit} fmt={p.fmt} />} cursor={false} isAnimationActive={false} />
          {p.series.map((s) => (
            <Line
              key={s.key} dataKey={s.key} name={s.label} type="linear" stroke={s.color} strokeWidth={1.4} strokeDasharray="4 3"
              dot={{ r: 2.6, fill: s.color, stroke: s.color }} activeDot={{ r: 4.5 }} isAnimationActive={false} connectNulls
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function EvidenceCharts({ field, evidence }: { field: Field; evidence: FieldEvidence }) {
  const [hoverT, setHoverT] = useState<number | null>(null);
  const optical = useMemo(() => evidence.optical.map((o) => ({ t: t(o.date), ndvi: o.ndvi, lswi: o.lswi })), [evidence]);
  const radar = useMemo(() => evidence.radar.map((r) => ({ t: t(r.date), vv: r.vv, vh: r.vh })), [evidence]);

  const nearest = <T extends { t: number }>(arr: T[], at: number) =>
    arr.reduce<T | null>((best, r) => (!best || Math.abs(r.t - at) < Math.abs(best.t - at) ? r : best), null);
  const nOpt = hoverT !== null ? nearest(optical, hoverT) : null;
  const nRad = hoverT !== null ? nearest(radar, hoverT) : null;
  const nearGap = field.maxGapNearHarvest > 15 ? evidence.gaps.find((g) => g.days === field.maxGapNearHarvest) ?? evidence.gaps[0] : null;
  const common = { field, evidence, hoverT, setHoverT };
  const iso = (v: number) => new Date(v).toISOString().slice(0, 10);

  return (
    <section className="ev-block" aria-labelledby="ts-h">
      <div className="ev-head">
        <h3 id="ts-h">Time series</h3>
        <span className="muted small">{season().label} · Sentinel-2 clear looks, Sentinel-1 orbit passes</span>
      </div>
      {nearGap && (
        <div className="callout callout-warn" role="note">
          <AlertTriangle size={15} aria-hidden />
          <span>
            <strong>Observation gap near harvest:</strong> no clear optical image for {nearGap.days} days
            ({fmtDate(nearGap.start)} – {fmtDate(nearGap.end)}). Radar passes continued, but optical values in this
            stretch are not observed.
          </span>
        </div>
      )}
      <div className="readout" aria-live="polite">
        {hoverT === null ? (
          <span className="muted">Hover any chart to compare dates across sensors.</span>
        ) : (
          <>
            {nOpt && <span>Optical {fmtDate(iso(nOpt.t))}: NDVI <b>{nOpt.ndvi.toFixed(2)}</b>, LSWI <b>{nOpt.lswi.toFixed(2)}</b></span>}
            {nRad && <span>Radar {fmtDate(iso(nRad.t))}: VV <b>{nRad.vv.toFixed(1)}</b>, VH <b>{nRad.vh.toFixed(1)}</b> dB</span>}
          </>
        )}
      </div>
      <div className="charts">
        <Panel {...common} id="ndvi" title="NDVI · vegetation greenness" unit="index" data={optical}
          series={[{ key: 'ndvi', label: 'NDVI (Sentinel-2)', color: C.ndvi }]} domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]}
          fmt={(v) => v.toFixed(2)} showX={false} showGaps />
        <Panel {...common} id="lswi" title="LSWI · canopy moisture" unit="index" data={optical}
          series={[{ key: 'lswi', label: 'LSWI (Sentinel-2)', color: C.lswi }]} domain={[-0.2, 0.4]} ticks={[-0.2, 0, 0.2, 0.4]}
          fmt={(v) => v.toFixed(2)} showX={false} showGaps />
        <Panel {...common} id="radar" title="Radar backscatter" unit="dB" data={radar}
          series={[{ key: 'vv', label: 'VV', color: C.vv }, { key: 'vh', label: 'VH', color: C.vh }]} domain={[-28, -2]} ticks={[-26, -20, -14, -8, -2]}
          fmt={(v) => v.toFixed(0)} showX showGaps={false} />
      </div>
      <div className="chart-key">
        <span><svg width="26" height="8" aria-hidden><circle cx="4" cy="4" r="2.6" fill="var(--ink)" /><line x1="8" y1="4" x2="26" y2="4" stroke="var(--ink)" strokeDasharray="3 2" /></svg> Dots are observations; dashes only connect them</span>
        <span><i className="key-window" /> Estimated harvest window</span>
        <span><i className="key-gap" /> No clear optical image (&gt; 10 days)</span>
      </div>
      <details className="explain">
        <summary>What am I looking at?</summary>
        <ul>
          <li><b>NDVI</b> measures green leaf cover. Corn cut for silage stays green, then drops sharply in one pass. Corn left for grain dries down over several weeks first.</li>
          <li><b>LSWI</b> tracks water in leaves and soil. Grain corn loses moisture well before harvest; silage is chopped while still moist.</li>
          <li><b>Radar (Sentinel-1)</b> sees through cloud every 12 days. VH responds to canopy volume and drops when the crop is removed. VV reflects soil and stalks.</li>
          <li>The <b>harvest window</b> is the span between the last image before the change and the first one after. The actual date is somewhere inside it.</li>
          <li>Values between dots are not measured. Dashed lines connect observations for readability only.</li>
        </ul>
      </details>
    </section>
  );
}
