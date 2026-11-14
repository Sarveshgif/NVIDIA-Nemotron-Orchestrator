import { useEffect, useState } from 'react';
import { AlertTriangle, ArrowRight, ImageOff } from 'lucide-react';
import type { Field, FieldEvidence } from '../types';
import { NDVI_GRADIENT } from '../lib/imagery';
import { daysBetween, fmtDate } from '../lib/format';
import { activeDataset } from '../data/datasets';
import { Segmented } from './ui';

type ImageMode = 'natural' | 'ndvi';

/** Real Sentinel-2 L2A chip for one clear date, cut from the field's strip made by scripts/fetch-sentinel.py. */
function SceneChip({ field, date, mode }: { field: Field; date: string; mode: ImageMode }) {
  const dates = activeDataset().chipDates(field.id);
  const i = dates.indexOf(date);
  const label = `Sentinel-2 ${mode === 'ndvi' ? 'NDVI' : 'true-colour'} image of ${field.id} on ${fmtDate(date, true)}, field outlined in white`;
  if (i < 0) {
    return (
      <div className="chip-missing" role="img" aria-label={`No image available for ${field.id} on ${fmtDate(date, true)}`}>
        <ImageOff size={18} aria-hidden /> No image for this date
      </div>
    );
  }
  const n = dates.length;
  return (
    <div className="scene-chip" role="img" aria-label={label} style={{
      backgroundImage: `url(${import.meta.env.BASE_URL}${activeDataset().chipBase}${field.id}_${mode === 'ndvi' ? 'ndvi' : 'rgb'}.jpg)`,
      backgroundSize: `${n * 100}% 100%`,
      backgroundPosition: `${n > 1 ? (i / (n - 1)) * 100 : 0}% 0`,
    }} />
  );
}

export function EvidenceImagery({ field, evidence }: { field: Field; evidence: FieldEvidence }) {
  const [mode, setMode] = useState<ImageMode>('natural');
  const [layout, setLayout] = useState<'side' | 'swipe'>('side');
  const dates = evidence.optical.map((o) => o.date);
  const [before, setBefore] = useState(evidence.before?.date ?? dates[0]);
  const [after, setAfter] = useState(evidence.after?.date ?? dates[dates.length - 1]);
  const [swipe, setSwipe] = useState(50);

  useEffect(() => {
    setBefore(evidence.before?.date ?? dates[0]);
    setAfter(evidence.after?.date ?? dates[dates.length - 1]);
  }, [evidence]); // eslint-disable-line react-hooks/exhaustive-deps

  const gap = daysBetween(before, after);
  const isDefault = before === evidence.before?.date && after === evidence.after?.date;
  const obsAt = (d: string) => evidence.optical.find((o) => o.date === d);

  const DateSelect = ({ value, onChange, id, label }: { value: string; onChange: (v: string) => void; id: string; label: string }) => (
    <label className="img-date">
      <span className="sr-only">{label}</span>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
        {dates.map((d) => (
          <option key={d} value={d}>{fmtDate(d, true)}</option>
        ))}
      </select>
    </label>
  );

  return (
    <section className="ev-block" aria-labelledby="img-h">
      <div className="ev-head">
        <h3 id="img-h">Before and after</h3>
        <div className="ev-tools">
          <Segmented<ImageMode> size="sm" label="Image view" value={mode} onChange={setMode}
            options={[{ value: 'natural', label: 'Natural colour' }, { value: 'ndvi', label: 'NDVI' }]} />
          <Segmented<'side' | 'swipe'> size="sm" label="Comparison layout" value={layout} onChange={setLayout}
            options={[{ value: 'side', label: 'Side by side' }, { value: 'swipe', label: 'Swipe' }]} />
        </div>
      </div>

      {gap > 15 && (
        <div className="callout callout-warn" role="note">
          <AlertTriangle size={15} aria-hidden />
          <span>
            <strong>{gap} days between these images.</strong> No clear optical observation exists in between, so the
            exact harvest date and speed of change are not known.
          </span>
        </div>
      )}

      {layout === 'side' ? (
        <div className="img-pair">
          <figure className="img-panel">
            <div className="img-frame">
              <SceneChip field={field} date={before} mode={mode} />
              <span className="img-tag">Sentinel-2</span>
            </div>
            <figcaption>
              <span className="img-cap-label">Before</span>
              <DateSelect id="img-before" label="Before image date" value={before} onChange={setBefore} />
              <span className="muted">NDVI {obsAt(before)?.ndvi.toFixed(2)}</span>
            </figcaption>
          </figure>
          <figure className="img-panel">
            <div className="img-frame">
              <SceneChip field={field} date={after} mode={mode} />
              <span className="img-tag">Sentinel-2</span>
            </div>
            <figcaption>
              <span className="img-cap-label">After</span>
              <DateSelect id="img-after" label="After image date" value={after} onChange={setAfter} />
              <span className="muted">NDVI {obsAt(after)?.ndvi.toFixed(2)}</span>
            </figcaption>
          </figure>
        </div>
      ) : (
        <figure className="img-swipe">
          <div className="img-frame">
            <SceneChip field={field} date={before} mode={mode} />
            <div className="img-swipe-top" style={{ clipPath: `inset(0 0 0 ${swipe}%)` }}>
              <SceneChip field={field} date={after} mode={mode} />
            </div>
            <span className="img-swipe-line" style={{ left: `${swipe}%` }} aria-hidden />
            <span className="img-tag">Sentinel-2</span>
            <span className="img-corner img-corner-l">{fmtDate(before)}</span>
            <span className="img-corner img-corner-r">{fmtDate(after)}</span>
          </div>
          <label className="swipe-ctl">
            <span className="sr-only">Swipe position</span>
            <input id="img-swipe" type="range" min={0} max={100} value={swipe} onChange={(e) => setSwipe(Number(e.target.value))} />
          </label>
          <figcaption className="swipe-dates">
            <DateSelect id="img-before-s" label="Before image date" value={before} onChange={setBefore} />
            <ArrowRight size={14} aria-hidden />
            <DateSelect id="img-after-s" label="After image date" value={after} onChange={setAfter} />
          </figcaption>
        </figure>
      )}

      <div className="img-meta">
        <span><strong>{gap} days</strong> between observations</span>
        {field.harvestWindow && (
          <span>
            Estimated harvest window <strong>{fmtDate(field.harvestWindow.start)} – {fmtDate(field.harvestWindow.end)}</strong>
          </span>
        )}
        {!isDefault && evidence.before && evidence.after && (
          <button type="button" className="link-btn" onClick={() => { setBefore(evidence.before!.date); setAfter(evidence.after!.date); }}>
            Reset to harvest window
          </button>
        )}
      </div>
      {mode === 'ndvi' && (
        <div className="ndvi-scale" aria-label="NDVI colour scale from 0 to 0.9">
          <span>0</span>
          <span className="ndvi-ramp" style={{ background: `linear-gradient(90deg, ${NDVI_GRADIENT})` }} />
          <span>0.9</span>
          <span className="muted">bare soil → dense green canopy</span>
        </div>
      )}
      <p className="fine">
        Real Sentinel-2 L2A images (10 m pixels) for clear dates only; the field outline is drawn in white. Each pixel is
        about 10 × 10 m, so row-level detail is not visible. {activeDataset().attribution}.
      </p>
    </section>
  );
}
