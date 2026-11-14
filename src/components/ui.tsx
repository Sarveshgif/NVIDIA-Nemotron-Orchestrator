import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { Check, CircleHelp, Leaf, Split, Wheat, Info, Ban, Clock } from 'lucide-react';
import type { ReviewStatus } from '../types';
import type { EffectiveClass } from '../lib/acreage';
import { CLASS_LABEL, STATUS_LABEL, confidenceBand, fmtPct } from '../lib/format';
import { patternMarkup, type PatternKey } from '../lib/patterns';

export function TypeIcon({ cls, size = 14 }: { cls: EffectiveClass | 'unclassified'; size?: number }) {
  if (cls === 'silage') return <Leaf size={size} aria-hidden />;
  if (cls === 'grain') return <Wheat size={size} aria-hidden />;
  if (cls === 'mixed') return <Split size={size} aria-hidden />;
  return <CircleHelp size={size} aria-hidden />;
}

export function TypePill({ cls, source, compact }: { cls: EffectiveClass | 'unclassified'; source?: 'model' | 'analyst'; compact?: boolean }) {
  const k = cls === 'unclassified' ? 'unresolved' : cls;
  return (
    <span className={`pill pill-${k} ${source === 'model' ? 'pill-model' : ''}`} title={source ? (source === 'model' ? 'Model prediction' : 'Analyst decision') : undefined}>
      <TypeIcon cls={cls} size={12} />
      {!compact && CLASS_LABEL[cls]}
    </span>
  );
}

export function StatusPill({ status }: { status: ReviewStatus }) {
  const Icon = status === 'pending' ? Clock : status === 'insufficient' ? Ban : Check;
  return (
    <span className={`status status-${status}`}>
      <Icon size={11} aria-hidden />
      {STATUS_LABEL[status]}
    </span>
  );
}

export function ConfidenceMeter({ value, label = true }: { value: number | null; label?: boolean }) {
  const band = confidenceBand(value);
  const pct = value === null ? 0 : ((value - 0.5) / 0.5) * 100;
  return (
    <span className={`conf conf-${band}`} title={value === null ? 'No demo score: unclassified' : `Demo score ${fmtPct(value)} (${band})`}>
      <span className="conf-track" aria-hidden>
        <span className="conf-fill" style={{ width: `${Math.max(4, pct)}%` }} />
      </span>
      {label && <span className="conf-num">{fmtPct(value)}</span>}
    </span>
  );
}

export function Swatch({ k, w = 22, h = 14 }: { k: PatternKey; w?: number; h?: number }) {
  const id = `sw-${useId().replace(/:/g, '')}`;
  return (
    <svg width={w} height={h} className="swatch" aria-hidden>
      <defs dangerouslySetInnerHTML={{ __html: patternMarkup(k, id) }} />
      <rect x="0.5" y="0.5" width={w - 1} height={h - 1} rx="2" fill={`url(#${id})`} stroke="rgba(20,30,24,.45)" />
    </svg>
  );
}

export function OutlineSwatch({ dash, weight }: { dash?: string; weight: number }) {
  return (
    <svg width={22} height={14} className="swatch" aria-hidden>
      <rect x="2" y="2" width="18" height="10" rx="1" fill="none" stroke="var(--ink)" strokeWidth={weight} strokeDasharray={dash} strokeLinecap="round" />
    </svg>
  );
}

/** Small click/focus popover for explanations. */
export function InfoTip({ children, label = 'More information', icon }: { children: ReactNode; label?: string; icon?: ReactNode }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const id = useId();
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
    document.addEventListener('mousedown', close);
    document.addEventListener('keydown', esc);
    return () => {
      document.removeEventListener('mousedown', close);
      document.removeEventListener('keydown', esc);
    };
  }, [open]);
  return (
    <span className="infotip" ref={ref}>
      <button
        type="button"
        className="icon-btn icon-btn-sm"
        aria-label={label}
        aria-expanded={open}
        aria-describedby={open ? id : undefined}
        onClick={() => setOpen((o) => !o)}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
      >
        {icon ?? <Info size={14} aria-hidden />}
      </button>
      {open && (
        <span role="tooltip" id={id} className="infotip-pop">
          {children}
        </span>
      )}
    </span>
  );
}

export function Segmented<T extends string>({
  value, options, onChange, label, size,
}: {
  value: T;
  options: { value: T; label: ReactNode; title?: string; disabled?: boolean }[];
  onChange: (v: T) => void;
  label: string;
  size?: 'sm';
}) {
  return (
    <div className={`seg ${size === 'sm' ? 'seg-sm' : ''}`} role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          aria-disabled={o.disabled || undefined}
          disabled={o.disabled}
          className={value === o.value ? 'on' : ''}
          title={o.title}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Chip({ on, onClick, children }: { on: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button type="button" className={`chip ${on ? 'on' : ''}`} aria-pressed={on} onClick={onClick}>
      {on && <Check size={11} aria-hidden />}
      {children}
    </button>
  );
}
