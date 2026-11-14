// Season of the active dataset. Tulare 2023 (Apr–Dec) by default; the Netherlands 2025 dataset sets Mar–Dec 2025.
// Day numbers are only ever differenced, so the fixed base below never biases gaps or windows.

export interface SeasonConfig {
  year: number;
  start: string;
  end: string;
  label: string;
  /** "Large" field threshold in acres, relative to typical field size in the dataset. */
  largeAcres: number;
  /** Summer peak search window and latest harvest drop (month-day) used by the rule set. */
  peakFrom: string;
  peakTo: string;
  dropTo: string;
}

const CA_2023: SeasonConfig = {
  year: 2023, start: '2023-04-01', end: '2023-12-31', label: 'Apr – Dec 2023', largeAcres: 100,
  peakFrom: '06-01', peakTo: '09-30', dropTo: '11-15',
};
let cfg: SeasonConfig = CA_2023;

export const setSeason = (c: SeasonConfig) => { cfg = c; };
export const season = () => cfg;
export const CA_SEASON = CA_2023;

const BASE = Date.parse('2000-01-01T00:00:00Z');
export const dayToIso = (t: number) => new Date(BASE + t * 864e5).toISOString().slice(0, 10);
export const isoToDay = (iso: string) => Math.round((Date.parse(iso + 'T00:00:00Z') - BASE) / 864e5);
export const toMs = (iso: string) => Date.parse(iso + 'T00:00:00Z');

/** Chart x-axis: season start to end, ticks on the first of each month after the start month. */
export const seasonDomain = (): [number, number] => [toMs(cfg.start), toMs(cfg.end)];
export const seasonTicks = () => {
  const m0 = Number(cfg.start.slice(5, 7)); // 1-based month of the start
  return Array.from({ length: 12 - m0 }, (_, k) => Date.UTC(cfg.year, m0 + k, 1));
};
