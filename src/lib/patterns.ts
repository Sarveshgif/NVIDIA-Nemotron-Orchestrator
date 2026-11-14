import { C } from './palette';

// SVG fill patterns shared by the map and the legend, so meaning never relies on colour alone.
export type PatternKey =
  | 'silage-m' | 'silage-r' | 'grain-m' | 'grain-r' | 'mixed' | 'unres'
  | 'pri-high' | 'pri-med' | 'pri-low' | 'pri-done'
  | 'cov-ok' | 'cov-gap' | 'cov-none';

export function patternMarkup(key: PatternKey, id: string): string {
  const P = (w: number, body: string, rot = 0) =>
    `<pattern id="${id}" patternUnits="userSpaceOnUse" width="${w}" height="${w}"${rot ? ` patternTransform="rotate(${rot})"` : ''}>${body}</pattern>`;
  switch (key) {
    case 'silage-m':
      return P(8, `<rect width="8" height="8" fill="${C.silage}" fill-opacity="0.32"/><rect width="2.6" height="8" fill="${C.silage}" fill-opacity="0.85"/>`, 45);
    case 'silage-r':
      return P(8, `<rect width="8" height="8" fill="${C.silage}" fill-opacity="0.88"/><rect width="1.4" height="8" fill="#ffffff" fill-opacity="0.45"/>`, 45);
    case 'grain-m':
      return P(7, `<rect width="7" height="7" fill="${C.grain}" fill-opacity="0.3"/><circle cx="3.5" cy="3.5" r="1.7" fill="${C.grain}" fill-opacity="0.95"/>`);
    case 'grain-r':
      return P(7, `<rect width="7" height="7" fill="${C.grain}" fill-opacity="0.9"/><circle cx="3.5" cy="3.5" r="1.2" fill="#fff8e6" fill-opacity="0.75"/>`);
    case 'mixed':
      return P(10, `<rect width="5" height="10" fill="${C.silage}" fill-opacity="0.9"/><rect x="5" width="5" height="10" fill="${C.grain}" fill-opacity="0.9"/>`, 45);
    case 'unres':
      return P(8, `<rect width="8" height="8" fill="${C.unresolved}" fill-opacity="0.45"/><path d="M0 0L8 8M8 0L0 8" stroke="#3e4440" stroke-width="0.9" stroke-opacity="0.7"/>`);
    case 'pri-high':
      return P(8, `<rect width="8" height="8" fill="${C.amber}" fill-opacity="0.88"/><rect width="2.2" height="8" fill="#3a1f00" fill-opacity="0.35"/>`, -45);
    case 'pri-med':
      return P(8, `<rect width="8" height="8" fill="${C.amberSoft}" fill-opacity="0.85"/>`);
    case 'pri-low':
      return P(8, `<rect width="8" height="8" fill="${C.sand}" fill-opacity="0.6"/>`);
    case 'pri-done':
      return P(8, `<rect width="8" height="8" fill="#6f8f7a" fill-opacity="0.55"/><circle cx="4" cy="4" r="1" fill="#ffffff" fill-opacity="0.8"/>`);
    case 'cov-ok':
      return P(8, `<rect width="8" height="8" fill="${C.covered}" fill-opacity="0.55"/>`);
    case 'cov-gap':
      return P(6, `<rect width="6" height="6" fill="${C.coveredSoft}" fill-opacity="0.65"/><rect width="6" height="2" fill="${C.covered}" fill-opacity="0.9"/>`);
    case 'cov-none':
      return patternMarkup('unres', id);
  }
}

export const ALL_PATTERNS: PatternKey[] = [
  'silage-m', 'silage-r', 'grain-m', 'grain-r', 'mixed', 'unres',
  'pri-high', 'pri-med', 'pri-low', 'pri-done', 'cov-ok', 'cov-gap', 'cov-none',
];
