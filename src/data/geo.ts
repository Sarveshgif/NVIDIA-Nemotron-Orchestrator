import type { Region, Season } from '../types';

// Approximate outlines for display only. Not survey-grade boundaries.
export const REGIONS: Region[] = [
  {
    id: 'nl-se',
    name: 'South-east Netherlands (validation)',
    state: 'NL',
    center: [51.4, 5.9],
    defaultYear: 2025,
    // filled from the dataset's block outline at load; rough hull of Limburg / Noord-Brabant / Achterhoek blocks
    outline: [[50.83, 5.64], [50.83, 6.03], [51.2, 6.06], [51.74, 6.13], [52.03, 6.78], [51.9, 6.78], [51.77, 5.48], [51.63, 5.02], [51.36, 5.02], [51.2, 5.7]],
  },
  {
    id: 'tulare-ca',
    name: 'Tulare County',
    state: 'CA',
    center: [36.22, -119.0],
    outline: [
      [36.744, -119.305], [36.665, -119.312], [36.585, -119.462], [36.49, -119.545],
      [36.27, -119.545], [35.79, -119.545], [35.79, -118.005], [36.03, -117.985],
      [36.2, -118.06], [36.38, -118.13], [36.55, -118.35], [36.66, -118.37],
      [36.744, -118.66], [36.744, -118.98],
    ],
  },
  {
    id: 'kings-ca',
    name: 'Kings County',
    state: 'CA',
    center: [36.07, -119.82],
    outline: [
      [36.49, -119.96], [36.49, -119.545], [35.79, -119.545], [35.79, -119.96],
      [35.98, -120.19], [36.2, -120.14], [36.4, -119.96],
    ],
  },
  {
    id: 'fresno-ca',
    name: 'Fresno County',
    state: 'CA',
    center: [36.75, -119.65],
    outline: [
      [37.1, -120.5], [37.22, -119.65], [37.4, -119.3], [37.45, -118.8], [36.95, -118.4],
      [36.744, -118.66], [36.744, -119.305], [36.49, -119.545], [36.4, -119.96], [36.2, -120.14],
      [36.0, -120.6], [36.48, -120.92], [36.9, -120.65],
    ],
  },
  {
    id: 'kern-ca',
    name: 'Kern County',
    state: 'CA',
    center: [35.34, -118.73],
    outline: [
      [35.79, -119.96], [35.79, -117.63], [34.82, -117.63], [34.82, -118.88],
      [35.0, -119.47], [35.43, -119.88],
    ],
  },
  {
    id: 'merced-ca',
    name: 'Merced County',
    state: 'CA',
    center: [37.19, -120.72],
    outline: [
      [37.63, -120.39], [37.18, -120.05], [37.1, -120.5], [36.9, -120.65], [36.75, -120.92],
      [36.95, -121.23], [37.18, -121.23], [37.42, -120.95],
    ],
  },
];

export const SEASONS: Season[] = [
  { id: 'summer-fall', label: 'Main season', start: '04-01', end: '12-31' },
  { id: 'spring', label: 'Spring silage', start: '04-01', end: '06-30' },
];

export const CROP_YEARS = [2025, 2023, 2022, 2021];

/** The single fully prepared demonstration selection. */
export const DEMO_SELECTION = { regionId: 'tulare-ca', cropYear: 2023, seasonId: 'summer-fall' };

// Reference lines for the offline schematic basemap.
export const HIGHWAYS: { name: string; path: [number, number][] }[] = [
  {
    name: 'SR 99',
    path: [
      [36.8, -119.85], [36.74, -119.79], [36.6, -119.67], [36.51, -119.55], [36.43, -119.48],
      [36.35, -119.42], [36.21, -119.345], [36.06, -119.31], [35.97, -119.29], [35.88, -119.27],
      [35.77, -119.25], [35.6, -119.2],
    ],
  },
  { name: 'SR 198', path: [[36.33, -119.8], [36.33, -119.65], [36.325, -119.42], [36.33, -119.29], [36.36, -119.1], [36.38, -119.03], [36.44, -118.9]] },
  { name: 'SR 190', path: [[36.06, -119.55], [36.06, -119.31], [36.065, -119.1], [36.07, -119.02], [36.12, -118.82]] },
  { name: 'SR 137', path: [[36.21, -119.56], [36.21, -119.345], [36.205, -119.09]] },
  { name: 'SR 65', path: [[35.77, -119.1], [36.07, -119.02], [36.2, -119.09], [36.32, -119.13]] },
  { name: 'SR 63', path: [[36.6, -119.42], [36.33, -119.29], [36.21, -119.31]] },
];

export const TOWNS: { name: string; at: [number, number]; size: number }[] = [
  { name: 'Visalia', at: [36.33, -119.29], size: 3 },
  { name: 'Tulare', at: [36.21, -119.345], size: 2.4 },
  { name: 'Porterville', at: [36.065, -119.02], size: 2.2 },
  { name: 'Hanford', at: [36.327, -119.645], size: 2.2 },
  { name: 'Dinuba', at: [36.543, -119.387], size: 1.4 },
  { name: 'Lindsay', at: [36.203, -119.088], size: 1.2 },
  { name: 'Exeter', at: [36.296, -119.142], size: 1.1 },
  { name: 'Tipton', at: [36.059, -119.312], size: 0.7 },
  { name: 'Pixley', at: [35.969, -119.291], size: 0.8 },
  { name: 'Earlimart', at: [35.884, -119.271], size: 0.9 },
  { name: 'Traver', at: [36.455, -119.483], size: 0.6 },
  { name: 'Delano', at: [35.769, -119.247], size: 2 },
];

// --- geometry helpers ---
const M_PER_DEG_LAT = 110574;
const mPerDegLng = (lat: number) => 111320 * Math.cos((lat * Math.PI) / 180);

/** Planar area of a small lat/lng ring, in acres. */
export function ringAcres(ring: [number, number][]): number {
  const lat0 = ring.reduce((s, p) => s + p[0], 0) / ring.length;
  const kx = mPerDegLng(lat0);
  let a = 0;
  for (let i = 0; i < ring.length; i++) {
    const [y1, x1] = ring[i];
    const [y2, x2] = ring[(i + 1) % ring.length];
    a += x1 * kx * (y2 * M_PER_DEG_LAT) - x2 * kx * (y1 * M_PER_DEG_LAT);
  }
  return Math.abs(a / 2) / 4046.8564;
}

export function ringCentroid(ring: [number, number][]): [number, number] {
  let lat = 0;
  let lng = 0;
  for (const p of ring) {
    lat += p[0];
    lng += p[1];
  }
  return [lat / ring.length, lng / ring.length];
}

export function ringBounds(ring: [number, number][]) {
  let minLat = Infinity, maxLat = -Infinity, minLng = Infinity, maxLng = -Infinity;
  for (const [la, ln] of ring) {
    minLat = Math.min(minLat, la); maxLat = Math.max(maxLat, la);
    minLng = Math.min(minLng, ln); maxLng = Math.max(maxLng, ln);
  }
  return { minLat, maxLat, minLng, maxLng };
}

/** Quarter-section (½ mile) cell size in degrees near 36°N. */
export const CELL_LAT = 804.67 / M_PER_DEG_LAT;
export const CELL_LNG = 804.67 / mPerDegLng(36.1);
