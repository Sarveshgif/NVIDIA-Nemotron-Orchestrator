import type { Region, Season } from '../types';

// Approximate outlines for display only. Not survey-grade boundaries.
export const REGIONS: Region[] = [
  { id: 'barc-md', name: 'BARC corn fields', state: 'MD', defaultYear: 2023, center: [39.03, -76.85], outline: [[39.00,-76.90],[39.00,-76.80],[39.07,-76.80],[39.07,-76.90]] },
  { id: 'nl-se', name: 'South-east Netherlands', state: 'NL', defaultYear: 2025, center: [51.43, 6.21], outline: [[50.80,5.60],[50.80,6.82],[52.06,6.82],[52.06,5.60]] },
];

export const SEASONS: Season[] = [
  { id: 'summer-fall', label: 'Main season', start: '04-01', end: '12-31' },
];

export const CROP_YEARS = [2025, 2023, 2022, 2021];

/** The single fully prepared demonstration selection. */
export const DEMO_SELECTION = { regionId: 'barc-md', cropYear: 2023, seasonId: 'summer-fall' };

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

export const TOWNS: { name: string; at: [number, number]; size: number }[] = [];

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
