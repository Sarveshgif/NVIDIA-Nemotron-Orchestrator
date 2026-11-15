import type { DatasetInfo } from './datasets';
import type { Field, FieldEvidence } from '../types';
interface RecordRow {
  id: string; year: number; geometryRings: [number, number][][][]; centroid: [number, number]; acres: number;
  rf: { p_silage: number; p_grain: number; predicted_label: 'GRAIN' | 'SILAGE'; model_version: string };
  harvest: { start: string; end: string } | null;
}
let cached: Promise<{ fields: RecordRow[] }> | null = null;
export async function loadBarc(year: number) {
  if (!cached) cached = fetch(`${import.meta.env.BASE_URL}barc/barc.json`).then(r => { if (!r.ok) throw new Error('BARC dataset could not be loaded.'); return r.json(); });
  const file = await cached;
  const rows = file.fields.filter(f => f.year === year);
  if (!rows.length) throw new Error('BARC is available for 2021, 2022 and 2023.');
  const info: DatasetInfo = {
    key: 'barc', regionId: 'barc-md', cropYear: year, validated: false,
    outlineSource: 'BARC Corn 2021–2023 field polygons', obsSource: 'Sentinel-2 L2A / Sentinel-1 RTC on demand',
    attribution: 'BARC field outlines. Contains modified Copernicus Sentinel data when retrieved.',
    chipBase: 'barc/', mlAvailable: true,
    mlNote: 'Saved BARC-recalibrated RF. Development data; reference labels were used in recalibration. Below 80%, Nemotron can assess dated satellite and weather evidence.',
    rulesNote: 'Rules are unavailable without a complete observation series.', scopeNote: '44 field-years across 34 distinct fields; development only.',
    season: { year, start: `${year}-03-01`, end: `${year}-12-31`, label: `Mar – Dec ${year}`, largeAcres: 20, peakFrom: '06-01', peakTo: '09-30', dropTo: '11-15' },
    chipDates: () => [],
  };
  const evidence: Record<string, FieldEvidence> = {};
  const fields: Field[] = rows.map((f, i) => {
    const label = f.rf.predicted_label.toLowerCase() as 'grain' | 'silage';
    const confidence = label === 'silage' ? f.rf.p_silage : f.rf.p_grain;
    evidence[f.id] = { fieldId: f.id, optical: [], radar: [], gaps: [], patches: [], before: null, after: null, pattern: 'sparse' };
    return { id: f.id, polygon: f.geometryRings[0][0], geometryRings: f.geometryRings, centroid: f.centroid,
      sourceId: f.id.split('OBJ')[1], acres: f.acres, cluster: `BARC ${year}`,
      prediction: { source: 'ml', predicted: label, confidence, pSilage: f.rf.p_silage, modelVersion: f.rf.model_version },
      barc: { year, rfLabel: label, rfConfidence: confidence, pSilage: f.rf.p_silage, pGrain: f.rf.p_grain },
      reasons: confidence < .8 ? ['low_confidence', 'observation_gap'] : ['observation_gap'],
      harvestWindow: f.harvest ? { ...f.harvest, gapDays: Math.round((Date.parse(f.harvest.end)-Date.parse(f.harvest.start))/864e5) } : null,
      coverage: 'insufficient', maxGapNearHarvest: 0,
      evidenceSummary: 'Original RF prediction from saved features. The candidate harvest window is unverified. Retrieve satellite evidence for the dated optical/radar observations; no imagery is simulated.', seed: i };
  });
  return { info, fields, evidence };
}
