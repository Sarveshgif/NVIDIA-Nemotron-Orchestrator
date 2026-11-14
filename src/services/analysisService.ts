// Data access layer. Components call only these functions, so a real API can replace
// the mock implementation without touching presentation code.

import type { AnalysisSelection, AnalysisStatus, Field, FieldEvidence } from '../types';
import { DATASET, LATEST_OBSERVATION } from '../data/generate';
import { DEMO_SELECTION, REGIONS, SEASONS } from '../data/geo';
import { RULESET_VERSION } from '../lib/rules';
import { CA_DATASET, loadNl, setActiveDataset } from '../data/datasets';

const latency = (ms: number) => new Promise((r) => setTimeout(r, ms));

export const selectionKey = (s: AnalysisSelection) => `${s.regionId}:${s.cropYear}:${s.seasonId}`;
export const isDemoSelection = (s: AnalysisSelection) => selectionKey(s) === selectionKey(DEMO_SELECTION);
export const NL_SELECTION: AnalysisSelection = { regionId: 'nl-se', cropYear: 2025, seasonId: 'summer-fall' };
export const isNlSelection = (s: AnalysisSelection) => selectionKey(s) === selectionKey(NL_SELECTION);

export const regionName = (id: string) => REGIONS.find((r) => r.id === id)?.name ?? id;
export const seasonLabel = (id: string) => SEASONS.find((s) => s.id === id)?.label ?? id;

export async function getAnalysisStatus(sel: AnalysisSelection): Promise<AnalysisStatus> {
  await latency(220);
  if (isNlSelection(sel)) {
    try {
      const nl = await loadNl();
      setActiveDataset(nl.info);
      const latest = Object.values(nl.evidence).flatMap((e) => e.optical.map((o) => o.date)).sort().pop() ?? null;
      return {
        selection: sel, state: 'ready', latestObservation: latest, modelVersion: nl.info.nl!.model.version,
        message: 'Random Forest predictions (Sentinel-2 + Sentinel-1) for held-out test regions are ready.',
      };
    } catch (e) {
      return { selection: sel, state: 'unavailable', latestObservation: null, modelVersion: null, message: e instanceof Error ? e.message : 'Dataset missing.' };
    }
  }
  if (isDemoSelection(sel)) {
    setActiveDataset(CA_DATASET);
    return {
      selection: sel, state: 'ready', latestObservation: LATEST_OBSERVATION, modelVersion: RULESET_VERSION,
      message: 'Rule-based estimates from Sentinel-2 and Sentinel-1 are ready.',
    };
  }
  if (sel.regionId === 'kings-ca' && sel.cropYear === 2023 && sel.seasonId === 'summer-fall') {
    return {
      selection: sel, state: 'processing', latestObservation: '2023-12-27', modelVersion: null, progress: 0.62,
      message: 'Model inference is running. Results are not available yet.',
    };
  }
  return {
    selection: sel, state: 'unavailable', latestObservation: null, modelVersion: null,
    message: 'No analysis has been prepared for this area and period.',
  };
}

export async function listFields(sel: AnalysisSelection): Promise<Field[]> {
  await latency(260);
  if (isNlSelection(sel)) return (await loadNl()).fields;
  if (!isDemoSelection(sel)) throw new Error('No prepared analysis for this selection.');
  return DATASET.fields;
}

export async function getSeedDecisions(sel: AnalysisSelection) {
  return isDemoSelection(sel) ? DATASET.seedDecisions : [];
}

const evidenceCache = new Map<string, FieldEvidence>();

/** All evidence for a selection, used to run the rule set over every field at once. */
export async function listEvidence(sel: AnalysisSelection): Promise<Record<string, FieldEvidence>> {
  await latency(120);
  if (isNlSelection(sel)) {
    const nl = await loadNl();
    Object.entries(nl.evidence).forEach(([id, ev]) => evidenceCache.set(id, ev));
    return nl.evidence;
  }
  if (!isDemoSelection(sel)) return {};
  DATASET.evidence.forEach((ev, id) => evidenceCache.set(id, ev));
  return Object.fromEntries(DATASET.evidence);
}

/** Synchronous cache read so field switching can render instantly when prefetched. */
export const peekEvidence = (fieldId: string) => evidenceCache.get(fieldId);

export async function getFieldEvidence(fieldId: string): Promise<FieldEvidence> {
  const hit = evidenceCache.get(fieldId);
  if (hit) return hit;
  await latency(140);
  const ev = DATASET.evidence.get(fieldId);
  if (!ev) throw new Error(`No evidence found for ${fieldId}.`);
  evidenceCache.set(fieldId, ev);
  return ev;
}

export function prefetchEvidence(ids: string[]) {
  ids.forEach((id) => void getFieldEvidence(id).catch(() => undefined));
}

export const PREP_STEPS = [
  'Collect Sentinel-2 scenes for the season',
  'Collect Sentinel-1 radar passes',
  'Mask cloud, shadow and smoke',
  'Load field outlines (DWR crop map / BRP parcels)',
  'Extract field time series',
  'Apply harvest-type rules',
];

/** Explicitly simulated. Nothing is sent anywhere. */
export function simulatePreparation(onStep: (index: number) => void, onDone: () => void) {
  let i = 0;
  onStep(0);
  const timer = setInterval(() => {
    i++;
    if (i >= PREP_STEPS.length) {
      clearInterval(timer);
      onDone();
    } else onStep(i);
  }, 900);
  return () => clearInterval(timer);
}
