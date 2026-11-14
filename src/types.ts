// Domain model for Harvest Console. These types are the contract a real API would satisfy.

export type HarvestType = 'silage' | 'grain';
/** What the model can say about a whole field. Mixed is never a model output. */
export type PredictedClass = HarvestType | 'unclassified';
/** What an analyst can decide. Mixed requires an acreage allocation. */
export type DecisionClass = HarvestType | 'mixed';

export type ReviewStatus = 'pending' | 'accepted' | 'changed' | 'insufficient';

export type ReviewReason =
  | 'low_confidence'
  | 'observation_gap'
  | 'sensor_disagreement'
  | 'within_field_variation'
  | 'large_acreage'
  | 'audit_sample';

export type AnalysisState = 'ready' | 'processing' | 'unavailable';

export interface Region {
  id: string;
  name: string;
  state: string;
  center: [number, number];
  /** Crop year with a prepared analysis, picked when the area is chosen. */
  defaultYear?: number;
  /** Approximate outline for display only. */
  outline: [number, number][];
}

export interface AnalysisSelection {
  regionId: string;
  cropYear: number;
  seasonId: string;
}

export interface Season {
  id: string;
  label: string;
  start: string; // ISO date
  end: string;
}

export interface AnalysisStatus {
  selection: AnalysisSelection;
  state: AnalysisState;
  latestObservation: string | null;
  modelVersion: string | null;
  message: string;
  /** 0–1 when processing. */
  progress?: number;
}

/** Where a field's prediction comes from. Rules run until a trained ML model is ready. */
export type PredictionSource = 'rules' | 'ml';

export interface Prediction {
  source: PredictionSource;
  predicted: PredictedClass;
  /** Demo score, 0.5–1. Null when unclassified. */
  confidence: number | null;
  /** Model probability that the field was harvested for silage. */
  pSilage: number | null;
  modelVersion: string;
}

export interface HarvestWindow {
  start: string; // last clear observation before change
  end: string; // first clear observation after change
  gapDays: number;
}

export interface Subregion {
  id: string;
  label: string;
  acres: number;
  /** Polygon in field-local unit coordinates (0–1, x east, y south). */
  shape: [number, number][];
  type: HarvestType;
  reviewedBy: string;
  note: string;
}

export interface Field {
  id: string;
  /** Polygon ring, [lat, lng]. Real parcel outline (DWR 2023 crop map, or PDOK BRP 2025 parcel). */
  polygon: [number, number][];
  /** Source ID of the outline (DWR UniqueID or BRP parcel UUID). */
  sourceId?: string;
  centroid: [number, number];
  acres: number;
  cluster: string;
  prediction: Prediction;
  reasons: ReviewReason[];
  harvestWindow: HarvestWindow | null;
  coverage: 'sufficient' | 'gap' | 'insufficient';
  /** Longest gap between clear optical observations within ±20 days of change. */
  maxGapNearHarvest: number;
  evidenceSummary: string;
  /** Short label for memorable demo cases. */
  showcase?: string;
  /** Defined, reviewed subregions (only the explicit mixed demo case). */
  subregions?: Subregion[];
  /** Stable per-field number (legacy; unused by real data). */
  seed: number;
}

export interface OpticalObs {
  date: string;
  ndvi: number;
  lswi: number;
}

export interface RadarObs {
  date: string;
  vv: number; // dB
  vh: number; // dB
}

export interface PatchSeries {
  id: string;
  label: string;
  /** Cell in a 3×2 grid: col 0–2, row 0–1. */
  col: number;
  row: number;
  changeWindow: HarvestWindow | null;
  ndvi: { date: string; value: number }[];
}

export interface FieldEvidence {
  fieldId: string;
  optical: OpticalObs[];
  radar: RadarObs[];
  gaps: { start: string; end: string; days: number }[];
  patches: PatchSeries[];
  before: OpticalObs | null;
  after: OpticalObs | null;
  /** Free-form flags used by the imagery renderer. */
  pattern: FieldPattern;
}

export type FieldPattern =
  | 'silage'
  | 'grain'
  | 'conflict'
  | 'partial'
  | 'mixed'
  | 'sparse';

export interface Allocation {
  silage: number;
  grain: number;
  unresolved: number;
  source: 'subregions' | 'analyst';
}

export interface Decision {
  fieldId: string;
  status: Exclude<ReviewStatus, 'pending'>;
  /** Analyst's harvest type; null when evidence was marked insufficient. */
  decided: DecisionClass | null;
  allocation?: Allocation;
  note: string;
  analyst: string;
  at: string; // ISO timestamp
}

export interface HistoryEntry {
  id: string;
  fieldId: string;
  at: string;
  analyst: string;
  action: 'accepted' | 'changed' | 'insufficient' | 'note' | 'reset';
  original: PredictedClass;
  originalConfidence: number | null;
  /** Which prediction the analyst was looking at (missing on older entries = ML). */
  originalSource?: PredictionSource;
  before: Decision | null;
  after: Decision | null;
}
