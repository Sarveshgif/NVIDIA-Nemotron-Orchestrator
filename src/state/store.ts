import { create } from 'zustand';
import type {
  Allocation, AnalysisSelection, AnalysisStatus, Decision, DecisionClass, Field, FieldEvidence, HistoryEntry, PredictionSource,
  ReviewReason, ReviewStatus,
} from '../types';
import { getAnalysisStatus, getSeedDecisions, listEvidence, listFields, selectionKey, cacheFieldEvidence } from '../services/analysisService';
import { DEFAULT_THRESHOLDS, runRules, withPrediction, type RuleResult, type RuleThresholds } from '../lib/rules';
import type { EffectiveClass } from '../lib/acreage';
import { mlAvailable } from '../lib/format';

export type View = 'workspace' | 'labels' | 'reports' | 'history';
export type MapMode = 'type' | 'priority' | 'coverage';
export type Basemap = 'imagery' | 'schematic';
export type QueueTab = 'needs' | 'all' | 'reviewed';
export type SortKey = 'priority' | 'acres' | 'confidence' | 'id';
export type ConfBand = 'none' | 'low' | 'medium' | 'high';

export interface Filters {
  search: string;
  types: EffectiveClass[];
  statuses: ReviewStatus[];
  confidence: ConfBand[];
  reasons: ReviewReason[];
  minAcres: string;
  maxAcres: string;
}

export const EMPTY_FILTERS: Filters = {
  search: '', types: [], statuses: [], confidence: [], reasons: [], minAcres: '', maxAcres: '',
};

export const ANALYST = 'You (demo analyst)';

interface Toast {
  id: number;
  message: string;
  undoable: boolean;
}

interface UndoItem {
  fieldId: string;
  before: Decision | null;
  historyId: string;
}

interface State {
  view: View;
  selection: AnalysisSelection;
  status: AnalysisStatus | null;
  loading: boolean;
  loadError: string | null;
  /** Fields as served, carrying the ML prediction. */
  baseFields: Field[];
  /** Fields as shown: prediction from the active source. */
  fields: Field[];
  evidence: Record<string, FieldEvidence>;
  predictionSource: PredictionSource;
  ruleThresholds: RuleThresholds;
  ruleResults: Record<string, RuleResult>;
  decisions: Record<string, Decision>;
  history: HistoryEntry[];
  undoStack: UndoItem[];
  storageOk: boolean;

  selectedId: string | null;
  hoveredId: string | null;
  mapMode: MapMode;
  showConfidence: boolean;
  basemap: Basemap;
  basemapFailed: boolean;
  fitRequest: { kind: 'county' | 'fields' | 'field'; n: number };

  queueOpen: boolean;
  queueTab: QueueTab;
  filtersOpen: boolean;
  filters: Filters;
  sort: { key: SortKey; dir: 'asc' | 'desc' };
  scope: 'all' | 'filtered';
  hintDismissed: boolean;
  toast: Toast | null;

  applyInvestigation: (id: string, label: 'grain' | 'silage', model: string, evidence: FieldEvidence) => void;
  setView: (v: View) => void;
  setPredictionSource: (s: PredictionSource) => void;
  setRuleThresholds: (patch: Partial<RuleThresholds>) => void;
  resetRuleThresholds: () => void;
  setSelection: (s: AnalysisSelection) => Promise<void>;
  select: (id: string | null) => void;
  hover: (id: string | null) => void;
  setMapMode: (m: MapMode) => void;
  toggleConfidence: () => void;
  setBasemap: (b: Basemap) => void;
  markBasemapFailed: () => void;
  requestFit: (kind: 'county' | 'fields' | 'field') => void;
  toggleQueue: (open?: boolean) => void;
  setQueueTab: (t: QueueTab) => void;
  toggleFiltersOpen: () => void;
  setFilters: (patch: Partial<Filters>) => void;
  clearFilters: () => void;
  setSort: (key: SortKey) => void;
  setScope: (s: 'all' | 'filtered') => void;
  dismissHint: (dismissed?: boolean) => void;
  record: (fieldId: string, d: { status: Decision['status']; decided: DecisionClass | null; allocation?: Allocation; note: string }) => void;
  clearDecision: (fieldId: string) => void;
  undo: () => void;
  resetDemo: () => Promise<void>;
  showToast: (message: string, undoable?: boolean) => void;
  closeToast: () => void;
}

// ---------- persistence (browser only, per prototype scope) ----------
// v3: real Sentinel data and no seeded reviews; earlier versions reviewed synthetic evidence.
const storeKey = (s: AnalysisSelection) => `harvest-console:v3:${selectionKey(s)}`;
const PREFS_KEY = 'harvest-console:v2:prefs'; // v2: rule thresholds changed meaning with real data

function readJSON<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}
function writeJSON(key: string, value: unknown): boolean {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

interface Persisted {
  decisions: Record<string, Decision>;
  history: HistoryEntry[];
}

interface Prefs {
  hintDismissed?: boolean;
  mapMode?: MapMode;
  showConfidence?: boolean;
  queueOpen?: boolean;
  predictionSource?: PredictionSource;
  ruleThresholds?: Partial<RuleThresholds>;
}
const prefs = readJSON<Prefs>(PREFS_KEY) ?? {};
const savePrefs = (p: Prefs) => writeJSON(PREFS_KEY, { ...readJSON<Prefs>(PREFS_KEY), ...p });

function seedHistory(fields: Field[], seeds: Decision[]): HistoryEntry[] {
  return seeds
    .map((d) => {
      const f = fields.find((x) => x.id === d.fieldId)!;
      return {
        id: `seed-${d.fieldId}`,
        fieldId: d.fieldId,
        at: d.at,
        analyst: d.analyst,
        action: d.status,
        original: f.prediction.predicted,
        originalConfidence: f.prediction.confidence,
        originalSource: f.prediction.source,
        before: null,
        after: d,
      } as HistoryEntry;
    })
    .sort((a, b) => b.at.localeCompare(a.at));
}

const viewFromHash = (): View => {
  const h = location.hash.replace('#', '');
  return h === 'reports' ? h : 'workspace';
};

const effectiveFields = (base: Field[], src: PredictionSource, rules: Record<string, RuleResult>) =>
  src === 'rules' ? base.map((f) => withPrediction(f, rules[f.id])) : base;

let toastSeq = 0;
let loadSeq = 0;

export const useStore = create<State>((set, get) => {
  const persist = () => {
    const { selection, decisions, history } = get();
    const ok = writeJSON(storeKey(selection), { decisions, history } satisfies Persisted);
    if (ok !== get().storageOk) set({ storageOk: ok });
  };

  return {
    view: viewFromHash(),
    selection: { regionId:'barc-md', cropYear:2023, seasonId:'summer-fall' },
    status: null,
    loading: true,
    loadError: null,
    baseFields: [],
    fields: [],
    evidence: {},
    predictionSource: prefs.predictionSource ?? 'rules',
    ruleThresholds: { ...DEFAULT_THRESHOLDS, ...prefs.ruleThresholds },
    ruleResults: {},
    decisions: {},
    history: [],
    undoStack: [],
    storageOk: true,

    selectedId: null,
    hoveredId: null,
    mapMode: prefs.mapMode ?? 'type',
    showConfidence: prefs.showConfidence ?? false,
    basemap: 'imagery',
    basemapFailed: false,
    fitRequest: { kind: 'fields', n: 0 },

    queueOpen: prefs.queueOpen ?? true,
    queueTab: 'needs',
    filtersOpen: false,
    filters: EMPTY_FILTERS,
    sort: { key: 'priority', dir: 'desc' },
    scope: 'all',
    hintDismissed: prefs.hintDismissed ?? false,
    toast: null,

    applyInvestigation: (id, label, model, evidence) => {
      cacheFieldEvidence(evidence);
      const baseFields=get().baseFields.map(f=>f.id===id?{...f,prediction:{...f.prediction,source:'ml' as const,predicted:label,confidence:null,pSilage:null,modelVersion:model},evidenceSummary:'Nemotron prediction from the current evidence. Original RF scores are shown separately.'}:f);
      set({baseFields,fields:effectiveFields(baseFields,get().predictionSource,get().ruleResults),evidence:{...get().evidence,[id]:evidence}});
    },
    setView: (view) => {
      set({ view });
      const target = view === 'workspace' ? '' : `#${view}`;
      if (location.hash !== target) {
        if (target) location.hash = target;
        else history.replaceState(null, '', location.pathname + location.search);
      }
    },

    setSelection: async (selection) => {
      const seq = ++loadSeq;
      set({ selection, basemap:'imagery', loading: true, loadError: null, status: null, baseFields: [], fields: [], evidence: {}, ruleResults: {}, selectedId: null, hoveredId: null, undoStack: [] });
      try {
        const status = await getAnalysisStatus(selection);
        if (seq !== loadSeq) return;
        if (status.state !== 'ready') {
          set({ status, loading: false, decisions: {}, history: [] });
          return;
        }
        const [baseFields, evidence] = await Promise.all([listFields(selection), listEvidence(selection)]);
        if (seq !== loadSeq) return;
        const ruleResults = runRules(evidence, get().ruleThresholds);
        // Datasets with a trained model open on its predictions; others can only show rules.
        const predictionSource: PredictionSource = selection.regionId==='barc-md' ? 'ml' : mlAvailable() ? (prefs.predictionSource === 'rules' ? 'rules' : 'ml') : 'rules';
        const fields = effectiveFields(baseFields, predictionSource, ruleResults);
        const saved = readJSON<Persisted>(storeKey(selection));
        let decisions: Record<string, Decision>;
        let hist: HistoryEntry[];
        if (saved && saved.decisions && saved.history) {
          decisions = saved.decisions;
          hist = saved.history;
        } else {
          const seeds = await getSeedDecisions(selection);
          decisions = Object.fromEntries(seeds.map((d) => [d.fieldId, d]));
          hist = seedHistory(baseFields, seeds);
        }
        set({ status, baseFields, fields, evidence, ruleResults, predictionSource, decisions, history: hist, loading: false, fitRequest: { kind: 'fields', n: get().fitRequest.n + 1 } });
      } catch (e) {
        if (seq !== loadSeq) return;
        set({ loading: false, loadError: e instanceof Error ? e.message : 'Could not load fields.' });
      }
    },

    setPredictionSource: (predictionSource) => {
      if (predictionSource === 'ml' && !mlAvailable()) return;
      const { baseFields, ruleResults } = get();
      set({ predictionSource, fields: effectiveFields(baseFields, predictionSource, ruleResults) });
      prefs.predictionSource = predictionSource;
      savePrefs({ predictionSource });
    },
    setRuleThresholds: (patch) => {
      const ruleThresholds = { ...get().ruleThresholds, ...patch };
      const { evidence, baseFields, predictionSource } = get();
      const ruleResults = runRules(evidence, ruleThresholds);
      set({ ruleThresholds, ruleResults, fields: effectiveFields(baseFields, predictionSource, ruleResults) });
      savePrefs({ ruleThresholds });
    },
    resetRuleThresholds: () => get().setRuleThresholds(DEFAULT_THRESHOLDS),

    select: (id) => {
      set({ selectedId: id });
      if (id && !get().hintDismissed) get().dismissHint(true);
    },
    hover: (id) => set({ hoveredId: id }),
    setMapMode: (mapMode) => {
      set({ mapMode });
      savePrefs({ mapMode });
    },
    toggleConfidence: () => {
      const showConfidence = !get().showConfidence;
      set({ showConfidence });
      savePrefs({ showConfidence });
    },
    setBasemap: (basemap) => set({ basemap, basemapFailed: basemap === 'imagery' ? false : get().basemapFailed }),
    markBasemapFailed: () => set({ basemap: 'schematic', basemapFailed: true }),
    requestFit: (kind) => set({ fitRequest: { kind, n: get().fitRequest.n + 1 } }),
    toggleQueue: (open) => {
      const queueOpen = open ?? !get().queueOpen;
      set({ queueOpen });
      savePrefs({ queueOpen });
    },
    setQueueTab: (queueTab) => set({ queueTab }),
    toggleFiltersOpen: () => set({ filtersOpen: !get().filtersOpen }),
    setFilters: (patch) => set({ filters: { ...get().filters, ...patch } }),
    clearFilters: () => set({ filters: EMPTY_FILTERS }),
    setSort: (key) => {
      const { sort } = get();
      if (sort.key === key) set({ sort: { key, dir: sort.dir === 'asc' ? 'desc' : 'asc' } });
      else set({ sort: { key, dir: key === 'id' ? 'asc' : key === 'confidence' ? 'asc' : 'desc' } });
    },
    setScope: (scope) => set({ scope }),
    dismissHint: (dismissed = true) => {
      set({ hintDismissed: dismissed });
      savePrefs({ hintDismissed: dismissed });
    },

    record: (fieldId, d) => {
      const { decisions, history, fields, undoStack } = get();
      const field = fields.find((f) => f.id === fieldId);
      if (!field) return;
      const before = decisions[fieldId] ?? null;
      const after: Decision = {
        fieldId, status: d.status, decided: d.decided, allocation: d.allocation, note: d.note.trim(),
        analyst: ANALYST, at: new Date().toISOString(),
      };
      const sameOutcome =
        before && before.status === after.status && before.decided === after.decided &&
        JSON.stringify(before.allocation ?? null) === JSON.stringify(after.allocation ?? null);
      const entry: HistoryEntry = {
        id: `h-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        fieldId, at: after.at, analyst: ANALYST,
        action: sameOutcome ? 'note' : d.status,
        original: field.prediction.predicted,
        originalConfidence: field.prediction.confidence,
        originalSource: field.prediction.source,
        before, after,
      };
      set({
        decisions: { ...decisions, [fieldId]: after },
        history: [entry, ...history],
        undoStack: [...undoStack.slice(-19), { fieldId, before, historyId: entry.id }],
      });
      persist();
      const verb =
        entry.action === 'note' ? 'Note saved' :
        d.status === 'accepted' ? 'Prediction accepted' :
        d.status === 'insufficient' ? 'Marked insufficient evidence' :
        `Changed to ${d.decided}`;
      get().showToast(`${verb} for ${fieldId}`, true);
    },

    clearDecision: (fieldId) => {
      const { decisions, history, fields, undoStack } = get();
      const before = decisions[fieldId];
      const field = fields.find((f) => f.id === fieldId);
      if (!before || !field) return;
      const next = { ...decisions };
      delete next[fieldId];
      const entry: HistoryEntry = {
        id: `h-${Date.now()}`, fieldId, at: new Date().toISOString(), analyst: ANALYST, action: 'reset',
        original: field.prediction.predicted, originalConfidence: field.prediction.confidence, originalSource: field.prediction.source, before, after: null,
      };
      set({ decisions: next, history: [entry, ...history], undoStack: [...undoStack, { fieldId, before, historyId: entry.id }] });
      persist();
      get().showToast(`${fieldId} returned to the queue`, true);
    },

    undo: () => {
      const { undoStack, decisions, history } = get();
      const last = undoStack[undoStack.length - 1];
      if (!last) return;
      const next = { ...decisions };
      if (last.before) next[last.fieldId] = last.before;
      else delete next[last.fieldId];
      set({
        decisions: next,
        history: history.filter((h) => h.id !== last.historyId),
        undoStack: undoStack.slice(0, -1),
      });
      persist();
      get().showToast(`Undone. ${last.fieldId} restored to its previous state.`, false);
    },

    resetDemo: async () => {
      const { selection, baseFields } = get();
      const seeds = await getSeedDecisions(selection);
      try {
        localStorage.removeItem(storeKey(selection));
      } catch {
        /* storage unavailable */
      }
      set({
        decisions: Object.fromEntries(seeds.map((d) => [d.fieldId, d])),
        history: seedHistory(baseFields, seeds),
        undoStack: [],
      });
      get().showToast('Demo reviews reset to the starting state', false);
    },

    showToast: (message, undoable = false) => set({ toast: { id: ++toastSeq, message, undoable } }),
    closeToast: () => set({ toast: null }),
  };
});

if (typeof window !== 'undefined') {
  window.addEventListener('hashchange', () => {
    const v = viewFromHash();
    if (useStore.getState().view !== v) useStore.setState({ view: v });
  });
}
