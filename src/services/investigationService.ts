export interface SceneObservation {
  date: string; usable?: boolean; item_id?: string; clear_fraction?: number;
  indices?: Record<string, { mean: number } | null>;
  vv_db?: number; vh_db?: number; relative_orbit?: number; orbit_pass?: string;
  ndvi_zones_2x2?: { row: number; column: number; ndvi_mean: number | null; clear_pixels: number }[];
}
export interface RulebookV4 { label: 'GRAIN' | 'SILAGE' | 'UNRESOLVED'; tier: 'strong' | 'likely' | 'lean' | null; step: number; reason: string; version: string; }
export interface SentinelImagePreview { position: 'before' | 'after'; date: string; ndvi_mean?: number | null; true_colour_png?: string; ndvi_png?: string; unavailable?: boolean; }
export interface SensorEvidence {
  status?: string; observations?: SceneObservation[];
  image_previews?: { source: string; items: SentinelImagePreview[]; missing_positions?: string[]; reason?: string };
  summary?: { usable_observations: number; first_date: string | null; last_date: string | null };
}
export interface InvestigationReport {
  field_id: string; year: number; workflow_version: string; final_label: string; decision_source: string;
  explanation: string; final_probability: number | null; missing_evidence: string[]; exception_flags: string[];
  nemotron_used_for_prediction?: boolean; evidence_strength: string; elapsed_ms: number; spatial_evidence_cited?: boolean;
  agent: { status: string; model: string; calls: number; correction_attempts: number };
  nemotron_hypothesis: { label: string; evidence_ids: string[]; explanation: string } | null;
  evidence: { tool: string; status: string; data: { s2?: SensorEvidence; s1?: SensorEvidence; rulebook_v4?: RulebookV4; [key: string]: unknown }; errors: string[] }[];
}
const reports = new Map<string, InvestigationReport>();
const inFlight = new Map<string, Promise<InvestigationReport>>();
let inputs: Promise<Record<string, Record<string, unknown>>> | null = null;
export async function savedInvestigation(id: string): Promise<InvestigationReport | null> {
  if (reports.has(id)) return reports.get(id)!;
  const r = await fetch(`${import.meta.env.BASE_URL}barc/reports/${encodeURIComponent(id)}.json`);
  if (!r.ok) return null;
  const report = await r.json() as InvestigationReport; reports.set(id, report); return report;
}
export function investigateBarc(id: string): Promise<InvestigationReport> {
  const running = inFlight.get(id);
  if (running) return running;
  const pending = fetchInvestigation(id).finally(() => inFlight.delete(id));
  inFlight.set(id, pending);
  return pending;
}
async function fetchInvestigation(id: string): Promise<InvestigationReport> {
  if (!inputs) inputs = fetch(`${import.meta.env.BASE_URL}barc/inputs.json`).then(r => { if (!r.ok) throw new Error('Saved BARC inputs are unavailable.'); return r.json(); });
  const all = await inputs; const input = all[id];
  if (!input) throw new Error('The selected BARC field is unavailable.');
  const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), 240000);
  try {
    const response = await fetch('/api/investigate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...input, rf_model: 'barc_development' }), signal: controller.signal });
    if (!response.ok) throw new Error(`Analysis could not finish (HTTP ${response.status}). Check the local backend.`);
    const report = await response.json() as InvestigationReport;
    if (report.field_id !== id) throw new Error('The returned analysis belongs to a different field.');
    const spatial = report.evidence.find(e => e.tool === 'get_sentinel_spatial_evidence');
    if (!spatial?.data.rulebook_v4) throw new Error('The backend is running an older workflow. Restart it to load Sentinel v4.');
    if (!spatial.data.s2?.image_previews) throw new Error('Restart the backend to enable Sentinel before and after images.');
    reports.set(id, report); return report;
  } catch (e) {
    if (e instanceof Error && e.name === 'AbortError') throw new Error('Analysis timed out. The saved RF prediction is still available.');
    if (e instanceof TypeError) throw new Error('Cannot reach the local backend. Start or restart it, then try again.');
    throw e;
  } finally { clearTimeout(timer); }
}
