import { activeDataset } from '../data/datasets';
import { useState } from 'react';
import { ChevronDown, ChevronUp, CloudOff, Crosshair, FilterX, Layers, Maximize2 } from 'lucide-react';
import { useStore, type MapMode } from '../state/store';
import type { Derived } from '../state/derived';
import { OutlineSwatch, Segmented, Swatch } from './ui';

export function MapToolbar({ derived }: { derived: Derived }) {
  const mapMode = useStore((s) => s.mapMode);
  const setMapMode = useStore((s) => s.setMapMode);
  const showConfidence = useStore((s) => s.showConfidence);
  const toggleConfidence = useStore((s) => s.toggleConfidence);
  return (
    <div className="map-toolbar">
      <Segmented<MapMode>
        label="Map mode"
        value={mapMode}
        onChange={setMapMode}
        options={[
          { value: 'type', label: 'Harvest type', title: 'Predicted or reviewed silage and grain' },
          { value: 'priority', label: 'Review priority', title: 'Fields that need attention' },
          { value: 'coverage', label: 'Data coverage', title: 'Observation gaps near harvest' },
        ]}
      />
      <label className="toggle" title="Outline each field by prediction confidence">
        <input type="checkbox" checked={showConfidence} onChange={toggleConfidence} />
        <span className="toggle-ui" aria-hidden />
        Confidence outline
      </label>
      {derived.active && (
        <span className="map-filter-note">
          Showing {derived.filtered.length} of {derived.byId.size} fields
        </span>
      )}
    </div>
  );
}

export function MapActions({ derived }: { derived: Derived }) {
  const requestFit = useStore((s) => s.requestFit);
  const clearFilters = useStore((s) => s.clearFilters);
  const basemap = useStore((s) => s.basemap);
  const setBasemap = useStore((s) => s.setBasemap);
  const selectedId = useStore((s) => s.selectedId);
  const failed = useStore((s) => s.basemapFailed);
  return (
    <div className="map-actions">
      {failed && basemap === 'schematic' && (
        <div className="offline-chip" role="status" title="Online satellite imagery could not load, so the map uses an offline schematic basemap. Fields and evidence are unaffected.">
          <CloudOff size={14} aria-hidden /> Imagery unavailable · offline basemap
          <button type="button" className="link-btn" onClick={() => setBasemap('imagery')}>Retry</button>
        </div>
      )}
      <div className="map-btn-group">
        <label className="map-select" title="Basemap">
          <Layers size={14} aria-hidden />
          <span className="sr-only">Basemap</span>
          <select value={basemap} onChange={(e) => setBasemap(e.target.value as 'imagery' | 'schematic')}>
            <option value="imagery">Imagery (online)</option>
            <option value="schematic">Schematic (offline)</option>
          </select>
        </label>
      </div>
      <div className="map-btn-group">
        <button type="button" className="map-btn" onClick={() => requestFit('county')} title="Fit the whole county">
          <Maximize2 size={14} aria-hidden /> County
        </button>
        <button type="button" className="map-btn" onClick={() => requestFit('fields')} title="Reset to all loaded fields">
          <Crosshair size={14} aria-hidden /> Reset view
        </button>
        {selectedId && (
          <button type="button" className="map-btn" onClick={() => requestFit('field')} title="Zoom to the selected field">
            Zoom to {selectedId.replace('TUL-', '')}
          </button>
        )}
        <button type="button" className="map-btn" onClick={clearFilters} disabled={!derived.active} title={derived.active ? 'Clear all filters' : 'No filters applied'}>
          <FilterX size={14} aria-hidden /> Clear filters
        </button>
      </div>
    </div>
  );
}

export function Legend() {
  const mapMode = useStore((s) => s.mapMode);
  const showConfidence = useStore((s) => s.showConfidence);
  const [open, setOpen] = useState(() => window.innerWidth >= 1500);
  return (
    <section className={`legend ${open ? '' : 'legend-closed'}`} aria-label="Map legend">
      <button type="button" className="legend-head" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        Legend {open ? <ChevronDown size={14} aria-hidden /> : <ChevronUp size={14} aria-hidden />}
      </button>
      {open && (
        <div className="legend-body">
          {mapMode === 'type' && (
            <>
              <div className="legend-cols">
                <span />
                <span className="legend-colh">Predicted</span>
                <span className="legend-colh">Reviewed</span>
                <span>Silage</span>
                <Swatch k="silage-m" />
                <Swatch k="silage-r" />
                <span>Grain</span>
                <Swatch k="grain-m" />
                <Swatch k="grain-r" />
              </div>
              <ul>
                <li><Swatch k="mixed" /> Mixed (reviewed subregions)</li>
                <li><Swatch k="unres" /> Unresolved or unclassified</li>
                <li><span className="legend-check" aria-hidden>✓</span> Analyst decision recorded</li>
              </ul>
            </>
          )}
          {mapMode === 'priority' && (
            <ul>
              <li><Swatch k="pri-high" /> High priority (score ≥ 45)</li>
              <li><Swatch k="pri-med" /> Medium priority (25–44)</li>
              <li><Swatch k="pri-low" /> Low or not flagged</li>
              <li><Swatch k="pri-done" /> Reviewed</li>
            </ul>
          )}
          {mapMode === 'coverage' && (
            <ul>
              <li><Swatch k="cov-ok" /> Sufficient observations</li>
              <li><Swatch k="cov-gap" /> Gap over 15 days near harvest</li>
              <li><Swatch k="cov-none" /> Insufficient data</li>
            </ul>
          )}
          {showConfidence && (
            <>
              <div className="legend-sub">Prediction confidence (score)</div>
              <ul>
                <li><OutlineSwatch weight={2.6} /> High, 85% and above</li>
                <li><OutlineSwatch weight={2} dash="5 3" /> Medium, 70–84%</li>
                <li><OutlineSwatch weight={2} dash="0.5 3.5" /> Low, under 70%</li>
              </ul>
            </>
          )}
          <p className="legend-foot">Field outlines: {activeDataset().outlineSource}. Not ownership records.</p>
        </div>
      )}
    </section>
  );
}
