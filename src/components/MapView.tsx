import { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { Decision, Field } from '../types';
import { REGIONS, TOWNS } from '../data/geo';
import { allocate } from '../lib/acreage';
import { createSchematicLayer } from '../lib/basemap';
import { CLASS_LABEL, REASON_META, STATUS_LABEL, confidenceBand, fmtAcres, fmtPct } from '../lib/format';
import { ALL_PATTERNS, patternMarkup, type PatternKey } from '../lib/patterns';
import { priorityBand, type PriorityBreakdown } from '../lib/priority';
import { useStore, type MapMode } from '../state/store';
import type { Derived } from '../state/derived';

const IMAGERY_URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';

export function fillKey(mode: MapMode, f: Field, d: Decision | undefined, pri: PriorityBreakdown): PatternKey {
  const al = allocate(f, d);
  if (mode === 'type') {
    if (al.effective === 'mixed') return 'mixed';
    if (al.effective === 'unresolved') return 'unres';
    return `${al.effective}-${al.source === 'analyst' ? 'r' : 'm'}` as PatternKey;
  }
  if (mode === 'priority') {
    if (d) return 'pri-done';
    if (!f.reasons.length) return 'pri-low';
    const b = priorityBand(pri.score);
    return b === 'high' ? 'pri-high' : b === 'medium' ? 'pri-med' : 'pri-low';
  }
  return f.coverage === 'sufficient' ? 'cov-ok' : f.coverage === 'gap' ? 'cov-gap' : 'cov-none';
}

function tooltipHtml(f: Field, d: Decision | undefined, pri: PriorityBreakdown) {
  const al = allocate(f, d);
  const p = f.prediction;
  const model = p.predicted === 'unclassified' ? 'Unclassified' : `${CLASS_LABEL[p.predicted]} · ${fmtPct(p.confidence)} score`;
  const analyst = d
    ? `${STATUS_LABEL[d.status]}${d.decided && d.status === 'changed' ? ` → ${CLASS_LABEL[d.decided]}` : ''}`
    : 'Not reviewed';
  const reasons = f.reasons.length
    ? `<div class="tt-warn">⚠ ${REASON_META[f.reasons[0]].short}${f.reasons.length > 1 ? ` +${f.reasons.length - 1}` : ''}${!d ? ` · priority ${pri.score}` : ''}</div>`
    : '';
  return `<div class="tt"><div class="tt-h"><b>${f.id}</b><span>${fmtAcres(f.acres)} ac</span></div>
  <div class="tt-r"><span>${p.source === 'rules' ? 'Rules' : 'ML'}</span>${model}</div>
  <div class="tt-r"><span>Analyst</span>${analyst}${al.source === 'analyst' ? ' ✓' : ''}</div>${reasons}</div>`;
}

export function MapView({ derived }: { derived: Derived }) {
  const el = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layers = useRef(new Map<string, L.Polygon>());
  const groupRef = useRef<L.LayerGroup | null>(null);
  const labelGroup = useRef<L.LayerGroup | null>(null);
  const haloGroup = useRef<L.LayerGroup | null>(null);
  const outlineRef = useRef<L.Polygon | null>(null);
  const baseRefs = useRef<{ schematic: L.GridLayer; imagery: L.TileLayer } | null>(null);
  const rendererRef = useRef<L.SVG | null>(null);

  const fields = useStore((s) => s.fields);
  const decisions = useStore((s) => s.decisions);
  const selectedId = useStore((s) => s.selectedId);
  const hoveredId = useStore((s) => s.hoveredId);
  const mapMode = useStore((s) => s.mapMode);
  const showConfidence = useStore((s) => s.showConfidence);
  const basemap = useStore((s) => s.basemap);
  const fitRequest = useStore((s) => s.fitRequest);
  const regionId = useStore((s) => s.selection.regionId);

  // ---- init ----
  useEffect(() => {
    if (!el.current || mapRef.current) return;
    const map = L.map(el.current, { zoomControl: false, minZoom: 7, maxZoom: 17, zoomSnap: 0.25, attributionControl: true });
    map.attributionControl.setPrefix(false);
    L.control.zoom({ position: 'bottomright' }).addTo(map);
    L.control.scale({ position: 'bottomright', imperial: true, metric: true }).addTo(map);
    map.setView([36.17, -119.38], 10);
    const renderer = L.svg({ padding: 0.4 });
    map.addLayer(renderer);
    rendererRef.current = renderer;
    const svg: SVGSVGElement | undefined = (renderer as any)._container;
    if (svg) {
      const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
      defs.innerHTML = ALL_PATTERNS.map((k) => patternMarkup(k, `p-${k}`)).join('');
      svg.insertBefore(defs, svg.firstChild);
    }
    baseRefs.current = {
      schematic: createSchematicLayer(),
      imagery: L.tileLayer(IMAGERY_URL, { maxZoom: 17, attribution: 'Imagery © Esri, Maxar, Earthstar Geographics' }),
    };
    groupRef.current = L.layerGroup().addTo(map);
    haloGroup.current = L.layerGroup().addTo(map);
    labelGroup.current = L.layerGroup().addTo(map);
    const towns = L.layerGroup(
      TOWNS.map((t) => L.marker(t.at, { interactive: false, keyboard: false, icon: L.divIcon({ className: 'town-label', html: `<span>${t.name}</span>`, iconSize: [0, 0] }) })),
    );
    const updateTowns = () => (map.getZoom() >= 9.5 ? towns.addTo(map) : towns.remove());
    map.on('zoomend', updateTowns);
    updateTowns();
    mapRef.current = map;
    const ro = new ResizeObserver(() => map.invalidateSize({ animate: false }));
    ro.observe(el.current);
    return () => {
      ro.disconnect();
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // ---- basemap with automatic fallback ----
  useEffect(() => {
    const map = mapRef.current;
    const b = baseRefs.current;
    if (!map || !b) return;
    b.schematic.remove();
    b.imagery.remove();
    if (basemap === 'schematic') {
      b.schematic.addTo(map).bringToBack();
      return;
    }
    let loaded = false;
    let errors = 0;
    const fail = () => {
      if (!loaded) useStore.getState().markBasemapFailed();
    };
    const onLoad = () => (loaded = true);
    const onError = () => {
      errors++;
      if (errors >= 3) fail();
    };
    b.imagery.on('tileload', onLoad).on('tileerror', onError);
    b.imagery.addTo(map).bringToBack();
    const timer = setTimeout(fail, 6000);
    return () => {
      clearTimeout(timer);
      b.imagery.off('tileload', onLoad).off('tileerror', onError);
    };
  }, [basemap]);

  // ---- region outline ----
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    outlineRef.current?.remove();
    const r = REGIONS.find((x) => x.id === regionId);
    if (!r) return;
    outlineRef.current = L.polygon(r.outline, {
      interactive: false, fill: false, color: '#173b2c', weight: 2, dashArray: '8 6', opacity: 0.75,
    }).addTo(map);
    if (!fields.length) map.fitBounds(outlineRef.current.getBounds(), { padding: [30, 30] });
  }, [regionId, fields.length]);

  // ---- field polygons ----
  useEffect(() => {
    const map = mapRef.current;
    const group = groupRef.current;
    if (!map || !group) return;
    group.clearLayers();
    layers.current.clear();
    for (const f of fields) {
      const poly = L.polygon(f.geometryRings ?? f.polygon, { renderer: rendererRef.current ?? undefined, bubblingMouseEvents: false });
      poly.on('click', () => useStore.getState().select(f.id));
      poly.on('mouseover', () => useStore.getState().hover(f.id));
      poly.on('mouseout', () => useStore.getState().hover(null));
      poly.bindTooltip(
        () => {
          const st = useStore.getState();
          return tooltipHtml(f, st.decisions[f.id], derivedRef.current.pri.get(f.id)!);
        },
        { sticky: true, direction: 'top', offset: [0, -12], className: 'hc-tooltip', opacity: 1 },
      );
      poly.addTo(group);
      const path = poly.getElement();
      if (path) {
        path.setAttribute('tabindex', '0');
        path.setAttribute('role', 'button');
        path.setAttribute('aria-label', `Field ${f.id}, ${fmtAcres(f.acres)} acres`);
        path.addEventListener('keydown', (e) => {
          if ((e as KeyboardEvent).key === 'Enter' || (e as KeyboardEvent).key === ' ') {
            e.preventDefault();
            useStore.getState().select(f.id);
          }
        });
      }
      layers.current.set(f.id, poly);
    }
  }, [fields]);

  const derivedRef = useRef(derived);
  derivedRef.current = derived;

  // ---- styles ----
  useEffect(() => {
    for (const f of fields) {
      const poly = layers.current.get(f.id);
      if (!poly) continue;
      const d = decisions[f.id];
      const key = fillKey(mapMode, f, d, derived.pri.get(f.id)!);
      const dim = derived.active && !derived.filteredIds.has(f.id);
      const hovered = hoveredId === f.id;
      const band = confidenceBand(f.prediction.confidence);
      let stroke: L.PathOptions = { color: '#1d2b22', weight: 1, opacity: 0.75, dashArray: undefined };
      if (showConfidence) {
        stroke =
          band === 'high' ? { color: '#0e1d15', weight: 3.4, opacity: 1, dashArray: undefined }
          : band === 'medium' ? { color: '#0e1d15', weight: 2.4, opacity: 1, dashArray: '8 5' }
          : band === 'low' ? { color: '#0e1d15', weight: 2.4, opacity: 1, dashArray: '1 6' }
          : { color: '#0e1d15', weight: 1, opacity: 0.6, dashArray: '2 3' };
      }
      poly.setStyle({
        ...stroke,
        lineCap: 'round',
        fillColor: `url(#p-${key})`,
        fillOpacity: dim ? 0.18 : 1,
        opacity: dim ? 0.25 : stroke.opacity,
        ...(hovered ? { weight: (stroke.weight ?? 1) + 1.5, opacity: 1 } : {}),
      });
    }
  }, [fields, decisions, mapMode, showConfidence, derived, hoveredId]);

  // ---- labels and reviewed badges ----
  useEffect(() => {
    const map = mapRef.current;
    const lg = labelGroup.current;
    if (!map || !lg) return;
    const draw = () => {
      lg.clearLayers();
      const z = map.getZoom();
      if (z < 11) return;
      for (const f of fields) {
        const d = decisions[f.id];
        const showId = z >= 13;
        if (!showId && !d) continue;
        const html = `<span class="fl ${d ? 'fl-done' : ''}">${showId ? f.id.replace('TUL-', '') : ''}${d ? '<i aria-hidden="true">✓</i>' : ''}</span>`;
        L.marker(f.centroid, { interactive: false, keyboard: false, icon: L.divIcon({ className: 'field-label', html, iconSize: [0, 0] }) }).addTo(lg);
      }
    };
    draw();
    map.on('zoomend', draw);
    return () => {
      map.off('zoomend', draw);
    };
  }, [fields, decisions]);

  // ---- selection halo + bring into view ----
  useEffect(() => {
    const map = mapRef.current;
    const hg = haloGroup.current;
    if (!map || !hg) return;
    hg.clearLayers();
    const f = fields.find((x) => x.id === selectedId);
    if (!f) return;
    L.polygon(f.geometryRings ?? f.polygon, { interactive: false, fill: false, color: '#ffffff', weight: 7, opacity: 0.95 }).addTo(hg);
    L.polygon(f.geometryRings ?? f.polygon, { interactive: false, fill: false, color: '#173b2c', weight: 3, opacity: 1 }).addTo(hg);
    const b = L.latLngBounds(f.geometryRings ? f.geometryRings.flat(2) : f.polygon);
    // wait for the drawer to resize the map first
    const t = setTimeout(() => {
      map.invalidateSize({ animate: false });
      const view = map.getBounds().pad(-0.12);
      if (!view.contains(b) || map.getZoom() < 11.5) {
        map.flyToBounds(b, { maxZoom: Math.max(map.getZoom(), 13.5), padding: [80, 80], duration: 0.6 });
      }
    }, 60);
    return () => clearTimeout(t);
  }, [selectedId, fields]);

  // ---- fit requests ----
  useEffect(() => {
    const map = mapRef.current;
    if (!map || fitRequest.n === 0) return;
    if (fitRequest.kind === 'county' && outlineRef.current) {
      map.flyToBounds(outlineRef.current.getBounds(), { padding: [24, 24], duration: 0.6 });
    } else if (fitRequest.kind === 'fields' && fields.length) {
      const b = L.latLngBounds(fields.flatMap((f) => f.geometryRings ? f.geometryRings.flat(2) : f.polygon));
      map.fitBounds(b, { padding: [40, 40] });
    } else if (fitRequest.kind === 'field') {
      const f = fields.find((x) => x.id === useStore.getState().selectedId);
      if (f) map.flyToBounds(L.latLngBounds(f.geometryRings ? f.geometryRings.flat(2) : f.polygon), { maxZoom: 15, padding: [60, 60], duration: 0.5 });
    }
  }, [fitRequest]); // eslint-disable-line react-hooks/exhaustive-deps

  return <div ref={el} className="map-canvas" aria-label="Field map" role="region" />;
}
