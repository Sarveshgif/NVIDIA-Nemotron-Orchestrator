import { useEffect, useState } from 'react';
import { AlertTriangle, Loader2 } from 'lucide-react';
import { BARC_SELECTION } from './services/analysisService';
import { useStore } from './state/store';
import { useDerived } from './state/derived';
import { MapView } from './components/MapView';
import { Legend, MapActions, MapToolbar } from './components/MapChrome';
import { AcreageSummary } from './components/AcreageSummary';
import { ReviewQueue } from './components/ReviewQueue';
import { FieldDrawer, neighbours } from './components/FieldDrawer';
import { ContextBar, FirstUseHint, NavRail, NotPrepared, Toast } from './components/Shell';
import { ReportsView } from './components/ReportsView';

function useWidth() {
  const [w, setW] = useState(() => window.innerWidth);
  useEffect(() => {
    const on = () => setW(window.innerWidth);
    window.addEventListener('resize', on);
    return () => window.removeEventListener('resize', on);
  }, []);
  return w;
}

export function App() {
  const derived = useDerived();
  const view = useStore((s) => s.view);
  const status = useStore((s) => s.status);
  const loading = useStore((s) => s.loading);
  const loadError = useStore((s) => s.loadError);
  const selectedId = useStore((s) => s.selectedId);
  const queueOpen = useStore((s) => s.queueOpen);
  const setSelection = useStore((s) => s.setSelection);
  const selection = useStore((s) => s.selection);
  const width = useWidth();

  useEffect(() => {
    void setSelection(BARC_SELECTION);
  }, [setSelection]);

  // keyboard shortcuts: Esc closes details, N moves to the next field
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag) || e.metaKey || e.ctrlKey || e.altKey) return;
      const st = useStore.getState();
      if (st.view !== 'workspace') return;
      if (e.key === 'Escape' && st.selectedId) st.select(null);
      if ((e.key === 'n' || e.key === 'N') && st.fields.length) {
        const cur = st.selectedId;
        const next = cur ? neighbours(derivedRef, cur).next : derivedRef.queue[0];
        if (next) st.select(next.id);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });
  const derivedRef = derived;

  const selected = selectedId ? derived.byId.get(selectedId) : undefined;
  const ready = status?.state === 'ready' && !loading;
  // On smaller laptops, fold the queue to a slim rail while a field is open so the map stays usable.
  const queueVisible = queueOpen && !(selected && width < 1640);

  return (
    <div className="app">
      <NavRail />
      <div className="main">
        <ContextBar derived={derived} />
        {view === 'workspace' && (
          <div className={`ws ${selected ? 'has-drawer' : ''}`}>
            {ready && (queueVisible ? <ReviewQueue derived={derived} /> : <QueueRail count={derived.tabCounts.needs} />)}
            <div className="center">
              {ready && <AcreageSummary derived={derived} />}
              <div className="map-wrap">
                <MapView derived={derived} />
                <div className="map-top">
                  {ready ? <MapToolbar derived={derived} /> : <span />}
                  <MapActions derived={derived} />
                </div>
                {ready && <Legend />}
                {ready && !selected && <FirstUseHint derived={derived} />}
                {loading && (
                  <div className="map-loading" role="status">
                    <Loader2 size={16} className="spin" aria-hidden /> Loading fields for the selected area…
                  </div>
                )}
                {loadError && (
                  <div className="np-wrap">
                    <div className="np-card" role="alert">
                      <div className="np-badge"><AlertTriangle size={13} /> Could not load fields</div>
                      <p>{loadError}</p>
                      <button type="button" className="btn btn-primary" onClick={() => setSelection(selection)}>Try again</button>
                    </div>
                  </div>
                )}
                {!loading && <NotPrepared />}
              </div>
            </div>
            {selected && <FieldDrawer key={selected.id} field={selected} derived={derived} />}
          </div>
        )}
        {view === 'reports' && <ReportsView derived={derived} />}
      </div>
      <Toast />
    </div>
  );
}

function QueueRail({ count }: { count: number }) {
  const toggleQueue = useStore((s) => s.toggleQueue);
  const select = useStore((s) => s.select);
  const queueOpen = useStore((s) => s.queueOpen);
  return (
    <aside className="queue queue-collapsed" aria-label="Review queue (collapsed)">
      <button
        type="button"
        className="queue-rail-btn"
        onClick={() => {
          if (queueOpen) select(null);
          else toggleQueue(true);
        }}
        aria-label="Open review queue"
      >
        <span className="queue-rail-count">{count}</span>
        <span className="queue-rail-label">Review queue</span>
      </button>
    </aside>
  );
}
