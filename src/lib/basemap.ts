// Offline schematic basemap drawn on canvas tiles. Used when online imagery is unavailable.
import L from 'leaflet';
import { CELL_LAT, CELL_LNG, HIGHWAYS, TOWNS } from '../data/geo';
import { hash2 } from './imagery';

const LNG0 = -120;
const LAT0 = 37;
const PARCEL_TONES = [
  [92, 118, 76], [112, 138, 84], [126, 146, 92], [196, 184, 150], [182, 166, 132],
  [168, 156, 122], [146, 152, 104], [204, 194, 160], [100, 124, 82],
];

const foothillLng = (lat: number) => -118.98 - (lat - 36) * 0.12;

export function createSchematicLayer(): L.GridLayer {
  const Layer = L.GridLayer.extend({
    createTile(this: L.GridLayer, coords: L.Coords) {
      const tile = document.createElement('canvas');
      const size = this.getTileSize();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      tile.width = size.x * dpr;
      tile.height = size.y * dpr;
      const ctx = tile.getContext('2d')!;
      ctx.scale(dpr, dpr);
      const z = coords.z;
      const map = (this as any)._map as L.Map;
      const origin = L.point(coords.x * size.x, coords.y * size.y);
      const px = (lat: number, lng: number) => map.project(L.latLng(lat, lng), z).subtract(origin);
      const nw = map.unproject(origin, z);
      const se = map.unproject(origin.add(size), z);

      // valley floor
      ctx.fillStyle = '#dcd5c1';
      ctx.fillRect(0, 0, size.x, size.y);

      // foothills and Sierra to the east, a soft gradient by longitude
      for (let x = 0; x < size.x; x += 4) {
        const lng = nw.lng + ((se.lng - nw.lng) * x) / size.x;
        const lat = (nw.lat + se.lat) / 2;
        const d = lng - foothillLng(lat);
        if (d > 0) {
          const t = Math.min(1, d / 0.6);
          ctx.fillStyle = `rgb(${Math.round(206 - 70 * t)},${Math.round(200 - 52 * t)},${Math.round(170 - 52 * t)})`;
          ctx.fillRect(x, 0, 4, size.y);
        }
      }

      // quarter-section parcels on the valley floor
      if (z >= 10) {
        const i0 = Math.floor((nw.lng - LNG0) / CELL_LNG) - 1, i1 = Math.ceil((se.lng - LNG0) / CELL_LNG) + 1;
        const j0 = Math.floor((LAT0 - nw.lat) / CELL_LAT) - 1, j1 = Math.ceil((LAT0 - se.lat) / CELL_LAT) + 1;
        const gap = z >= 13 ? 1.5 : z >= 12 ? 1 : 0.5;
        for (let i = i0; i <= i1; i++) {
          for (let j = j0; j <= j1; j++) {
            const west = LNG0 + i * CELL_LNG, north = LAT0 - j * CELL_LAT;
            if (west > foothillLng(north) - 0.02) continue;
            const a = px(north, west), b = px(north - CELL_LAT, west + CELL_LNG);
            const tone = PARCEL_TONES[Math.floor(hash2(i, j) * PARCEL_TONES.length)];
            ctx.fillStyle = `rgb(${tone[0]},${tone[1]},${tone[2]})`;
            ctx.fillRect(a.x + gap / 2, a.y + gap / 2, b.x - a.x - gap, b.y - a.y - gap);
            if (z >= 13 && hash2(i, j) < 0.36) {
              ctx.fillStyle = 'rgba(40,60,30,0.28)';
              for (let x = a.x + 3; x < b.x - 2; x += 4) for (let y = a.y + 3; y < b.y - 2; y += 4) ctx.fillRect(x, y, 1.6, 1.6);
            }
          }
        }
        // mile roads
        if (z >= 11) {
          ctx.strokeStyle = 'rgba(246,242,230,0.9)';
          ctx.lineWidth = z >= 13 ? 2.2 : 1.2;
          ctx.beginPath();
          for (let i = i0 - (i0 % 2); i <= i1; i += 2) {
            const lng = LNG0 + i * CELL_LNG;
            const p = px(nw.lat, lng);
            ctx.moveTo(p.x, 0); ctx.lineTo(p.x, size.y);
          }
          for (let j = j0 - (j0 % 2); j <= j1; j += 2) {
            const lat = LAT0 - j * CELL_LAT;
            const p = px(lat, nw.lng);
            ctx.moveTo(0, p.y); ctx.lineTo(size.x, p.y);
          }
          ctx.stroke();
        }
      } else {
        // low zoom: subtle texture
        ctx.fillStyle = 'rgba(120,140,90,0.18)';
        for (let k = 0; k < 220; k++) {
          const x = hash2(coords.x * 31 + k, coords.y) * size.x;
          const y = hash2(coords.y * 17 + k, coords.x) * size.y;
          ctx.fillRect(x, y, 6, 6);
        }
      }

      // towns
      for (const t of TOWNS) {
        const c = px(t.at[0], t.at[1]);
        const r = (t.size * 1000) / (156543 * Math.cos((t.at[0] * Math.PI) / 180) / 2 ** z);
        if (c.x < -r || c.y < -r || c.x > size.x + r || c.y > size.y + r) continue;
        ctx.fillStyle = 'rgba(190,184,174,0.95)';
        ctx.beginPath();
        ctx.ellipse(c.x, c.y, r, r * 0.8, 0, 0, Math.PI * 2);
        ctx.fill();
      }

      // highways
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      for (const [w, col] of [[z >= 12 ? 5 : 3.5, 'rgba(120,96,60,0.55)'], [z >= 12 ? 3 : 2, '#f1d48a']] as const) {
        ctx.strokeStyle = col;
        ctx.lineWidth = w;
        for (const h of HIGHWAYS) {
          ctx.beginPath();
          h.path.forEach(([la, ln], k) => {
            const p = px(la, ln);
            if (k) ctx.lineTo(p.x, p.y);
            else ctx.moveTo(p.x, p.y);
          });
          ctx.stroke();
        }
      }
      return tile;
    },
  });
  return new (Layer as any)({ attribution: 'Offline schematic basemap (illustrative)', maxZoom: 17 });
}
