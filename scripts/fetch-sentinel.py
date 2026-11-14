"""Extract real Sentinel-2 / Sentinel-1 evidence for the DWR corn fields (2023 season).

Reads Copernicus Sentinel data from Microsoft Planetary Computer (free, no account):
  * sentinel-2-l2a: per-field cloud-masked mean NDVI and LSWI for every clear look,
    a 3 x 2 sub-field NDVI grid, and small true-colour / NDVI image chips.
  * sentinel-1-rtc: per-field mean VV / VH backscatter (dB) from one relative orbit.

Outputs
  src/data/sentinel2023.json          time series used by the app and the rules
  public/s2/<field>_rgb.jpg           true-colour chips for every clear date, side by side (field outlined)
  public/s2/<field>_ndvi.jpg          NDVI chips, same order, same colour ramp as the app
  (chip order is the "chips" date list per field in the JSON; one strip per field keeps the file count small)

Setup (once):  python3 -m venv .venv && .venv/bin/pip install pystac-client planetary-computer rasterio numpy pyproj shapely pillow
Run:           .venv/bin/python scripts/fetch-sentinel.py
Contains modified Copernicus Sentinel data 2023.
"""
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from PIL import Image, ImageDraw
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry import Polygon, box

DATES = '2023-04-01/2023-12-31'
MAX_SCENE_CLOUD = 80          # skip scenes that are almost entirely cloud
MIN_FIELD_CLEAR = 0.9         # share of field pixels that must be clear (SCL veg / bare / water)
CLEAR_SCL = (4, 5, 6)
INTERIOR_BUFFER_M = -10       # drop edge pixels mixed with roads and neighbours
CHIP_PX = 220
PAD_M = 120

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, 'src', 'data', 'sentinel2023.json')
CHIP_DIR = os.path.join(ROOT, 'public', 's2')
TMP_CHIPS = os.path.join(ROOT, '.cache', 's2-chips')

os.environ.setdefault('GDAL_HTTP_MAX_RETRY', '4')
os.environ.setdefault('GDAL_HTTP_RETRY_DELAY', '2')
os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif,.TIF,.tiff')

NDVI_STOPS = [(0.0, (120, 78, 44)), (0.2, (168, 120, 62)), (0.35, (222, 196, 96)),
              (0.55, (150, 196, 86)), (0.75, (58, 148, 64)), (0.9, (20, 96, 44))]


def load_fields():
    src = open(os.path.join(ROOT, 'src', 'data', 'dwrFields.ts')).read()
    counters, out = defaultdict(int), []
    for m in re.finditer(r"cluster: '(\w)', dwrId: '(\d+)', ring: (\[\[.*?\]\]) ", src):
        c = m.group(1)
        counters[c] += 1
        ring = json.loads(m.group(3))  # [lat, lng]
        out.append(dict(id=f'TUL-{c}{counters[c]:02d}', cluster=c, lnglat=[(p[1], p[0]) for p in ring]))
    return out


def ndvi_rgb(v):
    v = np.clip(v, 0, 0.9)
    xs = [s[0] for s in NDVI_STOPS]
    return np.stack([np.interp(v, xs, [s[1][k] for s in NDVI_STOPS]) for k in range(3)], -1).astype(np.uint8)


class Geo:
    """Field geometry projected to one UTM zone, with cluster read boxes."""

    def __init__(self, fields, epsg):
        self.epsg = epsg
        tr = Transformer.from_crs(4326, epsg, always_xy=True)
        self.fields = fields
        self.poly = {f['id']: Polygon([tr.transform(*p) for p in f['lnglat']]) for f in fields}
        self.interior = {k: (p.buffer(INTERIOR_BUFFER_M) if p.buffer(INTERIOR_BUFFER_M).area > 0 else p) for k, p in self.poly.items()}
        self.clusters = defaultdict(list)
        for f in fields:
            self.clusters[f['cluster']].append(f['id'])
        self.cbox = {}
        for c, ids in self.clusters.items():
            b = np.array([self.poly[i].bounds for i in ids])
            self.cbox[c] = (b[:, 0].min() - PAD_M - 200, b[:, 1].min() - PAD_M - 200, b[:, 2].max() + PAD_M + 200, b[:, 3].max() + PAD_M + 200)


def read_window(href, bounds, shape=None, resampling=Resampling.nearest):
    with rasterio.open(href) as ds:
        win = from_bounds(*bounds, transform=ds.transform)
        win = Window(round(win.col_off), round(win.row_off), round(win.width), round(win.height))
        if shape is None:
            arr = ds.read(1, window=win, boundless=True, fill_value=0)
            return arr, ds.window_transform(win)
        return ds.read(1, window=win, out_shape=shape, boundless=True, fill_value=0, resampling=resampling), None


def cell_masks(poly, interior, transform, shape):
    """Interior mask plus a 3 x 2 grid of sub-masks over the field's bounding box."""
    inside = geometry_mask([interior], out_shape=shape, transform=transform, invert=True)
    x0, y0, x1, y1 = poly.bounds
    cells = []
    for row in range(2):
        for col in range(3):
            cb = box(x0 + (x1 - x0) * col / 3, y1 - (y1 - y0) * (row + 1) / 2, x0 + (x1 - x0) * (col + 1) / 3, y1 - (y1 - y0) * row / 2)
            g = interior.intersection(cb)
            m = geometry_mask([g], out_shape=shape, transform=transform, invert=True) if not g.is_empty else np.zeros(shape, bool)
            cells.append((col, row, m))
    return inside, cells


def chip(poly, transform, rgb, ndvi, path_base):
    x0, y0, x1, y1 = poly.bounds
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2 + PAD_M
    inv = ~transform
    c0, r0 = inv * (cx - half, cy + half)
    c1, r1 = inv * (cx + half, cy - half)
    c0, r0, c1, r1 = int(np.floor(c0)), int(np.floor(r0)), int(np.ceil(c1)), int(np.ceil(r1))
    if c0 < 0 or r0 < 0 or r1 > rgb.shape[0] or c1 > rgb.shape[1]:
        return
    scale = CHIP_PX / (c1 - c0)
    ring = [((inv * pt)[0] - c0, (inv * pt)[1] - r0) for pt in poly.exterior.coords]
    ring = [(x * scale, y * scale) for x, y in ring]
    for kind, arr in (('rgb', rgb[r0:r1, c0:c1]), ('ndvi', ndvi_rgb(ndvi[r0:r1, c0:c1]))):
        im = Image.fromarray(arr).resize((CHIP_PX, round((r1 - r0) * scale)), Image.BILINEAR)
        d = ImageDraw.Draw(im)
        d.line(ring, fill=(20, 24, 20), width=4)
        d.line(ring, fill=(255, 255, 255), width=2)
        im.save(f'{path_base}_{kind}.jpg', quality=82)


def process_s2(item, geo, results):
    date = item.datetime.strftime('%Y-%m-%d')
    baseline = float(item.properties.get('s2:processing_baseline', '0'))
    offset = 1000 if baseline >= 4 else 0
    a = item.assets
    for c, bounds in geo.cbox.items():
        try:
            b4, tf = read_window(a['B04'].href, bounds)
            if not b4.any():
                continue
            shape = b4.shape
            b2, _ = read_window(a['B02'].href, bounds, shape)
            b3, _ = read_window(a['B03'].href, bounds, shape)
            b8, _ = read_window(a['B08'].href, bounds, shape)
            b11, _ = read_window(a['B11'].href, bounds, shape, Resampling.bilinear)
            scl, _ = read_window(a['SCL'].href, bounds, shape)
        except Exception as e:  # network hiccup: skip this scene/cluster
            print('  skip', date, c, type(e).__name__, file=sys.stderr)
            continue
        valid = b4 > 0
        refl = {k: np.clip((v.astype('float32') - offset) / 10000, 0, 1) for k, v in (('b2', b2), ('b3', b3), ('b4', b4), ('b8', b8), ('b11', b11))}
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = np.where(valid, (refl['b8'] - refl['b4']) / (refl['b8'] + refl['b4'] + 1e-6), np.nan)
            lswi = np.where(valid, (refl['b8'] - refl['b11']) / (refl['b8'] + refl['b11'] + 1e-6), np.nan)
        clear = valid & np.isin(scl, CLEAR_SCL)
        rgb = (np.clip(np.stack([refl['b4'], refl['b3'], refl['b2']], -1) * 3.2, 0, 1) ** 0.85 * 255).astype(np.uint8)
        for fid in geo.clusters[c]:
            inside, cells = cell_masks(geo.poly[fid], geo.interior[fid], tf, shape)
            npx = inside.sum()
            if npx < 20:
                continue
            frac = (clear & inside).sum() / npx
            if frac < MIN_FIELD_CLEAR:
                continue
            m = clear & inside
            rec = dict(date=date, ndvi=round(float(np.nanmean(ndvi[m])), 3), lswi=round(float(np.nanmean(lswi[m])), 3), clear=round(float(frac), 2),
                       patches=[round(float(np.nanmean(ndvi[clear & cm])), 3) if (clear & cm).sum() >= 8 else None for _, _, cm in cells])
            prev = results[fid].get(date)
            if prev and prev['clear'] >= rec['clear']:
                continue
            results[fid][date] = rec
            os.makedirs(os.path.join(TMP_CHIPS, fid), exist_ok=True)
            chip(geo.poly[fid], tf, rgb, ndvi, os.path.join(TMP_CHIPS, fid, date))
    print('  s2', date, item.properties.get('s2:mgrs_tile'), flush=True)


def process_s1(item, geos, results):
    date = item.datetime.strftime('%Y-%m-%d')
    epsg = int(str(item.properties.get('proj:code') or item.properties.get('proj:epsg')).split(':')[-1])
    geo = geos(epsg)
    for c, bounds in geo.cbox.items():
        try:
            vv, tf = read_window(item.assets['vv'].href, bounds)
            vh, _ = read_window(item.assets['vh'].href, bounds, vv.shape)
        except Exception as e:
            print('  skip s1', date, c, type(e).__name__, file=sys.stderr)
            continue
        for fid in geo.clusters[c]:
            inside = geometry_mask([geo.interior[fid]], out_shape=vv.shape, transform=tf, invert=True)
            m = inside & (vv > 0) & (vh > 0) & np.isfinite(vv) & np.isfinite(vh)
            if m.sum() < 20:
                continue
            results[fid][date] = dict(date=date, vv=round(float(10 * np.log10(vv[m].mean())), 2), vh=round(float(10 * np.log10(vh[m].mean())), 2))
    print('  s1', date, flush=True)


def pack_chips(fid, dates):
    """Join a field's per-date chips into one strip per view. Returns the dates in strip order."""
    have = [d for d in dates if all(os.path.exists(os.path.join(TMP_CHIPS, fid, f'{d}_{k}.jpg')) for k in ('rgb', 'ndvi'))]
    os.makedirs(CHIP_DIR, exist_ok=True)
    for kind in ('rgb', 'ndvi'):
        strip = Image.new('RGB', (CHIP_PX * max(1, len(have)), CHIP_PX))
        for i, d in enumerate(have):
            im = Image.open(os.path.join(TMP_CHIPS, fid, f'{d}_{kind}.jpg')).resize((CHIP_PX, CHIP_PX), Image.BILINEAR)
            strip.paste(im, (i * CHIP_PX, 0))
        strip.save(os.path.join(CHIP_DIR, f'{fid}_{kind}.jpg'), quality=80, optimize=True)
    return have


def main():
    fields = load_fields()
    lng = [p[0] for f in fields for p in f['lnglat']]
    lat = [p[1] for f in fields for p in f['lnglat']]
    bbox = [min(lng) - 0.01, min(lat) - 0.01, max(lng) + 0.01, max(lat) + 0.01]
    cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)

    s2 = [i for i in cat.search(collections=['sentinel-2-l2a'], bbox=bbox, datetime=DATES,
                                query={'eo:cloud_cover': {'lt': MAX_SCENE_CLOUD}}).items()]
    print(f'Sentinel-2: {len(s2)} scenes', sorted({i.properties["s2:mgrs_tile"] for i in s2}))
    geo_cache = {}

    def geo_for(epsg):
        if epsg not in geo_cache:
            geo_cache[epsg] = Geo(fields, epsg)
        return geo_cache[epsg]

    s2_results = defaultdict(dict)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda it: process_s2(it, geo_for(int(str(it.properties.get('proj:code') or it.properties.get('proj:epsg')).split(':')[-1])), s2_results), s2))

    s1 = list(cat.search(collections=['sentinel-1-rtc'], bbox=bbox, datetime=DATES).items())
    orbits = defaultdict(list)
    for i in s1:
        orbits[i.properties['sat:relative_orbit']].append(i)
    orbit, s1 = max(orbits.items(), key=lambda kv: len(kv[1]))
    print(f'Sentinel-1: {len(s1)} passes from relative orbit {orbit}')
    s1_results = defaultdict(dict)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda it: process_s1(it, geo_for, s1_results), s1))

    out = dict(
        chipPx=CHIP_PX,
        source='Copernicus Sentinel-2 L2A and Sentinel-1 RTC via Microsoft Planetary Computer, 2023',
        attribution='Contains modified Copernicus Sentinel data 2023',
        dates=DATES, minFieldClear=MIN_FIELD_CLEAR, s1RelativeOrbit=orbit,
        fields={},
    )
    for f in fields:
        opt = [s2_results[f['id']][d] for d in sorted(s2_results[f['id']])]
        rad = [s1_results[f['id']][d] for d in sorted(s1_results[f['id']])]
        chips = pack_chips(f['id'], [o['date'] for o in opt])
        out['fields'][f['id']] = dict(optical=opt, radar=rad, chips=chips)
        print(f"{f['id']}: {len(opt)} clear S2, {len(rad)} S1")
    with open(OUT_JSON, 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))
    print('wrote', OUT_JSON)


if __name__ == '__main__':
    main()
