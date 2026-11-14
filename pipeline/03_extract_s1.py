"""Sentinel-1 RTC extraction per block (Microsoft Planetary Computer, anonymous SAS signing).

Product: sentinel-1-rtc = GRD radiometrically terrain-corrected gamma0, linear power, 10 m, per-item UTM CRS.
Each item is read in its native grid (no resampling): field interiors (10 m inward buffer) are rasterised
into that grid, VV and VH are averaged in LINEAR power over valid interior pixels (speckle handled by
field aggregation, >= 50 px), then converted: dB = 10*log10(mean_linear).
Cross-ratio: VH/VV computed in linear units, i.e. vh_db - vv_db in dB.
All orbits are kept and tagged (relative orbit, pass direction, platform); features later use one
consistent geometry per block (the orbit with most acquisitions). PC RTC has no incidence-angle layer;
fixing the relative orbit holds the viewing geometry approximately constant per field.
Output: data/s1/s1_<block>.parquet (field x acquisition).
"""
import os, sys, json, time
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                  GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2")
import numpy as np, pandas as pd, rasterio, pystac_client, planetary_computer as pc
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely.geometry import shape, box, mapping
from shapely.ops import transform as stransform
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor
from common import ROOT, SEASON, BUFFER_M, log

OUT = ROOT / "s1"; OUT.mkdir(exist_ok=True)
PC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def process(block):
    out = OUT / f"s1_{block}.parquet"
    if out.exists():
        log(f"S1 {block}: already done", "s1.log"); return
    t0 = time.time()
    fields = pd.read_parquet(ROOT / "s2" / f"fields_{block}.parquet")
    fields = fields[fields.eligible].reset_index(drop=True)
    bbox = json.load(open(ROOT / "blocks" / f"{block}.json"))["bbox"]
    cat = pystac_client.Client.open(PC)
    items = list(cat.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime="/".join(SEASON)).items())
    geoms = [shape(json.loads(g)) for g in fields.geom]
    cache = {}

    def grid_for(crs):
        if crs not in cache:
            tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
            cache[crs] = [(stransform(tr, g).buffer(BUFFER_M), k + 1) for k, g in enumerate(geoms)]
            cache[crs] = [(mapping(g), k) for g, k in cache[crs] if not g.is_empty]
        return cache[crs]

    def do(item):
        for attempt in range(4):
            try:
                it = pc.sign(item)
                res = {}
                for pol in ("vv", "vh"):
                    with rasterio.open(it.assets[pol].href) as src:
                        tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
                        bb = stransform(tr, box(*bbox)).bounds
                        win = from_bounds(*bb, src.transform).round_offsets().round_lengths()
                        arr = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float64)
                        wtr = rasterio.windows.transform(win, src.transform)
                        shapes = grid_for(src.crs.to_string())
                        lab = rasterize(shapes, out_shape=arr.shape, transform=wtr, fill=0, dtype="int32")
                    ok = np.isfinite(arr) & (arr > 0) & (lab > 0)
                    L = np.where(ok, lab, 0).ravel()
                    n = np.bincount(L, minlength=len(fields) + 1)
                    s = np.bincount(L, weights=np.where(ok, arr, 0).ravel(), minlength=len(fields) + 1)
                    res[pol] = (s, n)
                break
            except Exception as e:
                if attempt == 3:
                    log(f"S1 {block} {item.id}: failed {e}", "s1.log"); return None
                time.sleep(3 * (attempt + 1))
        (sv, nv), (sh, nh) = res["vv"], res["vh"]
        p = item.properties
        df = pd.DataFrame({"field_id": fields.field_id, "n_px": nv[1:], "vv_lin": sv[1:] / np.maximum(nv[1:], 1),
                           "vh_lin": sh[1:] / np.maximum(nh[1:], 1)})
        df = df[(nv[1:] >= 20) & (nh[1:] >= 20)]
        df["date"] = item.datetime.strftime("%Y-%m-%d")
        df["datetime"] = item.datetime.isoformat()
        df["rel_orbit"] = p.get("sat:relative_orbit"); df["pass"] = p.get("sat:orbit_state"); df["platform"] = p.get("platform")
        return df

    with ThreadPoolExecutor(int(os.environ.get("HTH_THREADS", 8))) as ex:
        parts = [d for d in ex.map(do, items) if d is not None and len(d)]
    df = pd.concat(parts, ignore_index=True)
    df["vv_db"] = 10 * np.log10(df.vv_lin); df["vh_db"] = 10 * np.log10(df.vh_lin)
    df["cr_db"] = df.vh_db - df.vv_db   # = 10*log10(VH_lin/VV_lin)
    df["block"] = block
    df.to_parquet(out)
    log(f"S1 {block}: {len(items)} items, {df.date.nunique()} dates, orbits {df.groupby('rel_orbit').date.nunique().to_dict()}, {time.time()-t0:.0f}s", "s1.log")


if __name__ == "__main__":
    for b in sys.argv[1:]:
        try:
            process(b)
        except Exception as e:
            import traceback; traceback.print_exc(); log(f"S1 {b}: FAILED {e}", "s1.log")
