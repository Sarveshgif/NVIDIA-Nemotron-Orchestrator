"""Sentinel-2 L2A extraction on a common 20 m grid, for the spectral-index experiment (NDRE, PSRI, NDTI).

Same blocks, same MGRS tile per block and same acquisitions as 02_extract_s2.py (tile read from fields_<block>.parquet);
same field set (the 10 m-eligible fields), so the comparison differs only in grid and features.

Grid: the tile's native 20 m grid (B05/B06/B8A/B11/B12/SCL read without resampling).
10 m bands (B02, B04, B08) are aggregated to 20 m by 2x2 mean of the native 10 m pixels (area-weighted average,
the appropriate downsampling for reflectance); a 20 m pixel is invalid if any of its four 10 m pixels is nodata.
Scaling: reflectance = DN * scale + offset from asset raster:bands (DN * 0.0001 - 0.1), identical for all bands.
Cloud mask: SCL in {4, 5} (native 20 m, no resampling). Pixel valid = clear AND every band DN > 0 (not nodata).
Field-date observed if >= 60 % of interior 20 m pixels are clear and >= 3 valid pixels (same 60 % rule as 10 m).
Interior = parcel buffered 10 m inward, rasterised at 20 m (pixel centre inside).

Indices (per pixel, then field mean over valid interior pixels):
  NDVI = (B08 - B04) / (B08 + B04)            B08 aggregated to 20 m
  LSWI = (B8A - B11) / (B8A + B11)
  NDRE = (B8A - B05) / (B8A + B05)
  PSRI = (B04 - B02) / B06                     plant senescence reflectance index; pixel invalid if B06 < 0.01
  NDTI = (B11 - B12) / (B11 + B12)             normalised difference tillage index (residue / bare soil)
Any non-finite value (zero denominator) is treated as missing for that index only.
Output: <HTH_DATA>/s2_20m/obs20_<block>.parquet
"""
import os, sys, json, time
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES",
                  CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_MAX_RETRY="5",
                  GDAL_HTTP_RETRY_DELAY="2", VSI_CACHE="TRUE",
                  GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_CONNECTTIMEOUT="20")   # a stalled read fails and is retried instead of hanging
import numpy as np, pandas as pd, rasterio, pystac_client
from rasterio.features import rasterize
from rasterio.windows import from_bounds, Window
from shapely.geometry import shape, mapping, box
from shapely.ops import transform as stransform
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor
from common import ROOT, SEASON, BUFFER_M, log

B10 = {"B02": "blue", "B04": "red", "B08": "nir"}
B20 = {"B05": "rededge1", "B06": "rededge2", "B8A": "nir08", "B11": "swir16", "B12": "swir22"}
MIN_CLEAR, MIN_VALID_PX, MIN_ZONE_PX = 0.6, 3, 3
OUT = ROOT / "s2_20m"; OUT.mkdir(exist_ok=True)
ES = "https://earth-search.aws.element84.com/v1"


def process(block):
    if (OUT / f"obs20_{block}.parquet").exists():
        log(f"{block}: 20m already done", "s2_20m.log"); return
    t0 = time.time()
    fl = pd.read_parquet(ROOT / "s2" / f"fields_{block}.parquet")
    tile = fl.tile.iloc[0]
    elig = fl[fl.eligible].reset_index(drop=True)
    bbox = json.load(open(ROOT / "blocks" / f"{block}.json"))["bbox"]
    cat = pystac_client.Client.open(ES)
    items = list(cat.search(collections=["sentinel-2-c1-l2a"], bbox=bbox, datetime="/".join(SEASON),
                            query={"eo:cloud_cover": {"lt": 95}}).items())
    items = sorted([i for i in items if i.properties["grid:code"] == tile], key=lambda i: i.datetime)
    seen, uniq = set(), []
    for i in items:
        k = i.datetime.strftime("%Y-%m-%d")
        if k not in seen: seen.add(k); uniq.append(i)
    items = uniq
    with rasterio.open(items[0].assets["scl"].href) as src:
        crs, t20 = src.crs, src.transform
    with rasterio.open(items[0].assets["red"].href) as src:
        t10 = src.transform
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    bpoly = stransform(to_utm, box(*bbox))
    win = from_bounds(*bpoly.bounds, t20).round_offsets().round_lengths()
    H, W = int(win.height), int(win.width)
    wtr = rasterio.windows.transform(win, t20)
    # matching 10 m window: same bounds, exactly 2x the pixels (10 m and 20 m grids share the tile origin)
    c10, r10 = ~t10 * (wtr.c, wtr.f)
    win10 = Window(int(round(c10)), int(round(r10)), 2 * W, 2 * H)

    shapes = []
    for k, r in enumerate(elig.itertuples(), start=1):
        gi = stransform(to_utm, shape(json.loads(r.geom))).buffer(BUFFER_M)
        if not gi.is_empty: shapes.append((mapping(gi), k))
    lab = rasterize(shapes, out_shape=(H, W), transform=wtr, fill=0, dtype="int32") if shapes else np.zeros((H, W), "int32")
    nfield = len(elig) + 1
    npx = np.bincount(lab.ravel(), minlength=nfield)
    # 2x2 zones on the 20 m grid (median row / column split, pixel-unique)
    zone = np.zeros_like(lab, dtype=np.int8)
    rr, cc = np.nonzero(lab); ids = lab[rr, cc]; o = np.argsort(ids); rr, cc, ids = rr[o], cc[o], ids[o]
    bnd = np.searchsorted(ids, np.arange(1, nfield + 1))
    for k in range(1, nfield):
        a, b = bnd[k - 1], bnd[k]
        if b > a:
            r, c = rr[a:b], cc[a:b]
            zone[r, c] = 1 + (r > np.median(r)).astype(np.int8) * 2 + (c > np.median(c)).astype(np.int8)
    log(f"{block}: tile {tile} 20m grid {H}x{W} items {len(items)} fields {len(elig)} "
        f"(no 20m interior px: {(npx[1:] == 0).sum()}, median px {int(np.median(npx[1:]))})", "s2_20m.log")

    def read(item, key, w, shape_=None):
        a = item.assets[key]; rb = (a.extra_fields.get("raster:bands") or [{}])[0]
        with rasterio.open(a.href) as src:
            arr = src.read(1, window=w, boundless=True, fill_value=0)
        return arr, rb.get("scale", 1.0), rb.get("offset", 0.0)

    def do_item(item):
        date = item.datetime.strftime("%Y-%m-%d")
        for attempt in range(4):
            try:
                scl, _, _ = read(item, "scl", win); break
            except Exception as e:
                if attempt == 3: log(f"{block} {date}: SCL read failed {e}", "s2_20m.log"); return None
                time.sleep(3 * (attempt + 1))
        clear = np.isin(scl, (4, 5))
        frac = np.bincount(lab.ravel(), weights=clear.ravel(), minlength=nfield) / np.maximum(npx, 1)
        base = pd.DataFrame({"k": np.arange(1, nfield), "clear_frac": frac[1:]}); base["date"] = date
        good = frac >= MIN_CLEAR; good[0] = False
        if not good.any(): return base
        refl, nodata = {}, np.zeros((H, W), bool)
        for attempt in range(4):
            try:
                for b, key in B10.items():
                    dn, s, o = read(item, key, win10)
                    dn = dn.astype(np.float32).reshape(H, 2, W, 2)
                    nodata |= (dn == 0).any(axis=(1, 3))
                    refl[b] = dn.mean(axis=(1, 3)) * s + o                    # 2x2 mean of native 10 m pixels
                for b, key in B20.items():
                    dn, s, o = read(item, key, win)
                    nodata |= dn == 0
                    refl[b] = dn.astype(np.float32) * s + o
                break
            except Exception as e:
                if attempt == 3: log(f"{block} {date}: band read failed {e}", "s2_20m.log"); return base
                time.sleep(3 * (attempt + 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            nd = lambda x, y: (x - y) / (x + y)
            idx = {"ndvi": nd(refl["B08"], refl["B04"]), "lswi": nd(refl["B8A"], refl["B11"]),
                   "ndre": nd(refl["B8A"], refl["B05"]),
                   "psri": np.where(refl["B06"] >= 0.01, (refl["B04"] - refl["B02"]) / refl["B06"], np.nan),
                   "ndti": nd(refl["B11"], refl["B12"])}
        valid = clear & ~nodata & (lab > 0)
        out = {}
        def fmean(x, extra=None):
            m = valid & np.isfinite(x) if extra is None else valid & extra
            L = np.where(m, lab, 0).ravel(); cnt = np.bincount(L, minlength=nfield)
            s = np.bincount(L, weights=np.where(m, x, 0).ravel(), minlength=nfield)
            return np.where(cnt >= MIN_VALID_PX, s / np.maximum(cnt, 1), np.nan), cnt
        for k, v in idx.items():
            out[k], cnt = fmean(v)
            if k == "ndvi": out["n_valid_px"] = cnt
        sq, _ = fmean(idx["ndvi"] ** 2)
        out["ndvi_sd"] = np.sqrt(np.maximum(sq - out["ndvi"] ** 2, 0))
        for b in ("B04", "B11", "B12"):
            out[b], _ = fmean(refl[b])
        for z in range(1, 5):
            m = valid & (zone == z) & np.isfinite(idx["ndvi"])
            L = np.where(m, lab, 0).ravel(); cz = np.bincount(L, minlength=nfield)
            sz = np.bincount(L, weights=np.where(m, idx["ndvi"], 0).ravel(), minlength=nfield)
            out[f"ndvi_z{z}"] = np.where(cz >= MIN_ZONE_PX, sz / np.maximum(cz, 1), np.nan)
        df = pd.DataFrame({k: v[1:] for k, v in out.items()}); df["k"] = np.arange(1, nfield)
        base = base.merge(df, on="k", how="left")
        bad = ~good[base.k.values] | ~np.isfinite(base.ndvi.values)
        base.loc[bad, [c for c in out if c != "n_valid_px"]] = np.nan
        return base

    obs = []
    with ThreadPoolExecutor(int(os.environ.get("HTH_THREADS", 8))) as ex:
        for res in ex.map(do_item, items):
            if res is not None: obs.append(res)
    obs = pd.concat(obs, ignore_index=True)
    obs["field_id"] = elig.field_id.values[obs.k.values - 1]
    obs = obs.drop(columns="k")
    obs.to_parquet(OUT / f"obs20_{block}.parquet")
    log(f"{block}: 20m done in {time.time() - t0:.0f}s, rows {len(obs)}, observed field-dates {obs.ndvi.notna().sum()}, "
        f"psri valid {obs.psri.notna().sum()}, ndti valid {obs.ndti.notna().sum()}", "s2_20m.log")


if __name__ == "__main__":
    for b in sys.argv[1:]:
        try:
            process(b)
        except Exception as e:
            import traceback; traceback.print_exc(); log(f"{b}: 20m FAILED {e}", "s2_20m.log")
