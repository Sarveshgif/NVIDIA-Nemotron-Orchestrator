"""Sentinel-2 L2A extraction per block (Element84 Earth Search, anonymous AWS COGs).

For each block: pick the one MGRS tile that covers it, build a 10 m grid in that tile's UTM CRS,
rasterise every silage/grain parcel (10 m inward buffer), then for each acquisition read the block
window once and summarise every field.

Outputs (data/s2/):
  fields_<block>.parquet  one row per parcel: id, class code, areas (projected CRS), pixel counts, tile
  obs_<block>.parquet     one row per field x acquisition (all acquisitions, incl. cloudy ones, so coverage
                          and gaps are measurable): clear fraction, band means, NDVI/LSWI/NDRE, 2x2 zone NDVI
  chips_<block>.npz       24x24 px (240 m) chips centred on each chip-field, clear dates only
Scaling: reflectance = DN * 0.0001 - 0.1 (asset raster:bands scale/offset, baseline >= 04.00).
Bands: B02 blue, B03 green, B04 red, B08 nir (10 m); B05 rededge1, B8A nir08, B11 swir16, B12 swir22,
SCL (20 m, read onto the 10 m grid by nearest neighbour - no added detail).
NDVI = (B08-B04)/(B08+B04)   LSWI = (B8A-B11)/(B8A+B11)   NDRE = (B8A-B05)/(B8A+B05)
Clear pixel = SCL in {4 vegetation, 5 not-vegetated}. A field-date counts as observed if >= 60% of its
interior pixels are clear; stats use clear pixels only.
"""
import os, sys, json, time
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES",
                  CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_MAX_RETRY="5",
                  GDAL_HTTP_RETRY_DELAY="2", VSI_CACHE="TRUE")
import numpy as np, pandas as pd, rasterio, pystac_client
from rasterio.features import rasterize
from rasterio.windows import from_bounds, Window
from rasterio.enums import Resampling
from shapely.geometry import shape, box, mapping
from shapely.ops import transform as stransform
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor
from common import ROOT, SEASON, SILAGE, GRAIN, BUFFER_M, MIN_INTERIOR_PX, log

BANDS10 = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}
BANDS20 = {"B05": "rededge1", "B8A": "nir08", "B11": "swir16", "B12": "swir22"}
BAND_ORDER = ["B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12"]
CHIP = 24
MIN_CLEAR = 0.6
OUT = ROOT / "s2"; OUT.mkdir(exist_ok=True)
ES = "https://earth-search.aws.element84.com/v1"
# Default = Collection-1 reprocessing. Override only where it has gaps (e.g. 2022 over Maryland -> "sentinel-2-l2a");
# scaling still comes from each asset's raster:bands, so DN -> reflectance stays per-item correct.
S2_COLLECTION = os.environ.get("HTH_S2_COLLECTION", "sentinel-2-c1-l2a")
# Earth Search legacy "sentinel-2-l2a" COGs already have the BOA offset removed (DN is exactly 1000 lower than
# Collection-1 on the same scene, verified 2022-03-21/12-06/12-11) while raster:bands still says offset -0.1.
# Set HTH_S2_OFFSET_OVERRIDE=0 for that collection; default None = use the asset metadata.
OFFSET_OVERRIDE = os.environ.get("HTH_S2_OFFSET_OVERRIDE")


def load_parcels(block):
    d = json.load(open(ROOT / "blocks" / f"{block}.json"))
    bb = box(*d["bbox"])
    keep = []
    for f in d["features"]:
        code = f["properties"]["gewascode"]
        if code not in SILAGE and code not in GRAIN:
            continue
        g = shape(f["geometry"])
        if not bb.contains(g.centroid):  # parcel belongs to the block holding its centroid
            continue
        keep.append((f["id"], code, f["properties"]["gewas"], g))
    return d["bbox"], keep


def pick_tile(items, block_poly):
    by = {}
    for it in items:
        by.setdefault(it.properties["grid:code"], []).append(it)
    best = None
    for code, its in by.items():
        cover = max(shape(i.geometry).intersection(block_poly).area / block_poly.area for i in its[:10])
        score = (round(cover, 3), len(its))
        if best is None or score > best[0]:
            best = (score, code)
    return best[1], best[0][0]


def process(block, chip_ids=None):
    if (OUT / f"obs_{block}.parquet").exists():
        log(f"{block}: already done"); return
    t0 = time.time()
    bbox, parcels = load_parcels(block)
    block_poly = box(*bbox)
    cat = pystac_client.Client.open(ES)
    items = list(cat.search(collections=[S2_COLLECTION], bbox=bbox, datetime="/".join(SEASON),
                            query={"eo:cloud_cover": {"lt": 95}}).items())
    tile, cover = pick_tile(items, block_poly)
    items = sorted([i for i in items if i.properties["grid:code"] == tile], key=lambda i: i.datetime)
    # de-duplicate reprocessed items of the same acquisition
    seen, uniq = set(), []
    for i in items:
        k = i.datetime.strftime("%Y-%m-%d")
        if k not in seen:
            seen.add(k); uniq.append(i)
    items = uniq
    with rasterio.open(items[0].assets["red"].href) as src:
        crs, ttr = src.crs, src.transform
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True).transform
    foot = stransform(to_utm, shape(items[0].geometry))
    if os.environ.get("HTH_FOOTPRINT") == "union":   # first item may be a partial swath; use all acquisitions' data footprint
        from shapely.ops import unary_union
        foot = stransform(to_utm, unary_union([shape(i.geometry) for i in items]))
    bpoly = stransform(to_utm, block_poly)
    xmin, ymin, xmax, ymax = bpoly.bounds
    win = from_bounds(xmin, ymin, xmax, ymax, ttr).round_offsets().round_lengths()
    H, W = int(win.height), int(win.width)
    wtr = rasterio.windows.transform(win, ttr)

    rows, interior, full = [], [], []
    for k, (pid, code, name, g) in enumerate(parcels, start=1):
        gu = stransform(to_utm, g)
        gi = gu.buffer(BUFFER_M)
        rec = dict(field_id=pid, block=block, code=code, crop_name=name, label="silage" if code in SILAGE else "grain",
                   area_ha=stransform(to_rd, g).area / 1e4, area_utm_ha=gu.area / 1e4, tile=tile,
                   cx=gu.centroid.x, cy=gu.centroid.y, lon=g.centroid.x, lat=g.centroid.y, in_tile=foot.contains(gu),
                   geom=json.dumps(mapping(g)), idx=k)
        rows.append(rec)
        if not gi.is_empty:
            interior.append((mapping(gi), k))
        full.append((mapping(gu), k))
    lab = rasterize(interior, out_shape=(H, W), transform=wtr, fill=0, dtype="int32")
    lab_full = rasterize(full, out_shape=(H, W), transform=wtr, fill=0, dtype="int32")
    fields = pd.DataFrame(rows)
    npx = np.bincount(lab.ravel(), minlength=len(rows) + 1)
    fields["n_interior_px"] = npx[fields.idx]
    fields["eligible"] = (fields.n_interior_px >= MIN_INTERIOR_PX) & fields.in_tile
    fields["excl_reason"] = np.where(~fields.in_tile, "outside_tile",
                                     np.where(fields.n_interior_px < MIN_INTERIOR_PX, "too_small_interior", ""))
    elig = fields[fields.eligible]
    # 2x2 zones: split each field's interior pixels at its median row and median column (pixel-unique).
    rr, cc = np.nonzero(lab)
    ids = lab[rr, cc]
    zone = np.zeros_like(lab, dtype=np.int8)
    order = np.argsort(ids)
    rr, cc, ids = rr[order], cc[order], ids[order]
    bounds_ = np.searchsorted(ids, np.arange(1, len(rows) + 2))
    for k in range(1, len(rows) + 1):
        a, b = bounds_[k - 1], bounds_[k]
        if b - a == 0: continue
        r, c = rr[a:b], cc[a:b]
        zone[r, c] = 1 + (r > np.median(r)).astype(np.int8) * 2 + (c > np.median(c)).astype(np.int8)
    # chip centres (pixel coords) for chip fields
    chip_set = set(elig.field_id) if chip_ids is None else set(chip_ids) & set(elig.field_id)
    centres = {}
    for _, r in elig[elig.field_id.isin(chip_set)].iterrows():
        col, row = ~wtr * (r.cx, r.cy)
        r0 = int(np.clip(round(row) - CHIP // 2, 0, H - CHIP)); c0 = int(np.clip(round(col) - CHIP // 2, 0, W - CHIP))
        centres[r.idx] = (r0, c0)
    fmask_chips = {k: (lab_full[r0:r0 + CHIP, c0:c0 + CHIP] == k) for k, (r0, c0) in centres.items()}
    imask_chips = {k: (lab[r0:r0 + CHIP, c0:c0 + CHIP] == k) for k, (r0, c0) in centres.items()}
    nfield = len(rows) + 1
    log(f"{block}: tile {tile} cover {cover:.2f} grid {H}x{W} items {len(items)} parcels {len(rows)} eligible {len(elig)} chips {len(centres)}")

    def read(item, key, out_shape=None):
        a = item.assets[key]
        rb = (a.extra_fields.get("raster:bands") or [{}])[0]
        with rasterio.open(a.href) as src:
            if out_shape is None:
                arr = src.read(1, window=win, boundless=True, fill_value=0)   # identical inside the tile; zero (= nodata) past its edge
            else:
                w20 = from_bounds(*rasterio.windows.bounds(win, ttr), src.transform)
                arr = src.read(1, window=w20, out_shape=out_shape, resampling=Resampling.nearest, boundless=True, fill_value=0)
        return arr, rb.get("scale", 1.0), (float(OFFSET_OVERRIDE) if OFFSET_OVERRIDE is not None else rb.get("offset", 0.0))

    def do_item(item):
        date = item.datetime.strftime("%Y-%m-%d")
        for attempt in range(4):
            try:
                scl, _, _ = read(item, "scl", (H, W))
                break
            except Exception as e:
                if attempt == 3:
                    log(f"{block} {date}: SCL read failed {e}"); return None
                time.sleep(3 * (attempt + 1))
        clear = np.isin(scl, (4, 5))
        valid = scl > 0
        n_clear = np.bincount(lab.ravel(), weights=clear.ravel(), minlength=nfield)
        n_valid = np.bincount(lab.ravel(), weights=valid.ravel(), minlength=nfield)
        frac = np.where(npx > 0, n_clear / np.maximum(npx, 1), 0)
        base = pd.DataFrame({"idx": np.arange(nfield), "clear_frac": frac, "valid_frac": n_valid / np.maximum(npx, 1)}).iloc[1:]
        base["date"] = date
        good = frac >= MIN_CLEAR
        good[0] = False
        if not good[elig.idx.values].any():
            return base, None
        bands = {}
        for attempt in range(4):
            try:
                for b, key in BANDS10.items():
                    arr, s, o = read(item, key); bands[b] = arr.astype(np.float32) * s + o
                for b, key in BANDS20.items():
                    arr, s, o = read(item, key, (H, W)); bands[b] = arr.astype(np.float32) * s + o
                break
            except Exception as e:
                if attempt == 3:
                    log(f"{block} {date}: band read failed {e}"); return base, None
                time.sleep(3 * (attempt + 1))
        eps = 1e-6
        ndvi = (bands["B08"] - bands["B04"]) / (bands["B08"] + bands["B04"] + eps)
        lswi = (bands["B8A"] - bands["B11"]) / (bands["B8A"] + bands["B11"] + eps)
        ndre = (bands["B8A"] - bands["B05"]) / (bands["B8A"] + bands["B05"] + eps)
        m = clear & (lab > 0)
        L = np.where(m, lab, 0).ravel()
        cnt = np.bincount(L, minlength=nfield)
        def fmean(x):
            return np.bincount(L, weights=np.where(m, x, 0).ravel(), minlength=nfield) / np.maximum(cnt, 1)
        out = {"ndvi": fmean(ndvi), "lswi": fmean(lswi), "ndre": fmean(ndre)}
        out["ndvi_sd"] = np.sqrt(np.maximum(fmean(ndvi ** 2) - out["ndvi"] ** 2, 0))
        for b in BAND_ORDER:
            out[b] = fmean(bands[b])
        for z in range(1, 5):
            mz = m & (zone == z)
            Lz = np.where(mz, lab, 0).ravel()
            cz = np.bincount(Lz, minlength=nfield)
            sz = np.bincount(Lz, weights=np.where(mz, ndvi, 0).ravel(), minlength=nfield)
            out[f"ndvi_z{z}"] = np.where(cz >= 10, sz / np.maximum(cz, 1), np.nan)
        df = pd.DataFrame(out)
        df["idx"] = np.arange(nfield)
        base = base.merge(df, on="idx", how="left")
        base.loc[~good[base.idx.values], [c for c in out]] = np.nan
        chips = []
        for k, (r0, c0) in centres.items():
            if not good[k]: continue
            stack = np.stack([bands[b][r0:r0 + CHIP, c0:c0 + CHIP] for b in BAND_ORDER]).astype(np.float16)
            chips.append((k, date, stack, clear[r0:r0 + CHIP, c0:c0 + CHIP]))
        return base, chips

    obs, chip_rec = [], []
    with ThreadPoolExecutor(int(os.environ.get("HTH_THREADS", 8))) as ex:
        for res in ex.map(do_item, items):
            if res is None: continue
            base, chips = res
            obs.append(base)
            if chips: chip_rec += chips
    obs = pd.concat(obs, ignore_index=True)
    obs = obs.merge(fields[["idx", "field_id"]], on="idx").drop(columns="idx")
    obs = obs[obs.field_id.isin(elig.field_id)]
    fields.drop(columns=["idx"]).to_parquet(OUT / f"fields_{block}.parquet")
    idx2id = dict(zip(fields.idx, fields.field_id))
    if chip_rec:
        np.savez_compressed(
            OUT / f"chips_{block}.npz",
            field_id=np.array([idx2id[k] for k, *_ in chip_rec]), date=np.array([d for _, d, *_ in chip_rec]),
            x=np.stack([s for *_, s, _ in chip_rec]), clear=np.stack([c for *_, c in chip_rec]),
            chip_field_id=np.array([idx2id[k] for k in centres]),
            field_mask=np.stack([fmask_chips[k] for k in centres]), interior_mask=np.stack([imask_chips[k] for k in centres]),
            bands=np.array(BAND_ORDER))
    obs.to_parquet(OUT / f"obs_{block}.parquet")
    log(f"{block}: done in {time.time() - t0:.0f}s, obs rows {len(obs)}, observed field-dates {obs.ndvi.notna().sum()}, chips {len(chip_rec)}")


if __name__ == "__main__":
    for b in sys.argv[1:]:
        try:
            process(b)
        except Exception as e:
            import traceback; traceback.print_exc(); log(f"{b}: FAILED {e}")
