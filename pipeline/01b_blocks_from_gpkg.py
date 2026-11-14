"""Build per-block parcel files for a past year from the PDOK BRP annual GeoPackage (anonymous download).

Uses the SAME block windows as the 2025 run (read from <ref_root>/blocks/*.json) so regions and splits are identical.
Writes <HTH_DATA>/blocks/<block>.json with the same structure as 01_scan_blocks.py (GeoJSON-like features, lon/lat).
Usage: HTH_YEAR=2024 HTH_DATA=.../data2024 python 01b_blocks_from_gpkg.py <gpkg> <ref_root>
"""
import sys, json, glob, os, numpy as np, pyogrio
from shapely import from_wkb
from shapely.geometry import mapping, box
from shapely.ops import transform as stransform
from pyproj import Transformer
from common import ROOT, YEAR, log

gpkg, ref = sys.argv[1], sys.argv[2]
layers = pyogrio.list_layers(gpkg)
layer = layers[0][0]
info = pyogrio.read_info(gpkg, layer=layer)
log(f"gpkg {os.path.basename(gpkg)}: layers {layers.tolist()} crs {info['crs']} fields {list(info['fields'])} features {info['features']}")
fields = list(info["fields"])
pick = lambda *names: next((f for f in fields if f.lower() in names), None)
F_CODE, F_NAME, F_CAT, F_YEAR, F_STATUS = pick("gewascode"), pick("gewas"), pick("category", "categorie"), pick("jaar"), pick("status")
F_ID = pick("id", "identificatie", "uuid", "perceel_id")
to_rd = Transformer.from_crs("EPSG:4326", info["crs"], always_xy=True).transform
to_ll = Transformer.from_crs(info["crs"], "EPSG:4326", always_xy=True).transform
(ROOT / "blocks").mkdir(exist_ok=True)
for p in sorted(glob.glob(os.path.join(ref, "blocks", "*.json"))):
    bid = os.path.basename(p)[:-5]
    out = ROOT / "blocks" / f"{bid}.json"
    if out.exists(): continue
    bb = json.load(open(p))["bbox"]
    rb = stransform(to_rd, box(*bb)).bounds
    meta, fids, geom, vals = pyogrio.raw.read(gpkg, layer=layer, bbox=rb, return_fids=True)
    cols = {n: v for n, v in zip(meta["fields"], vals)}
    feats = []
    for k in range(len(fids)):
        g = from_wkb(geom[k])
        if g is None or g.is_empty: continue
        if g.geom_type == "MultiPolygon" and len(g.geoms) == 1: g = g.geoms[0]
        if g.geom_type != "Polygon": continue        # keep single-part parcels, as the 2025 API returned
        props = {"category": str(cols[F_CAT][k]) if F_CAT else None, "gewas": str(cols[F_NAME][k]),
                 "gewascode": int(cols[F_CODE][k]), "jaar": int(cols[F_YEAR][k]) if F_YEAR else YEAR,
                 "status": str(cols[F_STATUS][k]) if F_STATUS else None}
        fid = str(cols[F_ID][k]) if F_ID else f"{YEAR}-{fids[k]}"
        feats.append({"type": "Feature", "id": fid, "properties": props, "geometry": mapping(stransform(to_ll, g))})
    json.dump({"id": bid, "bbox": bb, "features": feats, "source": os.path.basename(gpkg)}, open(out, "w"))
    years = {f["properties"]["jaar"] for f in feats}
    log(f"{bid}: {len(feats)} parcels from gpkg (jaar {years}), silage {sum(f['properties']['gewascode'] == 259 for f in feats)}, grain {sum(f['properties']['gewascode'] == 316 for f in feats)}")
