"""BARC (Beltsville, MD) external check, step 1: field-year selection and geometry, with labels kept apart.

Input : BARC_Corn 2021-2023.gpkg, layer barc_silage_grain_corn20212023 (NAD83 / Maryland ftUS).
Select: every (OBJECTID, year) with Corn_<year> in {Grain, Silage}. Other / empty values are not used.
Output (HTH_BARC, default <repo>/data_barc):
  <year>/blocks/BARC.json   label-free parcel file in the same layout as the BRP block files read by
                            02_extract_s2.py / 03_extract_s1.py: id = "OBJ<OBJECTID>", WGS84 (EPSG:4326)
                            polygon, properties = OBJECTID, year, ACRES only. No crop-use column.
  reference_labels.csv      OBJECTID, year, field_id, label, ACRES. Read ONLY by the evaluation step.
The whole polygon is kept; extraction rasterises the full parcel interior (10 m inward buffer), not a centre point.
"""
import os, sys, json, sqlite3, pathlib
import pandas as pd
from shapely import wkb
from shapely.geometry import mapping, box
from shapely.ops import transform as stransform
from pyproj import CRS, Transformer

GPKG = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "BARC Corn 2021-2023.gpkg")
LAYER = "barc_silage_grain_corn20212023"
OUT = pathlib.Path(os.environ.get("HTH_BARC", pathlib.Path(__file__).resolve().parent.parent / "data_barc"))
YEARS = (2021, 2022, 2023)
PAD_DEG = 0.01   # ~1 km around the fields for the read window


def gpkg_geom(blob):
    """GeoPackage binary -> shapely: 8-byte header + envelope (size from flag bits 1-3) + WKB."""
    flags = blob[3]
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 0b111]
    return wkb.loads(bytes(blob[8 + env:]))


con = sqlite3.connect(GPKG)
srs = con.execute("select srs_id from gpkg_geometry_columns where table_name=?", (LAYER,)).fetchone()[0]
wkt = con.execute("select definition from gpkg_spatial_ref_sys where srs_id=?", (srs,)).fetchone()[0]
to_ll = Transformer.from_crs(CRS.from_wkt(wkt), "EPSG:4326", always_xy=True).transform
rows = con.execute(f'select OBJECTID, ACRES, Corn_2021, Corn_2022, Corn_2023, geom from "{LAYER}"').fetchall()

ref, per_year = [], {y: [] for y in YEARS}
for oid, acres, c21, c22, c23, g in rows:
    geom = stransform(to_ll, gpkg_geom(g))
    if not geom.is_valid: geom = geom.buffer(0)
    for y, v in zip(YEARS, (c21, c22, c23)):
        if v not in ("Grain", "Silage"): continue
        fid = f"OBJ{int(oid)}"
        ref.append(dict(OBJECTID=int(oid), year=y, field_id=fid, label=v.lower(), ACRES=acres))
        per_year[y].append(dict(type="Feature", id=fid, geometry=mapping(geom),
                                properties=dict(OBJECTID=int(oid), year=y, ACRES=acres)))

for y, feats in per_year.items():
    if not feats: continue
    from shapely.geometry import shape
    b = [shape(f["geometry"]).bounds for f in feats]
    bbox = [min(x[0] for x in b) - PAD_DEG, min(x[1] for x in b) - PAD_DEG, max(x[2] for x in b) + PAD_DEG, max(x[3] for x in b) + PAD_DEG]
    d = OUT / str(y) / "blocks"; d.mkdir(parents=True, exist_ok=True)
    json.dump(dict(id="BARC", bbox=bbox, crs="EPSG:4326", features=feats), open(d / "BARC.json", "w"))
    print(y, len(feats), "field-years, bbox", [round(v, 4) for v in bbox])
r = pd.DataFrame(ref)
r.to_csv(OUT / "reference_labels.csv", index=False)
print(r.groupby(["year", "label"]).size().unstack(fill_value=0))
