"""Label audit (writes data/results/label_audit.md + label_audit.json). Run after 04_build_features.py."""
import json, glob, os, numpy as np, pandas as pd
from collections import Counter
from shapely.geometry import shape
from common import ROOT, YEAR, SILAGE, GRAIN, EXCLUDED_MAIZE

RES = ROOT / "results"; RES.mkdir(exist_ok=True)
blocks = sorted(os.path.basename(p)[4:-8] for p in glob.glob(str(ROOT / "s2" / "obs_*.parquet")))
maize_names, ids, invalid = Counter(), Counter(), 0
for b in blocks:
    d = json.load(open(ROOT / "blocks" / f"{b}.json"))
    for f in d["features"]:
        p = f["properties"]
        if "ais" in p["gewas"].lower():
            maize_names[(p["gewascode"], p["gewas"])] += 1
            ids[f["id"]] += 1
            if not shape(f["geometry"]).is_valid: invalid += 1
cand = pd.read_parquet(ROOT / "features" / "fields_all_candidates.parquet")
lab = pd.read_parquet(ROOT / "features" / "labels.parquet")
spl = pd.read_csv(ROOT / "features" / "splits.csv").merge(lab, on="field_id")
fl = pd.read_parquet(ROOT / "features" / "fields.parquet")[["field_id", "area_ha", "n_interior_px"]]
spl = spl.merge(fl, on="field_id")
dup_bbox = sum(1 for v in ids.values() if v > 1)
excl = cand.drop_duplicates("field_id").assign(reason=lambda d: np.where(d.excl_reason != "", d.excl_reason, np.where(d.block.isin(spl.block.unique()), "", "block_not_in_split")))
excl_tab = excl[excl.reason != ""].groupby(["label", "reason"]).size().unstack(fill_value=0)
cand_by_class = cand.drop_duplicates("field_id").groupby("label").size()
by_split = spl.groupby(["split", "label"]).agg(fields=("field_id", "nunique"), area_ha=("area_ha", "sum")).round(1)
by_region = spl.groupby(["split", "region", "label"]).field_id.nunique().unstack(fill_value=0)

audit = dict(
    source=(f"PDOK Basisregistratie Gewaspercelen (BRP) {YEAR}, status 'Definitief', OGC API api.pdok.nl/rvo/gewaspercelen (anonymous)"
            if YEAR == 2025 else
            f"PDOK Basisregistratie Gewaspercelen (BRP) {YEAR}, status 'Definitief', annual GeoPackage service.pdok.nl/rvo/gewaspercelen/atom/downloads/brpgewaspercelen_definitief_{YEAR}.gpkg (anonymous); same block windows as 2025"),
    license="BRP open data from RVO via PDOK (published as open data; verify current terms on the PDOK/NGR metadata record b812a145-b4fe-4331-8dc6-d914327a87ff)",
    year=YEAR, positive_class="silage (259 'Mais, snij-')", negative_class="grain (316 'Mais, korrel-')",
    excluded_maize_codes=EXCLUDED_MAIZE,
    maize_categories_in_scanned_blocks={f"{k[0]} {k[1]}": v for k, v in maize_names.most_common()},
    parcels_seen_in_more_than_one_block_bbox=dup_bbox, invalid_geometries=invalid,
    candidates_by_class=cand_by_class.to_dict(),
)
json.dump(audit, open(RES / "label_audit.json", "w"), indent=1, default=str)
md = f"""# Label audit — Netherlands BRP {YEAR}

**Source.** {audit['source']}. Gewaspercelen are the parcels farmers declare each spring for CAP payments (RVO).
The `Definitief` status means RVO finalised the registration; it is **not** an independent check of how the crop
was finally harvested. Labels are therefore *farmer-declared intended crop use*. A field declared `korrelmais`
can still be chopped as silage in autumn (and vice versa); that is irreducible label noise for this study.

**License.** {audit['license']}.

**Class definitions (official BRP codes, read from the {YEAR} records — not guessed).**

| Code | BRP name | Used as |
|---|---|---|
| 259 | Mais, snij- | **silage** (positive class) |
| 316 | Mais, korrel- | **grain** |
| 317 | Mais, corncob mix | excluded (ear-only harvest; neither class) |
| 1935 | Maiskolvensilage | excluded (ear silage) |
| 814 | Mais, suiker- | excluded (sweet corn) |

Maize categories found in the scanned blocks: {', '.join(f'{k}: {v}' for k, v in audit['maize_categories_in_scanned_blocks'].items())}.

**EuroCrops.** EuroCrops (Zenodo 14094196, CC-BY-SA-4.0) re-publishes the same RVO declarations (NL only for 2020)
with harmonised HCAT classes. It is *not* an independent second source. Its HCAT mapping merges 317 corncob mix,
814 sweet corn and 2032 energy maize into `grain_maize_corn_popcorn`, and 1935 into `green_silo_maize`, so using
HCAT classes would contaminate both labels. We use original BRP codes only, and only PDOK {YEAR} (no duplication).

**German FOR1695 / BonaRes (doi:10.20387/bonares-a0qc-46jc, CC-BY-4.0).** Investigated via DataCite metadata:
a handful of eddy-covariance research fields in Kraichgau and the Swabian Jura (2009–2018). Too few field-seasons
overlapping Sentinel-2 (2015+) to contribute a held-out class-balanced test; not used. Not a blocker.

**Not used as labels:** USDA CDL (corn only), our California rule outputs (weak labels), satellite crop maps.

## Counts

Candidate silage/grain parcels in the {len(blocks)} blocks (centroid-in-block de-duplication): {cand_by_class.to_dict()}.
Parcels whose geometry touched two block bounding boxes (assigned once, by centroid): {dup_bbox}. Invalid geometries: {invalid}.

Exclusions (field counts), checked **per class** so filtering does not silently remove one class:

{excl_tab.to_markdown()}

Eligibility: inside one Sentinel-2 tile footprint, and >= 50 clear-able interior 10 m pixels after a 10 m inward buffer
(>= 0.5 ha interior). Share excluded as too small: {', '.join(f'{c} {excl_tab.loc[c].sum() / cand_by_class[c]:.1%}' for c in excl_tab.index)} — similar for both classes.

**Unique fields by split (area from polygon in EPSG:28992 RD New):**

{by_split.to_markdown()}

**By region:**

{by_region.to_markdown()}

**Sample design.** Blocks were chosen deliberately where grain maize occurs (south-east NL), and every eligible
silage and grain parcel inside them is used. The sample is therefore *grain-enriched relative to NL as a whole*
(nationally grain maize is a small minority of maize area), but within blocks it is complete, not class-balanced.
Prevalence-dependent numbers (precision, acreage) apply to these regions, not to the whole country.

**Geometry quality.** BRP parcels are digitised on aerial imagery by RVO / farmers; boundaries are parcel-level
(not field-operation level). A parcel may contain field margins; the 10 m inward buffer removes mixed edge pixels.
"""
open(RES / "label_audit.md", "w").write(md)
print(md[:3000])
