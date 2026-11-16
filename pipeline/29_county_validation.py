"""County-level validation (no model changes): predicted silage share per county vs FSA 2025 crop acreage data.

FSA 2025 acreage data as of January 12, 2026 (county_data sheet): planted acres by county x crop x crop type x
intended use; corn rows with intended use Grain (Gr) or Silage (Sg); sweet / pop corn crop types excluded.
FSA silage share = silage / (grain + silage) planted acres.
Fields -> counties: field centroid in Census cartographic-boundary counties (cb_2023_us_county_500k).
Predictions per field (already computed, unchanged):
  console NL (thr 0.88) / console + BARC US threshold (0.86) / merged NL (0.90) / merged + BARC US threshold (0.87)
  from data_us/merged_rf/state_*_scores.csv; v4 rulebook from data_us/<st>/results/v4_fields.csv (WV, RI)
Our share per county = predicted-silage acres / scored acres (CDL field areas). Counties with few sampled fields are
noisy; results are reported with field counts and summarised over counties with >= MIN_FIELDS fields.
Output: data_us/county_validation/{county_table.csv, summary.json}
"""
import os, json, zipfile, urllib.request, pathlib, numpy as np, pandas as pd
import shapefile                       # pyshp
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
US = pathlib.Path(os.environ["HTH_US"]); OUT = US / "county_validation"; OUT.mkdir(exist_ok=True)
STATES = {"wv": ("54", "West Virginia"), "ri": ("44", "Rhode Island"), "de": ("10", "Delaware")}
MIN_FIELDS = 5
THR = {"console_nl": ("p_console", 0.88), "console_us": ("p_console", 0.86), "merged_nl": ("p_merged", 0.90), "merged_us": ("p_merged", 0.87)}

# ---- FSA ----
fsa = pd.read_excel(US / "fsa/fsa_2025/2025_fsa_acres_web_010526.xlsx", sheet_name="county_data", header=1, dtype={"State Code": str, "County Code": str})
corn = fsa[(fsa.Crop == "CORN") & fsa["Intended Use"].isin(["Grain", "Silage"]) & ~fsa["Crop Type"].astype(str).str.contains("SWEET|POP", case=False)]
corn = corn.assign(fips=corn["State Code"].str.zfill(2) + corn["County Code"].str.zfill(3))
F = corn.pivot_table(index=["fips", "State", "County"], columns="Intended Use", values="Planted Acres", aggfunc="sum", fill_value=0).reset_index()
F["fsa_corn_acres"] = F.Grain + F.Silage; F["fsa_silage_share"] = F.Silage / F.fsa_corn_acres
F = F[F.fips.str[:2].isin([v[0] for v in STATES.values()])]
crop_types = sorted(corn[corn.fips.str[:2].isin([v[0] for v in STATES.values()])]["Crop Type"].astype(str).unique())

# ---- counties ----
zp = US / "fsa/cb_2023_us_county_500k.zip"
if not zp.exists():
    urllib.request.urlretrieve("https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip", zp)
zipfile.ZipFile(zp).extractall(US / "fsa/counties")
r = shapefile.Reader(str(US / "fsa/counties/cb_2023_us_county_500k"))
fi = [f[0] for f in r.fields[1:]]
cty = [(rec[fi.index("GEOID")], rec[fi.index("NAME")], shape(sh.__geo_interface__)) for rec, sh in zip(r.records(), r.shapes())
       if rec[fi.index("STATEFP")] in [v[0] for v in STATES.values()]]
tree = STRtree([c[2] for c in cty])


def county_of(lon, lat):
    p = Point(lon, lat)
    for i in tree.query(p):
        if cty[i][2].contains(p): return cty[i][0], cty[i][1]
    return None, None


# ---- predictions per field ----
rows = []
for st in STATES:
    sc = pd.read_csv(US / "merged_rf" / f"state_{st}_scores.csv")
    blocks = json.load(open(US / st / "prepare.json"))["blocks"]
    geo = pd.concat([pd.read_parquet(US / st / "2025" / "s2" / f"fields_{b}.parquet")[["field_id", "lon", "lat"]]
                     for b in blocks if (US / st / "2025" / "s2" / f"fields_{b}.parquet").exists()])
    d = sc.merge(geo, on="field_id")
    v4p = US / st / "results" / "v4_fields.csv"
    if v4p.exists():
        v4 = pd.read_csv(v4p)[["field_id", "final"]]; d = d.merge(v4, on="field_id", how="left")
    else:
        d["final"] = np.nan
    for m, (col, t) in THR.items(): d[m] = d[col] >= t
    d["v4"] = np.where(d.final.isna(), np.nan, (d.final == "SILAGE").astype(float))
    d["v4_decided"] = d.final.isin(["SILAGE", "GRAIN"])
    d[["fips", "county"]] = [county_of(x, y) for x, y in zip(d.lon, d.lat)]
    d["state"] = st; rows.append(d)
D = pd.concat(rows, ignore_index=True)
models = list(THR) + ["v4"]
agg = []
for (st, fips, cn), g in D.groupby(["state", "fips", "county"]):
    a = g.acres
    rec = dict(state=st, fips=fips, county=cn, fields=len(g), acres=round(a.sum(), 1))
    for m in THR: rec[m] = round(float(a[g[m]].sum() / a.sum()), 3)
    dv = g[g.v4_decided]
    rec["v4"] = round(float(dv.acres[dv.final == "SILAGE"].sum() / dv.acres.sum()), 3) if len(dv) else np.nan
    agg.append(rec)
C = pd.DataFrame(agg).merge(F[["fips", "Grain", "Silage", "fsa_corn_acres", "fsa_silage_share"]], on="fips", how="left")
C.to_csv(OUT / "county_table.csv", index=False)


def summ(sub):
    out = {}
    for m in models:
        s = sub[[m, "fsa_silage_share", "acres"]].dropna()
        if len(s) < 2: continue
        err = (s[m] - s.fsa_silage_share).abs()
        out[m] = dict(counties=len(s), mae_pts=round(100 * float(err.mean()), 1), acre_weighted_mae_pts=round(100 * float((err * s.acres).sum() / s.acres.sum()), 1),
                      spearman=round(float(s[m].rank().corr(s.fsa_silage_share.rank())), 2) if len(s) >= 4 else None,
                      bias_pts=round(100 * float((s[m] - s.fsa_silage_share).mean()), 1))
    return out


res = dict(source="FSA 2025 crop acreage data as of 2026-01-12 (county_data; corn, intended use Grain/Silage)",
           corn_crop_types_kept=crop_types, min_fields=MIN_FIELDS,
           fields_without_county=int(D.fips.isna().sum()),
           all_counties=summ(C), counties_with_min_fields=summ(C[C.fields >= MIN_FIELDS]),
           fsa_state_totals={st: dict(grain=float(F[F.fips.str[:2] == v[0]].Grain.sum()), silage=float(F[F.fips.str[:2] == v[0]].Silage.sum()),
                                      silage_share=round(float(F[F.fips.str[:2] == v[0]].Silage.sum() / F[F.fips.str[:2] == v[0]].fsa_corn_acres.sum()), 3))
                             for st, v in STATES.items()})
json.dump(res, open(OUT / "summary.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
show = C[C.fields >= MIN_FIELDS].sort_values(["state", "fsa_silage_share"])[["state", "county", "fields", "acres", "fsa_silage_share", "fsa_corn_acres"] + models]
print(show.to_string(index=False))
