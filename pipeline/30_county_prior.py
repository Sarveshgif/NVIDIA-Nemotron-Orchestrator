"""County-prior calibration (option 2), validated on a later year.

Prior   : each county's historical silage share from FSA crop acreage data 2022, 2023, 2024 (final January releases),
          corn intended use Silage / (Grain + Silage), planted acres pooled over the 3 years. Counties with < 50
          pooled acres fall back to their state's pooled share. 2025 is never used to build the prior.
Model   : per-field probabilities already computed (no retraining): console RF and merged RF, each first mapped to US
          probabilities with the BARC Platt recalibration (28_recalibrate_merged.py), whose base rate is BARC's
          12/44 silage field-years. Bayes prior shift per field from that base rate to the county prior:
              p_c = r p / (r p + s (1 - p)),  r = pi_c / pi_0,  s = (1 - pi_c) / (1 - pi_0)
County share: calls (p_c >= 0.5) and probability-weighted (sum p_c x acres / acres).
Truth   : FSA 2025 county silage share (as 29_county_validation.py). Baseline: county history alone (2022-24 share).
Output  : data_us/county_prior/{county_table.csv, summary.json}
"""
import os, json, math, pathlib, numpy as np, pandas as pd
US = pathlib.Path(os.environ["HTH_US"]); OUT = US / "county_prior"; OUT.mkdir(exist_ok=True)
FSA = US / "fsa"
PLATT = {"console": (-9.053, 4.752), "merged": (-9.723, 5.12)}       # from 28_recalibrate_merged.py (fitted on all 44 BARC)
PI0 = 12 / 44
STATES = {"54": "wv", "44": "ri", "10": "de"}
MIN_FIELDS, MIN_PRIOR_ACRES, EPS = 5, 50.0, 0.005


def fsa_corn(path):
    x = pd.ExcelFile(path)
    sheet = next(s for s in x.sheet_names if "county" in s.lower() and "data" in s.lower())
    raw = pd.read_excel(x, sheet, header=None, dtype=str, nrows=6)
    hdr = int(next(i for i in range(6) if "State Code" in raw.iloc[i].astype(str).tolist()))
    d = pd.read_excel(x, sheet, header=hdr, dtype={"State Code": str, "County Code": str})
    d = d[(d.Crop == "CORN") & d["Intended Use"].isin(["Grain", "Silage"]) & ~d["Crop Type"].astype(str).str.contains("SWEET|POP", case=False)]
    d = d.assign(fips=d["State Code"].str.zfill(2) + d["County Code"].str.zfill(3))
    d = d[d.fips.str[:2].isin(STATES)]
    p = d.pivot_table(index="fips", columns="Intended Use", values="Planted Acres", aggfunc="sum", fill_value=0)
    for c in ("Grain", "Silage"):
        if c not in p: p[c] = 0.0
    return p[["Grain", "Silage"]]


hist = {y: fsa_corn(next((FSA / f"fsa_{y}").glob("*.xlsx"))) for y in (2022, 2023, 2024)}
H = sum(hist.values()) if False else pd.concat(hist.values()).groupby(level=0).sum()
H["hist_acres"] = H.Grain + H.Silage; H["hist_share"] = H.Silage / H.hist_acres
st_prior = {st: float(H[H.index.str[:2] == st].Silage.sum() / H[H.index.str[:2] == st].hist_acres.sum()) for st in STATES}
T25 = fsa_corn(next((FSA / "fsa_2025").glob("*.xlsx"))); T25["fsa2025_share"] = T25.Silage / (T25.Grain + T25.Silage)
per_year = {y: (h.Silage / (h.Grain + h.Silage)).rename(f"share_{y}") for y, h in hist.items()}

# fields with county (from 29) and model probabilities
fields = []
for st in ("wv", "ri", "de"):
    sc = pd.read_csv(US / "merged_rf" / f"state_{st}_scores.csv")
    blocks = json.load(open(US / st / "prepare.json"))["blocks"]
    geo = pd.concat([pd.read_parquet(US / st / "2025" / "s2" / f"fields_{b}.parquet")[["field_id", "lon", "lat"]]
                     for b in blocks if (US / st / "2025" / "s2" / f"fields_{b}.parquet").exists()])
    fields.append(sc.merge(geo, on="field_id").assign(state=st))
Fd = pd.concat(fields, ignore_index=True)
ns = {"__file__": str(pathlib.Path(__file__).resolve().parent / "29_county_validation.py")}
src = open(pathlib.Path(__file__).resolve().parent / "29_county_validation.py").read()
start = src.index("# ---- counties ----"); end = src.index("# ---- predictions per field ----")
ns["US"] = US; ns["STATES"] = {"wv": ("54", ""), "ri": ("44", ""), "de": ("10", "")}
exec("import zipfile, urllib.request, shapefile\nfrom shapely.geometry import shape, Point\nfrom shapely.strtree import STRtree\n" + src[start:end], ns)
Fd[["fips", "county"]] = [ns["county_of"](x, y) for x, y in zip(Fd.lon, Fd.lat)]

lgt = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))
Fd["prior"] = [H.hist_share.get(f, np.nan) if H.hist_acres.get(f, 0) >= MIN_PRIOR_ACRES else st_prior[f[:2]] for f in Fd.fips]
Fd["prior"] = Fd.prior.clip(EPS, 1 - EPS)
for m, (a, b) in PLATT.items():
    pus = sig(a + b * lgt(Fd[f"p_{m}"].values)); Fd[f"{m}_us"] = pus
    r, s = Fd.prior / PI0, (1 - Fd.prior) / (1 - PI0)
    Fd[f"{m}_prior"] = r * pus / (r * pus + s * (1 - pus))
# field ids (CDL00001...) restart in every state: always join on (state, field_id)
v4 = pd.concat([pd.read_csv(US / st / "results" / "v4_fields.csv")[["field_id", "final"]].assign(state=st) for st in ("wv", "ri")])
n0 = len(Fd); Fd = Fd.merge(v4, on=["state", "field_id"], how="left"); assert len(Fd) == n0, "join duplicated fields"

rows = []
for (fips, cn), g in Fd.groupby(["fips", "county"]):
    a = g.acres; rec = dict(state=STATES[fips[:2]], fips=fips, county=cn, fields=len(g), acres=round(a.sum(), 1),
                            history_only=round(float(g.prior.iloc[0]), 3))
    for m in PLATT:
        rec[f"{m}_us_calls"] = round(float(a[g[f"{m}_us"] >= 0.5].sum() / a.sum()), 3)
        rec[f"{m}_us_weighted"] = round(float((g[f"{m}_us"] * a).sum() / a.sum()), 3)
        rec[f"{m}_prior_calls"] = round(float(a[g[f"{m}_prior"] >= 0.5].sum() / a.sum()), 3)
        rec[f"{m}_prior_weighted"] = round(float((g[f"{m}_prior"] * a).sum() / a.sum()), 3)
    dv = g[g.final.isin(["SILAGE", "GRAIN"])]
    rec["v4"] = round(float(dv.acres[dv.final == "SILAGE"].sum() / dv.acres.sum()), 3) if len(dv) else np.nan
    rows.append(rec)
C = pd.DataFrame(rows).merge(T25[["fsa2025_share"]], left_on="fips", right_index=True, how="left")
for y, s in per_year.items(): C = C.merge(s, left_on="fips", right_index=True, how="left")
C.to_csv(OUT / "county_table.csv", index=False)
METHODS = ["history_only"] + [f"{m}_{k}" for m in PLATT for k in ("us_calls", "us_weighted", "prior_calls", "prior_weighted")] + ["v4"]


def summ(sub):
    out = {}
    for m in METHODS:
        s = sub[[m, "fsa2025_share"]].dropna()
        if len(s) < 3: continue
        e = s[m] - s.fsa2025_share
        out[m] = dict(counties=len(s), mae_pts=round(100 * float(e.abs().mean()), 1), bias_pts=round(100 * float(e.mean()), 1),
                      spearman=round(float(s[m].rank().corr(s.fsa2025_share.rank())), 2))
    return out


big = C[C.fields >= MIN_FIELDS]
res = dict(prior_years=[2022, 2023, 2024], truth="FSA 2025", pi0_barc=round(PI0, 3), state_priors={STATES[k]: round(v, 3) for k, v in st_prior.items()},
           counties_ge5_all=summ(big), counties_ge5_wv_ri=summ(big[big.state != "de"]), counties_ge5_same_as_v4=summ(big[big.v4.notna()]))
json.dump(res, open(OUT / "summary.json", "w"), indent=1, default=str)
for k in ("counties_ge5_all", "counties_ge5_same_as_v4"):
    print(f"\n== {k}")
    for m, v in res[k].items(): print(f"  {m:24s} n={v['counties']:2d}  MAE {v['mae_pts']:5.1f}  bias {v['bias_pts']:+6.1f}  spearman {v['spearman']:.2f}")
print("\nstate priors (2022-24):", res["state_priors"])
print(big.sort_values(["state", "fsa2025_share"])[["state", "county", "fields", "fsa2025_share", "share_2022", "share_2023", "share_2024", "history_only",
      "console_us_weighted", "console_prior_weighted", "merged_prior_weighted", "merged_prior_calls", "v4"]].to_string(index=False))
