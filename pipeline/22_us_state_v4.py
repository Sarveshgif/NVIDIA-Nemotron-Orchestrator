"""Frozen lenient-v4 rulebook on a US state's CDL corn fields, compared with the RF and NASS (aggregate shares).

Usage: HTH_US=<data_us> python 22_us_state_v4.py <st> "<NASS STATE NAME>" <nass.csv>
Reads the extraction written for 21_us_state.py (no labels exist). Optical = 10 m field-mean NDVI from
02_extract_s2.py (v4 was developed on the native-20 m NDVI; the two agree at r = 0.9999 in NL).
Radar = per block, the relative orbit with the most field-date rows (as for BARC). RF predictions from
<st>/results/fields.csv. Output: <st>/results/v4_vs_rf.json, v4_fields.csv
"""
import os, sys, json, csv, pathlib, importlib.util
from collections import Counter
import pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
st, nass_state, nass_csv = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [sys.argv[0], "--v4", "--no-eval"]
spec = importlib.util.spec_from_file_location("lenient", HERE / "17_rulebook_lenient.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
assert L.VERSION == "lenient-v4"
US = pathlib.Path(os.environ["HTH_US"]); R = US / st / "2025"; OUT = US / st / "results"
blocks = json.load(open(US / st / "prepare.json"))["blocks"]
rf = pd.read_csv(OUT / "fields.csv").set_index("field_id")
rows = []
for b in blocks:
    if not (R / "s2" / f"obs_{b}.parquet").exists(): continue
    fl = pd.read_parquet(R / "s2" / f"fields_{b}.parquet"); obs = pd.read_parquet(R / "s2" / f"obs_{b}.parquet")
    if "ndvi" not in obs.columns: continue
    s1p = R / "s1" / f"s1_{b}.parquet"; s1 = None
    if s1p.exists():
        s1 = pd.read_parquet(s1p); s1 = s1[s1.rel_orbit == int(s1.groupby("rel_orbit").size().idxmax())]
    for fid in fl.field_id:
        if rf.loc[fid, "pred"] == "not_scored": continue          # same field set as the RF
        o = obs[(obs.field_id == fid) & obs.ndvi.notna()].sort_values("date")
        r = s1[s1.field_id == fid].sort_values("date") if s1 is not None else pd.DataFrame(columns=["date", "vv_db", "vh_db"])
        d = L.decide(dict(year=2025, optical=[dict(date=str(x)[:10], ndvi=float(v)) for x, v in zip(o.date, o.ndvi)],
                          radar=[dict(date=str(x)[:10], vv=float(a), vh=float(c)) for x, a, c in zip(r.date, r.vv_db, r.vh_db)]))
        rows.append(dict(field_id=fid, acres=rf.loc[fid, "cdl_ha"] / 0.404686, rf=rf.loc[fid, "pred"], rf_p=rf.loc[fid, "p_silage"], **d))
D = pd.DataFrame(rows); D.to_csv(OUT / "v4_fields.csv", index=False)
nass = {("grain" if "GRAIN" in x["Data Item"] else "silage"): x["Value"] for x in csv.DictReader(open(nass_csv))
        if x["Geo Level"] == "STATE" and x["State"] == nass_state and x["Year"] == "2025"}
tot = D.acres.sum()
acr = lambda m: round(float(D.acres[m].sum()))
res = dict(state=nass_state, fields=len(D), acres=round(tot), nass=nass,
           v4=dict(silage_acres=acr(D.final == "SILAGE"), grain_acres=acr(D.final == "GRAIN"), unresolved_acres=acr(D.final == "UNRESOLVED"),
                   silage_share_of_all=round(D.acres[D.final == "SILAGE"].sum() / tot, 3),
                   silage_share_of_decided=round(D.acres[D.final == "SILAGE"].sum() / D.acres[D.final != "UNRESOLVED"].sum(), 3),
                   fields_by_outcome=dict(Counter(D.final + " " + D.tier.fillna(""))), fields_by_step=dict(Counter(D.step.astype(int)))),
           rf=dict(silage_acres=acr(D.rf == "silage"), grain_acres=acr(D.rf == "grain"), silage_share=round(D.acres[D.rf == "silage"].sum() / tot, 3)),
           agreement=dict(Counter(D.final.str.lower() + " / rf " + D.rf)))
import numpy as np
rng = np.random.default_rng(2025); av = D.acres.values; sv = (D.final == "SILAGE").values; dv = (D.final != "UNRESOLVED").values; rv = (D.rf == "silage").values
bs = [(av[i][sv[i]].sum() / av[i].sum(), av[i][sv[i]].sum() / max(av[i][dv[i]].sum(), 1e-9), av[i][rv[i]].sum() / av[i].sum()) for i in (rng.integers(0, len(av), len(av)) for _ in range(2000))]
res["ci95"] = {k: [round(float(np.percentile([b[j] for b in bs], 2.5)), 3), round(float(np.percentile([b[j] for b in bs], 97.5)), 3)] for j, k in enumerate(("v4_silage_share_of_all", "v4_silage_share_of_decided", "rf_silage_share"))}
json.dump(res, open(OUT / "v4_vs_rf.json", "w"), indent=1, default=str); print(json.dumps(res, indent=1, default=str))
print("v4 GRAIN examples:", D[D.final == "GRAIN"][["field_id", "step", "why"]].head(6).values.tolist())
