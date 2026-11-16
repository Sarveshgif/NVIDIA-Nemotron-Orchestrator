"""Frozen lenient-v4 rulebook on 40 unseen Dutch field-years (20 silage + 20 grain, 2025 test regions).

v4 was developed on BARC only; it never saw Dutch data. Labels are used only to draw the stratified sample and to
score afterwards; the rules receive date/NDVI optical looks and date/VV/VH radar passes only.
Inputs (HTH_DATA): features/{splits.csv,labels.parquet}, s2_20m/obs20_<block>.parquet, s1/s1_<block>.parquet,
         and E2 console predictions for comparison: E2_RESULTS/predictions_all.parquet.
Output: <HTH_DATA>/results_lenient_nl/{predictions.csv, per_field_eval.csv, eval.json}
"""
import os, sys, json, pathlib, importlib.util, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
VER = os.environ.get("RB_VERSION", "v4"); SEED = int(os.environ.get("RB_SEED", 20251114))
EXCLUDE = os.environ.get("RB_EXCLUDE")                   # csv of an earlier sample to keep out
sys.argv = [sys.argv[0], f"--{VER}", "--no-eval"]
spec = importlib.util.spec_from_file_location("lenient", HERE / "17_rulebook_lenient.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
assert L.VERSION == f"lenient-{VER}"
R = pathlib.Path(os.environ["HTH_DATA"]); E2 = pathlib.Path(os.environ["E2_RESULTS"])
OUT = R / f"results_lenient_nl_{VER}_seed{SEED}"; OUT.mkdir(exist_ok=True)
N = 20

spl = pd.read_csv(R / "features/splits.csv").merge(pd.read_parquet(R / "features/labels.parquet")[["field_id", "label"]])
test = spl[spl.split == "test"]
if EXCLUDE: test = test[~test.field_id.isin(pd.read_csv(EXCLUDE).field_id)]
sample = pd.concat([g.sample(N, random_state=SEED) for _, g in test.groupby("label")]).reset_index(drop=True)
truth = sample[["field_id", "block", "region", "label"]].copy()

preds = []
for b, g in sample.groupby("block"):
    obs = pd.read_parquet(R / "s2_20m" / f"obs20_{b}.parquet")
    s1 = pd.read_parquet(R / "s1" / f"s1_{b}.parquet")
    orbit = int(s1.groupby("rel_orbit").size().idxmax())          # coverage rule, as for BARC
    s1 = s1[s1.rel_orbit == orbit]
    for fid in g.field_id:                                         # no label passed below
        o = obs[(obs.field_id == fid) & obs.ndvi.notna()].sort_values("date")
        r = s1[s1.field_id == fid].sort_values("date")
        f = dict(year=2025, optical=[dict(date=str(d)[:10], ndvi=float(v)) for d, v in zip(o.date, o.ndvi)],
                 radar=[dict(date=str(d)[:10], vv=float(a), vh=float(c)) for d, a, c in zip(r.date, r.vv_db, r.vh_db)])
        d = L.decide(f)
        preds.append(dict(field_id=fid, block=b, orbit=orbit, n_optical=len(f["optical"]), n_radar=len(f["radar"]), **d))
P = pd.DataFrame(preds); P.to_csv(OUT / "predictions.csv", index=False)       # written before scoring

# ---- scoring ----
ml = pd.read_parquet(E2 / "predictions_all.parquet"); ml = ml[ml.year == 2025][["field_id", "pred"]].rename(columns={"pred": "ml"})
E = truth.merge(P, on=["field_id", "block"]).merge(ml, on="field_id", how="left")
E["rb"] = np.where(E.final == "UNRESOLVED", None, E.final.str.lower())
E["correct"] = np.where(E.rb.isna(), None, E.rb == E.label); E["ml_correct"] = E.ml == E.label
def conf(d, k):
    d = d[d[k].isin(["silage", "grain"])]
    tp = int(((d.label == "silage") & (d[k] == "silage")).sum()); fn = int(((d.label == "silage") & (d[k] == "grain")).sum())
    fp = int(((d.label == "grain") & (d[k] == "silage")).sum()); tn = int(((d.label == "grain") & (d[k] == "grain")).sum())
    rs = tp / (tp + fn) if tp + fn else None; rg = tn / (tn + fp) if tn + fp else None
    return dict(n=len(d), tp=tp, fn=fn, fp=fp, tn=tn, recall_silage=rs, recall_grain=rg,
                precision_silage=tp / (tp + fp) if tp + fp else None, precision_grain=tn / (tn + fn) if tn + fn else None,
                balanced_accuracy=None if rs is None or rg is None else (rs + rg) / 2)
dec = E[E.rb.notna()]
res = dict(version=L.VERSION, excluded_previous_sample=bool(EXCLUDE), sample=dict(n=len(E), silage=int((E.label == "silage").sum()), grain=int((E.label == "grain").sum()), seed=SEED,
                                          regions=E.region.value_counts().to_dict()),
           decided=len(dec), correct=int(dec.correct.sum()), outcomes=E.final.str.cat(E.tier.fillna(""), sep=" ").value_counts().to_dict(),
           by_step={int(s): dict(n=len(g), correct=int(g.correct.fillna(False).sum())) for s, g in E.groupby("step")},
           rulebook=conf(E, "rb"), ml_all=conf(E, "ml"), ml_same_subset=conf(dec, "ml"))
json.dump(res, open(OUT / "eval.json", "w"), indent=1, default=str)
E.drop(columns=["rb"]).to_csv(OUT / "per_field_eval.csv", index=False)
print(json.dumps({k: res[k] for k in ("sample", "decided", "correct", "outcomes", "by_step")}, indent=1, default=str))
print("rulebook", res["rulebook"]); print("ML all 40", res["ml_all"]); print("ML same subset", res["ml_same_subset"])
for r in E[E.correct == False].itertuples(): print("WRONG", r.field_id, r.region, r.label, "->", r.final, r.tier, "step", r.step, "|", r.why)
print("UNRESOLVED", E[E.rb.isna()][["field_id", "label", "why"]].values.tolist())
