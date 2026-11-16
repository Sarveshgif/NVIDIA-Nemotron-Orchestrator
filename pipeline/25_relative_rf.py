"""Relative-feature Random Forest: no calendar dates, no absolute index values on fixed dates.
Every feature is a ratio to the field's own peak, a change across the field's own harvest-like transition, or a
duration measured from the field's own events. Derived from the standard 04_build_features columns.

Trained exactly like the console (05_models.py): NL 2024+2025 train regions, the same 12-setting grid chosen by
validation ROC-AUC, Platt calibration and balanced-accuracy threshold on the validation regions.
Evaluated on: NL 2025 test regions (1,847), BARC 44 field labels, and state silage shares vs NASS 2025 for
West Virginia (300-field sample), Rhode Island (all 102), Delaware (105-field sample). Console RF = comparison.
Output: <HTH_US>/relative_rf/{results.json, importance.csv, features.json}
"""
import os, json, pathlib, importlib.util, numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from datetime import date, timedelta
HERE = pathlib.Path(__file__).resolve().parent
US = pathlib.Path(os.environ["HTH_US"]); BARC = pathlib.Path(os.environ["HTH_BARC"])
OUT = US / "relative_rf"; OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("bf", HERE / "04_build_features.py"); bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)
from multiyear import load_tables
console = joblib.load(pathlib.Path(os.environ["RF_BUNDLE"]) / "rf_nl_console.joblib"); CCOLS = console["cols"]
SEED = 20251114
GRID = [dict(max_depth=d, min_samples_leaf=m, max_features=f) for d in (None, 16) for m in (1, 3, 8) for f in ("sqrt", 0.3)]
NASS = {"wv": 9000 / 40000, "ri": 1.0, "de": 2000 / 174000}


def relative(F):
    F = F.copy(); pk = F.peak_ndvi
    g = lambda c: F[c] if c in F else pd.Series(np.nan, index=F.index)
    hm = lambda var: [c for c in F.columns if c.startswith(f"{var}_hm") and c[-3:-1] in ("06", "07", "08", "09")]
    R = pd.DataFrame(index=F.index)
    R["r_season_len"] = g("tr_start_doy") - g("greenup_doy")
    R["r_peak_to_tr_days"] = g("tr_start_doy") - g("peak_doy")
    R["r_greenup_to_peak_days"] = g("peak_doy") - g("greenup_doy")
    R["r_ndvi_before"] = g("tr_ndvi_before") / pk; R["r_ndvi_after"] = g("tr_ndvi_after") / pk
    R["r_drop"] = g("tr_fall") / pk; R["r_drop_rate"] = g("tr_rate") / pk; R["tr_gap"] = g("tr_gap")
    R["r_predrop_decline"] = g("predrop_decline") / pk; R["r_predrop_rate"] = g("predrop_decline_rate") / pk
    R["r_slope30_pre"] = g("slope30_pre") / pk
    R["r_post_20_60"] = g("post_ndvi_20_60") / pk; R["r_post_min30"] = g("post_min_ndvi_30") / pk
    R["r_first_below6_days"] = g("first_below_6_doy") - g("peak_doy"); R["r_first_below4_days"] = g("first_below_4_doy") - g("peak_doy")
    R["r_ndre_before"] = g("tr_ndre_before") / F[hm("ndre")].max(axis=1)
    R["r_lswi_change"] = g("tr_lswi_after") - g("tr_lswi_before"); R["r_lswi_before_vs_peak"] = g("tr_lswi_before") - F[hm("lswi")].max(axis=1)
    R["zone_tr_spread"] = g("zone_tr_spread"); R["r_zone_sd_before"] = g("zone_sd_before") / pk; R["r_zone_sd_after"] = g("zone_sd_after") / pk
    R["r_ndvi_sd_before"] = g("ndvi_sd_before") / pk
    for c in ("s1_vh_db_change_at_tr", "s1_vv_db_change_at_tr", "s1_cr_db_change_at_tr", "s1_vh_pre60_slope", "s1_vh_maxdrop"): R[c] = g(c)
    R["r_s1_maxdrop_vs_tr_days"] = g("s1_vh_maxdrop_doy") - g("tr_mid_doy")
    # radar level in the month of the transition relative to the field's own Jul-Aug level (dB difference)
    mo = (pd.to_datetime("2025-01-01") + pd.to_timedelta(g("tr_mid_doy").fillna(0), unit="D")).dt.month.where(g("tr_mid_doy").notna())
    for v in ("vh_db", "cr_db"):
        summer = F[[c for c in (f"{v}_m07", f"{v}_m08") if c in F]].mean(axis=1)
        at = pd.Series([F.at[i, f"{v}_m{int(m):02d}"] if pd.notna(m) and f"{v}_m{int(m):02d}" in F else np.nan for i, m in mo.items()], index=F.index)
        R[f"r_{v}_tr_vs_summer"] = at - summer
    return R


def feats(root, blocks, year):
    bf.YEAR = year; rows = []
    for b in blocks:
        fp, op = root / "s2" / f"fields_{b}.parquet", root / "s2" / f"obs_{b}.parquet"
        if not (fp.exists() and op.exists()): continue
        fl, obs = pd.read_parquet(fp), pd.read_parquet(op)
        if "ndvi" not in obs.columns: continue
        og = dict(tuple(obs.groupby("field_id"))); s1g = {}
        sp = root / "s1" / f"s1_{b}.parquet"
        if sp.exists():
            s1 = pd.read_parquet(sp); s1g = dict(tuple(s1[s1.rel_orbit == int(s1.groupby("rel_orbit").size().idxmax())].groupby("field_id")))
        for r in fl.itertuples():
            gdf = og.get(r.field_id)
            if not r.eligible or gdf is None or gdf.ndvi.notna().sum() == 0: continue
            F = dict(field_id=r.field_id); F.update(bf.s2_features(gdf)); F.update(bf.s1_features(s1g.get(r.field_id), F.get("tr_mid_doy", np.nan)))
            rows.append(F)
    X = pd.DataFrame(rows)
    for c in CCOLS:
        if c not in X: X[c] = np.nan
    return X


def state(st):
    p = json.load(open(US / st / "prepare.json")); X = feats(US / st / "2025", p["blocks"], 2025)
    cdl = {f["id"]: f["properties"]["cdl_ha"] for b in p["blocks"] for f in json.load(open(US / st / "2025" / "blocks" / f"{b}.json"))["features"]}
    X["acres"] = X.field_id.map(cdl) / 0.404686; return X


lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def binm(y, call, score):
    y, call = np.asarray(y, bool), np.asarray(call, bool)
    tp, fn, fp, tn = int((y & call).sum()), int((y & ~call).sum()), int((~y & call).sum()), int((~y & ~call).sum())
    f = lambda a, b: round(a / b, 3) if b else None
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, recall_silage=f(tp, tp + fn), recall_grain=f(tn, tn + fp), precision_silage=f(tp, tp + fp),
                precision_grain=f(tn, tn + fn), balanced_accuracy=round(((f(tp, tp + fn) or 0) + (f(tn, tn + fp) or 0)) / 2, 3),
                roc_auc=round(float(roc_auc_score(y, score)), 3) if len(set(y)) == 2 else None)


def share(call, p, acres):
    call, p, a = np.asarray(call, bool), np.asarray(p), np.asarray(acres); rng = np.random.default_rng(2025)
    bt = [a[i][call[i]].sum() / a[i].sum() for i in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
    return dict(threshold_share=round(float(a[call].sum() / a.sum()), 3), threshold_ci95=[round(float(np.percentile(bt, 2.5)), 3), round(float(np.percentile(bt, 97.5)), 3)],
                weighted_share=round(float((p * a).sum() / a.sum()), 3))


# ---------------- train (NL) ----------------
df, _ = load_tables(); df = df[df.role != "unused"].reset_index(drop=True)
RC = list(relative(df).columns)
RX = relative(df); TR, VA, TE = (df.role == s for s in ("train", "val", "test"))
best = None
for prm in GRID:
    m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                      RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED, **prm)).fit(RX[TR], df.y[TR])
    auc = roc_auc_score(df.y[VA], m.predict_proba(RX[VA])[:, 1])
    if best is None or auc > best[0]: best = (auc, prm, m)
val_auc, prm, model = best
pv = model.predict_proba(RX[VA])[:, 1]; cal = LogisticRegression(C=1e6).fit(lg(pv)[:, None], df.y[VA])
calib = lambda p: cal.predict_proba(lg(p)[:, None])[:, 1]
cv = calib(pv); ths = np.linspace(0.05, 0.95, 91)
thr = float(ths[int(np.argmax([binm(df.y[VA], cv >= t, cv)["balanced_accuracy"] for t in ths]))])
P = lambda X: calib(model.predict_proba(relative(X)[RC])[:, 1])
Pc = lambda X: console["calibration"].predict_proba(lg(console["model"].predict_proba(X[CCOLS])[:, 1])[:, None])[:, 1]
print(f"relative RF: {len(RC)} features, val AUC {val_auc:.3f}, params {prm}, threshold {thr:.2f}", flush=True)
# ---------------- evaluate ----------------
res = dict(features=RC, n_features=len(RC), params=prm, val_auc=round(val_auc, 3), threshold=thr, console_threshold=console["threshold"])
te = df[TE]
res["NL_test_1847"] = dict(console=binm(te.y, Pc(te) >= console["threshold"], Pc(te)), relative=binm(te.y, P(te) >= thr, P(te)))
bl = []
for y in (2021, 2022, 2023):
    X = feats(BARC / str(y), ["BARC"], y); X["year"] = y; bl.append(X)
BA = pd.concat(bl, ignore_index=True).merge(pd.read_csv(BARC / "reference_labels.csv")[["field_id", "year", "label"]], on=["field_id", "year"])
assert len(BA) == 44
yb = BA.label == "silage"
res["BARC_44"] = dict(console=binm(yb, Pc(BA) >= console["threshold"], Pc(BA)), relative=binm(yb, P(BA) >= thr, P(BA)))
for st in ("wv", "ri", "de"):
    X = state(st)
    res[f"state_{st}"] = dict(nass_silage_share=round(NASS[st], 3), fields=len(X),
                              console=share(Pc(X) >= console["threshold"], Pc(X), X.acres), relative=share(P(X) >= thr, P(X), X.acres))
imp = pd.Series(model.named_steps["randomforestclassifier"].feature_importances_[:len(RC)], index=RC).sort_values(ascending=False)
imp.to_csv(OUT / "importance.csv"); res["top_features"] = imp.head(10).round(4).to_dict()
json.dump(res, open(OUT / "results.json", "w"), indent=1, default=str)
joblib.dump(dict(model=model, cols=RC, threshold=thr, calibration=cal, name="rf_relative_nl", note="features from relative() in 25_relative_rf.py"), OUT / "rf_relative_nl.joblib", compress=3)
print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=1, default=str))
