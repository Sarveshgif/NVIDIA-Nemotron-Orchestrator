"""BARC external check, step 3: features -> console model inference -> (separately) comparison with BARC labels.

Model: the console model (E2: Random Forest, S2 + zones + S1, trained on 2024+2025 NL train regions, tuned and
calibrated on NL validation regions). 05_models.py never saved the fitted forest, so it is re-fitted here with
the E2 hyper-parameters and seed, and checked against the saved E2 test predictions before use.
Nothing is tuned on BARC: threshold, calibration and review band are the NL ones.

Inputs : <HTH_BARC>/<year>/{s2/obs_BARC.parquet, s2/fields_BARC.parquet, s1/s1_BARC.parquet}
Labels : <HTH_BARC>/reference_labels.csv is read ONLY in the evaluation section, after predictions are written.
Outputs: <HTH_BARC>/results/{predictions_barc.csv, barc_eval.json, models/console_rf_s2_s1.joblib}
Env    : HTH_ROOTS / HTH_TRAIN_YEARS / HTH_TEST_YEAR as for E2; E2_RESULTS = E2 results directory.
"""
import os, json, time, pathlib, importlib.util, warnings, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
BARC = pathlib.Path(os.environ.get("HTH_BARC", HERE.parent / "data_barc"))
E2 = pathlib.Path(os.environ["E2_RESULTS"])
OUT = BARC / "results"; (OUT / "models").mkdir(parents=True, exist_ok=True)
YEARS = (2021, 2022, 2023)
SEED = 20251114
from multiyear import load_tables

spec = importlib.util.spec_from_file_location("bf", HERE / "04_build_features.py")
bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)

# ---------------- 1. console model (re-fit exactly as 05_models.py, E2 setting) ----------------
M = json.load(open(E2 / "metrics.json"))
assert M["selected_model"] == "rf_s2_s1"
cfg = M["rf_s2_s1_config"]
df, schema = load_tables()
df = df[df.role != "unused"].reset_index(drop=True)
cols = schema["s2_summary"] + schema["s2_patch"] + schema["s1"]
assert len(cols) == cfg["n_features"], (len(cols), cfg["n_features"])
TR, VA, TE = (df.role == s for s in ("train", "val", "test"))
model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                      RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1,
                                             random_state=SEED, **cfg["params"])).fit(df.loc[TR, cols], df.y[TR])
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
pv = model.predict_proba(df.loc[VA, cols])[:, 1]
val_auc = roc_auc_score(df.y[VA], pv)
cal = LogisticRegression(C=1e6).fit(lg(pv)[:, None], df.y[VA])
calib = lambda p: cal.predict_proba(lg(p)[:, None])[:, 1]
thr, (lo, hi) = cfg["threshold"], cfg["review_band"]
# reproduction check against the saved console predictions (E2 test fields)
pt = calib(model.predict_proba(df.loc[TE, cols])[:, 1])
saved = pd.read_parquet(E2 / "predictions_all.parquet")
chk = df.loc[TE, ["field_id", "year"]].assign(p_new=pt).merge(saved[["field_id", "year", "p_silage"]], on=["field_id", "year"])
repro = dict(val_auc_saved=cfg["val_auc"], val_auc_refit=float(val_auc), n_compared=len(chk),
             max_abs_diff_p=float((chk.p_new - chk.p_silage).abs().max()),
             same_class_share=float(((chk.p_new >= thr) == (chk.p_silage >= thr)).mean()))
print("reproduction:", repro, flush=True)
assert repro["max_abs_diff_p"] < 1e-6, "re-fitted model does not reproduce the console predictions"
joblib.dump(dict(model=model, cols=cols, threshold=thr, review_band=[lo, hi], calibration=cal, source="E2 rf_s2_s1"),
            OUT / "models" / "console_rf_s2_s1.joblib")

# ---------------- 2. BARC features (no labels anywhere in this section) ----------------
rows = []
for y in YEARS:
    R = BARC / str(y)
    if not (R / "s2" / "obs_BARC.parquet").exists(): continue
    bf.YEAR = y                                   # day-of-year helpers read the module-level YEAR
    cal_orig = dict(peak_from=f"{y}-06-01", peak_to=f"{y}-09-30", drop_to=f"{y}-11-15")
    ac = M["rules_adapted_config"]
    cal_adapt = {k: f"{y}{v[4:]}" for k, v in ac["calendar"].items()}
    fl = pd.read_parquet(R / "s2" / "fields_BARC.parquet")
    obs = pd.read_parquet(R / "s2" / "obs_BARC.parquet")
    og = dict(tuple(obs.groupby("field_id")))
    s1g, orbit = {}, None
    if (R / "s1" / "s1_BARC.parquet").exists():
        s1 = pd.read_parquet(R / "s1" / "s1_BARC.parquet")
        orbit = int(s1.groupby("rel_orbit").date.nunique().idxmax()); s1g = dict(tuple(s1[s1.rel_orbit == orbit].groupby("field_id")))
    for r in fl.itertuples():
        F = dict(field_id=r.field_id, year=y, OBJECTID=int(r.field_id[3:]), eligible=bool(r.eligible), excl_reason=r.excl_reason,
                 area_ha=r.area_utm_ha, n_interior_px=int(r.n_interior_px), s1_orbit=orbit)
        g = og.get(r.field_id)
        if r.eligible and g is not None:
            rad = s1g.get(r.field_id)
            F.update(bf.s2_features(g)); F.update(bf.s1_features(rad, F.get("tr_mid_doy", np.nan)))
            opt = g[g.ndvi.notna()][["date", "ndvi", "lswi"]]
            for name, c, th, veto in (("rules_orig", cal_orig, bf.ORIG_TH, True), ("rules_adapted", cal_adapt, ac["thresholds"], ac["radar_veto"])):
                w, reason, lean, sc = bf.apply_rules(bf.rule_features(opt, rad, c), th, veto)
                F[f"{name}_weak"], F[f"{name}_reason"] = w, reason or ""
        rows.append(F)
X = pd.DataFrame(rows)
for c in cols:
    if c not in X: X[c] = np.nan
ok = X.eligible & X.n_obs.fillna(0).gt(0)
t0 = time.perf_counter()
p = np.full(len(X), np.nan); p[ok.values] = calib(model.predict_proba(X.loc[ok, cols])[:, 1])
t_inf = time.perf_counter() - t0
X["p_silage"] = p
X["pred"] = np.where(~ok, "not_scored", np.where(p >= thr, "silage", "grain"))
X["review"] = ok & (p >= lo) & (p < hi)
X["no_transition"] = ok & X.tr_mid_doy.isna()
keep = ["OBJECTID", "year", "field_id", "area_ha", "eligible", "excl_reason", "n_obs", "max_gap_aug15_oct", "s1_n", "s1_orbit",
        "peak_doy", "tr_start_doy", "tr_end_doy", "tr_ndvi_before", "tr_ndvi_after", "p_silage", "pred", "review", "no_transition",
        "rules_orig_weak", "rules_orig_reason", "rules_adapted_weak", "rules_adapted_reason"]
X[[c for c in keep if c in X]].to_csv(OUT / "predictions_barc.csv", index=False)
print(f"predictions written: {len(X)} field-years, scored {int(ok.sum())}, inference {t_inf*1e3:.0f} ms", flush=True)
# feature coverage relative to NL training: share of model inputs that are missing on BARC vs NL test
miss_barc = float(X.loc[ok, cols].isna().mean().mean()); miss_nl = float(df.loc[TE, cols].isna().mean().mean())


# ---------------- 3. evaluation: labels joined only now ----------------
def metrics(y, pred, score=None):
    y, pred = np.asarray(y), np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sd = lambda a, b: a / b if b else float("nan")
    m = dict(n=int(len(y)), n_silage=int(y.sum()), n_grain=int((1 - y).sum()), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
             precision_silage=sd(tp, tp + fp), recall_silage=sd(tp, tp + fn), precision_grain=sd(tn, tn + fn),
             recall_grain=sd(tn, tn + fp), fpr_silage=sd(fp, fp + tn), accuracy=sd(tp + tn, len(y)))
    m["f1_silage"] = sd(2 * m["precision_silage"] * m["recall_silage"], m["precision_silage"] + m["recall_silage"])
    m["f1_grain"] = sd(2 * m["precision_grain"] * m["recall_grain"], m["precision_grain"] + m["recall_grain"])
    m["balanced_accuracy"] = (m["recall_silage"] + m["recall_grain"]) / 2
    if score is not None and len(set(y)) == 2: m["roc_auc"] = float(roc_auc_score(y, score))
    return m


ref = pd.read_csv(BARC / "reference_labels.csv")
E = ref.merge(X[["OBJECTID", "year", "eligible", "excl_reason", "n_obs", "p_silage", "pred", "review", "no_transition",
                 "rules_orig_weak", "rules_adapted_weak", "area_ha"]], on=["OBJECTID", "year"], how="left")
E["y"] = (E.label == "silage").astype(int)
S = E[E.pred.isin(["silage", "grain"])].reset_index(drop=True)
yhat = (S.pred == "silage").astype(int).values
res = dict(model="console RF (E2 rf_s2_s1), NL threshold/calibration, nothing tuned on BARC", threshold=thr, review_band=[lo, hi],
           reproduction=repro, n_reference=len(E), n_scored=len(S),
           not_scored=E[~E.pred.isin(["silage", "grain"])][["OBJECTID", "year", "label", "excl_reason"]].to_dict("records"),
           overall=metrics(S.y, yhat, S.p_silage),
           by_year={int(y): metrics(g.y, (g.pred == "silage").astype(int), g.p_silage) for y, g in S.groupby("year")},
           missing_input_share=dict(barc=miss_barc, nl_test=miss_nl),
           inference=dict(seconds=t_inf, n=int(ok.sum()), note="from cached features, lab CPU; excludes imagery retrieval"))
auto = ~S.review.values.astype(bool)
res["auto_decided"] = dict(coverage=float(auto.mean()), **metrics(S.y[auto], yhat[auto], S.p_silage[auto]))
for name in ("rules_orig", "rules_adapted"):
    w = S[f"{name}_weak"].values; cov = np.isin(w, ["silage", "grain"])
    res[name] = dict(coverage=float(cov.mean()), **(metrics(S.y[cov], (w[cov] == "silage").astype(int)) if cov.any() else {}))
a = S.area_ha.values
res["acreage_ha"] = dict(declared_silage=float(a[S.y == 1].sum()), declared_grain=float(a[S.y == 0].sum()),
                         threshold_silage=float(a[yhat == 1].sum()), threshold_grain=float(a[yhat == 0].sum()),
                         expected_silage=float((S.p_silage * a).sum()), expected_grain=float(((1 - S.p_silage) * a).sum()))
# bootstrap clustered by field (OBJECTID): the same parcel appears in up to three years
rng = np.random.default_rng(SEED); ids = S.OBJECTID.unique(); ix = {k: np.flatnonzero(S.OBJECTID.values == k) for k in ids}
ba, au = [], []
for _ in range(2000):
    ii = np.concatenate([ix[k] for k in rng.choice(ids, len(ids))])
    if len(set(S.y.values[ii])) < 2: continue
    mm = metrics(S.y.values[ii], yhat[ii], S.p_silage.values[ii]); ba.append(mm["balanced_accuracy"]); au.append(mm["roc_auc"])
res["ci95_field_cluster_bootstrap"] = dict(balanced_accuracy=[float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))],
                                          roc_auc=[float(np.percentile(au, 2.5)), float(np.percentile(au, 97.5))])
E.to_csv(OUT / "barc_matched.csv", index=False)
json.dump(res, open(OUT / "barc_eval.json", "w"), indent=1, default=float)
o = res["overall"]
print(f"BARC: scored {len(S)}/{len(E)}; bal_acc {o['balanced_accuracy']:.3f} AUC {o.get('roc_auc', float('nan')):.3f} "
      f"CI {res['ci95_field_cluster_bootstrap']}; TP {o['tp']} FN {o['fn']} FP {o['fp']} TN {o['tn']}")
print("BARC_EVAL_DONE")
