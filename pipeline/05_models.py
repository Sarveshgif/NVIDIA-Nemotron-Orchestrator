"""Model A (rules, original + adapted), Model B (Random Forest ablations), evaluation.

Convention: positive class = SILAGE. FPR = share of grain fields predicted silage.
Split: fixed spatial regions (04_build_features.SPLIT). Hyper-parameters, rule thresholds, decision threshold,
calibration and review band are chosen on TRAIN/VAL only; TEST is touched once per final model.
Outputs: data/results/*.json|csv|png, data/results/predictions_all.parquet (leave-region-out, for the console).
"""
import json, time, itertools, warnings, importlib.util, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import ROOT, log

warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("bf", __file__.replace("05_models.py", "04_build_features.py"))
bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)
from multiyear import load_tables, roots, RESULTS, TRAIN_YEARS, TEST_YEAR
RES = RESULTS
ROOTS = roots()
SEED = 20251114

df, schema = load_tables()
df = df[df.role != "unused"].reset_index(drop=True)
assert df.uid.is_unique
FEATURE_COLS = [c for g in schema.values() for c in g]
TR, VA, TE = (df.role == s for s in ("train", "val", "test"))
log(f"models[{RES.name}]: train years {TRAIN_YEARS} test year {TEST_YEAR}; fields train={TR.sum()} val={VA.sum()} test={TE.sum()}; "
    f"grain train={(df.y[TR] == 0).sum()} val={(df.y[VA] == 0).sum()} test={(df.y[TE] == 0).sum()}", "models.log")


# ---------- metrics ----------
def metrics(y, pred, score=None, area=None):
    """y, pred in {1 silage, 0 grain}; pred may contain -1 = abstain/unresolved."""
    y, pred = np.asarray(y), np.asarray(pred)
    cov = pred >= 0
    yc, pc = y[cov], pred[cov]
    tn, fp, fn, tp = confusion_matrix(yc, pc, labels=[0, 1]).ravel() if cov.any() else (0, 0, 0, 0)
    sd = lambda a, b: a / b if b else float("nan")
    m = dict(n=int(len(y)), n_silage=int(y.sum()), n_grain=int((1 - y).sum()), coverage=float(cov.mean()),
             n_covered=int(cov.sum()), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
             precision_silage=sd(tp, tp + fp), recall_silage=sd(tp, tp + fn),
             precision_grain=sd(tn, tn + fn), recall_grain=sd(tn, tn + fp),
             fpr_silage=sd(fp, fp + tn), fpr_grain=sd(fn, fn + tp),
             accuracy=sd(tp + tn, cov.sum()))
    m["f1_silage"] = sd(2 * m["precision_silage"] * m["recall_silage"], m["precision_silage"] + m["recall_silage"])
    m["f1_grain"] = sd(2 * m["precision_grain"] * m["recall_grain"], m["precision_grain"] + m["recall_grain"])
    m["balanced_accuracy"] = (m["recall_silage"] + m["recall_grain"]) / 2
    m["macro_f1"] = (m["f1_silage"] + m["f1_grain"]) / 2
    if score is not None:
        s = np.asarray(score, float); ok = np.isfinite(s)
        if ok.sum() and len(set(y[ok])) == 2:
            m["roc_auc"] = roc_auc_score(y[ok], s[ok]); m["pr_auc_silage"] = average_precision_score(y[ok], s[ok])
            m["pr_auc_grain"] = average_precision_score(1 - y[ok], -s[ok]); m["grain_prevalence"] = float(1 - y[ok].mean())
    if area is not None:
        a = np.asarray(area); m["unresolved_area_frac"] = float(a[~cov].sum() / a.sum())
        m["area_true_silage_ha"] = float(a[y == 1].sum()); m["area_true_grain_ha"] = float(a[y == 0].sum())
        m["area_pred_silage_ha"] = float(a[pred == 1].sum()); m["area_pred_grain_ha"] = float(a[pred == 0].sum())
        m["area_unresolved_ha"] = float(a[~cov].sum())
    return m


def block_bootstrap(d, pred, score=None, B=1000, keys=("balanced_accuracy", "recall_grain", "precision_grain", "recall_silage", "fpr_silage", "roc_auc")):
    rng = np.random.default_rng(SEED)
    blocks = d.block.unique(); idx = {b: np.flatnonzero(d.block.values == b) for b in blocks}
    vals = {k: [] for k in keys}
    for _ in range(B):
        ii = np.concatenate([idx[b] for b in rng.choice(blocks, len(blocks))])
        if len(set(d.y.values[ii])) < 2: continue
        mm = metrics(d.y.values[ii], np.asarray(pred)[ii], None if score is None else np.asarray(score)[ii])
        for k in keys:
            if k in mm: vals[k].append(mm[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in vals.items() if v}


# ---------- Model A: rules ----------
obs_cache = {}
def series(yr, b):
    key = (yr, b)
    if key not in obs_cache:
        R = ROOTS[yr]
        o = pd.read_parquet(R / "s2" / f"obs_{b}.parquet")
        p = R / "s1" / f"s1_{b}.parquet"
        s1 = None
        if p.exists():
            s1 = pd.read_parquet(p); s1 = s1[s1.rel_orbit == int(s1.groupby("rel_orbit").date.nunique().idxmax())]
        obs_cache[key] = (dict(tuple(o.groupby("field_id"))), dict(tuple(s1.groupby("field_id"))) if s1 is not None else {})
    return obs_cache[key]


CALS = {"orig": lambda y: dict(peak_from=f"{y}-06-01", peak_to=f"{y}-09-30", drop_to=f"{y}-11-15"),
        "ext": lambda y: dict(peak_from=f"{y}-06-01", peak_to=f"{y}-09-30", drop_to=f"{y}-12-31")}
rf_by_cal = {}
for cname, cal in CALS.items():
    rows = []
    for _, r in df.iterrows():
        bf.YEAR = int(r.year)   # day-of-year helpers in 04_build_features use the module-level YEAR
        og, sg = series(r.year, r.block)
        g = og[r.field_id]
        rows.append(bf.rule_features(g[g.ndvi.notna()][["date", "ndvi", "lswi"]], sg.get(r.field_id), cal(int(r.year))))
    rf_by_cal[cname] = rows
log("models: rule features computed", "models.log")


def run_rules(cname, th, veto=True):
    out = [bf.apply_rules(f, th, veto) for f in rf_by_cal[cname]]
    weak = np.array([1 if o[0] == "silage" else 0 if o[0] == "grain" else -1 for o in out])
    lean = np.array([1 if o[2] == "silage" else 0 if o[2] == "grain" else -1 for o in out])
    score = np.array([o[3] for o in out], float)
    reason = np.array([o[1] or "" for o in out])
    return weak, lean, score, reason


results, preds_test = {}, {}
dt = df[TE].reset_index(drop=True)
w, l, s, reason = run_rules("orig", bf.ORIG_TH, True)
results["rules_original_weak"] = metrics(df.y[TE], w[TE], s[TE], df.area_ha[TE])
results["rules_original_lean"] = metrics(df.y[TE], l[TE], s[TE], df.area_ha[TE])
results["rules_original_abstain_reasons_test"] = pd.Series(reason[TE]).value_counts().to_dict()
results["rules_original_abstain_by_class_test"] = pd.crosstab(reason[TE], df.label[TE]).to_dict()
preds_test["rules_original"] = (w[TE], s[TE])
log(f"rules original test: {json.dumps({k: round(v, 3) for k, v in results['rules_original_weak'].items() if isinstance(v, float)})}", "models.log")

# adapted rules: grid on TRAIN, selection on VAL (balanced accuracy of weak labels x coverage^0.5)
grid = dict(silageMinNdvi=[0.5, 0.6, 0.7], silageMaxDryDown=[7, 14, 21], grainMinDryDown=[14, 25, 35],
            grainMaxNdvi=[0.45, 0.55, 0.65], maxGapAtDrop=[15, 25], minClearObs=[14], minRadarDrop=[1.5])
best = None
t0 = time.time()
for cname in CALS:
    for veto in (True, False):
        for vals in itertools.product(*grid.values()):
            th = dict(zip(grid.keys(), vals))
            if th["silageMaxDryDown"] >= th["grainMinDryDown"]: continue
            w, l, s, _ = run_rules(cname, th, veto)
            mv = metrics(df.y[VA], w[VA])
            obj = (mv["balanced_accuracy"] if np.isfinite(mv["balanced_accuracy"]) else 0) * np.sqrt(mv["coverage"])
            if best is None or obj > best[0]:
                best = (obj, cname, veto, th, mv)
log(f"adapted rule search {time.time()-t0:.0f}s: cal={best[1]} veto={best[2]} th={best[3]} val_bal_acc={best[4]['balanced_accuracy']:.3f} cov={best[4]['coverage']:.2f}", "models.log")
w, l, s, reason_a = run_rules(best[1], best[3], best[2])
results["rules_adapted_config"] = dict(calendar=CALS[best[1]](TEST_YEAR), radar_veto=best[2], thresholds=best[3], selected_on="val")
results["rules_adapted_weak"] = metrics(df.y[TE], w[TE], s[TE], df.area_ha[TE])
results["rules_adapted_lean"] = metrics(df.y[TE], l[TE], s[TE], df.area_ha[TE])
preds_test["rules_adapted"] = (w[TE], s[TE])
results["majority_silage"] = metrics(df.y[TE], np.ones(TE.sum(), int), None, df.area_ha[TE])

# ---------- Model B: Random Forest ablations ----------
G = schema
SETS = {
    "rf_s2_summary": G["s2_summary"],
    "rf_s2_summary_patch": G["s2_summary"] + G["s2_patch"],
    "rf_s1_only": G["s1"],
    "rf_s2_s1": G["s2_summary"] + G["s2_patch"] + G["s1"],
}
GRID = [dict(max_depth=d, min_samples_leaf=m, max_features=f) for d in (None, 16) for m in (1, 3, 8) for f in ("sqrt", 0.3)]


def make_rf(p):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED, **p))


def fit_select(cols, Xdf=df):
    best = None
    for p in GRID:
        m = make_rf(p).fit(Xdf.loc[TR, cols], df.y[TR])
        auc = roc_auc_score(df.y[VA], m.predict_proba(Xdf.loc[VA, cols])[:, 1])
        if best is None or auc > best[0]: best = (auc, p, m)
    return best


def threshold_and_calibration(model, cols, Xdf=df):
    pv = model.predict_proba(Xdf.loc[VA, cols])[:, 1]
    cal = LogisticRegression(C=1e6).fit(np.log(np.clip(pv, 1e-4, 1 - 1e-4) / (1 - np.clip(pv, 1e-4, 1 - 1e-4)))[:, None], df.y[VA])
    calib = lambda p: cal.predict_proba(np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))[:, None])[:, 1]
    cv = calib(pv)
    ths = np.linspace(0.05, 0.95, 91)
    ba = [metrics(df.y[VA], (cv >= t).astype(int))["balanced_accuracy"] for t in ths]
    t = float(ths[int(np.argmax(ba))])
    # Review band = explicit budget: the REVIEW_SHARE of validation fields closest to the threshold (logit distance)
    # go to an analyst. Expressed as a probability interval [lo, hi) so it can be applied to any field.
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    margin = np.abs(lg(cv) - lg(np.array([t])))
    mq = float(np.quantile(margin, REVIEW_SHARE))
    inv = lambda z: 1 / (1 + np.exp(-z))
    band = (float(inv(lg(np.array([t]))[0] - mq)), float(inv(lg(np.array([t]))[0] + mq)))
    return calib, t, band


REVIEW_SHARE = 0.30
models_out = {}
for name, cols in SETS.items():
    t0 = time.time()
    auc, p, model = fit_select(cols)
    calib, t, band = threshold_and_calibration(model, cols)
    t_fit = time.time() - t0
    t1 = time.perf_counter(); raw = model.predict_proba(dt[cols])[:, 1]; t_inf = time.perf_counter() - t1
    t1 = time.perf_counter()
    for _ in range(50): model.predict_proba(dt[cols].iloc[[0]])
    t_one = (time.perf_counter() - t1) / 50
    ps = calib(raw)
    pred = (ps >= t).astype(int)
    rev = np.where(ps >= band[1], 1, np.where(ps < band[0], 0, -1))
    results[name] = metrics(dt.y, pred, ps, dt.area_ha)
    results[name + "_with_review"] = metrics(dt.y, rev, ps, dt.area_ha)
    # acreage: hard-threshold totals vs probability-weighted totals (calibrated on val)
    a = dt.area_ha.values
    results[name + "_acreage"] = dict(declared_silage_ha=float(a[dt.y == 1].sum()), declared_grain_ha=float(a[dt.y == 0].sum()),
        threshold_silage_ha=float(a[pred == 1].sum()), threshold_grain_ha=float(a[pred == 0].sum()),
        expected_silage_ha=float((ps * a).sum()), expected_grain_ha=float(((1 - ps) * a).sum()),
        note="expected = sum of calibrated p(silage) x area; calibration fit on validation regions")
    results[name + "_config"] = dict(params=p, val_auc=auc, threshold=t, review_band=band, n_features=len(cols),
                                     fit_select_seconds=t_fit, batch_inference_seconds=t_inf, n_test=len(dt),
                                     per_field_inference_ms=t_one * 1e3)
    results[name + "_ci"] = block_bootstrap(dt, pred, ps)
    preds_test[name] = (pred, ps)
    models_out[name] = (model, calib, t, band, cols)
    rf = model.named_steps["randomforestclassifier"]
    log(f"{name}: val_auc={auc:.3f} test bal_acc={results[name]['balanced_accuracy']:.3f} auc={results[name].get('roc_auc', float('nan')):.3f} "
        f"grain P/R={results[name]['precision_grain']:.2f}/{results[name]['recall_grain']:.2f} fit {t_fit:.0f}s", "models.log")

# common-subset comparison: test fields that have S1 and an S2 transition
common = dt.s1_n.notna().values & dt.tr_mid_doy.notna().values if "s1_n" in dt else np.ones(len(dt), bool)
results["common_subset_n"] = int(common.sum())
for name, (pred, ps) in preds_test.items():
    results[f"common__{name}"] = metrics(dt.y[common], np.asarray(pred)[common], np.asarray(ps)[common])

# paired comparison S2 vs S2+S1 (McNemar exact, on identical test fields)
from scipy.stats import binomtest
def mcnemar(a, b):
    ca, cb = (np.asarray(preds_test[a][0]) == dt.y.values), (np.asarray(preds_test[b][0]) == dt.y.values)
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return dict(a_only_correct=n01, b_only_correct=n10, p_value=binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0)
results["mcnemar_s2_vs_s2s1"] = mcnemar("rf_s2_summary_patch", "rf_s2_s1")
results["mcnemar_s2sum_vs_s2patch"] = mcnemar("rf_s2_summary", "rf_s2_summary_patch")

# ---------- timeliness: features available by Oct 31 only (in-season estimate) ----------
def doy_of(c):
    import re
    m = re.search(r"_hm(\d\d)([ab])$", c)
    if m: return (int(m.group(1)), m.group(2))
    m = re.search(r"_m(\d\d)$", c)
    if m: return (int(m.group(1)), "b")
    return None
early_cols = [c for c in SETS["rf_s2_s1"] if (doy_of(c) is None or doy_of(c)[0] <= 10)
              and not c.startswith(("post_", "first_below", "s1_vh_maxdrop", "tr_", "cycle_len", "predrop", "slope30", "zone_", "ndvi_sd_before"))]
auc, p, m_early = fit_select(early_cols)
calib_e, t_e, band_e = threshold_and_calibration(m_early, early_cols)
ps = calib_e(m_early.predict_proba(dt[early_cols])[:, 1])
results["rf_asof_oct31"] = metrics(dt.y, (ps >= t_e).astype(int), ps, dt.area_ha)
results["rf_asof_oct31_config"] = dict(params=p, val_auc=auc, n_features=len(early_cols),
    note="only half-month/monthly summaries up to Oct 31 and season-level features not depending on later looks")
log(f"as-of Oct31: test bal_acc={results['rf_asof_oct31']['balanced_accuracy']:.3f} auc={results['rf_asof_oct31'].get('roc_auc', float('nan')):.3f}", "models.log")

# ---------- simulated optical gap (clearly SIMULATED) ----------
# Remove all clear optical looks from Sep 1 - Oct 15 in test fields (persistent-cloud scenario) and recompute features.
def gapped_features(sub):
    rows = []
    for _, r in sub.iterrows():
        bf.YEAR = int(r.year); lo, hi = f"{r.year}-09-01", f"{r.year}-10-15"
        og, sg = series(r.year, r.block)
        g = og[r.field_id].copy()
        m = (g.date >= lo) & (g.date <= hi)
        g.loc[m, [c for c in g.columns if c not in ("date", "field_id", "clear_frac", "valid_frac")]] = np.nan
        F = {"uid": r.uid}; F.update(bf.s2_features(g)); F.update(bf.s1_features(sg.get(r.field_id), F.get("tr_mid_doy", np.nan)))
        rows.append(F)
    out = pd.DataFrame(rows)
    for c in FEATURE_COLS:
        if c not in out: out[c] = np.nan
    return sub[["uid", "y", "block", "area_ha"]].merge(out, on="uid")

gt = gapped_features(dt)
results["simulated_gap"] = {"window": [f"{TEST_YEAR}-09-01", f"{TEST_YEAR}-10-15"], "note": "SIMULATED: real clear looks removed from test fields"}
for name in ("rf_s2_summary_patch", "rf_s2_s1"):
    model, calib, t, band, cols = models_out[name]
    ps = calib(model.predict_proba(gt[cols])[:, 1])
    results["simulated_gap"][name] = metrics(gt.y, (ps >= t).astype(int), ps)
log(f"simulated gap: S2 bal_acc={results['simulated_gap']['rf_s2_summary_patch']['balanced_accuracy']:.3f} S1+S2={results['simulated_gap']['rf_s2_s1']['balanced_accuracy']:.3f}", "models.log")

# ---------- choose the console model (val AUC only), leave-region-out predictions for every field ----------
best_name = max(SETS, key=lambda n: results[n + "_config"]["val_auc"])
results["selected_model"] = best_name
model, calib, t, band, cols = models_out[best_name]
imp = pd.Series(model.named_steps["randomforestclassifier"].feature_importances_[:len(cols)], index=cols).sort_values(ascending=False)
imp.head(25).to_csv(RES / f"importance_{best_name}.csv")
# console predictions: test fields from the frozen train-only model (genuinely held out)
lro = df.loc[TE, ["field_id", "year"]].copy()
ps = calib(model.predict_proba(df.loc[TE, cols])[:, 1])
lro["p_silage"] = ps; lro["pred"] = np.where(ps >= t, "silage", "grain")
lro["review"] = (ps >= band[0]) & (ps < band[1]); lro["pred_source"] = f"train-only model ({'+'.join(map(str, TRAIN_YEARS))}), held-out test region"
rw, rl, rs, rr = run_rules("orig", bf.ORIG_TH, True)
rules_df = pd.DataFrame({"field_id": df.field_id, "year": df.year, "rule_weak": np.where(rw == 1, "silage", np.where(rw == 0, "grain", "abstain")),
                         "rule_pred": np.where(rl == 1, "silage", np.where(rl == 0, "grain", "unclassified")), "rule_score": rs, "rule_reason": rr})
lro.merge(rules_df, on=["field_id", "year"]).to_parquet(RES / "predictions_all.parquet")
results["experiment"] = dict(train_years=TRAIN_YEARS, test_year=TEST_YEAR, n_train=int(TR.sum()), n_val=int(VA.sum()), n_test=int(TE.sum()),
                             grain_train=int((df.y[TR] == 0).sum()), grain_val=int((df.y[VA] == 0).sum()), grain_test=int((df.y[TE] == 0).sum()))

# ---------- tables & plots ----------
comp_names = ["majority_silage", "rules_original_weak", "rules_original_lean", "rules_adapted_weak", "rules_adapted_lean"] + list(SETS)
comp = pd.DataFrame({n: results[n] for n in comp_names}).T
cols_show = ["n", "coverage", "accuracy", "balanced_accuracy", "precision_silage", "recall_silage", "fpr_silage", "precision_grain", "recall_grain", "f1_grain", "roc_auc", "pr_auc_grain", "unresolved_area_frac"]
comp[[c for c in cols_show if c in comp]].to_csv(RES / "model_comparison_test.csv")
json.dump(results, open(RES / "metrics.json", "w"), indent=1, default=float)

fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
for ax, n in zip(axs, ["rules_original_weak", "rules_adapted_weak", "rf_s2_summary_patch", "rf_s2_s1"]):
    m = results[n]; cm = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
    ax.imshow(cm, cmap="Blues"); ax.set_xticks([0, 1], ["pred silage", "pred grain"]); ax.set_yticks([0, 1], ["silage", "grain"])
    for i in range(2):
        for j in range(2): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
    ax.set_title(f"{n}\ncoverage {m['coverage']:.0%}, bal.acc {m['balanced_accuracy']:.2f}", fontsize=9)
plt.tight_layout(); plt.savefig(RES / "confusion_matrices.png", dpi=130)
log(f"models: done; selected {best_name}", "models.log")
