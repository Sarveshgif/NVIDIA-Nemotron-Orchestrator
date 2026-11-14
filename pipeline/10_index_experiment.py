"""Do NDRE, PSRI and NDTI improve the silage-vs-grain Random Forest? (20 m grid, E2 setting)

Configurations (each in two families: S2 = base + zones; S2+S1 = base + zones + S1):
  baseline, +NDRE, +PSRI, +NDTI, +all three
Everything else is held fixed: same fields, same region split (train/val = 2024+2025 train/val regions, test =
2025 test regions), same seed, same 12-setting hyper-parameter grid chosen by validation ROC-AUC, threshold chosen on
validation for balanced accuracy, Platt calibration on validation. The selected configuration is chosen on
VALIDATION ONLY; test metrics are computed once per configuration for the final comparison.
Outputs (HTH_RESULTS): index_experiment.json, index_experiment_test.csv, index_experiment_val.csv,
models/index_best.joblib, index_best_test_preds.parquet
"""
import os, json, time, pathlib, warnings, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.stats import binomtest
from multiyear import roots, TRAIN_YEARS, TEST_YEAR, RESULTS
warnings.filterwarnings("ignore")
SEED = 20251114                      # identical to 05_models.py
GRID = [dict(max_depth=d, min_samples_leaf=m, max_features=f) for d in (None, 16) for m in (1, 3, 8) for f in ("sqrt", 0.3)]
RES = RESULTS; (RES / "models").mkdir(parents=True, exist_ok=True)

parts, schema = [], None
for y, r in roots().items():
    F = r / "features"
    d = (pd.read_csv(F / "splits.csv").merge(pd.read_parquet(F / "labels.parquet")[["field_id", "label"]])
         .merge(pd.read_parquet(F / "fields.parquet")[["field_id", "area_ha"]]).merge(pd.read_parquet(F / "features20.parquet")))
    d.insert(0, "year", y); d.insert(0, "uid", [f"{y}:{i}" for i in d.field_id]); parts.append(d)
    sch = json.load(open(F / "feature_schema20.json"))["groups"]
    schema = sch if schema is None else {k: [c for c in schema[k] if c in sch[k]] for k in schema}
df = pd.concat(parts, ignore_index=True)
df["y"] = (df.label == "silage").astype(int)
df["role"] = "unused"
df.loc[df.year.isin(TRAIN_YEARS) & (df.split == "train"), "role"] = "train"
df.loc[df.year.isin(TRAIN_YEARS) & (df.split == "val"), "role"] = "val"
df.loc[(df.year == TEST_YEAR) & (df.split == "test"), "role"] = "test"
df = df[df.role != "unused"].reset_index(drop=True)
TR, VA, TE = (df.role == s for s in ("train", "val", "test"))
dt = df[TE].reset_index(drop=True)
print(f"fields train={TR.sum()} val={VA.sum()} test={TE.sum()}; grain {(df.y[TR]==0).sum()}/{(df.y[VA]==0).sum()}/{(df.y[TE]==0).sum()}", flush=True)


def metrics(y, pred, score=None):
    y, pred = np.asarray(y), np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sd = lambda a, b: a / b if b else float("nan")
    m = dict(n=len(y), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
             precision_silage=sd(tp, tp + fp), recall_silage=sd(tp, tp + fn),
             precision_grain=sd(tn, tn + fn), recall_grain=sd(tn, tn + fp), fpr_silage=sd(fp, fp + tn))
    m["f1_silage"] = sd(2 * m["precision_silage"] * m["recall_silage"], m["precision_silage"] + m["recall_silage"])
    m["f1_grain"] = sd(2 * m["precision_grain"] * m["recall_grain"], m["precision_grain"] + m["recall_grain"])
    m["macro_f1"] = (m["f1_silage"] + m["f1_grain"]) / 2
    m["balanced_accuracy"] = (m["recall_silage"] + m["recall_grain"]) / 2
    if score is not None: m["roc_auc"] = roc_auc_score(y, score)
    return m


def make_rf(p):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED, **p))


def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


FAMILIES = {"S2": ["base", "patch"], "S2+S1": ["base", "patch", "s1"]}
ADD = {"baseline": [], "+NDRE": ["ndre"], "+PSRI": ["psri"], "+NDTI": ["ndti"], "+all three": ["ndre", "psri", "ndti"]}
out, models = [], {}
for fam, groups in FAMILIES.items():
    for add, idx in ADD.items():
        cols = [c for g in groups + idx for c in schema[g]]
        t0 = time.time(); best = None
        for p in GRID:                                       # tuning budget: identical 12 settings for every config
            m = make_rf(p).fit(df.loc[TR, cols], df.y[TR])
            pv = m.predict_proba(df.loc[VA, cols])[:, 1]
            auc = roc_auc_score(df.y[VA], pv)
            if best is None or auc > best[0]: best = (auc, p, m, pv)
        auc_v, p, model, pv = best
        cal = LogisticRegression(C=1e6).fit(logit(pv)[:, None], df.y[VA])
        calib = lambda q, cal=cal: cal.predict_proba(logit(q)[:, None])[:, 1]
        cv = calib(pv); ths = np.linspace(0.05, 0.95, 91)
        t = float(ths[int(np.argmax([metrics(df.y[VA], (cv >= th).astype(int))["balanced_accuracy"] for th in ths]))])
        mv = metrics(df.y[VA], (cv >= t).astype(int), cv)
        fit_s = time.time() - t0
        X = dt[cols]
        bt = []
        for _ in range(5):
            t1 = time.perf_counter(); raw = model.predict_proba(X)[:, 1]; bt.append(time.perf_counter() - t1)
        t1 = time.perf_counter()
        for i in range(50): model.predict_proba(X.iloc[[i]])
        one_ms = (time.perf_counter() - t1) / 50 * 1e3
        ps = calib(raw); pred = (ps >= t).astype(int)
        mt = metrics(dt.y, pred, ps)
        key = f"{fam} | {add}"
        models[key] = (model, calib, t, cols, pred, ps)
        out.append(dict(config=key, family=fam, added=add, n_features=len(cols), params=p, threshold=t,
                        val=mv, test=mt, batch_inference_s=float(np.median(bt)), per_field_ms=one_ms, fit_select_s=fit_s))
        print(f"{key:24s} nfeat={len(cols):3d} val AUC={mv['roc_auc']:.3f} bal={mv['balanced_accuracy']:.3f} | test bal={mt['balanced_accuracy']:.3f} "
              f"AUC={mt['roc_auc']:.3f} gP/R={mt['precision_grain']:.2f}/{mt['recall_grain']:.2f} FPR={mt['fpr_silage']:.2f} "
              f"batch {np.median(bt):.3f}s single {one_ms:.1f}ms fit {fit_s:.0f}s", flush=True)

# ---- selection on validation only ----
by = {r["config"]: r for r in out}
sel_fam = {fam: max((r for r in out if r["family"] == fam), key=lambda r: r["val"]["roc_auc"])["config"] for fam in FAMILIES}
sel = max(out, key=lambda r: r["val"]["roc_auc"])["config"]


def paired(a, b, B=2000):
    """Block bootstrap over test blocks of (b - a) for balanced accuracy and AUC; McNemar on correctness."""
    rng = np.random.default_rng(SEED)
    _, _, _, _, pa, sa = models[a]; _, _, _, _, pb, sb = models[b]
    y = dt.y.values; blocks = dt.block.unique(); ix = {k: np.flatnonzero(dt.block.values == k) for k in blocks}
    dba, dauc = [], []
    for _ in range(B):
        ii = np.concatenate([ix[k] for k in rng.choice(blocks, len(blocks))])
        if len(set(y[ii])) < 2: continue
        dba.append(metrics(y[ii], pb[ii])["balanced_accuracy"] - metrics(y[ii], pa[ii])["balanced_accuracy"])
        dauc.append(roc_auc_score(y[ii], sb[ii]) - roc_auc_score(y[ii], sa[ii]))
    ca, cb = pa == y, pb == y
    n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    return dict(delta_bal_acc_ci95=[float(np.percentile(dba, 2.5)), float(np.percentile(dba, 97.5))],
                delta_auc_ci95=[float(np.percentile(dauc, 2.5)), float(np.percentile(dauc, 97.5))],
                mcnemar_only_new_correct=n01, mcnemar_only_base_correct=n10,
                mcnemar_p=float(binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0)


comparisons = {}
for fam in FAMILIES:
    base = f"{fam} | baseline"
    for add in ADD:
        k = f"{fam} | {add}"
        if k != base: comparisons[k] = paired(base, k)

KEYS = ["precision_silage", "recall_silage", "f1_silage", "precision_grain", "recall_grain", "f1_grain", "macro_f1",
        "fpr_silage", "balanced_accuracy", "roc_auc"]
rows_t, rows_v = [], []
for r in out:
    b = by[f"{r['family']} | baseline"]
    rt = dict(config=r["config"], n_features=r["n_features"], **{k: r["test"][k] for k in KEYS},
              batch_inference_s=r["batch_inference_s"], per_field_ms=r["per_field_ms"])
    for k in KEYS: rt["d_" + k] = r["test"][k] - b["test"][k]
    rt["d_batch_inference_s"] = r["batch_inference_s"] - b["batch_inference_s"]; rt["d_per_field_ms"] = r["per_field_ms"] - b["per_field_ms"]
    rows_t.append(rt)
    rows_v.append(dict(config=r["config"], val_roc_auc=r["val"]["roc_auc"], val_balanced_accuracy=r["val"]["balanced_accuracy"],
                       val_macro_f1=r["val"]["macro_f1"], d_val_roc_auc=r["val"]["roc_auc"] - b["val"]["roc_auc"],
                       selected_in_family=r["config"] in sel_fam.values(), selected_overall=r["config"] == sel))
pd.DataFrame(rows_t).to_csv(RES / "index_experiment_test.csv", index=False)
pd.DataFrame(rows_v).to_csv(RES / "index_experiment_val.csv", index=False)

model, calib, t, cols, pred, ps = models[sel]
joblib.dump(dict(model=model, threshold=t, cols=cols, config=sel, calibration=calib.__defaults__[0]), RES / "models" / "index_best.joblib")
pd.DataFrame({"uid": dt.uid, "field_id": dt.field_id, "year": dt.year, "p_silage": ps, "pred": pred}).to_parquet(RES / "index_best_test_preds.parquet")
json.dump(dict(setting=dict(train_years=TRAIN_YEARS, test_year=TEST_YEAR, n_train=int(TR.sum()), n_val=int(VA.sum()), n_test=int(TE.sum()),
                            grid="20 m", seed=SEED, tuning_grid=GRID, selection="validation ROC-AUC (same rule as 05_models.py)"),
               configs=out, selected_per_family=sel_fam, selected_overall=sel, paired_vs_baseline_test=comparisons),
          open(RES / "index_experiment.json", "w"), indent=1, default=float)
print("SELECTED per family (val):", sel_fam, "overall:", sel)
for k, v in comparisons.items():
    print(f"{k:24s} dBalAcc CI {v['delta_bal_acc_ci95'][0]:+.3f}..{v['delta_bal_acc_ci95'][1]:+.3f} dAUC CI {v['delta_auc_ci95'][0]:+.3f}..{v['delta_auc_ci95'][1]:+.3f} McNemar p={v['mcnemar_p']:.2f}")
print("INDEX_EXPERIMENT_DONE")
