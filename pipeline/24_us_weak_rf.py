"""Weak-label US Random Forest: Rhode Island fields = silage, Delaware fields = grain (state-level NASS shares
~100% / ~99%), versus the Dutch console RF. Same 135 features, same RF settings as the console (E2 parameters, seed).

Models
  A  NL console (rf_nl_console.joblib): NL calibration, threshold 0.88
  B  RI + DE only (weak labels), raw RF probability, threshold 0.5
  C  NL train regions (2024+2025) + RI + DE, raw RF probability, threshold 0.5
Evaluation (never used for training): BARC 44 field-years (field labels), West Virginia 300-field sample vs NASS
2025 (22.5% silage), NL 2025 test regions (1,847 fields; B and C). B's out-of-bag score on RI+DE shows how easily the
two states separate (a geography check, not accuracy).
Run on the lab machine with the E2 environment (HTH_ROOTS etc.), HTH_US, HTH_BARC, E2_RESULTS.
Output: <HTH_US>/weak_rf/{results.json, importance_B.csv}
"""
import os, sys, json, pathlib, importlib.util, numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "rf_bundle")) if (HERE / "rf_bundle").exists() else None
US = pathlib.Path(os.environ["HTH_US"]); BARC = pathlib.Path(os.environ["HTH_BARC"]); E2 = pathlib.Path(os.environ["E2_RESULTS"])
OUT = US / "weak_rf"; OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("bf", HERE / "04_build_features.py"); bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)
from multiyear import load_tables
cfg = json.load(open(E2 / "metrics.json"))["rf_s2_s1_config"]
console = joblib.load(pathlib.Path(os.environ.get("RF_BUNDLE", HERE.parent / "rf_bundle")) / "rf_nl_console.joblib")
COLS = console["cols"]


def feats(root, blocks, year):
    """Features exactly as 04_build_features / 21_us_state (radar: per block the orbit with most field-date rows)."""
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
            g = og.get(r.field_id)
            if not r.eligible or g is None or g.ndvi.notna().sum() == 0: continue
            F = dict(field_id=r.field_id, area_ha=r.area_utm_ha); F.update(bf.s2_features(g)); F.update(bf.s1_features(s1g.get(r.field_id), F.get("tr_mid_doy", np.nan)))
            rows.append(F)
    X = pd.DataFrame(rows)
    for c in COLS:
        if c not in X: X[c] = np.nan
    return X


def state(st):
    p = json.load(open(US / st / "prepare.json"))
    X = feats(US / st / "2025", p["blocks"], 2025)
    cdl = {f["id"]: f["properties"]["cdl_ha"] for b in p["blocks"] for f in json.load(open(US / st / "2025" / "blocks" / f"{b}.json"))["features"]}
    X["acres"] = X.field_id.map(cdl) / 0.404686
    return X


def console_p(X):
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    return console["calibration"].predict_proba(lg(console["model"].predict_proba(X[COLS])[:, 1])[:, None])[:, 1]


def rf(p=cfg["params"]):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=20251114, oob_score=True, **p))


def binm(y, call, score=None):
    y, call = np.asarray(y), np.asarray(call)
    tp, fn = int((y & call).sum()), int((y & ~call).sum()); fp, tn = int((~y & call).sum()), int((~y & ~call).sum())
    f = lambda a, b: round(a / b, 3) if b else None
    m = dict(tp=tp, fn=fn, fp=fp, tn=tn, precision_silage=f(tp, tp + fp), recall_silage=f(tp, tp + fn), precision_grain=f(tn, tn + fn), recall_grain=f(tn, tn + fp))
    m["balanced_accuracy"] = round(((m["recall_silage"] or 0) + (m["recall_grain"] or 0)) / 2, 3)
    if score is not None and len(set(y)) == 2: m["roc_auc"] = round(float(roc_auc_score(y, score)), 3)
    return m


def share(call, acres, seed=2025):
    call, acres = np.asarray(call, bool), np.asarray(acres, float); rng = np.random.default_rng(seed)
    bs = [acres[i][call[i]].sum() / acres[i].sum() for i in (rng.integers(0, len(acres), len(acres)) for _ in range(2000))]
    return dict(share=round(float(acres[call].sum() / acres.sum()), 3), ci95=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])


# ---------------- data ----------------
RI, DE, WV = state("ri"), state("de"), state("wv")
print("fields: RI", len(RI), "DE", len(DE), "WV", len(WV), flush=True)
bl, lab = [], {}
for y in (2021, 2022, 2023):
    Xb = feats(BARC / str(y), ["BARC"], y); Xb["year"] = y; bl.append(Xb)
BA = pd.concat(bl, ignore_index=True)
ref = pd.read_csv(BARC / "reference_labels.csv")
BA = BA.merge(ref[["field_id", "year", "label"]], on=["field_id", "year"]); BA["y"] = BA.label == "silage"
assert len(BA) == 44, len(BA)
df, _ = load_tables(); df = df[df.role != "unused"].reset_index(drop=True)
NLtr = df[df.role == "train"]; NLte = df[df.role == "test"]
# ---------------- models ----------------
W = pd.concat([RI.assign(y=1), DE.assign(y=0)], ignore_index=True)
mB = rf().fit(W[COLS], W.y)
C_X = pd.concat([NLtr[COLS], W[COLS]], ignore_index=True); C_y = np.r_[NLtr.y.values, W.y.values]
mC = rf().fit(C_X, C_y)
pB = lambda X: mB.predict_proba(X[COLS])[:, 1]; pC = lambda X: mC.predict_proba(X[COLS])[:, 1]
res = dict(training=dict(RI_silage_fields=len(RI), DE_grain_fields=len(DE), NL_train_fields=len(NLtr),
                         B_oob_accuracy_on_RI_DE=round(float(mB.named_steps["randomforestclassifier"].oob_score_), 3),
                         note="B separates RI from DE almost perfectly if OOB ~1.0: that measures state separability, not silage-vs-grain skill"))
for name, X in (("BARC_44_field_labels", BA),):
    pa = console_p(X)
    res[name] = dict(A_nl_console=binm(X.y.values, pa >= console["threshold"], pa), B_ri_de=binm(X.y.values, pB(X) >= 0.5, pB(X)),
                     C_nl_plus_ri_de=binm(X.y.values, pC(X) >= 0.5, pC(X)))
res["WV_vs_NASS_22.5pct"] = dict(A_nl_console=share(console_p(WV) >= console["threshold"], WV.acres), B_ri_de=share(pB(WV) >= 0.5, WV.acres),
                                 C_nl_plus_ri_de=share(pC(WV) >= 0.5, WV.acres))
res["NL_test_1847"] = dict(A_nl_console=binm(NLte.y.values.astype(bool), console_p(NLte) >= console["threshold"], console_p(NLte)),
                           B_ri_de=binm(NLte.y.values.astype(bool), pB(NLte) >= 0.5, pB(NLte)), C_nl_plus_ri_de=binm(NLte.y.values.astype(bool), pC(NLte) >= 0.5, pC(NLte)))
imp = pd.Series(mB.named_steps["randomforestclassifier"].feature_importances_[:len(COLS)], index=COLS).sort_values(ascending=False)
imp.head(25).to_csv(OUT / "importance_B.csv"); res["B_top_features"] = imp.head(10).round(4).to_dict()
json.dump(res, open(OUT / "results.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
