"""Merged RF: the console's 135 absolute features + the relative/GDD features of 26_relative_gdd_rf.py (variant a),
duplicates removed, one Random Forest trained exactly like the console (NL 2024+2025 train regions, 12-setting grid
on validation AUC, Platt calibration and threshold on validation). Same evaluation as 25/26.
Output: <HTH_US>/merged_rf/{results.json, importance.csv}
"""
import os, json, pathlib, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
src = open(HERE / "26_relative_gdd_rf.py").read().split("def train(Xd):")[0]
ns = {"__file__": str(HERE / "26_relative_gdd_rf.py")}; exec(compile(src, "26_relative_gdd_rf.py", "exec"), ns)
g = ns.get
df, TR, VA, TE, BA, ST, A_nl, A_ba, CCOLS, console = (g(k) for k in ("df", "TR", "VA", "TE", "BA", "ST", "A_nl", "A_ba", "CCOLS", "console"))
binm, share, lg, GRID, SEED, US, NASS, BARC = (g(k) for k in ("binm", "share", "lg", "GRID", "SEED", "US", "NASS", "BARC"))
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
OUT = US / "merged_rf"; OUT.mkdir(exist_ok=True)
extra = [c for c in A_nl.columns if c not in CCOLS]
merge = lambda X, A: pd.concat([X[CCOLS].reset_index(drop=True), A[extra].reset_index(drop=True)], axis=1)
M_nl = merge(df, A_nl); M_ba = merge(BA, A_ba); M_st = {s: merge(v[0], v[1]) for s, v in ST.items()}
print(f"merged features: {M_nl.shape[1]} ({len(CCOLS)} console + {len(extra)} relative/GDD)", flush=True)
best = None
for prm in GRID:
    m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                      RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED, **prm)).fit(M_nl[TR.values], df.y[TR])
    auc = roc_auc_score(df.y[VA], m.predict_proba(M_nl[VA.values])[:, 1])
    if best is None or auc > best[0]: best = (auc, prm, m)
val_auc, prm, model = best
pv = model.predict_proba(M_nl[VA.values])[:, 1]; cal = LogisticRegression(C=1e6).fit(lg(pv)[:, None], df.y[VA])
calib = lambda p: cal.predict_proba(lg(p)[:, None])[:, 1]; cv = calib(pv); ths = np.linspace(0.05, 0.95, 91)
thr = float(ths[int(np.argmax([binm(df.y[VA], cv >= t, cv)["balanced_accuracy"] for t in ths]))])
P = lambda X: calib(model.predict_proba(X)[:, 1])
Pc = lambda X: console["calibration"].predict_proba(lg(console["model"].predict_proba(X[CCOLS])[:, 1])[:, None])[:, 1]
te = df[TE]; yb = BA.label == "silage"
res = dict(n_features=M_nl.shape[1], val_auc=round(val_auc, 3), params=prm, threshold=thr,
           merged=dict(NL_test=binm(te.y, P(M_nl[TE.values]) >= thr, P(M_nl[TE.values])), BARC_44=binm(yb, P(M_ba) >= thr, P(M_ba)),
                       **{f"state_{s}": share(P(X) >= thr, P(X), ST[s][0].acres) for s, X in M_st.items()}),
           console=dict(NL_test=binm(te.y, Pc(te) >= console["threshold"], Pc(te)), BARC_44=binm(yb, Pc(BA) >= console["threshold"], Pc(BA)),
                        **{f"state_{s}": share(Pc(v[0]) >= console["threshold"], Pc(v[0]), v[0].acres) for s, v in ST.items()}),
           nass={s: round(v, 3) for s, v in NASS.items()})
import joblib
joblib.dump(dict(model=model, cols=list(M_nl.columns), threshold_nl=float(thr), calibration_nl=cal, val_auc=float(val_auc), params=prm,
                 console_cols=list(CCOLS), extra_cols=extra), OUT / "rf_merged_nl.joblib", compress=3)
M_ba.assign(field_id=BA.field_id.values, year=BA.year.values, p_merged=P(M_ba)).to_parquet(OUT / "barc_merged_features_check.parquet")
imp = pd.Series(model.named_steps["randomforestclassifier"].feature_importances_[:M_nl.shape[1]], index=M_nl.columns).sort_values(ascending=False)
imp.to_csv(OUT / "importance.csv"); res["top_features"] = imp.head(10).round(4).to_dict()
res["share_of_importance_from_relative_gdd"] = round(float(imp[extra].sum()), 3)
json.dump(res, open(OUT / "results.json", "w"), indent=1, default=str)
# per-field scores for post-hoc recalibration (23-style): BARC with labels, states with acres
ref = pd.read_csv(BARC / "reference_labels.csv")
pd.DataFrame({"field_id": BA.field_id, "year": BA.year, "label": BA.label, "p_merged": P(M_ba), "p_console": Pc(BA)}).merge(
    ref[["field_id", "year", "OBJECTID", "ACRES"]], on=["field_id", "year"]).to_csv(OUT / "barc_scores.csv", index=False)
for s, X in M_st.items():
    pd.DataFrame({"field_id": ST[s][0].field_id, "acres": ST[s][0].acres, "p_merged": P(X), "p_console": Pc(ST[s][0])}).to_csv(OUT / f"state_{s}_scores.csv", index=False)
for k in ("merged", "console"):
    r = res[k]; b = r["BARC_44"]
    print(f"{k:8s} NL {r['NL_test']['balanced_accuracy']}/{r['NL_test']['roc_auc']} | BARC {b['balanced_accuracy']}/{b['roc_auc']} TP{b['tp']} FN{b['fn']} FP{b['fp']} TN{b['tn']} | "
          + " ".join(f"{s} {r['state_' + s]['threshold_share']} {r['state_' + s]['threshold_ci95']}" for s in ("wv", "ri", "de")))
print("threshold", thr, "val_auc", res["val_auc"], "importance share relative/GDD", res["share_of_importance_from_relative_gdd"])
print("top", res["top_features"])
