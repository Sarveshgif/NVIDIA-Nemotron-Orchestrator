"""Relative RF with growing degree days (GDD) instead of calendar days.

Weather: Open-Meteo historical archive (ERA5-based), daily 2 m Tmax/Tmin (F), one point per block (centre, rounded to
0.25 deg), cached in <HTH_US>/weather_gdd/. GDD (corn, 86/50 F): (min(Tmax,86) + max(Tmin,50))/2 - 50 with both
clipped to [50, 86]; cumulative from Jan 1. BARC uses its saved rulebook weather.

Variants (trained exactly like the console: NL 2024+2025 train regions, 12-setting grid on validation AUC, Platt
calibration and threshold on validation):
  a  relative + GDD : the 25_relative_rf relative features with every day-count replaced by GDD between the same events
                      (green-up -> peak, peak -> transition, green-up -> transition, peak -> NDVI<0.6/<0.4,
                      radar max-drop vs transition) plus accumulated GDD at green-up and at the transition
  b  a + absolute crop state at the transition: NDVI before/after, LSWI before/after, NDRE before, post-harvest
     NDVI 20-60 d, season peak NDVI
Evaluation as 25_relative_rf.py (NL test, BARC 44, WV / RI / DE shares vs NASS), with the console for comparison.
Output: <HTH_US>/relative_gdd_rf/results.json, importance_a.csv, importance_b.csv
"""
import os, json, time, pathlib, urllib.request, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
src = open(HERE / "25_relative_rf.py").read().split("# ---------------- train (NL) ----------------")[0]
src = src.replace('            F = dict(field_id=r.field_id); F.update', '            F = dict(field_id=r.field_id, block=b); F.update')
ns = {"__file__": str(HERE / "25_relative_rf.py")}; exec(compile(src, "25_relative_rf.py", "exec"), ns)
relative, feats, state, binm, share, lg = ns["relative"], ns["feats"], ns["state"], ns["binm"], ns["share"], ns["lg"]
console, CCOLS, GRID, SEED, NASS, US, BARC = ns["console"], ns["CCOLS"], ns["GRID"], ns["SEED"], ns["NASS"], ns["US"], ns["BARC"]
load_tables = ns["load_tables"]
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
OUT = US / "relative_gdd_rf"; OUT.mkdir(exist_ok=True)
WC = US / "weather_gdd"; WC.mkdir(exist_ok=True)


def gdd_day(tx, tn):
    tx, tn = np.clip(tx, 50, 86), np.clip(tn, 50, 86)
    return (tx + tn) / 2 - 50


def cum_gdd(lat, lon, year):
    lat, lon = round(lat * 4) / 4, round(lon * 4) / 4
    p = WC / f"{lat:.2f}_{lon:.2f}_{year}.json"
    if not p.exists():
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={year}-01-01&end_date={year}-12-31"
               "&daily=temperature_2m_max,temperature_2m_min&temperature_unit=fahrenheit&timezone=auto")
        for k in range(5):
            try:
                d = json.load(urllib.request.urlopen(url, timeout=60))["daily"]; json.dump(d, open(p, "w")); break
            except Exception:
                time.sleep(5 * (k + 1))
        time.sleep(0.3)
    d = json.load(open(p))
    return np.cumsum(gdd_day(np.array(d["temperature_2m_max"], float), np.array(d["temperature_2m_min"], float)))


def block_centre(path):
    b = json.load(open(path))["bbox"]; return (b[1] + b[3]) / 2, (b[0] + b[2]) / 2


def add_gdd(F, cums):
    """cums: Series of cumulative-GDD arrays aligned with F (index 0 = Jan 1, matching 0-based doy)."""
    def at(col):
        out = []
        for c, v in zip(cums, F[col] if col in F else [np.nan] * len(F)):
            out.append(c[int(min(max(v, 0), len(c) - 1))] if v == v and c is not None else np.nan)
        return np.array(out, float)
    G = pd.DataFrame(index=F.index)
    gu, pk, tr, b6, b4, mid, md = (at(c) for c in ("greenup_doy", "peak_doy", "tr_start_doy", "first_below_6_doy", "first_below_4_doy", "tr_mid_doy", "s1_vh_maxdrop_doy"))
    G["g_greenup_to_peak"] = pk - gu; G["g_peak_to_tr"] = tr - pk; G["g_season"] = tr - gu
    G["g_peak_to_below6"] = b6 - pk; G["g_peak_to_below4"] = b4 - pk
    G["g_s1_maxdrop_vs_tr"] = md - mid; G["g_at_greenup"] = gu; G["g_at_tr"] = tr
    return G


DAY_COLS = ["r_season_len", "r_peak_to_tr_days", "r_greenup_to_peak_days", "r_first_below6_days", "r_first_below4_days", "r_s1_maxdrop_vs_tr_days"]
ABS = ["tr_ndvi_before", "tr_ndvi_after", "tr_lswi_before", "tr_lswi_after", "tr_ndre_before", "post_ndvi_20_60", "peak_ndvi"]


def design(F, cums):
    R = relative(F).drop(columns=DAY_COLS)
    A = pd.concat([R, add_gdd(F, cums)], axis=1)
    B = pd.concat([A, F[ABS].reset_index(drop=True).set_index(F.index)], axis=1)
    return A, B


# ---------------- weather lookups ----------------
roots = {2024: pathlib.Path(os.environ["HTH_ROOTS"].split(",")[0].split(":")[1]), 2025: pathlib.Path(os.environ["HTH_ROOTS"].split(",")[1].split(":")[1])}
def nl_cums(df):
    cache = {}
    def one(y, b):
        if (y, b) not in cache: cache[(y, b)] = cum_gdd(*block_centre(roots[y] / "blocks" / f"{b}.json"), y)
        return cache[(y, b)]
    return pd.Series([one(int(y), b) for y, b in zip(df.year, df.block)], index=df.index)


def us_cums(st, X):
    cache = {}
    def one(b):
        if b not in cache: cache[b] = cum_gdd(*block_centre(US / st / "2025" / "blocks" / f"{b}.json"), 2025)
        return cache[b]
    return pd.Series([one(b) for b in X.block], index=X.index)


def barc_cum(y):
    d = json.load(open(BARC / "rulebook" / f"weather_{y}.json"))["daily"]
    return np.cumsum(gdd_day(np.array(d["tmax_f"], float), np.array(d["tmin_f"], float)))


# ---------------- data ----------------
t0 = time.time()
df, _ = load_tables(); df = df[df.role != "unused"].reset_index(drop=True)
cn = nl_cums(df); A_nl, B_nl = design(df, cn)
print(f"NL designed in {time.time() - t0:.0f}s ({len(df)} rows, a={A_nl.shape[1]} b={B_nl.shape[1]} features)", flush=True)
TR, VA, TE = (df.role == s for s in ("train", "val", "test"))
bl = []
for y in (2021, 2022, 2023):
    X = feats(BARC / str(y), ["BARC"], y); X["year"] = y; bl.append(X)
BA = pd.concat(bl, ignore_index=True).merge(pd.read_csv(BARC / "reference_labels.csv")[["field_id", "year", "label"]], on=["field_id", "year"])
cb = pd.Series([barc_cum(y) for y in BA.year], index=BA.index); A_ba, B_ba = design(BA, cb)
ST = {}
for st in ("wv", "ri", "de"):
    X = state(st); ST[st] = (X,) + design(X, us_cums(st, X))
print(f"all designed in {time.time() - t0:.0f}s", flush=True)


def train(Xd):
    best = None
    for prm in GRID:
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                          RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED, **prm)).fit(Xd[TR], df.y[TR])
        auc = roc_auc_score(df.y[VA], m.predict_proba(Xd[VA])[:, 1])
        if best is None or auc > best[0]: best = (auc, prm, m)
    auc, prm, m = best
    pv = m.predict_proba(Xd[VA])[:, 1]; cal = LogisticRegression(C=1e6).fit(lg(pv)[:, None], df.y[VA])
    calib = lambda p: cal.predict_proba(lg(p)[:, None])[:, 1]; cv = calib(pv); ths = np.linspace(0.05, 0.95, 91)
    thr = float(ths[int(np.argmax([binm(df.y[VA], cv >= t, cv)["balanced_accuracy"] for t in ths]))])
    return dict(model=m, calib=calib, thr=thr, val_auc=round(auc, 3), params=prm, cols=list(Xd.columns))


Pc = lambda X: console["calibration"].predict_proba(lg(console["model"].predict_proba(X[CCOLS])[:, 1])[:, None])[:, 1]
res = {}
yb = BA.label == "silage"; te = df[TE]
res["console"] = dict(NL_test=binm(te.y, Pc(te) >= console["threshold"], Pc(te)), BARC_44=binm(yb, Pc(BA) >= console["threshold"], Pc(BA)),
                      **{f"state_{s}": share(Pc(v[0]) >= console["threshold"], Pc(v[0]), v[0].acres) for s, v in ST.items()})
for name, k in (("a_relative_gdd", 1), ("b_relative_gdd_plus_abs", 2)):
    Xn = A_nl if k == 1 else B_nl; M = train(Xn)
    P = lambda Xd: M["calib"](M["model"].predict_proba(Xd[M["cols"]])[:, 1])
    pb = P(A_ba if k == 1 else B_ba)
    r = dict(n_features=len(M["cols"]), val_auc=M["val_auc"], threshold=M["thr"], params=M["params"],
             NL_test=binm(te.y, P(Xn[TE]) >= M["thr"], P(Xn[TE])), BARC_44=binm(yb, pb >= M["thr"], pb))
    for s, v in ST.items():
        ps = P(v[k]); r[f"state_{s}"] = share(ps >= M["thr"], ps, v[0].acres)
    imp = pd.Series(M["model"].named_steps["randomforestclassifier"].feature_importances_[:len(M["cols"])], index=M["cols"]).sort_values(ascending=False)
    imp.to_csv(OUT / f"importance_{name[0]}.csv"); r["top_features"] = imp.head(8).round(4).to_dict()
    res[name] = r; print(name, "done", flush=True)
res["nass"] = {s: round(v, 3) for s, v in NASS.items()}
res["weather"] = "Open-Meteo archive (ERA5-based), one point per block rounded to 0.25 deg; BARC: saved rulebook weather"
json.dump(res, open(OUT / "results.json", "w"), indent=1, default=str)
summary = {m: {"NL_balacc_auc": (res[m]["NL_test"]["balanced_accuracy"], res[m]["NL_test"]["roc_auc"]),
               "BARC_balacc_auc": (res[m]["BARC_44"]["balanced_accuracy"], res[m]["BARC_44"]["roc_auc"]),
               **{s: res[m][f"state_{s}"]["threshold_share"] for s in ("wv", "ri", "de")}} for m in ("console", "a_relative_gdd", "b_relative_gdd_plus_abs")}
print(json.dumps(dict(summary=summary, nass=res["nass"], top_a=res["a_relative_gdd"]["top_features"], top_b=res["b_relative_gdd_plus_abs"]["top_features"]), indent=1, default=str))
