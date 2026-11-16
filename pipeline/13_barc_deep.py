"""BARC external check with the CNN -> temporal Transformer models (no retraining, nothing tuned on BARC).

Uses the 9 trained E2 checkpoints (variants s2, s2s1, s2s1_nocnn x seeds 1-3) from 06_deep.py.
For each checkpoint the normalisation statistics, Platt calibration and threshold are rebuilt exactly as in
06_deep.run (same seed -> same random training-token subset; calibration + threshold on NL validation regions),
and the rebuilt model is checked against its saved NL test metrics before it scores BARC.

BARC inputs: chips from 11_barc_extract.py with HTH_CHIP=N, in <HTH_BARC_CHIPS>/<N>/<year>/s2/chips_BARC.npz.
  N = 24: identical to training (240 m, centred on the parcel).
  N = 64: 640 m, covers each whole BARC parcel. The CNN is fully convolutional and pools only over valid interior
          pixels, so it accepts a larger chip; pixels outside the parcel interior are zeroed as in training.
S1: <HTH_BARC>/<year>/s1/s1_BARC.parquet, orbit with the most field-date rows (as in 12_barc_infer.py).
Labels: reference_labels.csv is joined only after all predictions are made.
Output: <HTH_BARC>/results/deep/{barc_deep_predictions.csv, barc_deep_eval.json}
"""
import os, sys, json, pathlib, importlib.util, numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
HERE = pathlib.Path(__file__).resolve().parent
BARC = pathlib.Path(os.environ["HTH_BARC"]); CHIPS = pathlib.Path(os.environ["HTH_BARC_CHIPS"])
E2 = pathlib.Path(os.environ["E2_RESULTS"])
OUT = BARC / "results" / "deep"; OUT.mkdir(parents=True, exist_ok=True)
VARIANTS, SEEDS, SIZES, YEARS = ("s2", "s2s1", "s2s1_nocnn"), (1, 2, 3), (24, 64), (2021, 2022, 2023)
spec = importlib.util.spec_from_file_location("d6", HERE / "06_deep.py")
d6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d6)
DEV = d6.DEV

# ---------- NL data (same loader as training) ----------
splits, X, C, T, starts, fmask, imask, s1 = d6.load()
y = (splits.label == "silage").astype(int).values
tr, va, te = (np.flatnonzero(splits.role.values == s) for s in ("train", "val", "test"))
has = np.diff(starts) >= 3
tr, va, te = tr[has[tr]], va[has[va]], te[has[te]]
tok_field = np.repeat(np.arange(len(splits)), np.diff(starts))
s1all = np.concatenate([s1[i][1] for i in tr if i in s1]); s1mu, s1sd = s1all.mean(0), s1all.std(0)
print(f"NL loaded: tokens {len(X)} train {len(tr)} val {len(va)} test {len(te)}", flush=True)


def stats(seed):
    """Exactly 06_deep.run: np.random seeded at import, first draw = 20,000 training tokens."""
    np.random.seed(seed)
    sub = np.flatnonzero(np.isin(tok_field, tr)); sub = np.random.choice(sub, min(20000, len(sub)), replace=False)
    xs = X[sub].astype(np.float32); vm = C[sub] & imask[tok_field[sub]]
    return (np.array([xs[:, c][vm].mean() for c in range(8)], np.float32), np.array([xs[:, c][vm].std() for c in range(8)], np.float32))


def predict(net, loader):
    net.eval(); out = []
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
        for bt in loader: out.append(net(*[t.to(DEV) for t in bt[:6]]).float().cpu())
    return torch.cat(out).numpy()


# ---------- BARC inputs (label-free) ----------
def barc_tokens(size):
    """Per field-year: S2 tokens (n, 10, size, size) built like 06_deep.DS (eval path) minus normalisation."""
    recs = []
    for yr in YEARS:
        z = np.load(CHIPS / str(size) / str(yr) / "s2" / "chips_BARC.npz")
        d0 = pd.Timestamp(f"{yr}-01-01")
        fm = dict(zip(z["chip_field_id"], z["field_mask"])); im = dict(zip(z["chip_field_id"], z["interior_mask"]))
        fl = pd.read_parquet(BARC / str(yr) / "s2" / "fields_BARC.parquet").set_index("field_id")
        s1p = pd.read_parquet(BARC / str(yr) / "s1" / "s1_BARC.parquet")
        orbit = int(s1p.groupby("rel_orbit").size().idxmax()); s1p = s1p[s1p.rel_orbit == orbit]
        s1g = dict(tuple(s1p.groupby("field_id")))
        for fid in z["chip_field_id"]:
            k = np.flatnonzero(z["field_id"] == fid)
            t = np.array([(pd.Timestamp(str(d)) - d0).days for d in z["date"][k]]); o = np.argsort(t); k, t = k[o], t[o]
            g = s1g.get(fid)
            sv = None if g is None else (np.array([(pd.Timestamp(str(d)) - d0).days for d in g.sort_values("date").date]),
                                          g.sort_values("date")[["vv_db", "vh_db", "cr_db"]].values.astype(np.float32))
            recs.append(dict(field_id=fid, year=yr, OBJECTID=int(fid[3:]), x=z["x"][k].astype(np.float32), clear=z["clear"][k],
                             t=t, fmask=fm[fid], imask=im[fid], s1=sv,
                             interior_cover=float(im[fid].sum() / max(int(fl.loc[fid, "n_interior_px"]), 1))))
    return recs


def barc_batch(recs, mu, sd, size):
    B = len(recs); M2, M1 = d6.MAX_S2, d6.MAX_S1
    xs = np.zeros((B, M2, 10, size, size), np.float32); ts = np.zeros((B, M2), np.float32); m2 = np.zeros((B, M2), bool)
    s1v = np.zeros((B, M1, 3), np.float32); s1t = np.zeros((B, M1), np.float32); m1 = np.zeros((B, M1), bool)
    for b, r in enumerate(recs):
        sel = np.arange(len(r["t"]))
        if len(sel) > M2: sel = np.linspace(0, len(sel) - 1, M2).astype(int)
        x = (r["x"][sel] - mu[None, :, None, None]) / sd[None, :, None, None]
        valid = (r["clear"][sel] & r["imask"][None]).astype(np.float32)
        x = np.concatenate([x * valid[:, None], valid[:, None], np.repeat(r["fmask"][None, None].astype(np.float32), len(sel), 0)], 1)
        n = len(sel); xs[b, :n] = x; ts[b, :n] = r["t"][sel]; m2[b, :n] = True
        if r["s1"] is not None:
            tt, vv = r["s1"]
            if len(tt) > M1: p = np.linspace(0, len(tt) - 1, M1).astype(int); tt, vv = tt[p], vv[p]
            s1v[b, :len(tt)] = (vv - s1mu) / s1sd; s1t[b, :len(tt)] = tt; m1[b, :len(tt)] = True
    return [torch.from_numpy(a) for a in (xs, ts, m2, s1v, s1t, m1)]


barc = {n: barc_tokens(n) for n in SIZES}
cover = {n: float(np.mean([r["interior_cover"] for r in barc[n]])) for n in SIZES}
cover_min = {n: float(np.min([r["interior_cover"] for r in barc[n]])) for n in SIZES}
print("BARC interior pixels inside chip: mean", cover, "min", cover_min, flush=True)

rows, checks = [], []
for v in VARIANTS:
    for seed in SEEDS:
        saved = json.load(open(E2 / f"deep_{v}_seed{seed}.json"))
        mu, sd = stats(seed)
        net = d6.Net(use_s1="s1" in v, use_cnn="nocnn" not in v).to(DEV)
        net.load_state_dict(torch.load(E2 / "models" / f"deep_{v}_seed{seed}.pt", map_location=DEV))
        mk = lambda idx: torch.utils.data.DataLoader(d6.DS(idx, y[idx], X, C, T, starts, fmask, imask, s1, mu, sd, s1mu, s1sd, False, "s1" in v),
                                                      batch_size=128, num_workers=8)
        lv = predict(net, mk(va))
        cal = LogisticRegression(C=1e6).fit(lv[:, None], y[va]); calib = lambda l: cal.predict_proba(l[:, None])[:, 1]
        pv = calib(lv); ths = np.linspace(0.05, 0.95, 91)
        thr = float(ths[int(np.argmax([d6.metrics(y[va], (pv >= t).astype(int))["balanced_accuracy"] for t in ths]))])
        pt = calib(predict(net, mk(te)))
        mt = d6.metrics(y[te], (pt >= thr).astype(int), pt)
        ck = dict(variant=v, seed=seed, thr_saved=saved["threshold"], thr_rebuilt=thr,
                  nl_bal_saved=saved["test"]["balanced_accuracy"], nl_bal_rebuilt=mt["balanced_accuracy"],
                  nl_auc_saved=saved["test"]["roc_auc"], nl_auc_rebuilt=mt["roc_auc"])
        checks.append(ck); print("check", ck, flush=True)
        for n in SIZES:
            with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
                net.eval(); lb = net(*[t.to(DEV) for t in barc_batch(barc[n], mu, sd, n)]).float().cpu().numpy()
            pb = calib(lb)
            for r, p in zip(barc[n], pb):
                rows.append(dict(variant=v, seed=seed, chip=n, field_id=r["field_id"], year=r["year"], OBJECTID=r["OBJECTID"],
                                 p_silage=float(p), threshold=thr, pred="silage" if p >= thr else "grain", interior_cover=r["interior_cover"]))
P = pd.DataFrame(rows); P.to_csv(OUT / "barc_deep_predictions.csv", index=False)

# ---------- evaluation: labels joined now ----------
ref = pd.read_csv(BARC / "reference_labels.csv")
E = P.merge(ref[["OBJECTID", "year", "label"]], on=["OBJECTID", "year"]); E["y"] = (E.label == "silage").astype(int)
res = dict(checks=checks, interior_cover_mean=cover, interior_cover_min=cover_min, per_seed={}, summary={}, ensemble={})
for (v, n), g in E.groupby(["variant", "chip"]):
    key = f"{v} | chip {n}"
    ms = {int(s): d6.metrics(h.y, (h.pred == "silage").astype(int), h.p_silage) for s, h in g.groupby("seed")}
    res["per_seed"][key] = ms
    res["summary"][key] = {k: [float(np.mean([m[k] for m in ms.values()])), float(np.std([m[k] for m in ms.values()]))]
                           for k in ("balanced_accuracy", "roc_auc", "precision_silage", "recall_silage", "precision_grain", "recall_grain", "fpr_silage")}
    # seed ensemble: mean calibrated probability vs mean threshold
    e = g.groupby(["OBJECTID", "year"]).agg(p=("p_silage", "mean"), thr=("threshold", "mean"), y=("y", "first")).reset_index()
    res["ensemble"][key] = d6.metrics(e.y, (e.p >= e.thr).astype(int), e.p)
E.to_csv(OUT / "barc_deep_matched.csv", index=False)
json.dump(res, open(OUT / "barc_deep_eval.json", "w"), indent=1, default=float)
for k, s in res["summary"].items():
    en = res["ensemble"][k]
    print(f"{k:24s} bal {s['balanced_accuracy'][0]:.3f}±{s['balanced_accuracy'][1]:.3f} AUC {s['roc_auc'][0]:.3f}±{s['roc_auc'][1]:.3f} | "
          f"ensemble bal {en['balanced_accuracy']:.3f} AUC {en['roc_auc']:.3f} TP {en['tp']} FN {en['fn']} FP {en['fp']} TN {en['tn']}")
print("BARC_DEEP_DONE")
