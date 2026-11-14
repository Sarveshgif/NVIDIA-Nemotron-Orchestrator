"""Export held-out (TEST region) fields to the analyst console.

Writes data/console/nl2025.json (fields + evidence + evaluation summary), data/console/nl2025_reference.json
(BRP declared labels, used only by the evaluation panel) and data/console/s2/<id>_{rgb,ndvi}.jpg strips.
Predictions shown are from the model trained on TRAIN regions only (test regions never seen in training/tuning).
"""
import json, glob, os, hashlib, numpy as np, pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import shape, mapping
from common import ROOT, YEAR, log

from multiyear import RESULTS as RES
FEAT = ROOT / "features"
OUT = ROOT / "console"; (OUT / "s2").mkdir(parents=True, exist_ok=True)
M = json.load(open(RES / "metrics.json"))
best = M["selected_model"]; cfg = M[best + "_config"]
lo, hi = cfg["review_band"]; thr = cfg["threshold"]
MODEL_TXT = {'rf_s2_summary': 'Random Forest (S2)', 'rf_s2_summary_patch': 'Random Forest (S2 + within-field zones)', 'rf_s2_s1': 'Random Forest (S2 + zones + S1)', 'rf_s1_only': 'Random Forest (S1)'}[best]
fields = pd.read_parquet(FEAT / "fields.parquet")
splits = pd.read_csv(FEAT / "splits.csv")
feats = pd.read_parquet(FEAT / "features.parquet")
labels = pd.read_parquet(FEAT / "labels.parquet")
preds = pd.read_parquet(RES / "predictions_all.parquet")
preds = preds[preds.year == YEAR].drop(columns="year")
dom = json.load(open(FEAT / "s1_dominant_orbit.json"))
df = fields.drop(columns=[c for c in ("region", "split") if c in fields]).merge(splits[["field_id", "region", "split"]], on="field_id").merge(feats, on="field_id").merge(preds, on="field_id")
df = df[df.split == "test"].reset_index(drop=True)
ABBR = lambda b: b.split("-", 1)[1][:3].upper()
df["cid"] = [f"NL-{ABBR(b)}-{i:04d}" for b, i in zip(df.block, df.groupby("block").cumcount() + 1)]
df["acres"] = df.area_ha * 2.4710538
LARGE = float(np.percentile(df.acres, 90))
fmt = lambda d: pd.Timestamp(d).strftime("%b %-d")
CHIP_SCALE, MAX_CHIPS = 3, 10


def rgb_img(x, clear, fmask):
    r = np.clip(np.stack([x[2], x[1], x[0]], -1).astype(np.float32) / 0.18, 0, 1) ** (1 / 1.6)
    img = (r * 255).astype(np.uint8)
    img[~clear] = (img[~clear] * 0.45 + 90).astype(np.uint8)
    return outline(img, fmask)


def ndvi_img(x, clear, fmask):
    nd = (x[4].astype(np.float32) - x[2]) / (x[4].astype(np.float32) + x[2] + 1e-6)
    t = np.clip((nd + 0.1) / 1.0, 0, 1)
    stops = np.array([[0.55, 0.36, 0.22], [0.85, 0.75, 0.45], [0.75, 0.85, 0.35], [0.25, 0.6, 0.2], [0.05, 0.35, 0.12]])
    pos = t * (len(stops) - 1); i = np.clip(pos.astype(int), 0, len(stops) - 2); f = (pos - i)[..., None]
    img = ((stops[i] * (1 - f) + stops[i + 1] * f) * 255).astype(np.uint8)
    img[~clear] = 128
    return outline(img, fmask)


def outline(img, fmask):
    big = np.kron(img, np.ones((CHIP_SCALE, CHIP_SCALE, 1), np.uint8))
    m = np.kron(fmask.astype(np.uint8), np.ones((CHIP_SCALE, CHIP_SCALE), np.uint8)).astype(bool)
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    big[edge] = 255
    return big


out_fields, ref = [], {}
for b, g in df.groupby("block"):
    obs = pd.read_parquet(ROOT / "s2" / f"obs_{b}.parquet")
    obs = obs[obs.ndvi.notna()]
    og = dict(tuple(obs.groupby("field_id")))
    s1 = pd.read_parquet(ROOT / "s1" / f"s1_{b}.parquet")
    s1 = s1[s1.rel_orbit == dom[b]]
    sg = dict(tuple(s1.groupby("field_id")))
    zf = np.load(ROOT / "s2" / f"chips_{b}.npz")
    z = {k: zf[k] for k in ("field_id", "date", "x", "clear", "chip_field_id", "field_mask")}  # decompress once
    cidx = {}
    for k, f in enumerate(z["field_id"]): cidx.setdefault(f, []).append(k)
    fm = {f: z["field_mask"][k] for k, f in enumerate(z["chip_field_id"])}
    for _, r in g.iterrows():
        o = og.get(r.field_id, pd.DataFrame(columns=obs.columns)).sort_values("date")
        rad = sg.get(r.field_id)
        tr = None
        if pd.notna(r.tr_start_doy):
            d0 = pd.Timestamp(f"{YEAR}-01-01")
            tr = [(d0 + pd.Timedelta(days=int(r.tr_start_doy))).strftime("%Y-%m-%d"), (d0 + pd.Timedelta(days=int(r.tr_end_doy))).strftime("%Y-%m-%d"), int(r.tr_gap)]
        dates = o.date.values
        maxgap = 0
        if tr:
            mid = pd.Timestamp(tr[0]) + (pd.Timestamp(tr[1]) - pd.Timestamp(tr[0])) / 2
            dd = pd.to_datetime(pd.Series(dates))
            for k in range(1, len(dd)):
                if dd[k] >= mid - pd.Timedelta(days=20) and dd[k - 1] <= mid + pd.Timedelta(days=20):
                    maxgap = max(maxgap, (dd[k] - dd[k - 1]).days)
        p = float(r.p_silage); conf = max(p, 1 - p)
        reasons = []
        if lo <= p < hi: reasons.append("low_confidence")
        # Review triggers for NL chosen on VALIDATION regions: gap length and zone variation did not predict errors
        # there (Dutch autumns: median gap ~23 d), so only extreme gaps (> 45 d) queue a field; zone variation is shown, not queued.
        if maxgap > 45: reasons.append("observation_gap")
        dvh, dvv = r.get("s1_vh_db_change_at_tr", np.nan), r.get("s1_vv_db_change_at_tr", np.nan)
        if np.isfinite(dvh) and abs(dvh) < 0.5 and np.isfinite(dvv) and abs(dvv) < 0.5: reasons.append("sensor_disagreement")
        if r.acres >= LARGE and conf < 0.9: reasons.append("large_acreage")
        if not reasons and int(hashlib.md5(r.field_id.encode()).hexdigest(), 16) % 20 == 0: reasons.append("audit_sample")
        cls = "silage" if p >= thr else "grain"
        parts = [f"{MODEL_TXT} (trained on other regions): calibrated silage score {p:.2f} → {cls} (threshold {thr:.2f})."]
        if tr:
            parts.append(f"Largest NDVI fall between {fmt(tr[0])} and {fmt(tr[1])} ({tr[2]} d between clear looks), NDVI {r.tr_ndvi_before:.2f} → {r.tr_ndvi_after:.2f}.")
        late = np.nanmean([r.get("ndvi_hm11a", np.nan), r.get("ndvi_hm12a", np.nan)])
        if np.isfinite(late):
            parts.append(f"Early Nov–Dec NDVI {late:.2f} ({'green cover after harvest' if late > 0.55 else 'little green cover after harvest'}).")
        if np.isfinite(dvh): parts.append(f"Radar VH change across the fall: {dvh:+.1f} dB.")
        if maxgap > 15: parts.append(f"Longest clear-sky gap near the change: {maxgap} d.")
        # chips
        ks = cidx.get(r.field_id, [])
        chip_dates = []
        if ks and r.field_id in fm:
            cd = {str(z["date"][k]): k for k in ks}
            all_d = sorted(cd)
            want = set([tr[0], tr[1]] if tr else [])
            pool = [d for d in all_d if d >= f"{YEAR}-05-15" and d not in want]
            pick = [pool[int(i)] for i in np.linspace(0, len(pool) - 1, max(0, MAX_CHIPS - len(want)))] if pool else []
            chip_dates = sorted(set(pick) | (want & set(all_d)))
            rgb = [rgb_img(z["x"][cd[d]], z["clear"][cd[d]], fm[r.field_id]) for d in chip_dates]
            nd = [ndvi_img(z["x"][cd[d]], z["clear"][cd[d]], fm[r.field_id]) for d in chip_dates]
            Image.fromarray(np.concatenate(rgb, 1)).save(OUT / "s2" / f"{r.cid}_rgb.jpg", quality=85)
            Image.fromarray(np.concatenate(nd, 1)).save(OUT / "s2" / f"{r.cid}_ndvi.jpg", quality=85)
        geom = shape(json.loads(r.geom)).simplify(0.000004, preserve_topology=True)
        ring = [[round(y, 6), round(x, 6)] for x, y in geom.exterior.coords[:-1]]
        zones = o[["date", "ndvi_z1", "ndvi_z2", "ndvi_z3", "ndvi_z4"]].values.tolist()
        out_fields.append(dict(
            id=r.cid, brp=r.field_id, ring=ring, ha=round(r.area_ha, 2), ac=round(r.acres, 2), block=b, region=r.region,
            p=round(p, 4), pred=cls, review=bool(lo <= p < hi), reasons=reasons, summary=" ".join(parts), tr=tr,
            maxGap=int(maxgap), nObs=int(len(o)),
            opt=[[d, round(float(a), 3), round(float(c), 3)] for d, a, c in zip(o.date, o.ndvi, o.lswi)],
            rad=[] if rad is None else [[d, round(float(a), 2), round(float(c), 2)] for d, a, c in zip(rad.sort_values("date").date, rad.sort_values("date").vv_db, rad.sort_values("date").vh_db)],
            zones=[[row[0]] + [None if not np.isfinite(v) else round(float(v), 3) for v in row[1:]] for row in zones],
            chips=chip_dates))
    log(f"console: block {b} exported", "console.log")

lab = labels.set_index("field_id").label
ref = {f["id"]: lab[f["brp"]] for f in out_fields}
# showcase (prediction-based only, never reference labels)
byp = sorted(out_fields, key=lambda f: f["p"])
def tag(f, t):
    if f and "showcase" not in f: f["showcase"] = t
tag(next((f for f in reversed(byp) if f["chips"] and f["tr"] and not f["reasons"]), None), "Confident silage")
tag(next((f for f in byp if f["chips"] and f["tr"]), None), "Strongest grain call")
tag(min((f for f in out_fields if f["review"]), key=lambda f: -f["ac"], default=None), "Uncertain: in review band")
tag(next((f for f in out_fields if "observation_gap" in f["reasons"]), None), "Long cloud gap at harvest")
tag(next((f for f in out_fields if "sensor_disagreement" in f["reasons"]), None), "Radar does not confirm harvest")
tag(next((f for f in sorted(out_fields, key=lambda f: -f["ac"]) if "large_acreage" in f["reasons"]), None), "Large uncertain field")

rows = []
LBL = {"majority_silage": "Majority class (always silage)", "rules_original_weak": "Rules v1 (Tulare), unchanged",
       "rules_adapted_weak": "Rules adapted (tuned on train/val)", "rf_s2_summary": "RF · S2 field summaries",
       "rf_s2_summary_patch": "RF · S2 + within-field zones", "rf_s1_only": "RF · S1 radar only", "rf_s2_s1": "RF · S2 + zones + S1",
       "rf_asof_oct31": "RF · S2+S1, data to Oct 31 only"}
for k, l in LBL.items():
    if k in M:
        m = M[k]; rows.append(dict(model=k, label=l, **{c: m.get(c) for c in ["n", "coverage", "accuracy", "balanced_accuracy", "precision_silage", "recall_silage", "fpr_silage", "precision_grain", "recall_grain", "f1_grain", "roc_auc", "pr_auc_grain", "unresolved_area_frac"]}))
deep = {}
for p in sorted(glob.glob(str(RES / "deep_*_seed*.json"))):
    d = json.load(open(p)); deep.setdefault(d["variant"], []).append(d)
deep_summary = {v: dict(n_seeds=len(ds), balanced_accuracy=[x["test"]["balanced_accuracy"] for x in ds], roc_auc=[x["test"]["roc_auc"] for x in ds],
                        gap_balanced_accuracy=[x["simulated_gap_sep1_oct15"]["balanced_accuracy"] for x in ds], params=ds[0]["params"],
                        train_seconds=[x["train_seconds"] for x in ds], hardware=ds[0]["hardware"]) for v, ds in deep.items()}
for v, s in deep_summary.items():
    rows.append(dict(model=f"deep_{v}", label={"s2": "CNN→Transformer · S2", "s2s1": "CNN→Transformer · S2+S1", "s2s1_nocnn": "Transformer (field means) · S2+S1"}.get(v, v) + f" (mean of {s['n_seeds']} seeds)",
                     n=deep[v][0]["n_test"], coverage=1.0, balanced_accuracy=float(np.mean(s["balanced_accuracy"])), roc_auc=float(np.mean(s["roc_auc"])),
                     **{c: float(np.mean([x["test"][c] for x in deep[v]])) for c in ["accuracy", "precision_silage", "recall_silage", "fpr_silage", "precision_grain", "recall_grain", "f1_grain", "pr_auc_grain"]}))
te = M[best]
EXP = M.get("experiment", {})
YEARS_TABLE = []
for d in sorted(RES.parent.glob("results_e*")):
    mp = d / "metrics.json"
    if not mp.exists(): continue
    mm = json.load(open(mp)); ex = mm.get("experiment", {})
    for k in ("rf_s2_summary_patch", "rf_s2_s1"):
        if k in mm:
            YEARS_TABLE.append(dict(experiment=d.name, train_years="+".join(map(str, ex.get("train_years", []))), test_year=ex.get("test_year"),
                                    n_train=ex.get("n_train"), grain_train=ex.get("grain_train"), model=k, n_test=mm[k]["n"],
                                    balanced_accuracy=mm[k]["balanced_accuracy"], roc_auc=mm[k].get("roc_auc"), recall_grain=mm[k]["recall_grain"],
                                    precision_grain=mm[k]["precision_grain"], gap_balanced_accuracy=mm.get("simulated_gap", {}).get(k, {}).get("balanced_accuracy")))
meta = dict(
    title="South-east Netherlands (held-out test regions)", year=YEAR,
    season=dict(year=YEAR, start=f"{YEAR}-03-01", end=f"{YEAR}-12-31", label=f"Mar – Dec {YEAR}", largeAcres=round(LARGE, 1),
                peakFrom="06-01", peakTo="09-30", dropTo="11-15"),
    provenance=dict(outlines=f"PDOK BRP {YEAR} gewaspercelen (RVO parcel declarations)",
                    observations=f"Copernicus Sentinel-2 L2A (Element84 Earth Search) / Sentinel-1 RTC (Planetary Computer), {YEAR}",
                    attribution=f"Contains modified Copernicus Sentinel data {YEAR}; BRP © RVO/PDOK",
                    scope=f"{len(out_fields)} silage/grain maize parcels in 3 held-out test regions of south-east Netherlands (Limburg south & north, Gelderland east). Not national coverage; regions were chosen where grain maize occurs.",
                    labels="Farmer-declared crop use (BRP codes 259 snijmais / 316 korrelmais), kept separate; used only for evaluation."),
    model=dict(name=best, version=f"{best}-{YEAR}.1", threshold=thr, review_band=[lo, hi], features=cfg["n_features"],
               note=f"Random Forest trained on {EXP.get('n_train', int((splits.split == 'train').sum()))} field-seasons ({'+'.join(map(str, EXP.get('train_years', [YEAR])))}) in other regions; tuned and calibrated on validation regions; these test regions were never used for training or tuning."),
    evaluation=dict(split=f"Fixed spatial split by region (same every year): train = Limburg-mid, Peel, Kempen; val = NE Brabant, Echt; test = Limburg-south, Limburg-north, Gelderland-east. Training years: {'+'.join(map(str, EXP.get('train_years', [YEAR])))}; test year: {YEAR}",
                    years=YEARS_TABLE,
                    positive="silage", test_fields=te["n"], test_silage=te["n_silage"], test_grain=te["n_grain"],
                    comparison=rows,
                    confusion={k: {c: M[k][c] for c in ("tp", "fn", "fp", "tn")} for k in ["rules_original_weak", "rules_adapted_weak", "rf_s2_summary_patch", "rf_s2_s1", best] if k in M},
                    ci={k[:-3]: v for k, v in M.items() if k.endswith("_ci")},
                    latency=dict(rf_per_field_ms=cfg["per_field_inference_ms"], rf_batch_seconds=cfg["batch_inference_seconds"], rf_batch_fields=cfg["n_test"],
                                 note="Inference on the lab workstation CPU (64 cores) from cached features; excludes Sentinel retrieval."),
                    findings=[], deep=deep_summary, gap=M.get("simulated_gap"), with_review=M.get(best + "_with_review")),
    chipPx=24 * CHIP_SCALE,
)
def clean(o):
    """Strict JSON: NaN/inf -> null (browsers reject NaN)."""
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)): return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
json.dump(clean(dict(meta=meta, fields=out_fields)), open(OUT / "nl2025.json", "w"), separators=(",", ":"), allow_nan=False)
json.dump(dict(source="PDOK BRP 2025 definitief, farmer declarations; evaluation only", labels=ref), open(OUT / "nl2025_reference.json", "w"))
log(f"console: {len(out_fields)} fields, {sum(1 for f in out_fields if f['chips'])} with chips, large≥{LARGE:.1f} ac", "console.log")
