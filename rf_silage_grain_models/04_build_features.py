"""Build per-field tables: labels (kept separate), spatial split, rule features, RF features.

Outputs in data/features/:
  labels.parquet    field_id, label, code, crop_name, source (BRP 2025 farmer declaration)
  fields.parquet    eligible fields: geometry, area, block, region, split (no label column)
  splits.csv        field_id, block, region, split
  features.parquet  one row per field, NO label/ID-derived predictors
  feature_schema.json
  s1_dominant_orbit.json
Transitions are estimated from observations only (labels are never read here except to write labels.parquet).
"""
import json, numpy as np, pandas as pd, glob, os
from common import ROOT, YEAR, log

FEAT = ROOT / "features"; FEAT.mkdir(exist_ok=True)
D = lambda s: (pd.Timestamp(s) - pd.Timestamp(f"{YEAR}-01-01")).days  # day of year (0-based)

# Spatial split: whole regions (groups of adjacent blocks). Test regions are separated from training blocks.
REGIONS = {
    "LI-south": ["LI-sittard", "LI-maastricht", "LI-heerlen"],
    "LI-mid": ["LI-roermond", "LI-weert"],
    "LI-north": ["LI-venray", "LI-horst", "LI-gennep"],
    "NB-peel": ["NB-asten", "NB-deurne", "NB-peel", "NB-eindhoven"],
    "NB-northeast": ["NB-cuijk", "NB-uden", "NB-oss"],
    "LI-echt": ["LI-echt"],
    "NB-kempen": ["NB-kempen"],
    "GE-east": ["GE-achterhoek", "GE-doetinchem", "GE-winterswijk"],
}
SPLIT = {"LI-mid": "train", "NB-peel": "train", "NB-kempen": "train",
         "NB-northeast": "val", "LI-echt": "val",
         "LI-south": "test", "LI-north": "test", "GE-east": "test"}
BLOCK_REGION = {b: r for r, bs in REGIONS.items() for b in bs}

# ---- faithful port of src/lib/rules.ts (rules-v1) ----
ORIG_TH = dict(silageMinNdvi=0.6, silageMaxDryDown=14, grainMinDryDown=25, grainMaxNdvi=0.55,
               minClearObs=14, maxGapAtDrop=15, minRadarDrop=1.5)
ORIG_CAL = dict(peak_from=f"{YEAR}-06-01", peak_to=f"{YEAR}-09-30", drop_to=f"{YEAR}-11-15")
DECLINE_SHARE = 1 / 3


def rule_features(opt, rad, cal=ORIG_CAL):
    """opt: DataFrame(date, ndvi, lswi) of clear looks; rad: DataFrame(date, vh_db) single orbit."""
    opt = opt.sort_values("date").reset_index(drop=True)
    f = dict(clearObs=len(opt), peakNdvi=np.nan, dropStart=None, dropEnd=None, gapAtDrop=np.nan, ndviBeforeDrop=np.nan,
             lswiBeforeDrop=np.nan, dryDownDays=np.nan, radarDrop=np.nan, ndviAfterDrop=np.nan, dropFall=np.nan)
    if len(opt) < 3: return f
    dates, nd = opt.date.values, opt.ndvi.values
    pk = -1
    for i in range(len(opt)):
        if cal["peak_from"] <= dates[i] <= cal["peak_to"] and (pk < 0 or nd[i] > nd[pk]): pk = i
    if pk < 0: return f
    f["peakNdvi"] = round(nd[pk], 2)
    best, bestFall = -1, 0.1
    i = pk + 1
    while i < len(opt) and dates[i] <= cal["drop_to"]:
        fall = nd[i - 1] - nd[i]
        if fall > bestFall: bestFall, best = fall, i
        i += 1
    if best < 0: return f
    b, a = best - 1, best
    f.update(dropStart=dates[b], dropEnd=dates[a], gapAtDrop=D(dates[a]) - D(dates[b]), ndviBeforeDrop=round(nd[b], 2),
             lswiBeforeDrop=round(opt.lswi.values[b], 2), ndviAfterDrop=nd[a], dropFall=nd[b] - nd[a])
    line = nd[pk] - (nd[pk] - nd[a]) * DECLINE_SHARE
    k = best - 1
    while k - 1 > pk and nd[k - 1] < line: k -= 1
    f["dryDownDays"] = 0 if nd[b] >= line else D(dates[b]) - D(dates[k])
    if rad is not None and len(rad):
        rd = rad.sort_values("date")
        s0, s1 = D(dates[b]), D(dates[a])
        dd = rd.date.map(D).values
        pre = rd[(rd.date.values <= dates[b]) & (s0 - dd <= 30)].tail(2)
        post = rd[(rd.date.values >= dates[a]) & (dd - s1 <= 30)].head(2)
        if len(pre) and len(post):
            f["radarDrop"] = round(pre.vh_db.mean() - post.vh_db.mean(), 1)
    return f


def apply_rules(f, th, radar_veto=True):
    """Returns (weak_label, abstain_reason, predicted_with_lean, score_silage)."""
    if f["clearObs"] < th["minClearObs"]: return "abstain", "few_obs", "unclassified", np.nan
    if np.isnan(f["ndviBeforeDrop"]) or np.isnan(f["dryDownDays"]): return "abstain", "no_drop", "unclassified", np.nan
    if f["gapAtDrop"] > th["maxGapAtDrop"]: return "abstain", "gap", "unclassified", np.nan
    if radar_veto and not np.isnan(f["radarDrop"]) and f["radarDrop"] < th["minRadarDrop"]:
        return "abstain", "sensor_conflict", "unclassified", np.nan
    nd, dd = f["ndviBeforeDrop"], f["dryDownDays"]
    clamp = lambda v: max(0, min(1, v))
    if nd >= th["silageMinNdvi"] and dd <= th["silageMaxDryDown"]:
        m = min(clamp((nd - th["silageMinNdvi"]) / 0.2), clamp((th["silageMaxDryDown"] - dd) / max(1, th["silageMaxDryDown"])))
        return "silage", None, "silage", round(0.7 + 0.27 * m, 2)
    if dd >= th["grainMinDryDown"] and nd <= th["grainMaxNdvi"]:
        m = min(clamp((dd - th["grainMinDryDown"]) / 15), clamp((th["grainMaxNdvi"] - nd) / 0.15))
        return "grain", None, "grain", round(1 - (0.7 + 0.27 * m), 2)
    mid = (th["silageMaxDryDown"] + th["grainMinDryDown"]) / 2
    cls = "silage" if dd < mid else "grain"
    return "abstain", "in_between", cls, 0.55 if cls == "silage" else 0.45


# ---- RF features ----
HALF_MONTHS = [(m, h) for m in range(4, 13) for h in (0, 1)]


def hm_key(date):
    t = pd.Timestamp(date); return (t.month, 0 if t.day <= 15 else 1)


def s2_features(opt_all):
    """opt_all: all acquisitions of one field (clear_frac, ndvi... NaN when not observed)."""
    o = opt_all.sort_values("date")
    obs = o[o.ndvi.notna()].reset_index(drop=True)
    F = {}
    F["n_acq"] = len(o); F["n_obs"] = len(obs)
    F["mean_clear_frac"] = o.clear_frac.mean()
    doy = obs.date.map(D).values
    for lo, hi, name in [(D(f"{YEAR}-07-01"), D(f"{YEAR}-11-30"), "jul_nov"), (D(f"{YEAR}-08-15"), D(f"{YEAR}-10-31"), "aug15_oct")]:
        sel = np.r_[lo, doy[(doy >= lo) & (doy <= hi)], hi]
        F[f"max_gap_{name}"] = np.diff(np.sort(sel)).max()
        F[f"n_obs_{name}"] = int(((doy >= lo) & (doy <= hi)).sum())
    # half-month medians of observed values only (NaN = no clear look in that half-month; never interpolated)
    keys = obs.date.map(hm_key)
    for var in ("ndvi", "lswi", "ndre"):
        med = obs.groupby(keys)[var].median()
        for m, h in HALF_MONTHS:
            F[f"{var}_hm{m:02d}{'ab'[h]}"] = med.get((m, h), np.nan)
    # VIS / SWIR at half-month resolution for Aug-Nov (senescence / residue brightness)
    for var in ("B04", "B11", "B12"):
        med = obs.groupby(keys)[var].median()
        for m, h in [(m, h) for m in range(8, 12) for h in (0, 1)]:
            F[f"{var}_hm{m:02d}{'ab'[h]}"] = med.get((m, h), np.nan)
    if len(obs) < 3:
        return F
    nd = obs.ndvi.values
    summer = (doy >= D(f"{YEAR}-06-01")) & (doy <= D(f"{YEAR}-09-30"))
    if not summer.any():
        return F
    pk = int(np.flatnonzero(summer)[np.argmax(nd[summer])])
    F["peak_ndvi"] = nd[pk]; F["peak_doy"] = doy[pk]
    gu = np.flatnonzero((doy >= D(f"{YEAR}-04-15")) & (nd >= 0.5) & (np.arange(len(nd)) <= pk))
    F["greenup_doy"] = doy[gu[0]] if len(gu) else np.nan
    # transition = largest fall between consecutive clear looks after peak, searched to Dec 31 (no label used)
    falls = nd[pk:-1] - nd[pk + 1:]
    if len(falls):
        j = int(np.argmax(falls)); b, a = pk + j, pk + j + 1
        F.update(tr_start_doy=doy[b], tr_end_doy=doy[a], tr_gap=doy[a] - doy[b], tr_fall=nd[b] - nd[a],
                 tr_rate=(nd[b] - nd[a]) / max(1, doy[a] - doy[b]), tr_ndvi_before=nd[b], tr_ndvi_after=nd[a],
                 tr_lswi_before=obs.lswi.values[b], tr_ndre_before=obs.ndre.values[b], tr_lswi_after=obs.lswi.values[a],
                 tr_mid_doy=(doy[a] + doy[b]) / 2, cycle_len=(doy[b] - F["greenup_doy"]) if len(gu) else np.nan)
        # gradual decline: NDVI loss between peak and the last look before the transition, per day
        F["predrop_decline"] = nd[pk] - nd[b]
        F["predrop_decline_rate"] = (nd[pk] - nd[b]) / max(1, doy[b] - doy[pk])
        # 30-day pre-transition slope from looks in [b-30d, b]
        w = (doy >= doy[b] - 30) & (doy <= doy[b])
        F["slope30_pre"] = np.polyfit(doy[w], nd[w], 1)[0] if w.sum() >= 2 else np.nan
        # post-transition persistence/regrowth (catch crop) 20-60 d after
        p = (doy >= doy[a] + 20) & (doy <= doy[a] + 60)
        F["post_ndvi_20_60"] = np.median(nd[p]) if p.any() else np.nan
        F["post_min_ndvi_30"] = nd[(doy >= doy[a]) & (doy <= doy[a] + 30)].min()
        # within-field (2x2 zone) agreement around the transition
        z = obs[[f"ndvi_z{i}" for i in range(1, 5)]].values
        F["zone_sd_before"] = np.nanstd(z[b]) if np.isfinite(z[b]).sum() >= 2 else np.nan
        F["zone_sd_after"] = np.nanstd(z[a]) if np.isfinite(z[a]).sum() >= 2 else np.nan
        zd = []
        for zi in range(4):
            v = z[pk:, zi]; ok = np.isfinite(v)
            if ok.sum() >= 3:
                vv, dz = v[ok], doy[pk:][ok]
                fz = vv[:-1] - vv[1:]
                if fz.max() > 0.15: jj = int(np.argmax(fz)); zd.append((dz[jj] + dz[jj + 1]) / 2)
        F["zone_tr_spread"] = (max(zd) - min(zd)) if len(zd) >= 2 else np.nan
        F["ndvi_sd_before"] = obs.ndvi_sd.values[b]; F["ndvi_sd_peak"] = obs.ndvi_sd.values[pk]
    # senescence: first clear look after peak with NDVI < 0.6 / 0.4
    for thr in (0.6, 0.4):
        s = np.flatnonzero((np.arange(len(nd)) > pk) & (nd < thr))
        F[f"first_below_{int(thr*10)}_doy"] = doy[s[0]] if len(s) else np.nan
    return F


def s1_features(rad, tr_mid):
    """rad: one field, dominant orbit, sorted. Features in dB; changes are dB differences (= linear ratios)."""
    F = {}
    if rad is None or len(rad) < 5: return F
    rad = rad.sort_values("date")
    doy = rad.date.map(D).values
    F["s1_n"] = len(rad)
    keys = rad.date.map(lambda s: pd.Timestamp(s).month)
    for var in ("vv_db", "vh_db", "cr_db"):
        med = rad.groupby(keys)[var].median()
        for m in range(5, 13):
            F[f"{var}_m{m:02d}"] = med.get(m, np.nan)
    # largest drop in 3-pass running mean of VH between Aug 1 and Dec 15
    vh = rad.vh_db.rolling(3, center=True, min_periods=2).mean().values
    sel = np.flatnonzero((doy >= D(f"{YEAR}-08-01")) & (doy <= D(f"{YEAR}-12-15")))
    if len(sel) > 3:
        d = vh[sel[:-1]] - vh[sel[1:]]
        j = int(np.argmax(d)); F["s1_vh_maxdrop"] = d[j]; F["s1_vh_maxdrop_doy"] = (doy[sel[j]] + doy[sel[j + 1]]) / 2
    if np.isfinite(tr_mid):
        pre = rad[(doy < tr_mid) & (doy >= tr_mid - 30)]; post = rad[(doy > tr_mid) & (doy <= tr_mid + 30)]
        if len(pre) and len(post):
            for var in ("vv_db", "vh_db", "cr_db"):
                F[f"s1_{var}_change_at_tr"] = post[var].mean() - pre[var].mean()
        F["s1_vh_pre60_slope"] = np.nan
        w = (doy >= tr_mid - 60) & (doy < tr_mid)
        if w.sum() >= 3: F["s1_vh_pre60_slope"] = np.polyfit(doy[w], rad.vh_db.values[w], 1)[0]
    return F


if __name__ == "__main__":
    blocks = sorted(os.path.basename(p)[4:-8] for p in glob.glob(str(ROOT / "s2" / "obs_*.parquet")))
    fields = pd.concat([pd.read_parquet(ROOT / "s2" / f"fields_{b}.parquet") for b in blocks], ignore_index=True)
    fields["region"] = fields.block.map(BLOCK_REGION)
    fields["split"] = fields.region.map(SPLIT)
    excl = fields[~fields.eligible | fields.split.isna()]
    fields_all = fields.copy()
    fields = fields[fields.eligible & fields.split.notna()].drop_duplicates("field_id").reset_index(drop=True)
    fields_all.drop(columns=["geom"]).to_parquet(FEAT / "fields_all_candidates.parquet")
    labels = fields[["field_id", "label", "code", "crop_name"]].assign(source=f"PDOK BRP {YEAR} definitief (farmer declaration)")
    labels.to_parquet(FEAT / "labels.parquet")
    fields[["field_id", "block", "region", "split"]].to_csv(FEAT / "splits.csv", index=False)
    fields.drop(columns=["label", "code", "crop_name"]).to_parquet(FEAT / "fields.parquet")
    log(f"features: {len(fields)} eligible fields in {len(blocks)} blocks; excluded {len(excl)}")

    rows, rule_rows, dom = [], [], {}
    for b in blocks:
        obs = pd.read_parquet(ROOT / "s2" / f"obs_{b}.parquet")
        s1p = ROOT / "s1" / f"s1_{b}.parquet"
        s1 = pd.read_parquet(s1p) if s1p.exists() else None
        if s1 is not None:
            o = int(s1.groupby("rel_orbit").date.nunique().idxmax()); dom[b] = o
            s1 = s1[s1.rel_orbit == o]
            s1g = dict(tuple(s1.groupby("field_id")))
        else:
            s1g = {}
        keep = set(fields.field_id[fields.block == b])
        for fid, g in obs.groupby("field_id"):
            if fid not in keep: continue
            rad = s1g.get(fid)
            opt = g[g.ndvi.notna()][["date", "ndvi", "lswi"]]
            rf = rule_features(opt, rad)
            wl, reason, pred, score = apply_rules(rf, ORIG_TH)
            rule_rows.append(dict(field_id=fid, **{k: v for k, v in rf.items()}, weak_label=wl, abstain_reason=reason,
                                  rule_pred=pred, rule_score_silage=score, has_s1=rad is not None))
            F = {"field_id": fid}
            F.update(s2_features(g))
            F.update(s1_features(rad, F.get("tr_mid_doy", np.nan)))
            rows.append(F)
        log(f"features: block {b} done")
    feats = pd.DataFrame(rows)
    feats = fields[["field_id"]].merge(feats, on="field_id", how="left")
    feats.to_parquet(FEAT / "features.parquet")
    pd.DataFrame(rule_rows).to_parquet(FEAT / "rules_original_features.parquet")
    json.dump(dom, open(FEAT / "s1_dominant_orbit.json", "w"), indent=1)
    # Acquisition counts / clear fractions / gap lengths depend on tile overlap and weather, not on the crop:
    # kept for coverage reporting and review routing, excluded as predictors (source artifacts).
    ARTIFACT = ("n_acq", "n_obs", "mean_clear_frac", "max_gap_", "n_obs_", "s1_n")
    groups = {
        "s2_summary": [c for c in feats.columns if c != "field_id" and not c.startswith(("s1_", "vv_", "vh_", "cr_")) and not c.startswith(("zone_", "ndvi_sd")) and not c.startswith(ARTIFACT)],
        "s2_patch": [c for c in feats.columns if c.startswith(("zone_", "ndvi_sd"))],
        "s1": [c for c in feats.columns if c.startswith(("s1_", "vv_", "vh_", "cr_")) and not c.startswith(ARTIFACT)],
        "coverage_only": [c for c in feats.columns if c.startswith(ARTIFACT)],
    }
    json.dump({"rows": "one per eligible field (field-season 2025)", "groups": groups,
               "notes": "NaN = not observed (no interpolation). doy = day of year from Jan 1 (0-based). dB features from linear-mean field backscatter."},
              open(FEAT / "feature_schema.json", "w"), indent=1)
    log(f"features: wrote {feats.shape}, groups " + str({k: len(v) for k, v in groups.items()}))
