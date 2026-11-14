"""Per-field features from the 20 m extraction, with the spectral indices as separable groups.

Temporal features are built exactly as in 04_build_features.s2_features (half-month medians of observed looks,
Apr-Dec, NaN when a half-month has no clear look; transition = largest NDVI fall after the summer peak).
Each added index X in {ndre, psri, ndti} contributes the same template:
  X_hm<MM><a|b>   18 half-month medians (Apr-Dec)
  tr_X_before / tr_X_after   value at the last clear look before / first after the NDVI transition
The baseline (NDVI + LSWI + B04/B11/B12, transition, zones) contains no NDRE, PSRI or NDTI feature.
Same field set and split as the 10 m pipeline (features/splits.csv). S1 features are recomputed with the 20 m
transition date so that every configuration shares one consistent table.
Outputs: <HTH_DATA>/features/features20.parquet, feature_schema20.json
"""
import json, importlib.util, numpy as np, pandas as pd
from common import ROOT, YEAR, log

spec = importlib.util.spec_from_file_location("bf", __file__.replace("09_index_features.py", "04_build_features.py"))
bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)
bf.YEAR = YEAR
INDICES = ("ndre", "psri", "ndti")
FEAT = ROOT / "features"
spl = pd.read_csv(FEAT / "splits.csv")
dom = json.load(open(FEAT / "s1_dominant_orbit.json"))

rows = []
for b, ids in spl.groupby("block").field_id:
    obs = pd.read_parquet(ROOT / "s2_20m" / f"obs20_{b}.parquet")
    p = ROOT / "s1" / f"s1_{b}.parquet"
    s1g = {}
    if p.exists():
        s1 = pd.read_parquet(p); s1 = s1[s1.rel_orbit == dom[b]]; s1g = dict(tuple(s1.groupby("field_id")))
    og = dict(tuple(obs.groupby("field_id")))
    for fid in ids:
        g = og.get(fid)
        F = {"field_id": fid}
        if g is None or g.ndvi.notna().sum() == 0:
            rows.append(F); continue
        F.update(bf.s2_features(g))                       # baseline template; includes ndre_hm*, tr_ndre_before
        o = g[g.ndvi.notna()].sort_values("date").reset_index(drop=True)
        keys = o.date.map(bf.hm_key)
        for x in ("psri", "ndti"):                       # same half-month template for the new indices
            med = o.groupby(keys)[x].median()
            for m, h in bf.HALF_MONTHS:
                F[f"{x}_hm{m:02d}{'ab'[h]}"] = med.get((m, h), np.nan)
        if np.isfinite(F.get("tr_start_doy", np.nan)):
            doy = o.date.map(bf.D).values
            bi = int(np.flatnonzero(doy == F["tr_start_doy"])[0]); ai = int(np.flatnonzero(doy == F["tr_end_doy"])[0])
            for x in INDICES:
                F[f"tr_{x}_before"] = o[x].values[bi]; F[f"tr_{x}_after"] = o[x].values[ai]
        F.update(bf.s1_features(s1g.get(fid), F.get("tr_mid_doy", np.nan)))
        rows.append(F)
    log(f"features20: block {b} done", "s2_20m.log")

feats = spl[["field_id"]].merge(pd.DataFrame(rows), on="field_id", how="left")
feats.to_parquet(FEAT / "features20.parquet")
ARTIFACT = ("n_acq", "n_obs", "mean_clear_frac", "max_gap_", "n_obs_", "s1_n")
is_s1 = lambda c: c.startswith(("s1_", "vv_", "vh_", "cr_"))
idx_of = lambda c: next((x for x in INDICES if c.startswith(f"{x}_hm") or c in (f"tr_{x}_before", f"tr_{x}_after")), None)
cols = [c for c in feats.columns if c != "field_id"]
groups = {
    "base": [c for c in cols if not is_s1(c) and not c.startswith(("zone_", "ndvi_sd")) and not c.startswith(ARTIFACT) and idx_of(c) is None],
    "patch": [c for c in cols if c.startswith(("zone_", "ndvi_sd"))],
    "s1": [c for c in cols if is_s1(c) and not c.startswith(ARTIFACT)],
    **{x: [c for c in cols if idx_of(c) == x] for x in INDICES},
    "coverage_only": [c for c in cols if c.startswith(ARTIFACT)],
}
json.dump({"grid": "20 m", "groups": groups}, open(FEAT / "feature_schema20.json", "w"), indent=1)
miss = {x: float(feats[groups[x]].isna().mean().mean()) for x in INDICES}
log(f"features20: {feats.shape}, groups {({k: len(v) for k, v in groups.items()})}, mean missing share {miss}, "
    f"fields without any 20 m look {int(feats.tr_mid_doy.isna().sum())} (no transition)", "s2_20m.log")
