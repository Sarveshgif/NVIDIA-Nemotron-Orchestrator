"""US state test: CDL corn fields -> unchanged console RF (Dutch-trained, E2) -> state silage share vs NASS Quick Stats.

  prepare <ST> <FIPS> : CDL 2025 (CropScape) corn (code 1 only; sweet corn 12 and popcorn 13 excluded) ->
                        8-connected components >= MIN_PX pixels (30 m) -> polygons (EPSG:4326) ->
                        <HTH_US>/<st>/2025/blocks/BARC.json (label-free, same layout 11_barc_extract.py reads)
  infer <ST> <NASS_STATE_NAME> <nass.csv> :
                        features from the extraction, RF (re-fitted exactly as 12_barc_infer.py, checked against the
                        saved console predictions), threshold / calibration from NL; state acres vs NASS 2025 survey.
Extraction in between: HTH_BARC=<HTH_US>/<st> HTH_FOOTPRINT=union python 11_barc_extract.py 2025
This is an AGGREGATE comparison (state shares), not field-level validation: no field labels exist here.
"""
import os, sys, json, csv, pathlib, numpy as np
HERE = pathlib.Path(__file__).resolve().parent
US = pathlib.Path(os.environ.get("HTH_US", HERE.parent / "data_us"))
MIN_PX = 12          # >= 1.08 ha of CDL corn pixels
N_SAMPLE = int(os.environ.get("US_SAMPLE", 0)) or None   # e.g. 300: random sample of fields
SEED = int(os.environ.get("US_SEED", 2025))
YEAR = 2025


def prepare(st, fips):
    import rasterio, urllib.request, re
    from rasterio.features import shapes
    from scipy import ndimage
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union, transform as stransform
    from pyproj import Transformer
    root = US / st; root.mkdir(parents=True, exist_ok=True)
    tif = root / f"CDL_{YEAR}_{fips}.tif"
    if not tif.exists():
        x = urllib.request.urlopen(f"https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?year={YEAR}&fips={fips}", timeout=60).read().decode()
        urllib.request.urlretrieve(re.search(r"<returnURL>(.*?)</returnURL>", x).group(1), tif)
    with rasterio.open(tif) as s:
        a = s.read(1); tr = s.transform; crs = s.crs
    corn = a == 1
    lab, n = ndimage.label(corn, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    keep = np.flatnonzero(sizes >= MIN_PX); keep = keep[keep > 0]
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    polys = {}
    m = np.isin(lab, keep)
    for geom, v in shapes(lab.astype("int32"), mask=m, transform=tr, connectivity=8):
        polys.setdefault(int(v), []).append(shape(geom))
    feats = []
    for k, (v, ps) in enumerate(sorted(polys.items()), start=1):
        g = stransform(to_ll, unary_union(ps))
        feats.append(dict(type="Feature", id=f"CDL{k:05d}", geometry=mapping(g), properties=dict(cdl_px=int(sizes[v]), cdl_ha=round(sizes[v] * 0.09, 3))))
    from shapely.geometry import shape as sh
    pop_fields, pop_acres = len(feats), round(sum(f["properties"]["cdl_ha"] for f in feats) / 0.404686)
    if N_SAMPLE:
        import random
        feats = random.Random(SEED).sample(feats, min(N_SAMPLE, len(feats)))   # simple random sample of fields
    b = [sh(f["geometry"]).bounds for f in feats]
    bbox = [min(x[0] for x in b) - 0.01, min(x[1] for x in b) - 0.01, max(x[2] for x in b) + 0.01, max(x[3] for x in b) + 0.01]
    d = root / str(YEAR) / "blocks"; d.mkdir(parents=True, exist_ok=True)
    # small blocks: one per ~0.1 deg (~10 km) grid cell that contains corn fields (by centroid), so only the area around
    # the fields is read; each block picks its own best-covering Sentinel-2 tile; blocks can be extracted in parallel
    CELL = 0.1
    cells = {}
    for f in feats:
        c = sh(f["geometry"]).centroid
        cells.setdefault((int(c.x // CELL), int(c.y // CELL)), []).append(f)
    names = []
    for k, ((i, j), fq) in enumerate(sorted(cells.items()), start=1):
        q = f"C{k:03d}"
        bq = [sh(f["geometry"]).bounds for f in fq]
        bb = [min(x[0] for x in bq) - 0.003, min(x[1] for x in bq) - 0.003, max(x[2] for x in bq) + 0.003, max(x[3] for x in bq) + 0.003]
        json.dump(dict(id=q, bbox=bb, crs="EPSG:4326", source=f"USDA NASS CDL {YEAR} FIPS {fips}, code 1 (corn), components >= {MIN_PX} px", features=fq), open(d / f"{q}.json", "w"))
        names.append(q)
    tot_px = int(corn.sum()); kept_px = int(sizes[keep].sum())
    info = dict(state=st, fips=fips, cdl_corn_acres=round(tot_px * 0.09 / 0.404686), fields=len(feats), kept_acres=round(kept_px * 0.09 / 0.404686),
                components_total=int(n), min_px=MIN_PX, blocks=names, population_fields=pop_fields, population_acres_ge_min=pop_acres, sample_n=N_SAMPLE, sample_seed=SEED if N_SAMPLE else None, sweet_corn_acres_excluded=round(int((a == 12).sum()) * 0.09 / 0.404686), bbox=bbox)
    json.dump(info, open(root / "prepare.json", "w"), indent=1); print(info)


def infer(st, nass_state, nass_csv):
    import importlib.util, pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from multiyear import load_tables
    spec = importlib.util.spec_from_file_location("bf", HERE / "04_build_features.py")
    bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)
    E2 = pathlib.Path(os.environ["E2_RESULTS"]); M = json.load(open(E2 / "metrics.json")); cfg = M["rf_s2_s1_config"]
    df, schema = load_tables(); df = df[df.role != "unused"].reset_index(drop=True)
    cols = schema["s2_summary"] + schema["s2_patch"] + schema["s1"]
    TR, VA, TE = (df.role == s for s in ("train", "val", "test"))
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                          RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=20251114, **cfg["params"])).fit(df.loc[TR, cols], df.y[TR])
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    cal = LogisticRegression(C=1e6).fit(lg(model.predict_proba(df.loc[VA, cols])[:, 1])[:, None], df.y[VA])
    calib = lambda p: cal.predict_proba(lg(p)[:, None])[:, 1]
    saved = pd.read_parquet(E2 / "predictions_all.parquet")
    chk = df.loc[TE, ["field_id", "year"]].assign(p=calib(model.predict_proba(df.loc[TE, cols])[:, 1])).merge(saved, on=["field_id", "year"])
    assert (chk.p - chk.p_silage).abs().max() < 1e-6, "re-fit does not reproduce the console model"
    thr = cfg["threshold"]
    R = US / st / str(YEAR); bf.YEAR = YEAR
    blocks = json.load(open(US / st / "prepare.json"))["blocks"]
    fl = pd.concat([pd.read_parquet(R / "s2" / f"fields_{b}.parquet") for b in blocks if (R / "s2" / f"fields_{b}.parquet").exists()], ignore_index=True)
    ob = [pd.read_parquet(R / "s2" / f"obs_{b}.parquet") for b in blocks if (R / "s2" / f"obs_{b}.parquet").exists()]
    obs = pd.concat([o for o in ob if "ndvi" in o.columns], ignore_index=True)   # blocks with no eligible field have no observations
    og = dict(tuple(obs.groupby("field_id"))); s1g = {}
    for b in blocks:                                   # one relative orbit per block, chosen by coverage (field-date rows)
        p = R / "s1" / f"s1_{b}.parquet"
        if p.exists():
            s1 = pd.read_parquet(p); orbit = int(s1.groupby("rel_orbit").size().idxmax()); s1g.update(dict(tuple(s1[s1.rel_orbit == orbit].groupby("field_id"))))
    cdl = {f["id"]: f["properties"]["cdl_ha"] for b in blocks for f in json.load(open(R / "blocks" / f"{b}.json"))["features"]}
    missing_blocks = [b for b in blocks if not (R / "s2" / f"fields_{b}.parquet").exists()]
    rows = []
    for r in fl.itertuples():
        F = dict(field_id=r.field_id, eligible=bool(r.eligible), excl=r.excl_reason, cdl_ha=cdl[r.field_id])
        g = og.get(r.field_id)
        if r.eligible and g is not None and g.ndvi.notna().sum() > 0:
            F.update(bf.s2_features(g)); F.update(bf.s1_features(s1g.get(r.field_id), F.get("tr_mid_doy", np.nan)))
        rows.append(F)
    X = pd.DataFrame(rows)
    for c in cols:
        if c not in X: X[c] = np.nan
    ok = X.eligible & X.n_obs.fillna(0).gt(0)
    X["p_silage"] = np.nan; X.loc[ok, "p_silage"] = calib(model.predict_proba(X.loc[ok, cols])[:, 1])
    X["pred"] = np.where(~ok, "not_scored", np.where(X.p_silage >= thr, "silage", "grain"))
    ac = 1 / 0.404686
    scored = X[ok]; a = scored.cdl_ha * ac
    nass = {}
    for x in csv.DictReader(open(nass_csv)):
        if x["Geo Level"] == "STATE" and x["State"] == nass_state and x["Year"] == str(YEAR):
            nass["grain" if "GRAIN" in x["Data Item"] else "silage"] = float(x["Value"].replace(",", "")) if x["Value"].strip()[0].isdigit() else None
    ns, ng = nass.get("silage") or 0.0, nass.get("grain") or 0.0
    rng = np.random.default_rng(SEED); bs_t, bs_e = [], []
    av, sv, pv = a.values, (scored.pred == "silage").values, scored.p_silage.values
    for _ in range(2000):
        i = rng.integers(0, len(av), len(av)); bs_t.append(av[i][sv[i]].sum() / av[i].sum()); bs_e.append((pv[i] * av[i]).sum() / av[i].sum())
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
    res = dict(state=nass_state, year=YEAR, share_ci95=dict(threshold=ci(bs_t), expected=ci(bs_e)), sample=json.load(open(US / st / "prepare.json")), model="console RF E2 (NL-trained), NL threshold %.2f, NL calibration" % thr,
               fields_total=len(X), fields_in_sample=len(cdl), blocks_failed=missing_blocks, fields_scored=int(ok.sum()), not_scored=X[~ok].excl.replace("", "no clear looks").value_counts().to_dict(),
               cdl_corn_acres_all_fields=round(X.cdl_ha.sum() * ac), cdl_corn_acres_scored=round(a.sum()),
               pred_silage_acres_threshold=round(a[scored.pred == "silage"].sum()), pred_grain_acres_threshold=round(a[scored.pred == "grain"].sum()),
               pred_silage_share_threshold=round(float(a[scored.pred == "silage"].sum() / a.sum()), 3),
               pred_silage_share_expected=round(float((scored.p_silage * a).sum() / a.sum()), 3),
               nass_silage_acres=ns, nass_grain_acres=ng if "grain" in nass else "not published", nass_silage_share=round(ns / (ns + ng), 3) if ns + ng else None,
               p_silage_quantiles=scored.p_silage.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict(),
               caveats=["CDL = planted corn from a 30 m classification; NASS = harvested acres from a survey (CV not shown for small states)",
                        "aggregate share comparison only; no field labels",
                        "fields < ~0.5 ha interior after the 10 m inward buffer are not scored",
                        "model probabilities were miscalibrated in the US (BARC); the threshold share is the primary number"])
    out = US / st / "results"; out.mkdir(exist_ok=True)
    X[["field_id", "eligible", "excl", "cdl_ha", "n_obs", "p_silage", "pred", "tr_start_doy", "tr_ndvi_before", "tr_ndvi_after"]].to_csv(out / "fields.csv", index=False)
    json.dump(res, open(out / "state_vs_nass.json", "w"), indent=1, default=str); print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"prepare": lambda: prepare(sys.argv[2], sys.argv[3]), "infer": lambda: infer(sys.argv[2], sys.argv[3], sys.argv[4])}[sys.argv[1]]()
