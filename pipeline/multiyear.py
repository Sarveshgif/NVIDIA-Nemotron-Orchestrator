"""Multi-year experiment config. Each crop year has its own data root (blocks/, s2/, s1/, features/).

HTH_ROOTS        "2024:/path/data2024,2025:/path/data"   (year:root pairs)
HTH_TRAIN_YEARS  years whose TRAIN/VAL regions are used for fitting / tuning, e.g. "2024,2025"
HTH_TEST_YEAR    year whose TEST regions are scored, e.g. "2025"
HTH_RESULTS      output directory for this experiment
Regions -> split are identical every year, so a physical field seen in two years always stays in one split.
Field keys are "<year>:<BRP id>" so the two declarations of one parcel are separate field-seasons.
"""
import os, json, pathlib, pandas as pd
from common import ROOT

def roots():
    spec = os.environ.get("HTH_ROOTS")
    if not spec:
        return {int(os.environ.get("HTH_YEAR", 2025)): ROOT}
    return {int(k): pathlib.Path(v) for k, v in (x.split(":", 1) for x in spec.split(","))}

TRAIN_YEARS = [int(x) for x in os.environ.get("HTH_TRAIN_YEARS", "2025").split(",")]
TEST_YEAR = int(os.environ.get("HTH_TEST_YEAR", "2025"))
RESULTS = pathlib.Path(os.environ.get("HTH_RESULTS", ROOT / "results")); RESULTS.mkdir(parents=True, exist_ok=True)

def load_tables(years=None):
    """Returns (df, schema): one row per field-season with year, uid, raw field_id, split, region, block, label, area_ha, features."""
    parts, schema = [], None
    for y, r in roots().items():
        if years is not None and y not in years: continue
        F = r / "features"
        feats = pd.read_parquet(F / "features.parquet")
        lab = pd.read_parquet(F / "labels.parquet")[["field_id", "label"]]
        spl = pd.read_csv(F / "splits.csv")
        fl = pd.read_parquet(F / "fields.parquet")[["field_id", "area_ha"]]
        d = spl.merge(lab, on="field_id").merge(feats, on="field_id").merge(fl, on="field_id")
        d.insert(0, "year", y); d.insert(0, "uid", [f"{y}:{i}" for i in d.field_id])
        parts.append(d)
        sch = json.load(open(F / "feature_schema.json"))["groups"]
        schema = sch if schema is None else {k: [c for c in schema[k] if c in sch.get(k, [])] for k in schema}
    df = pd.concat(parts, ignore_index=True)
    df["y"] = (df.label == "silage").astype(int)
    df["role"] = "unused"
    df.loc[df.year.isin(TRAIN_YEARS) & (df.split == "train"), "role"] = "train"
    df.loc[df.year.isin(TRAIN_YEARS) & (df.split == "val"), "role"] = "val"
    df.loc[(df.year == TEST_YEAR) & (df.split == "test"), "role"] = "test"
    return df, schema
