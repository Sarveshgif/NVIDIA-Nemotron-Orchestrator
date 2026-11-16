"""BARC external check, step 2: Sentinel-2 and Sentinel-1 extraction for one labelled year, label-free.

Usage: HTH_BARC=<dir> python 11_barc_extract.py <year>   (optionally HTH_S2_COLLECTION=sentinel-2-l2a)
Runs the unchanged extraction code of 02_extract_s2.py and 03_extract_s1.py on <HTH_BARC>/<year>/blocks/BARC.json.
Only the parcel loader is replaced: it returns the BARC polygons with no crop-use code (the BRP loader
filters on BRP codes). Full polygons are rasterised (10 m inward buffer); no chips (chip_ids=[]), so nothing
is centre-sampled. The placeholder label/code columns 02 writes are dropped immediately.
"""
import os, sys, json, pathlib, importlib.util
YEAR = int(sys.argv[1])
BARC = pathlib.Path(os.environ.get("HTH_BARC", pathlib.Path(__file__).resolve().parent.parent / "data_barc"))
os.environ["HTH_DATA"] = str(BARC / str(YEAR)); os.environ["HTH_YEAR"] = str(YEAR)
import pandas as pd
from shapely.geometry import shape
HERE = pathlib.Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


s2 = load("s2x", "02_extract_s2.py")


def load_parcels(block):
    d = json.load(open(s2.ROOT / "blocks" / f"{block}.json"))
    return d["bbox"], [(f["id"], 0, "unlabelled", shape(f["geometry"])) for f in d["features"]]


s2.load_parcels = load_parcels
BLOCKS = os.environ.get("HTH_BLOCKS", "BARC").split(",")   # several blocks when fields span S2 tiles
# HTH_CHIP=N: also write N x N chips (10 m) centred on every field, for the CNN -> Transformer (13_barc_deep.py).
# Use a separate HTH_BARC root for chip runs; HTH_SKIP_S1=1 skips S1 (link the s1/ folder from the main root).
CHIP = os.environ.get("HTH_CHIP")
if CHIP: s2.CHIP = int(CHIP)
def run_block(blk):
    s2.process(blk, chip_ids=None if CHIP else [])
    p = s2.OUT / f"fields_{blk}.parquet"
    f = pd.read_parquet(p).drop(columns=["label", "code", "crop_name"], errors="ignore")
    f.to_parquet(p)
    s2.log(f"{blk} {YEAR}: collection {s2.S2_COLLECTION}; fields {len(f)}, eligible {int(f.eligible.sum())}, "
           f"excluded {f.excl_reason[f.excl_reason != ''].value_counts().to_dict()}")
    if not os.environ.get("HTH_SKIP_S1") and f.eligible.any():
        load("s1x", "03_extract_s1.py").process(blk)


for blk in BLOCKS:
    try:
        run_block(blk)
    except Exception as e:      # e.g. a block whose only field is too small: log it and continue with the next block
        s2.log(f"{blk}: FAILED {type(e).__name__}: {e}")
print("BARC_EXTRACT_DONE", YEAR)
