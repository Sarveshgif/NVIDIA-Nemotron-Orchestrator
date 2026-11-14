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
s2.process("BARC", chip_ids=[])
p = s2.OUT / "fields_BARC.parquet"
f = pd.read_parquet(p).drop(columns=["label", "code", "crop_name"], errors="ignore")
f.to_parquet(p)
s2.log(f"BARC {YEAR}: collection {s2.S2_COLLECTION}; fields {len(f)}, eligible {int(f.eligible.sum())}, "
       f"excluded {f.excl_reason[f.excl_reason != ''].value_counts().to_dict()}")
load("s1x", "03_extract_s1.py").process("BARC")
print("BARC_EXTRACT_DONE", YEAR)
