"""Shared config for the Netherlands silage/grain maize pipeline."""
import os, json, time, pathlib, requests

ROOT = pathlib.Path(os.environ.get("HTH_DATA", pathlib.Path(__file__).resolve().parent.parent / "data_nl"))
ROOT.mkdir(parents=True, exist_ok=True)
YEAR = int(os.environ.get("HTH_YEAR", 2025))
SEASON = (f"{YEAR}-03-01", f"{YEAR}-12-31")
BRP_API = "https://api.pdok.nl/rvo/gewaspercelen/ogc/v1/collections/brpgewas/items"
# Official BRP crop codes (verified against live 2025 records, see label_audit.md).
SILAGE = {259: "Mais, snij-"}
GRAIN = {316: "Mais, korrel-"}
# Maize categories deliberately excluded from the binary task.
EXCLUDED_MAIZE = {317: "Mais, corncob mix", 1935: "Maiskolvensilage", 814: "Mais, suiker-"}
BUFFER_M = -10          # inward buffer against mixed edge pixels
MIN_INTERIOR_PX = 50    # >= 0.5 ha of interior 10 m pixels

def log(msg, name="pipeline.log"):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ROOT / name, "a") as f:
        f.write(line + "\n")

def get_json(url, params=None, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 502, 503, 504):
                time.sleep(2 ** k); continue
            r.raise_for_status()
        except requests.RequestException:
            if k == tries - 1: raise
            time.sleep(2 ** k)
    raise RuntimeError(f"failed {url}")
