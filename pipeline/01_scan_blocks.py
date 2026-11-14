"""Scan candidate ~10 km blocks via the PDOK BRP OGC API (anonymous) and cache every parcel's
declaration (all crops, so maize categories can be audited). Output: blocks/<id>.json."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from common import ROOT, BRP_API, get_json, log, SILAGE, GRAIN

# Candidate block centres (lon, lat): maize regions spread across the country.
CANDIDATES = {
    "LI-venray": (6.00, 51.50), "LI-horst": (6.05, 51.42), "LI-weert": (5.80, 51.25), "LI-roermond": (5.98, 51.15),
    "LI-sittard": (5.85, 51.02), "NB-peel": (5.75, 51.52), "NB-deurne": (5.80, 51.42), "NB-asten": (5.70, 51.33),
    "NB-kempen": (5.25, 51.40), "NB-oss": (5.55, 51.72), "GE-achterhoek": (6.55, 51.95), "GE-winterswijk": (6.70, 51.98),
    "GE-gelderse-vallei": (5.60, 52.10), "OV-twente": (6.80, 52.30), "OV-salland": (6.30, 52.40),
    "DR-veenkolonien": (6.95, 52.85), "GR-oldambt": (7.05, 53.10), "DR-midden": (6.55, 52.85), "FR-zuid": (5.95, 52.95),
    "FL-noordoostpolder": (5.75, 52.70), "FL-dronten": (5.65, 52.50), "ZE-zuid": (3.85, 51.32), "NH-kop": (4.85, 52.80),
    "UT-lopik": (4.95, 51.98),
    # second pass: grain-maize belt (south-east)
    "LI-heerlen": (5.95, 50.88), "LI-maastricht": (5.72, 50.88), "LI-gennep": (6.05, 51.65), "NB-cuijk": (5.88, 51.65),
    "NB-uden": (5.65, 51.62), "NB-eindhoven": (5.50, 51.45), "NB-best": (5.35, 51.55), "NB-hilvarenbeek": (5.10, 51.45),
    "NB-west": (4.50, 51.55), "GE-rivierenland": (5.30, 51.85), "GE-doetinchem": (6.30, 51.95), "LI-echt": (5.82, 51.12),
}
HALF_LON, HALF_LAT = 0.072, 0.045   # ~10 x 10 km

def fetch_block(item):
    bid, (lon, lat) = item
    out = ROOT / "blocks" / f"{bid}.json"
    if out.exists():
        return bid, json.load(open(out))
    bbox = f"{lon-HALF_LON},{lat-HALF_LAT},{lon+HALF_LON},{lat+HALF_LAT}"
    feats, url, params = [], BRP_API, {"f": "json", "limit": 1000, "bbox": bbox}
    while url:
        d = get_json(url, params)
        feats += d["features"]
        url = next((l["href"] for l in d["links"] if l["rel"] == "next"), None)
        params = None
    rec = {"id": bid, "bbox": [lon-HALF_LON, lat-HALF_LAT, lon+HALF_LON, lat+HALF_LAT], "features": feats}
    out.parent.mkdir(exist_ok=True)
    json.dump(rec, open(out, "w"))
    return bid, rec

if __name__ == "__main__":
    rows = []
    with ThreadPoolExecutor(4) as ex:
        for bid, rec in ex.map(fetch_block, CANDIDATES.items()):
            c = Counter(f["properties"]["gewascode"] for f in rec["features"])
            rows.append((bid, len(rec["features"]), sum(c[k] for k in SILAGE), sum(c[k] for k in GRAIN)))
            log(f"block {bid}: parcels={rows[-1][1]} silage={rows[-1][2]} grain={rows[-1][3]}")
    rows.sort(key=lambda r: -r[3])
    for r in rows: print(*r, sep="\t")
