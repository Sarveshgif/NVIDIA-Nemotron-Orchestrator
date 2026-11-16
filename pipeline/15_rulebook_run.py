"""Rulebook v1, step B: run the frozen engine on every field-year input (no labels read in this step).

Inputs : <HTH_BARC>/rulebook/{inputs_<year>.json, weather_<year>.json, key_map.csv}, rulebook/config_v1.json,
         rulebook/evidence_registry_v1.json
Outputs: <HTH_BARC>/rulebook/results_v1/{traces.json, predictions.csv, predictions.geojson}
         <HTH_APP_PUBLIC>/barc/rulebook_v1.json (app: map, table, traces, series; no reference labels)
"""
import os, sys, csv, json, pathlib, hashlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rulebook.engine import decide
BARC = pathlib.Path(os.environ["HTH_BARC"])
IN = BARC / "rulebook"; OUT = IN / "results_v1"; OUT.mkdir(parents=True, exist_ok=True)
CFG_PATH = HERE / "rulebook" / "config_v1.json"
CFG = json.load(open(CFG_PATH)); REG = json.load(open(HERE / "rulebook" / "evidence_registry_v1.json"))
cfg_hash = hashlib.sha256(CFG_PATH.read_bytes()).hexdigest()[:12]
YEARS = (2021, 2022, 2023)

traces, inputs = [], {}
for y in YEARS:
    w = json.load(open(IN / f"weather_{y}.json"))
    for fy in json.load(open(IN / f"inputs_{y}.json")):
        inputs[fy["key"]] = fy
        traces.append(decide(fy, w, CFG, REG))
# opaque key -> OBJECTID only now, for exports (the engine never saw it)
km = {r["key"]: r for r in csv.DictReader(open(IN / "key_map.csv"))}
for t in traces: t["OBJECTID"] = int(km[t["key"]]["OBJECTID"])
json.dump(dict(config_version=CFG["version"], config_sha256_12=cfg_hash, traces=traces), open(OUT / "traces.json", "w"), indent=1, default=str)

COLS = ["OBJECTID", "year", "key", "crop_cycle_id", "final", "review", "reason_codes", "harvest_last_standing", "harvest_first_removed",
        "harvest_width_days", "harvest_midpoint_estimate", "harvest_source"] + \
       [f"{k}_state" for k in ("1_identity", "2_quality", "3_documents", "4_removal", "5_condition", "6_timing", "7_surface", "8_radar", "9_exceptions", "11_context")] + \
       [f"{k}_outcome" for k in ("4_removal", "5_condition", "6_timing", "7_surface", "8_radar")] + \
       ["exception_flags", "missing_evidence", "decision_path", "config_version"]
with open(OUT / "predictions.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, COLS); wr.writeheader()
    for t in traces:
        hi = t["harvest_interval"] or {}
        row = {"OBJECTID": t["OBJECTID"], "year": t["year"], "key": t["key"], "crop_cycle_id": t["crop_cycle_id"], "final": t["final"],
               "review": t["review"], "reason_codes": ";".join(t["reason_codes"]), "harvest_last_standing": hi.get("last_standing"),
               "harvest_first_removed": hi.get("first_removed"), "harvest_width_days": hi.get("width_days"),
               "harvest_midpoint_estimate": hi.get("midpoint_estimate"), "harvest_source": hi.get("source"),
               "exception_flags": ";".join(t["exception_flags"]), "missing_evidence": ";".join(t["missing_evidence"]),
               "decision_path": " | ".join(t["decision_path"]), "config_version": t["config_version"]}
        for k, c in t["checks"].items():
            if f"{k}_state" in COLS: row[f"{k}_state"] = c["state"]
            if f"{k}_outcome" in COLS: row[f"{k}_outcome"] = c["outcome"]
        wr.writerow(row)
gj = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "geometry": inputs[t["key"]]["geometry"],
     "properties": {"OBJECTID": t["OBJECTID"], "year": t["year"], "crop_cycle_id": t["crop_cycle_id"], "final": t["final"], "review": t["review"],
                    "reason_codes": ";".join(t["reason_codes"]), "harvest_interval": json.dumps(t["harvest_interval"]), "config_version": t["config_version"]}}
    for t in traces]}
json.dump(gj, open(OUT / "predictions.geojson", "w"))

# ---- app export (no labels) ----
app = os.environ.get("HTH_APP_PUBLIC")
if app:
    P = pathlib.Path(app) / "barc"; P.mkdir(parents=True, exist_ok=True)
    fields = []
    for t in traces:
        fy = inputs[t["key"]]
        fields.append(dict(trace=t, geometry=fy["geometry"], area_ha=fy["area_ha"],
                           optical=[{k: o[k] for k in ("date", "ndvi", "ndre", "ndti", "lswi", "clear_frac")} for o in fy["optical"] if o["ndvi"] is not None],
                           radar=[{k: r[k] for k in ("date", "vv", "vh")} for r in fy["radar"]["series"]],
                           radar_meta={k: fy["radar"][k] for k in ("orbit", "coverage_frac", "full_cover", "reason")}))
    json.dump(dict(config_version=CFG["version"], config_sha256_12=cfg_hash, config=CFG, registry=REG, fields=fields),
              open(P / "rulebook_v1.json", "w"), separators=(",", ":"), default=str)
counts = {}
for t in traces: counts[t["final"]] = counts.get(t["final"], 0) + 1
print("rulebook v1:", len(traces), "field-years", counts, "review", sum(t["review"] for t in traces))
print("RULEBOOK_RUN_DONE")
