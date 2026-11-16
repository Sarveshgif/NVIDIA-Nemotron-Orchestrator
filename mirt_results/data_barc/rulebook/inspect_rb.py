import json, sys
from collections import Counter
T = json.load(open("results_v1/traces.json"))["traces"]
I = {}
for y in (2021, 2022, 2023):
    for f in json.load(open("inputs_%d.json" % y)): I[f["key"]] = f
print(Counter(r for t in T if t["final"] == "UNRESOLVED" for r in t["reason_codes"]).most_common())
print(Counter(t["checks"]["4_removal"]["outcome"] for t in T))
print(Counter(t["checks"]["5_condition"]["outcome"] for t in T), Counter(t["checks"]["6_timing"]["outcome"] for t in T))
sel = sys.argv[1]
for t in [x for x in T if x["final"] == sel or sel in x["reason_codes"] or x["checks"]["4_removal"]["outcome"] == sel][:int(sys.argv[2])]:
    s = [(o["date"][5:], o["ndvi"]) for o in I[t["key"]]["optical"] if o["ndvi"] is not None and o["date"] >= "%d-07-01" % t["year"]]
    print(t["year"], t["final"], t["reason_codes"]); print("  ndvi", s)
    print("  cands", [(c["a"][5:], c["b"][5:], c["ndvi_a"], c["ndvi_b"], c["gap_days"], c["faster_than_senescence"], c["confirmed"], c["recovery"]) for c in t["checks"]["4_removal"]["values"].get("optical_candidates", [])])
    print("  radar", [(r["start"][5:], r["end"][5:], r["dVH_db"], r["dCR_db"], r["wetting"]) for r in t["checks"]["4_removal"]["values"].get("radar_changes", [])])
    print("  path", t["decision_path"])
