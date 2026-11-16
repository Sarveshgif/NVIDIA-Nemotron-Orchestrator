"""Rulebook v1, step C: evaluation. The ONLY step that reads BARC reference labels, after all predictions exist.

Inputs : <HTH_BARC>/rulebook/results_v1/traces.json, <HTH_BARC>/reference_labels.csv,
         ML predictions on the same field-years: <HTH_BARC>/results/barc_matched.csv (console RF, run 2)
Outputs: <HTH_BARC>/rulebook/results_v1/{eval_v1.json, per_field_eval.csv}
         <HTH_APP_PUBLIC>/barc/rulebook_v1_eval.json (labels; loaded only by the app's evaluation tab)
"""
import os, sys, csv, json, pathlib
from collections import Counter
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rulebook.evaluate import join_rows, summarize, bin_metrics
BARC = pathlib.Path(os.environ["HTH_BARC"])
OUT = BARC / "rulebook" / "results_v1"
T = json.load(open(OUT / "traces.json")); traces = T["traces"]
labels = list(csv.DictReader(open(BARC / "reference_labels.csv")))
ml = list(csv.DictReader(open(BARC / "results" / "barc_matched.csv")))

# population check (expected: 44 field-years, 32 grain, 12 silage, 34 polygons)
pop = dict(field_years=len(labels), grain=sum(l["label"] == "grain" for l in labels), silage=sum(l["label"] == "silage" for l in labels),
           polygons=len({l["OBJECTID"] for l in labels}))
pop["matches_expected"] = (pop["field_years"], pop["grain"], pop["silage"], pop["polygons"]) == (44, 32, 12, 34)
rows = join_rows(traces, labels, ml)
assert len(rows) == len(labels), "evaluation rows lost"
rb, mlsum = summarize(rows, "rb"), summarize(rows, "ml")
sub = [r for r in rows if r["rb"] in ("silage", "grain")]
same_subset = dict(n=len(sub), rulebook=bin_metrics([r["label"] for r in sub], [r["rb"] for r in sub]),
                   ml=bin_metrics([r["label"] for r in sub if r["ml"]], [r["ml"] for r in sub if r["ml"]]))
bykey = {(t["OBJECTID"], t["year"]): t for t in traces}
avail = {}
for t in traces:
    for k, c in t["checks"].items():
        avail.setdefault(k, Counter())[c["state"]] += 1
unres = Counter(rc for r in rows if r["rb"] is None for rc in r["reason_codes"].split(";") if rc)
outcome_by_class = {c: dict(Counter(r["final"] for r in rows if r["label"] == c)) for c in ("silage", "grain")}
per_field = []
for r in rows:
    t = bykey.get((r["OBJECTID"], r["year"]))
    per_field.append(dict(r, rb_correct=None if r["rb"] is None else r["rb"] == r["label"], ml_correct=None if r["ml"] is None else r["ml"] == r["label"],
                          harvest_interval=json.dumps(t["harvest_interval"]) if t else "", decision_path=" | ".join(t["decision_path"]) if t else ""))
with open(OUT / "per_field_eval.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, list(per_field[0].keys())); w.writeheader(); w.writerows(per_field)
res = dict(benchmark="BARC 2021-2023, development benchmark (data examined in earlier project work; not a pristine independent test)",
           config_version=T["config_version"], config_sha256_12=T["config_sha256_12"], population=pop,
           rulebook=rb, ml_all=mlsum, same_decided_subset=same_subset, outcome_by_class=outcome_by_class,
           check_availability={k: dict(v) for k, v in avail.items()}, unresolved_reasons=dict(unres.most_common()),
           errors=[p for p in per_field if p["rb_correct"] is False])
json.dump(res, open(OUT / "eval_v1.json", "w"), indent=1, default=str)
app = os.environ.get("HTH_APP_PUBLIC")
if app:
    json.dump(dict(res, per_field=per_field), open(pathlib.Path(app) / "barc" / "rulebook_v1_eval.json", "w"), separators=(",", ":"), default=str)
print("population", pop)
print("rulebook outcomes", rb["outcomes"]); print("coverage", rb["decided"], "/", rb["n"], "correct/all", rb["correct"], "/", rb["n"])
print("rulebook binary", rb["binary"]); print("ML all", mlsum["binary"]); print("same subset", same_subset)
print("availability", res["check_availability"]); print("unresolved reasons", res["unresolved_reasons"])
print("RULEBOOK_EVAL_DONE")
