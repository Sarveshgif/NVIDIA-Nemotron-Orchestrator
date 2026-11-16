"""Rulebook v7 = v6 (unchanged) + a separate dry-grain harvest pathway. Both pathways are evaluated before deciding.

Dry-grain pathway (GRAIN - likely), all required:
  G1 an earlier clear look establishes a green canopy in the same cycle (the season peak, NDVI >= 0.60 by v6 step 0)
  G2 >= 3 clear looks spanning >= 21 d after it establish drying (each <= peak - 0.15; not necessarily monotonic),
     ending at NDVI <= 0.50 (the last clear look up to the radar event)
  G3 a later radar event with VH decrease >= 1.5 dB (v6 two-before/two-after comparison and 24-day windows)
  G4 the last low-NDVI clear look is <= 20 d before the radar event (the event's last pre pass)
  G5 no earlier credible green-cut harvest (v6 credible optical event with green preharvest) and no intervening
     green-up (a clear look >= 0.55 after drying began and before the event)
  G6 VV increase >= 1 dB at the event = moisture ambiguity -> REVIEW unless resolved independently: the VH drop
     persists in the 3rd-4th passes after the event (still <= -1.5 dB vs the pre passes), i.e. not a transient wet spell
No >= 0.20 optical drop and no post-harvest NDVI <= 0.35 are required on this pathway.

Combination with v6 (the silage pathway, unchanged):
  - v6 step-0 outcomes (uncertain identity, cycle not separable, no green-up, no season data) stand.
  - v6 SILAGE + grain pathway holds          -> REVIEW (incompatible interpretations)
  - v6 GRAIN + grain pathway holds           -> v6 GRAIN (consistent; v6 tier kept)
  - v6 REVIEW/INSUFFICIENT for lack of a credible harvest + grain pathway holds -> GRAIN likely
  - v6 REVIEW because a credible harvest had an ambiguous preharvest condition + grain event overlaps that same
    harvest -> GRAIN likely (the drying trajectory resolves the condition); otherwise REVIEW (conflict)
  - any other v6 contradiction (zones, radar before harvest while green, questionable image, incompatible
    candidates) + grain pathway holds -> REVIEW with both explanations
  - grain pathway blocked only by moisture -> REVIEW (moisture ambiguity) when v6 itself did not decide

Run (lab machine): HTH_BARC=.. HTH_DATA=.. E2_RESULTS=.. python 20_rulebook_v7.py dev|final
  dev   = development diagnostics on BARC 44 and the NL seed-99 sample
  final = frozen run on an untouched NL sample (seed RB_SEED, excluding RB_EXCLUDE samples) + figure
"""
import os, sys, json, csv, hashlib, pathlib, importlib.util
from datetime import timedelta
from collections import Counter
HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v6", HERE / "19_rulebook_v6.py")
V6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(V6)
T = V6.T
G = dict(dry_end=0.50, decline_below_peak=0.15, min_looks=3, min_span=21, max_age=20, green=0.55,
         radar_drop=T["radar_drop"], wet_vv=T["wet_vv"])
VERSION = "rulebook-v7"
STEP0 = ("uncertain corn identity", "no green-up observed", "spring green cycle", "green-up not observed", "no clear observation in the corn season")
NO_HARVEST = ("radar change without a credible optical harvest", "observations adequate but no credible harvest event", "no credible harvest event and observations too sparse")


def grain_pathway(f):
    y = f["year"]
    looks = sorted([dict(o, d=V6.D(o["date"])) for o in f["optical"] if V6.ok(o.get("ndvi"))], key=lambda o: o["d"])
    rad = sorted([dict(r, d=V6.D(r["date"])) for r in f.get("radar", []) if V6.ok(r.get("vh")) and V6.ok(r.get("vv"))], key=lambda r: r["d"])
    s0, s1 = V6.md(y, T["season"][0]), V6.md(y, T["season"][1])
    season = [o for o in looks if s0 <= o["d"] <= s1]
    if not season: return dict(holds=False, why="no season data")
    pk = max(season, key=lambda o: o["ndvi"])
    if pk["ndvi"] < T["peak_min"]: return dict(holds=False, why="G1 no green canopy")
    tol = 12
    if len(rad) > 2:
        import statistics
        tol = int(statistics.median([(b["d"] - a["d"]).days for a, b in zip(rad, rad[1:])]))
    # G5 (part 1): earlier credible green-cut harvest from the v6 silage logic
    oe = V6.optical_events(looks, pk["d"]); re_ = V6.radar_events(rad, pk["d"])
    green_cuts = []
    for e in oe:
        corr = any(V6.overlaps(e["a"]["d"], e["b"]["d"], r["start"], r["end"], tol) for r in re_)
        if (corr or e["persistent"]) and e["a"]["ndvi"] >= G["green"]: green_cuts.append(e)
    tried = []
    for r in re_:
        upto = [o for o in looks if pk["d"] < o["d"] <= r["end"]]
        if not upto: continue
        last = upto[-1]
        drying = [o for o in upto if o["ndvi"] <= pk["ndvi"] - G["decline_below_peak"]]
        ev = dict(radar_event=(str(r["start"]), str(r["end"])), dVH=round(r["dvh"], 2), dVV=round(r["dvv"], 2),
                  last_look=str(last["d"]), last_ndvi=round(last["ndvi"], 3), last_look_age_days=(r["start"] - last["d"]).days,
                  drying_looks=len(drying), drying_span_days=(drying[-1]["d"] - drying[0]["d"]).days if drying else 0)
        fail = None
        if len(drying) < G["min_looks"] or ev["drying_span_days"] < G["min_span"]: fail = "G2 drying not established (>=3 looks over >=21 d)"
        elif last["ndvi"] > G["dry_end"]: fail = f"G2 drying does not end <= {G['dry_end']} (last look {last['ndvi']:.2f})"
        elif (r["start"] - last["d"]).days > G["max_age"]: fail = f"G4 last low-NDVI look {(r['start'] - last['d']).days} d before the radar event"
        elif any(gc["b"]["d"] <= r["end"] for gc in green_cuts): fail = "G5 earlier credible green-cut harvest"
        elif any(o["ndvi"] >= G["green"] for o in upto if o["d"] > drying[0]["d"]): fail = "G5 intervening green-up"
        if fail: tried.append(dict(ev, fail=fail)); continue
        if r["moisture"]:
            j = r["j"]; pre = [x for x in rad[:j] if (r["end"] - x["d"]).days <= T["radar_side_days"]][-2:]
            later = rad[j + 2:j + 4]
            persists = len(later) == 2 and len(pre) == 2 and (sum(x["vh"] for x in later) / 2 - sum(x["vh"] for x in pre) / 2) <= -G["radar_drop"]
            ev["moisture"] = True; ev["moisture_resolved_by_persistence"] = persists
            if not persists: return dict(holds=False, moisture_block=True, why="G6 VV rise >= 1 dB (moisture ambiguity), VH drop not persistent", evidence=ev, tried=tried)
        return dict(holds=True, why="dry-grain pathway G1-G6", evidence=ev, tried=tried, peak=str(pk["d"]))
    return dict(holds=False, why=(tried[0]["fail"] if tried else "G3 no radar VH decrease >= 1.5 dB after the peak"), tried=tried)


def decide(f):
    s = V6.decide(f); g = grain_pathway(f)
    out = dict(s, v6_final=s["final"], v6_tier=s["tier"], v6_why=s["why"], grain_pathway=g["why"], grain_evidence=g.get("evidence"))
    why = s["why"]
    if s["final"] in ("REVIEW", "INSUFFICIENT DATA") and why.startswith(STEP0): return out
    if not g["holds"]:
        if g.get("moisture_block") and s["final"] not in ("SILAGE", "GRAIN"):
            return dict(out, final="REVIEW", tier=None, why="moisture ambiguity on the only grain-pathway radar event; " + why)
        return out
    gw = f"dry-grain pathway: radar {g['evidence']['radar_event'][0]}..{g['evidence']['radar_event'][1]} dVH {g['evidence']['dVH']:+.1f} dB after {g['evidence']['drying_looks']} drying looks over {g['evidence']['drying_span_days']} d, last NDVI {g['evidence']['last_ndvi']:.2f}"
    if s["final"] == "SILAGE": return dict(out, final="REVIEW", tier=None, why=f"CONFLICT: silage pathway ({why}) vs {gw}")
    if s["final"] == "GRAIN": return dict(out, why=why + " | consistent with " + gw)
    if why.startswith(NO_HARVEST): return dict(out, final="GRAIN", tier="likely", why=gw)
    if why.startswith("credible harvest but preharvest condition"):
        hi = s.get("harvest_interval"); re_ = g["evidence"]["radar_event"]
        if hi and V6.overlaps(V6.D(hi[0]), V6.D(hi[1]), V6.D(re_[0]), V6.D(re_[1]), 12):
            return dict(out, final="GRAIN", tier="likely", why=gw + " (same event as the optical harvest; drying resolves its condition)")
    return dict(out, final="REVIEW", tier=None, why=f"CONFLICT: {why} | {gw}")


# ---------------------------------------------------------------- evaluation
def report(name, rows, out, ml_note):
    for r in rows:
        r["call"] = r["final"].lower() if r["final"] in ("SILAGE", "GRAIN") else ("review" if r["final"] == "REVIEW" else "insufficient")
        r["call_v6"] = r["v6_final"].lower() if r["v6_final"] in ("SILAGE", "GRAIN") else ("review" if r["v6_final"] == "REVIEW" else "insufficient")
    def summ(key):
        cm = {lab: dict(Counter(r[key] for r in rows if r["label"] == lab)) for lab in ("silage", "grain")}
        dec = [r for r in rows if r[key] in ("silage", "grain")]
        f = lambda a, b: round(a / b, 3) if b else None
        res = dict(confusion=cm, coverage=f"{len(dec)}/{len(rows)}", decided_accuracy=f(sum(r[key] == r["label"] for r in dec), len(dec)))
        for c in ("silage", "grain"):
            tp = sum(1 for r in dec if r[key] == c and r["label"] == c)
            res[f"precision_{c}"] = f(tp, sum(1 for r in dec if r[key] == c))
            res[f"recall_{c}_incl_abstentions"] = f(tp, sum(1 for r in rows if r["label"] == c))
        res["false_calls"] = {c: sum(1 for r in dec if r[key] == c and r["label"] != c) for c in ("silage", "grain")}
        return res
    ml = summ("ml")
    res = dict(dataset=name, version=VERSION, thresholds=dict(v6=T, grain_pathway=G), rf_exclusion=ml_note,
               v7=summ("call"), v6=summ("call_v6"), rf=ml, v7_by_tier={t: summ_t for t, summ_t in
               ((t, dict(n=sum(1 for r in rows if r["tier"] == t and r["call"] in ("silage", "grain")),
                         correct=sum(1 for r in rows if r["tier"] == t and r["call"] == r["label"]))) for t in ("strong", "likely"))},
               v7_review_reasons=dict(Counter(r["why"].split(":")[0].split(" (")[0][:70] for r in rows if r["call"] in ("review", "insufficient"))),
               grain_pathway_outcomes=dict(Counter(r["grain_pathway"].split(" (")[0][:60] for r in rows)))
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "eval.json", "w"), indent=1, default=str)
    keys = ["id", "year", "label", "final", "tier", "why", "v6_final", "v6_tier", "v6_why", "ml", "grain_pathway", "grain_evidence",
            "harvest_interval", "harvest_kind", "preharvest_ndvi", "preharvest_date", "closeness", "condition", "optical_drop", "radar_change", "peak", "peak_ndvi"]
    with open(out / "per_field.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"\n######## {name}")
    for k in ("v6", "v7", "rf"): print(f" {k:3s}", res[k])
    print(" v7 by tier", res["v7_by_tier"]); print(" v7 review reasons", res["v7_review_reasons"]); print(" grain pathway", res["grain_pathway_outcomes"])
    for r in rows:
        if r["call"] in ("silage", "grain") and r["call"] != r["label"]: print("  WRONG", r["id"], r["year"], r["label"], "->", r["final"], "|", r["why"][:160])
    return res


def barc_rows(B):
    km = {r["key"]: r for r in csv.DictReader(open(B / "rulebook/key_map.csv"))}
    labels = {(int(r["OBJECTID"]), int(r["year"])): r["label"] for r in csv.DictReader(open(B / "reference_labels.csv"))}
    ml = {(int(r["OBJECTID"]), int(r["year"])): r["pred"] for r in csv.DictReader(open(B / "results/barc_matched.csv"))}
    rows, inputs = [], {}
    for y in (2021, 2022, 2023):
        for fy in json.load(open(B / f"rulebook/inputs_{y}.json")):
            f = dict(year=y, optical=fy["optical"], radar=fy["radar"]["series"]); oid = int(km[fy["key"]]["OBJECTID"])
            rows.append(dict(id=oid, year=y, **decide(f))); inputs[(oid, y)] = f
    for r in rows: r["label"] = labels[(r["id"], r["year"])]; r["ml"] = ml[(r["id"], r["year"])]
    return rows, inputs


def nl_rows(R, E2, seed, exclude):
    import pandas as pd
    spl = pd.read_csv(R / "features/splits.csv").merge(pd.read_parquet(R / "features/labels.parquet")[["field_id", "label"]])
    test = spl[spl.split == "test"]
    for ex in exclude: test = test[~test.field_id.isin(pd.read_csv(ex).field_id)]
    sample = pd.concat([g.sample(20, random_state=seed) for _, g in test.groupby("label")])
    rows, inputs = [], {}
    for b, g in sample.groupby("block"):
        obs = pd.read_parquet(R / "s2_20m" / f"obs20_{b}.parquet"); s1 = pd.read_parquet(R / "s1" / f"s1_{b}.parquet")
        orbit = int(s1.groupby("rel_orbit").size().idxmax()); s1 = s1[s1.rel_orbit == orbit]
        for fid in g.field_id:
            o = obs[(obs.field_id == fid) & obs.ndvi.notna()].sort_values("date"); r = s1[s1.field_id == fid].sort_values("date")
            f = dict(year=2025, optical=[dict(date=str(x.date)[:10], ndvi=float(x.ndvi), clear_frac=float(x.clear_frac),
                                              **{f"z{k}": (None if x[f"ndvi_z{k}"] != x[f"ndvi_z{k}"] else float(x[f"ndvi_z{k}"])) for k in range(1, 5)})
                                         for _, x in o.iterrows()],
                     radar=[dict(date=str(d)[:10], vv=float(v), vh=float(h)) for d, v, h in zip(r.date, r.vv_db, r.vh_db)])
            rows.append(dict(id=fid, year=2025, block=b, region=g.region.iloc[0] if "region" in g else None, **decide(f))); inputs[(fid, 2025)] = f
    pd.DataFrame(rows).drop(columns=["grain_evidence"], errors="ignore").to_csv(R / f"results_rulebook_v7_seed{seed}_predictions.csv", index=False)  # before labels
    lab = dict(zip(sample.field_id, sample.label)); reg = dict(zip(sample.field_id, sample.region))
    mlp = pd.read_parquet(E2 / "predictions_all.parquet"); mlp = mlp[mlp.year == 2025]; ml = dict(zip(mlp.field_id, mlp.pred))
    for r in rows: r["label"] = lab[r["id"]]; r["ml"] = ml.get(r["id"]); r["region"] = reg[r["id"]]
    # RF exclusion check: sampled fields are 2025 TEST-region fields; E2 trained on train-region and tuned on
    # val-region field-seasons only (2024+2025); same regions map to the same split every year.
    import importlib
    s24 = pd.read_csv(pathlib.Path(str(R).replace("/data", "/data2024")) / "features/splits.csv") if (pathlib.Path(str(R).replace("/data", "/data2024")) / "features/splits.csv").exists() else None
    regions = set(sample.region)
    note = dict(sample_split=sorted(set(sample.split)), sample_regions=sorted(regions),
                regions_in_2025_train_val=sorted(set(spl[spl.split.isin(["train", "val"])].region) & regions),
                regions_in_2024_train_val=None if s24 is None else sorted(set(s24[s24.split.isin(["train", "val"])].region) & regions),
                all_have_rf_prediction=all(r["ml"] is not None for r in rows),
                statement="RF (E2) fitted on train-region and tuned/calibrated on val-region field-seasons of 2024+2025; these fields are 2025 test-region fields, never used for RF training or tuning.")
    return rows, inputs, note


def figure(rows, inputs, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from datetime import date as _d
    pick = []
    for cond, lab in ((lambda r: r["call"] == "silage" and r["label"] == "silage", "correct silage"),
                      (lambda r: r["call"] == "grain" and r["label"] == "grain" and "dry-grain" in r["why"], "correct grain (new pathway)"),
                      (lambda r: r["call"] in ("silage", "grain") and r["call"] != r["label"], "incorrect"),
                      (lambda r: r["call"] == "grain" and r["label"] != "grain", "false grain call"),
                      (lambda r: r["call"] == "review", "unresolved (REVIEW)"),
                      (lambda r: r["call"] == "review" and r is not None, "unresolved (REVIEW) #2")):
        c = [r for r in rows if cond(r) and r not in [p[0] for p in pick]]
        if c: pick.append((c[0], lab))
    fig, axs = plt.subplots(len(pick), 1, figsize=(11, 2.6 * len(pick)), squeeze=False)
    for ax, (r, lab) in zip(axs[:, 0], pick):
        f = inputs[(r["id"], r["year"])]; y = r["year"]
        od = [(V6.D(o["date"]), o["ndvi"]) for o in f["optical"] if V6.ok(o.get("ndvi"))]
        ax.plot([d for d, _ in od], [v for _, v in od], "o-", color="#24523e", ms=3, lw=1, label="NDVI (clear looks)")
        ax.set_ylim(-0.05, 1); ax.set_xlim(_d(y, 3, 1), _d(y, 12, 31)); ax.axhline(0.5, color="#b7801a", lw=0.6, ls=":"); ax.axhline(0.35, color="#7f8682", lw=0.6, ls=":")
        ax2 = ax.twinx(); rd = [(V6.D(x["date"]), x["vh"], x["vv"]) for x in f["radar"] if V6.ok(x.get("vh"))]
        ax2.plot([d for d, *_ in rd], [h for _, h, _ in rd], "s-", color="#7a5aa6", ms=2.5, lw=0.8, label="VH dB")
        ax2.plot([d for d, *_ in rd], [v for *_, v in rd], "^-", color="#4f6a8a", ms=2, lw=0.6, alpha=0.6, label="VV dB")
        if r.get("harvest_interval"): ax.axvspan(V6.D(r["harvest_interval"][0]), V6.D(r["harvest_interval"][1]), color="#c26a00", alpha=0.2)
        ge = r.get("grain_evidence")
        if ge: ax.axvspan(V6.D(ge["radar_event"][0]), V6.D(ge["radar_event"][1]), color="#7a5aa6", alpha=0.18)
        ax.set_title(f"{lab}: {r['id']} {y} | reference {r['label']} | v7 {r['final']} {r['tier'] or ''} | v6 {r['v6_final']} | RF {r['ml']}\n{r['why'][:150]}", fontsize=8, loc="left")
        ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
    axs[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle(title + "  (orange = optical harvest interval, purple = grain-pathway radar event; dotted 0.50 / 0.35)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    mode = sys.argv[1]
    B = pathlib.Path(os.environ["HTH_BARC"]); R = pathlib.Path(os.environ["HTH_DATA"]); E2 = pathlib.Path(os.environ["E2_RESULTS"])
    code_hash = hashlib.sha256((HERE / "20_rulebook_v7.py").read_bytes() + (HERE / "19_rulebook_v6.py").read_bytes()).hexdigest()[:12]
    print("code sha256[:12]", code_hash)
    if mode == "dev":
        rows, inp = barc_rows(B); report("DEV BARC 44", rows, B / "results/rulebook_v7_dev", "RF trained only on NL; BARC never used")
        rows, inp, note = nl_rows(R, E2, 99, [R / "results_lenient_nl/predictions.csv", R / "results_lenient_nl_v4_seed7/predictions.csv"])
        report("DEV NL seed-99 40", rows, R / "results_rulebook_v7_dev_seed99", note)
        for r in rows:
            if r["v6_final"] == "SILAGE" and r["label"] == "grain":
                f = inp[(r["id"], 2025)]
                print("  v6-silage-error", r["id"], r["region"], "| v7:", r["final"], "|", r["v6_why"],
                      "\n     NDVI", [(o["date"][5:], round(o["ndvi"], 2)) for o in f["optical"] if "07-15" <= o["date"][5:] <= "10-31"],
                      "\n     VH  ", [(x["date"][5:], round(x["vh"], 1)) for x in f["radar"] if "07-15" <= x["date"][5:] <= "10-31"],
                      "\n     zones on harvest dates", [(o["date"][5:], [None if o.get(f'z{k}') is None else round(o[f'z{k}'], 2) for k in range(1, 5)]) for o in f["optical"] if r.get("harvest_interval") and o["date"] in (r["harvest_interval"][0], r["harvest_interval"][1])])
    else:
        seed = int(os.environ["RB_SEED"]); ex = [pathlib.Path(p) for p in os.environ["RB_EXCLUDE"].split(",")]
        out = R / f"results_rulebook_v7_final_seed{seed}"
        rows, inp, note = nl_rows(R, E2, seed, ex)
        res = report(f"FINAL untouched NL 40 (seed {seed}), code {code_hash}", rows, out, note)
        print(" RF exclusion check", note)
        figure(rows, inp, out / "cases.png", f"Rulebook v7 on untouched NL sample (seed {seed})")
        json.dump(dict(code_sha256_12=code_hash, seed=seed, excluded=[str(p) for p in ex]), open(out / "freeze.json", "w"), indent=1)
