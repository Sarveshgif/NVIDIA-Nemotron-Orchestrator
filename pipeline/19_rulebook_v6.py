"""Rulebook v6 (the user's proposed "v5" spec): establish a credible harvest event, then classify from the
condition before it. Outputs SILAGE / GRAIN / REVIEW / INSUFFICIENT DATA with an evidence tier (strong / likely).
Tiers describe satellite evidence strength, not calibrated probabilities. Thresholds are provisional and frozen here.

Run on the lab machine:
  HTH_BARC=<data_barc> HTH_DATA=<data> E2_RESULTS=<E2 results> RB_SEED=99 RB_EXCLUDE=a.csv,b.csv python 19_rulebook_v6.py
Labels are used only to draw the Dutch sample and to score; decide() receives dates, NDVI, clear fraction,
zone NDVI and radar VV/VH only.
"""
import os, sys, json, csv, pathlib, statistics
from datetime import date, timedelta
from collections import Counter

VERSION = "rulebook-v6"
T = dict(
    season=("04-15", "10-15"), peak_min=0.60, greenup_below_peak=0.30, spring_window=("03-01", "05-15"), trough=0.35,
    drop=0.20, max_gap=15, after_max=0.35,                                  # optical candidate
    radar_drop=1.5, radar_side_days=24, wet_vv=1.0,                         # radar candidate
    persist_days=20, persist_max=0.40,                                      # persistent optical event
    close_days=10, weak_days=20,                                            # preharvest observation age
    green=0.55, dry=0.50, decline_below_peak=0.15, decline_looks=3, decline_span=21,
    adequate_looks=6, adequate_max_gap=30, adequate_until="11-30",
    regrow=0.55, zone_min_looks=6, zone_tol=10, questionable_clear=0.80,
)
md = lambda y, s: date(y, int(s[:2]), int(s[3:]))
D = lambda s: date.fromisoformat(str(s)[:10])
ok = lambda v: v is not None and v == v


def overlaps(a0, a1, b0, b1, tol):
    return a0 - timedelta(days=tol) <= b1 and b0 <= a1 + timedelta(days=tol)


def optical_events(looks, start):
    """Non-overlapping optical candidates after `start`: drop >= 0.20 within <= 15 d to NDVI <= 0.35 (narrowest interval)."""
    seq = [o for o in looks if o["d"] >= start]
    out, last_b = [], None
    for j, b in enumerate(seq):
        if b["ndvi"] > T["after_max"]: continue
        a = next((seq[i] for i in range(j - 1, -1, -1) if (b["d"] - seq[i]["d"]).days <= T["max_gap"]
                  and seq[i]["ndvi"] - b["ndvi"] >= T["drop"]), None)
        if a is None or (last_b is not None and a["d"] < last_b): continue
        nxt = [o for o in looks if b["d"] < o["d"] <= b["d"] + timedelta(days=T["persist_days"])]
        out.append(dict(a=a, b=b, drop=a["ndvi"] - b["ndvi"], persistent=bool(nxt) and nxt[0]["ndvi"] <= T["persist_max"],
                        follow=nxt[0] if nxt else None))
        last_b = b["d"]
    return out


def radar_events(rad, start):
    out = []
    for j, r in enumerate(rad):
        if r["d"] < start: continue
        pre = [x for x in rad[:j] if (r["d"] - x["d"]).days <= T["radar_side_days"]][-2:]
        post = [x for x in rad[j:] if (x["d"] - r["d"]).days <= T["radar_side_days"]][:2]
        if len(pre) < 2 or len(post) < 2: continue
        dvh = sum(x["vh"] for x in post) / 2 - sum(x["vh"] for x in pre) / 2
        dvv = sum(x["vv"] for x in post) / 2 - sum(x["vv"] for x in pre) / 2
        if dvh <= -T["radar_drop"]:
            e = dict(start=pre[-1]["d"], end=r["d"], dvh=dvh, dvv=dvv, moisture=dvv >= T["wet_vv"], j=j)
            if out and out[-1]["j"] == j - 1:                       # consecutive passes = one event; keep the deepest
                if dvh < out[-1]["dvh"]: out[-1] = e
                else: out[-1]["j"] = j
            else: out.append(e)
    return out


def decide(f):
    y = f["year"]
    looks = sorted([dict(o, d=D(o["date"])) for o in f["optical"] if ok(o.get("ndvi"))], key=lambda o: o["d"])
    rad = sorted([dict(r, d=D(r["date"])) for r in f.get("radar", []) if ok(r.get("vh")) and ok(r.get("vv"))], key=lambda r: r["d"])
    ev = {}
    res = lambda final, tier, why, **k: dict(final=final, tier=tier, why=why, **ev, **k)
    # ---- step 0: crop cycle ----
    s0, s1 = md(y, T["season"][0]), md(y, T["season"][1])
    season = [o for o in looks if s0 <= o["d"] <= s1]
    if not season: return res("INSUFFICIENT DATA", None, "no clear observation in the corn season window")
    pk = max(season, key=lambda o: o["ndvi"]); ev.update(peak=str(pk["d"]), peak_ndvi=round(pk["ndvi"], 3))
    if pk["ndvi"] < T["peak_min"]: return res("REVIEW", None, f"uncertain corn identity (season peak NDVI {pk['ndvi']:.2f})")
    before = [o for o in looks if md(y, T["spring_window"][0]) <= o["d"] < pk["d"]]
    if not before: return res("INSUFFICIENT DATA", None, "green-up not observed (no clear look before the peak)")
    if not any(o["ndvi"] <= pk["ndvi"] - T["greenup_below_peak"] for o in before):
        return res("REVIEW", None, "no green-up observed before the peak (cycle not identifiable)")
    sp = [o for o in before if o["d"] <= md(y, T["spring_window"][1]) and o["ndvi"] >= T["peak_min"]]
    if sp:
        hi = max(sp, key=lambda o: o["ndvi"])
        if not any(hi["d"] < o["d"] < pk["d"] and o["ndvi"] <= T["trough"] for o in looks):
            return res("REVIEW", None, f"spring green cycle ({hi['d']}) not separable from the corn cycle")
        ev["earlier_cycle"] = str(hi["d"])
    post = [o for o in looks if o["d"] > pk["d"]]
    until = [pk] + [o for o in post if o["d"] <= md(y, T["adequate_until"])]
    gaps = [(b["d"] - a["d"]).days for a, b in zip(until, until[1:])]
    adequate = len(post) >= T["adequate_looks"] and (max(gaps) if gaps else 999) <= T["adequate_max_gap"]
    ev.update(post_peak_looks=len(post), max_gap_to_nov30=max(gaps) if gaps else None)
    # ---- step 1: candidates ----
    oe = optical_events(looks, pk["d"])
    re_ = radar_events(rad, pk["d"])
    tol = int(statistics.median([(b["d"] - a["d"]).days for a, b in zip(rad, rad[1:])])) if len(rad) > 2 else 12
    ev.update(optical_candidates=[(str(e["a"]["d"]), str(e["b"]["d"]), round(e["drop"], 2), e["persistent"]) for e in oe],
              radar_candidates=[(str(e["start"]), str(e["end"]), round(e["dvh"], 1), e["moisture"]) for e in re_], radar_tol_days=tol)
    # ---- step 2: credibility ----
    for e in oe:
        e["radar"] = next((r for r in re_ if overlaps(e["a"]["d"], e["b"]["d"], r["start"], r["end"], tol)), None)
        e["kind"] = "corroborated" if e["radar"] else "persistent" if e["persistent"] else "unconfirmed"
    cred = [e for e in oe if e["kind"] != "unconfirmed"]
    if not cred:
        if any(not any(overlaps(e["a"]["d"], e["b"]["d"], r["start"], r["end"], tol) for e in oe) for r in re_):
            return res("REVIEW", None, "radar change without a credible optical harvest (possible disturbance)")
        if adequate: return res("REVIEW", None, "observations adequate but no credible harvest event" + (" (unconfirmed optical drop only)" if oe else ""))
        return res("INSUFFICIENT DATA", None, "no credible harvest event and observations too sparse")
    h = cred[0]
    if len(cred) > 1:
        between = [o for o in looks if h["b"]["d"] < o["d"] < cred[1]["a"]["d"]]
        if not any(o["ndvi"] >= T["regrow"] for o in between):
            return res("REVIEW", None, "several incompatible harvest candidates")
    a, b = h["a"], h["b"]
    ev.update(harvest_interval=(str(a["d"]), str(b["d"])), harvest_kind=h["kind"], optical_drop=round(h["drop"], 2),
              radar_change=None if not h["radar"] else round(h["radar"]["dvh"], 1), preharvest_ndvi=round(a["ndvi"], 3),
              preharvest_date=str(a["d"]), days_peak_to_harvest=(a["d"] - pk["d"]).days)
    # ---- step 3: condition before harvest ----
    age = (b["d"] - a["d"]).days
    closeness = "close" if age <= T["close_days"] else "weaker" if age <= T["weak_days"] else "inadequate"
    dec = [o for o in looks if pk["d"] < o["d"] <= a["d"] and o["ndvi"] <= pk["ndvi"] - T["decline_below_peak"]]
    drying_traj = len(dec) >= T["decline_looks"] and (dec[-1]["d"] - dec[0]["d"]).days >= T["decline_span"]
    if a["ndvi"] >= T["green"]: cond = "green"
    elif a["ndvi"] <= T["dry"] and drying_traj: cond = "drying"
    else: cond = "ambiguous"
    ev.update(preharvest_age_days=age, closeness=closeness, condition=cond)
    # ---- step 5: contradictions ----
    zone_iv = {}
    for z in ("z1", "z2", "z3", "z4"):
        zl = [dict(o, ndvi=o[z]) for o in looks if ok(o.get(z))]
        if len(zl) < T["zone_min_looks"]: continue
        ze = [e for e in optical_events(zl, pk["d"]) if e["persistent"]]
        if ze: zone_iv[z] = (ze[0]["a"]["d"], ze[0]["b"]["d"])
    iv = list(zone_iv.values())
    if any(not overlaps(p[0], p[1], q[0], q[1], T["zone_tol"]) for i, p in enumerate(iv) for q in iv[i + 1:]):
        return res("REVIEW", None, "field zones follow different harvest trajectories", zones={k: (str(v[0]), str(v[1])) for k, v in zone_iv.items()})
    early = [r for r in re_ if not r["moisture"] and r["end"] < a["d"] - timedelta(days=tol)
             and all(o["ndvi"] >= T["green"] for o in looks if r["end"] <= o["d"] <= a["d"])]
    if early: return res("REVIEW", None, f"radar event {early[0]['start']}..{early[0]['end']} before the optical harvest while the canopy stayed green")
    if h["kind"] == "persistent" and min(a.get("clear_frac") or 1, b.get("clear_frac") or 1) < T["questionable_clear"]:
        return res("REVIEW", None, "optical-only harvest rests on a partly cloudy image")
    # ---- step 4: classification ----
    if closeness == "inadequate" or cond == "ambiguous":
        return res("REVIEW", None, f"credible harvest but preharvest condition {cond} ({closeness} observation, NDVI {a['ndvi']:.2f})")
    cls = "SILAGE" if cond == "green" else "GRAIN"
    if h["kind"] == "corroborated":
        tier = "strong" if closeness == "close" else "likely"
    else:
        tier = "likely"
    return res(cls, tier, f"{h['kind']} harvest {a['d']}..{b['d']}; {cond} before ({a['ndvi']:.2f}, {closeness})")


# ------------------------------------------------------------------ data + evaluation
def metrics(rows, key):
    d = [r for r in rows if r[key] in ("silage", "grain")]
    c = Counter((r["label"], r[key]) for r in d)
    tp, fn, fp, tn = c[("silage", "silage")], c[("silage", "grain")], c[("grain", "silage")], c[("grain", "grain")]
    f = lambda a, b: round(a / b, 3) if b else None
    rs, rg = f(tp, tp + fn), f(tn, tn + fp)
    return dict(n=len(d), tp=tp, fn=fn, fp=fp, tn=tn, precision_silage=f(tp, tp + fp), recall_silage=rs, precision_grain=f(tn, tn + fn),
                recall_grain=rg, balanced_accuracy=None if rs is None or rg is None else round((rs + rg) / 2, 3))


def evaluate(name, rows, out):
    for r in rows: r["rb"] = r["final"].lower() if r["final"] in ("SILAGE", "GRAIN") else None
    dec = [r for r in rows if r["rb"]]
    res = dict(dataset=name, version=VERSION, thresholds=T, n=len(rows), labels=dict(Counter(r["label"] for r in rows)),
               outcomes=dict(Counter(r["final"] + (f" ({r['tier']})" if r["tier"] else "") for r in rows)),
               coverage=f"{len(dec)}/{len(rows)}", coverage_by_class={c: f"{sum(1 for r in dec if r['label'] == c)}/{sum(1 for r in rows if r['label'] == c)}" for c in ("silage", "grain")},
               correct=sum(r["rb"] == r["label"] for r in dec), rulebook=metrics(rows, "rb"),
               by_tier={t: metrics([r for r in rows if r["tier"] == t], "rb") for t in ("strong", "likely")},
               ml_all=metrics(rows, "ml"), ml_same_subset=metrics(dec, "ml"),
               review_reasons=dict(Counter(r["why"].split(" (")[0] for r in rows if r["final"] in ("REVIEW", "INSUFFICIENT DATA"))))
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "eval.json", "w"), indent=1, default=str)
    keys = ["id", "year", "label", "final", "tier", "why", "ml", "harvest_interval", "harvest_kind", "preharvest_ndvi", "preharvest_date",
            "closeness", "condition", "optical_drop", "radar_change", "days_peak_to_harvest", "peak", "peak_ndvi"]
    with open(out / "per_field.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"\n######## {name}: outcomes {res['outcomes']} | coverage {res['coverage']} {res['coverage_by_class']} | correct {res['correct']}")
    print(" rulebook", res["rulebook"]); print(" by tier", res["by_tier"]); print(" ML all", res["ml_all"]); print(" ML same subset", res["ml_same_subset"])
    print(" review/insufficient reasons", res["review_reasons"])
    for r in rows:
        if r["rb"] and r["rb"] != r["label"]: print("  WRONG", r["id"], r["year"], r["label"], "->", r["final"], r["tier"], "|", r["why"])
    return res


if __name__ == "__main__":
    import pandas as pd
    HERE = pathlib.Path(__file__).resolve().parent
    # ---- BARC (reference only: ideas partly came from studying BARC) ----
    B = pathlib.Path(os.environ["HTH_BARC"])
    km = {r["key"]: r for r in csv.DictReader(open(B / "rulebook/key_map.csv"))}
    preds = []
    for y in (2021, 2022, 2023):
        for fy in json.load(open(B / f"rulebook/inputs_{y}.json")):
            d = decide(dict(year=y, optical=fy["optical"], radar=fy["radar"]["series"]))
            preds.append(dict(id=int(km[fy["key"]]["OBJECTID"]), year=y, **d))
    labels = {(int(r["OBJECTID"]), int(r["year"])): r["label"] for r in csv.DictReader(open(B / "reference_labels.csv"))}
    ml = {(int(r["OBJECTID"]), int(r["year"])): r["pred"] for r in csv.DictReader(open(B / "results/barc_matched.csv"))}
    for p in preds: p["label"] = labels[(p["id"], p["year"])]; p["ml"] = ml[(p["id"], p["year"])]
    assert len(preds) == 44
    evaluate("BARC 2021-2023 (reference, not a clean test)", preds, B / "results/rulebook_v6")
    # ---- fresh Dutch sample (clean test) ----
    R = pathlib.Path(os.environ["HTH_DATA"]); seed = int(os.environ.get("RB_SEED", 99))
    spl = pd.read_csv(R / "features/splits.csv").merge(pd.read_parquet(R / "features/labels.parquet")[["field_id", "label"]])
    test = spl[spl.split == "test"]
    for ex in filter(None, os.environ.get("RB_EXCLUDE", "").split(",")):
        test = test[~test.field_id.isin(pd.read_csv(ex).field_id)]
    sample = pd.concat([g.sample(20, random_state=seed) for _, g in test.groupby("label")])
    lab = dict(zip(sample.field_id, sample.label))
    mlnl = pd.read_parquet(pathlib.Path(os.environ["E2_RESULTS"]) / "predictions_all.parquet"); mlnl = dict(zip(mlnl[mlnl.year == 2025].field_id, mlnl[mlnl.year == 2025].pred))
    preds = []
    for b, g in sample.groupby("block"):
        obs = pd.read_parquet(R / "s2_20m" / f"obs20_{b}.parquet"); s1 = pd.read_parquet(R / "s1" / f"s1_{b}.parquet")
        orbit = int(s1.groupby("rel_orbit").size().idxmax()); s1 = s1[s1.rel_orbit == orbit]   # one relative orbit (= one pass direction)
        for fid in g.field_id:
            o = obs[(obs.field_id == fid) & obs.ndvi.notna()].sort_values("date")
            r = s1[s1.field_id == fid].sort_values("date")
            f = dict(year=2025, optical=[dict(date=str(x.date)[:10], ndvi=float(x.ndvi), clear_frac=float(x.clear_frac),
                                              **{f"z{k}": (None if x[f"ndvi_z{k}"] != x[f"ndvi_z{k}"] else float(x[f"ndvi_z{k}"])) for k in range(1, 5)})
                                         for _, x in o.iterrows()],
                     radar=[dict(date=str(d)[:10], vv=float(v), vh=float(h)) for d, v, h in zip(r.date, r.vv_db, r.vh_db)])
            preds.append(dict(id=fid, year=2025, **decide(f)))
    pd.DataFrame(preds).to_csv(R / f"results_rulebook_v6_seed{seed}_predictions.csv", index=False)   # before scoring
    for p in preds: p["label"] = lab[p["id"]]; p["ml"] = mlnl.get(p["id"])
    evaluate(f"NL fresh 40 (seed {seed}, clean test)", preds, R / f"results_rulebook_v6_seed{seed}")
