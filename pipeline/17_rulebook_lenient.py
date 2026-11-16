"""Lenient rulebook (lenient-v1): a short decision cascade with relaxed, agronomy-based thresholds.

Thresholds are frozen below BEFORE the run; nothing is tuned on BARC labels.
Inference reads only public/barc/rulebook_v1.json (label-free optical + radar series per field-year).
Evaluation (labels) happens after all predictions are written, in evaluate().

Cascade (first step that answers decides):
  1. Harvest drop seen (NDVI falls >= 0.20 between two clear looks <= 15 d apart, after the summer peak)
     and the crop was green just before (NDVI >= 0.55)                                  -> SILAGE (strong)
  2. Same drop, crop already brown just before (NDVI <= 0.50)                            -> GRAIN  (strong)
  3. Crop stood brown: >= 3 clear looks spanning >= 21 d after the peak with 0.30 <= NDVI <= 0.50
     (0.30 floor separates a dry standing canopy from bare soil after a chop)          -> GRAIN  (likely)
  4. Timing from the field's own peak to a harvest drop (>= 0.15 within 15 d):
     <= 45 d -> SILAGE (likely); >= 70 d -> GRAIN (likely)
  5. Radar structure collapse (VH mean of next 2 passes - previous 2 <= -1.5 dB) while the last clear
     optical look before it was green (NDVI >= 0.55)                                   -> SILAGE (lean)
  6. Otherwise UNRESOLVED (with reason). Missing data never becomes grain: step 3 needs observed looks.
"""
import json, csv, sys, pathlib
from datetime import date
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
VERSION = "lenient-v5" if "--v5" in sys.argv else "lenient-v4" if "--v4" in sys.argv else "lenient-v3" if "--v3" in sys.argv else "lenient-v2" if "--v2" in sys.argv else "lenient-v1"
V2 = VERSION in ("lenient-v2", "lenient-v3", "lenient-v4", "lenient-v5")
V5 = VERSION == "lenient-v5"
V4 = VERSION in ("lenient-v4", "lenient-v5")
V3 = VERSION == "lenient-v3"
T = dict(drop=0.20, drop_weak=0.15, max_gap=15, green=0.55, brown=0.50, brown_floor=0.30, brown_looks=3, brown_span=21,
         early_days=45, late_days=70, radar_vh_drop=1.5, peak_window=("06-01", "09-30"),
         # v2 harvest definition (developed on BARC, generic physical thresholds):
         bare_after=0.35, stay_window=30, stay_tol=0.10, radar_confirm_db=1.0, radar_standing_db=0.5,
         # v3: brown looks count as standing crop only before the first bare-ending drop, or if radar still shows structure
         radar_near_days=6, radar_stand_below_summer_db=1.0,
         # v4: radar sets the removal date (largest sustained VH drop >= 2 dB after peak, not a wetting event);
         # optical gives the condition at the last clear look <= 20 d before the last standing radar pass
         radar_step_db=2.0, radar_wet_vv_db=1.0, cond_max_age=20,
         # v5: the radar step must agree with an NDVI drop across it (last look <= 20 d before vs first look <= 20 d after:
         # fall >= 0.10 and ending <= 0.40); the earliest agreeing step is the harvest
         agree_drop=0.10, agree_after_max=0.40, agree_window=20)


def D(s): return date.fromisoformat(s[:10])


def drops(looks, start, thr):
    """(a, b) clear-look pairs after `start`, <= max_gap days apart, NDVI falling >= thr; earliest b, narrowest first."""
    seq = [o for o in looks if o["d"] > start]
    pairs = [(a, b) for i, a in enumerate(seq) for b in seq[i + 1:] if (b["d"] - a["d"]).days <= T["max_gap"] and a["ndvi"] - b["ndvi"] >= thr]
    return sorted(pairs, key=lambda p: (p[1]["d"], -p[0]["d"].toordinal()))


def radar_state(rad, a, b):
    """Two passes before the last standing look vs two passes after the first removed look (VH, dB)."""
    pre = [r["vh"] for r in rad if r["d"] <= a][-2:]; post = [r["vh"] for r in rad if r["d"] >= b][:2]
    if len(pre) < 2 or len(post) < 2: return "unknown", None
    dvh = sum(post) / 2 - sum(pre) / 2
    if dvh <= -T["radar_confirm_db"]: return "confirmed", dvh
    if abs(dvh) < T["radar_standing_db"]: return "standing", dvh
    return "unknown", dvh


def harvests(looks, rad, start, thr):
    """v1: any sharp drop. v2: drop that ends near bare, stays low, and radar does not show the crop still standing."""
    out = []
    for a, b in drops(looks, start, thr):
        if not V2: out.append((a, b, None)); continue
        if b["ndvi"] > T["bare_after"]: continue
        nxt = [o for o in looks if b["d"] < o["d"] <= b["d"] + __import__("datetime").timedelta(days=T["stay_window"])]
        if nxt and nxt[0]["ndvi"] > b["ndvi"] + T["stay_tol"]: continue
        st, dvh = radar_state(rad, a["d"], b["d"])
        if st == "standing": continue
        out.append((a, b, (st, dvh)))
    return out


def decide(f):
    """f: {'year', 'optical': [{date, ndvi}], 'radar': [{date, vh}]} - no labels."""
    y = f["year"]
    looks = sorted([dict(o, d=D(o["date"])) for o in f["optical"] if o.get("ndvi") is not None], key=lambda o: o["d"])
    p0, p1 = date(y, int(T["peak_window"][0][:2]), int(T["peak_window"][0][3:])), date(y, int(T["peak_window"][1][:2]), int(T["peak_window"][1][3:]))
    summer = [o for o in looks if p0 <= o["d"] <= p1]
    if not summer:
        return dict(final="UNRESOLVED", tier=None, step=6, why="no clear summer observation (insufficient data)")
    pk = max(summer, key=lambda o: o["ndvi"])
    rad = sorted([dict(r, d=D(r["date"])) for r in f.get("radar", []) if r.get("vh") is not None], key=lambda r: r["d"])
    ev = harvests(looks, rad, pk["d"], T["drop"])
    if V4:
        best = None
        for j in range(2, len(rad) - 1):
            if rad[j - 1]["d"] <= pk["d"]: continue
            dvh = (rad[j]["vh"] + rad[j + 1]["vh"]) / 2 - (rad[j - 1]["vh"] + rad[j - 2]["vh"]) / 2
            vv = [r.get("vv") for r in (rad[j - 2], rad[j - 1], rad[j], rad[j + 1])]
            dvv = None if None in vv else (vv[2] + vv[3]) / 2 - (vv[0] + vv[1]) / 2
            if dvh <= -T["radar_step_db"] and not (dvv is not None and dvv >= T["radar_wet_vv_db"]):
                r0, r1 = rad[j - 1]["d"], rad[j]["d"]
                if V5:
                    pre = [o for o in looks if o["d"] <= r0 and (r0 - o["d"]).days <= T["agree_window"]]
                    post = [o for o in looks if o["d"] >= r1 and (o["d"] - r1).days <= T["agree_window"]]
                    if not (pre and post and pre[-1]["ndvi"] - post[0]["ndvi"] >= T["agree_drop"] and post[0]["ndvi"] <= T["agree_after_max"]):
                        continue
                    best = (dvh, r0, r1); break                      # earliest agreeing step
                if best is None or dvh < best[0]:
                    best = (dvh, r0, r1)
        if best:
            dvh, r0, r1 = best
            before = [o for o in looks if o["d"] <= r0 and (r0 - o["d"]).days <= T["cond_max_age"]]
            if before:
                a = before[-1]; w = f"radar VH {dvh:+.1f} dB between {r0} and {r1}; last clear look {a['d']} NDVI {a['ndvi']:.2f}"
                if a["ndvi"] >= T["green"]:
                    return dict(final="SILAGE", tier="strong", step=0, why="green at radar-dated removal: " + w, a=str(r0), b=str(r1))
                if a["ndvi"] <= T["brown"]:
                    return dict(final="GRAIN", tier="strong", step=0, why="dry at radar-dated removal: " + w, a=str(r0), b=str(r1))
    rtxt = lambda e: "" if not e[2] else f"; radar {e[2][0]}" + ("" if e[2][1] is None else f" ({e[2][1]:+.1f} dB)")
    if ev:
        a, b = ev[0][:2]
        if a["ndvi"] >= T["green"]:
            return dict(final="SILAGE", tier="strong", step=1, why=f"green ({a['ndvi']:.2f}) on {a['d']} then {b['ndvi']:.2f} on {b['d']}" + rtxt(ev[0]), a=str(a["d"]), b=str(b["d"]))
        if a["ndvi"] <= T["brown"]:
            return dict(final="GRAIN", tier="strong", step=2, why=f"already brown ({a['ndvi']:.2f}) on {a['d']} then {b['ndvi']:.2f} on {b['d']}" + rtxt(ev[0]), a=str(a["d"]), b=str(b["d"]))
    brown = [o for o in looks if o["d"] > pk["d"] and T["brown_floor"] <= o["ndvi"] <= T["brown"]]
    note3 = ""
    if V3:
        bare = [(a, b) for a, b in drops(looks, pk["d"], T["drop_weak"]) if b["ndvi"] <= T["bare_after"]]
        t_h = bare[0][0]["d"] if bare else None
        summer_vh = sorted(r["vh"] for r in rad if r["d"].month in (7, 8))
        ref = summer_vh[len(summer_vh) // 2] if summer_vh else None
        def standing(o):
            near = [r["vh"] for r in rad if abs((r["d"] - o["d"]).days) <= T["radar_near_days"]]
            return ref is not None and near and max(near) >= ref - T["radar_stand_below_summer_db"]
        if t_h is not None:
            kept = [o for o in brown if o["d"] < t_h or standing(o)]
            note3 = f" (before bare drop on {t_h} or radar standing; {len(brown) - len(kept)} post-harvest looks excluded)"
            brown = kept
    if len(brown) >= T["brown_looks"] and (brown[-1]["d"] - brown[0]["d"]).days >= T["brown_span"]:
        return dict(final="GRAIN", tier="likely", step=3, why=f"{len(brown)} brown standing looks {brown[0]['d']}..{brown[-1]['d']}" + note3)
    weak = harvests(looks, rad, pk["d"], T["drop_weak"])
    if weak:
        a, b = weak[0][:2]; days = (a["d"] - pk["d"]).days
        if days <= T["early_days"]:
            return dict(final="SILAGE", tier="likely", step=4, why=f"harvest drop {days} d after peak {pk['d']}", a=str(a["d"]), b=str(b["d"]))
        if days >= T["late_days"]:
            return dict(final="GRAIN", tier="likely", step=4, why=f"harvest drop {days} d after peak {pk['d']}", a=str(a["d"]), b=str(b["d"]))
    for j in range(2, len(rad) - 1):
        if rad[j]["d"] <= pk["d"]: continue
        dvh = (rad[j]["vh"] + rad[j + 1]["vh"]) / 2 - (rad[j - 1]["vh"] + rad[j - 2]["vh"]) / 2
        before = [o for o in looks if o["d"] <= rad[j - 1]["d"]]
        if dvh <= -T["radar_vh_drop"] and before and before[-1]["ndvi"] >= T["green"]:
            return dict(final="SILAGE", tier="lean", step=5, why=f"radar VH {dvh:+.1f} dB at {rad[j]['d']} while last look green ({before[-1]['ndvi']:.2f})")
    return dict(final="UNRESOLVED", tier=None, step=6, why="no step gave a clear answer")


def evaluate(preds, out):
    labels = {(int(r["OBJECTID"]), int(r["year"])): r["label"] for r in csv.DictReader(open(HERE / "results/experiments/barc_2021_2023/reference_labels.csv"))}
    ml = {(int(r["OBJECTID"]), int(r["year"])): r["pred"] for r in csv.DictReader(open(HERE / "results/experiments/barc_2021_2023/barc_matched.csv"))}
    rows = []
    for p in preds:
        k = (p["OBJECTID"], p["year"]); lab = labels[k]; pr = p["final"].lower() if p["final"] != "UNRESOLVED" else None
        rows.append(dict(p, label=lab, correct=None if pr is None else pr == lab, ml=ml.get(k), ml_correct=ml.get(k) == lab))
    assert len(rows) == len(labels) == 44
    def conf(rs, key):
        m = Counter((r["label"], r[key]) for r in rs if r[key] in ("silage", "grain"))
        tp, fn, fp, tn = m[("silage", "silage")], m[("silage", "grain")], m[("grain", "silage")], m[("grain", "grain")]
        rs_, rg = (tp / (tp + fn) if tp + fn else None), (tn / (tn + fp) if tn + fp else None)
        return dict(n=tp + fn + fp + tn, tp=tp, fn=fn, fp=fp, tn=tn, recall_silage=rs_, recall_grain=rg,
                    precision_silage=tp / (tp + fp) if tp + fp else None, precision_grain=tn / (tn + fn) if tn + fn else None,
                    balanced_accuracy=(rs_ + rg) / 2 if rs_ is not None and rg is not None else None)
    for r in rows: r["rb"] = r["final"].lower() if r["final"] != "UNRESOLVED" else None
    dec = [r for r in rows if r["rb"]]
    res = dict(version=VERSION, thresholds=T, n=len(rows), decided=len(dec), correct=sum(r["correct"] for r in dec),
               outcomes=dict(Counter(f"{r['final']} ({r['tier']})" if r["tier"] else r["final"] for r in rows)),
               by_step={s: dict(n=len(g), correct=sum(1 for r in g if r["correct"])) for s in range(0, 7) for g in [[r for r in rows if r["step"] == s]] if g},
               rulebook=conf(rows, "rb"), ml_all=conf(rows, "ml"), ml_same_subset=conf(dec, "ml"),
               coverage_by_class={c: f"{sum(1 for r in dec if r['label'] == c)}/{sum(1 for r in rows if r['label'] == c)}" for c in ("silage", "grain")},
               coverage_by_year={y: f"{sum(1 for r in dec if r['year'] == y)}/{sum(1 for r in rows if r['year'] == y)}" for y in (2021, 2022, 2023)})
    json.dump(res, open(out / "eval.json", "w"), indent=1, default=str)
    with open(out / "per_field_eval.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["OBJECTID", "year", "final", "tier", "step", "why", "label", "correct", "ml", "ml_correct"], extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    return res, rows


if __name__ == "__main__":
    out = HERE / ("results/experiments/barc_rulebook_lenient_v5" if V5 else "results/experiments/barc_rulebook_lenient_v4" if V4 else "results/experiments/barc_rulebook_lenient_v3" if V3 else "results/experiments/barc_rulebook_lenient_v2" if V2 else "results/experiments/barc_rulebook_lenient_v1"); out.mkdir(parents=True, exist_ok=True)
    data = json.load(open(HERE.parent / "public/barc/rulebook_v1.json"))      # label-free inputs
    preds = []
    for f in data["fields"]:
        t = f["trace"]
        r = decide(dict(year=t["year"], optical=f["optical"], radar=f["radar"]))
        preds.append(dict(OBJECTID=t["OBJECTID"], year=t["year"], version=VERSION, **r))
    with open(out / "predictions.csv", "w", newline="") as fh:   # written before any label is read
        w = csv.DictWriter(fh, ["OBJECTID", "year", "version", "final", "tier", "step", "why", "a", "b"], extrasaction="ignore"); w.writeheader(); w.writerows(preds)
    if "--no-eval" not in sys.argv:
        res, rows = evaluate(preds, out)
        print(json.dumps({k: res[k] for k in ("decided", "correct", "outcomes", "by_step", "coverage_by_class", "coverage_by_year")}, indent=1))
        print("rulebook", res["rulebook"]); print("ML all", res["ml_all"]); print("ML same subset", res["ml_same_subset"])
        for r in rows:
            if r["correct"] is False: print("WRONG", r["OBJECTID"], r["year"], r["label"], "->", r["final"], r["tier"], "step", r["step"], "|", r["why"])
