"""Rulebook v1: deterministic, auditable silage / grain decision workflow for one field-year (crop cycle).

Pure Python (no numpy/pandas) so it can be unit-tested anywhere. The engine never receives reference labels:
`decide(fy, weather, cfg, registry)` takes only the field-year input built by 14_rulebook_inputs.py.

Every check returns:
  {"state": supported|contradicted|unknown|not_applicable, "outcome": <label>, "values": {...}, "dates": {...},
   "sources": [...], "reasons": [...], "missing": [...]}
Final classes: CONFIRMED SILAGE, LIKELY SILAGE, CONFIRMED GRAIN, LIKELY GRAIN, MIXED, OTHER, NOT YET HARVESTED,
UNRESOLVED; plus a separate review flag and reason codes.
"""
from __future__ import annotations
import math
from datetime import date, timedelta

SUP, CON, UNK, NA = "supported", "contradicted", "unknown", "not_applicable"
FINAL = ("CONFIRMED SILAGE", "LIKELY SILAGE", "CONFIRMED GRAIN", "LIKELY GRAIN", "MIXED", "OTHER", "NOT YET HARVESTED", "UNRESOLVED")


def D(s): return s if isinstance(s, date) else date.fromisoformat(str(s)[:10])
def iso(d): return None if d is None else d.isoformat()
def md(year, s): return date(year, int(s[:2]), int(s[3:5]))
def ok(v): return v is not None and isinstance(v, (int, float)) and math.isfinite(v)
def r3(v): return None if not ok(v) else round(float(v), 3)


def check(state, outcome=None, values=None, dates=None, sources=None, reasons=None, missing=None):
    return {"state": state, "outcome": outcome, "values": values or {}, "dates": dates or {},
            "sources": sources or [], "reasons": reasons or [], "missing": missing or []}


# ------------------------------------------------------------------ helpers
def clear_looks(opt):
    """Clear optical looks: NDVI present (extraction already applied the >=60 % clear / >=3 valid px rule)."""
    out = [dict(o, date=D(o["date"])) for o in opt if ok(o.get("ndvi"))]
    return sorted(out, key=lambda o: o["date"])


class GDD:
    def __init__(self, weather):
        d = weather["daily"]
        self.day = {D(t): (tx, tn, p) for t, tx, tn, p in zip(d["date"], d["tmax_f"], d["tmin_f"], d["precip_mm"])}

    def gdd(self, day):
        v = self.day.get(day)
        if v is None or not ok(v[0]) or not ok(v[1]): return None
        tx, tn = min(max(v[0], 50.0), 86.0), min(max(v[1], 50.0), 86.0)
        return (tx + tn) / 2 - 50.0

    def cum(self, d0, d1):
        """GDD accumulated over days in (d0, d1]. None if any day is missing."""
        s, d = 0.0, d0 + timedelta(days=1)
        while d <= d1:
            g = self.gdd(d)
            if g is None: return None
            s += g; d += timedelta(days=1)
        return s

    def back_to(self, d_end, target):
        """Latest date P with cum(P, d_end) >= target, or None if weather does not reach back far enough."""
        s, d = 0.0, d_end
        while True:
            g = self.gdd(d)
            if g is None: return None
            s += g
            if s >= target: return d - timedelta(days=1)
            d -= timedelta(days=1)

    def rain(self, d, days=2):
        vals = [self.day.get(d - timedelta(days=k), (None, None, None))[2] for k in range(1, days + 1)]
        return None if any(not ok(v) for v in vals) else sum(vals)

    def min_tmin(self, d0, d1):
        v = [x[1] for k, x in self.day.items() if d0 <= k <= d1 and ok(x[1])]
        return min(v) if v else None

    def precip(self, d0, d1):
        v = [x[2] for k, x in self.day.items() if d0 <= k <= d1]
        return None if not v or any(not ok(x) for x in v) else sum(v)


# ------------------------------------------------------------------ step 1: identity and crop cycle
def step1_identity(fy, looks, cfg):
    y, ci, se = fy["year"], cfg["identity"], cfg["season"]
    s0, s1 = md(y, se["summer_peak_window"][0]), md(y, se["summer_peak_window"][1])
    summer = [o for o in looks if s0 <= o["date"] <= s1]
    cyc = {"peak": None, "spring_min": None, "detect": None, "detect_prev": None, "preceding_cover": None}
    crop_type = check(NA, "no crop-type record", reasons=["no_variety_or_crop_type_record"],
                      missing=["records identifying sweet corn / popcorn / other crop"])
    if not summer:
        return check(UNK, "IDENTITY UNKNOWN", reasons=["no_summer_observation"], missing=["clear optical looks Jun-Sep"]), crop_type, cyc
    pk = max(summer, key=lambda o: o["ndvi"]); cyc["peak"] = pk
    before = [o for o in looks if md(y, se["spring_window"][0]) <= o["date"] < pk["date"]]
    if before:
        sm = min(before, key=lambda o: o["ndvi"]); cyc["spring_min"] = sm
        cover = [o for o in before if o["date"] < sm["date"] and o["ndvi"] >= 0.5]
        if cover: cyc["preceding_cover"] = max(cover, key=lambda o: o["ndvi"])
        if sm["ndvi"] <= ci["spring_min_ndvi_max"]:
            after = [o for o in looks if sm["date"] < o["date"] <= pk["date"]]
            det = next((o for o in after if o["ndvi"] >= ci["canopy_detect_ndvi"]), None)
            if det:
                cyc["detect"] = det
                prev = [o for o in looks if o["date"] < det["date"]]
                cyc["detect_prev"] = prev[-1] if prev else None
    vals = {"summer_peak_ndvi": r3(pk["ndvi"]), "spring_min_ndvi": r3(cyc["spring_min"]["ndvi"]) if cyc["spring_min"] else None,
            "preceding_green_cover_ndvi": r3(cyc["preceding_cover"]["ndvi"]) if cyc["preceding_cover"] else None}
    dates = {"summer_peak": iso(pk["date"]), "spring_min": iso(cyc["spring_min"]["date"]) if cyc["spring_min"] else None,
             "canopy_detect": iso(cyc["detect"]["date"]) if cyc["detect"] else None}
    reasons = []
    if cyc["preceding_cover"]: reasons.append("spring_green_cover_before_corn_cycle")
    if not cyc["detect"]: reasons.append("cycle_start_not_observed")
    if pk["ndvi"] >= ci["corn_peak_ndvi_min"]:
        return check(SUP, "CORN-LIKE CYCLE", vals, dates, reasons=reasons), crop_type, cyc
    if pk["ndvi"] >= ci["uncertain_peak_ndvi_min"]:
        return check(UNK, "UNCERTAIN IDENTITY", vals, dates, reasons=reasons + ["uncertain_identity"]), crop_type, cyc
    return check(CON, "NOT CORN-LIKE", vals, dates, reasons=reasons + ["not_corn_like_cycle"]), crop_type, cyc


# ------------------------------------------------------------------ step 2: observation quality
def step2_quality(fy, looks, cfg):
    y, q, f = fy["year"], cfg["quality"], cfg["field"]
    jd = [o for o in looks if date(y, 6, 1) <= o["date"] <= date(y, 12, 31)]
    area_ok = ok(fy.get("interior_ha")) and fy["interior_ha"] >= f["min_interior_ha"]
    n_acq = len(fy.get("optical", []))
    vals = {"clear_looks_jun_dec": len(jd), "acquisitions": n_acq, "interior_ha": r3(fy.get("interior_ha")),
            "median_clear_frac_of_clear_looks": r3(sorted(o["clear_frac"] for o in jd)[len(jd) // 2]) if jd else None,
            "median_valid_px20": (sorted(o.get("n_valid_px") or 0 for o in jd)[len(jd) // 2]) if jd else None,
            "radar_orbit": fy["radar"].get("orbit"), "radar_field_coverage": r3(fy["radar"].get("coverage_frac")),
            "radar_dates": len(fy["radar"].get("series", []))}
    reasons, missing = [], []
    if not area_ok: reasons.append("insufficient_interior_area"); missing.append("interior area >= %.1f ha after %d m buffer" % (f["min_interior_ha"], f["inward_buffer_m"]))
    optical_ok = len(jd) >= q["min_clear_obs_jun_dec"]
    if not optical_ok: reasons.append("insufficient_optical_observations"); missing.append(">= %d clear looks Jun-Dec" % q["min_clear_obs_jun_dec"])
    radar_ok = bool(fy["radar"].get("full_cover")) and len(fy["radar"].get("series", [])) > 0
    if not radar_ok: reasons.append("radar_unavailable:" + (fy["radar"].get("reason") or "no_series")); missing.append("full-coverage consistent-orbit Sentinel-1")
    state = SUP if (area_ok and optical_ok) else (UNK if (area_ok and radar_ok) else CON)
    return check(state, "SUFFICIENT" if state == SUP else "LIMITED" if state == UNK else "INSUFFICIENT", vals, reasons=reasons, missing=missing,
                 sources=["Sentinel-2 L2A field means on native 20 m grid", "Sentinel-1 RTC field means"]), optical_ok, radar_ok, area_ok


# ------------------------------------------------------------------ step 3 / 11: documents
def step3_documents(fy, registry):
    items = [i for i in registry.get("items", []) if i.get("field_key") == fy["key"] and int(i.get("crop_year", -1)) == fy["year"]]
    actual = [i for i in items if i.get("actual_or_planned") == "actual" and i.get("url") and i.get("excerpt") and i.get("retrieved")]
    if not items:
        return check(UNK, "NO DIRECT RECORD", reasons=["no_field_linked_records_available"],
                     missing=["actual harvest log / delivery record / nutrient-management or dairy report linked to this field and year"],
                     sources=["rulebook/evidence_registry_v1.json"])
    claims = {i["claim"] for i in actual}
    src = [{"url": i["url"], "excerpt": i["excerpt"], "location": i.get("location"), "retrieved": i["retrieved"],
            "field_match": i.get("field_match"), "actual_or_planned": i["actual_or_planned"]} for i in items]
    if not actual: return check(UNK, "PLANNED/UNVERIFIED ONLY", sources=src, reasons=["records_planned_or_unverified"])
    if len(claims) > 1 and "mixed" not in claims: return check(CON, "CONFLICTING RECORDS", sources=src, reasons=["conflicting_direct_records"])
    c = claims.pop()
    return check(SUP, {"silage": "CONFIRMED SILAGE", "grain": "CONFIRMED GRAIN", "mixed": "MIXED"}[c], sources=src)


# ------------------------------------------------------------------ step 4: physical harvest
def optical_candidates(looks, start, cutoff, cfg):
    r = cfg["removal"]; out = []
    seq = [o for o in looks if start <= o["date"] <= cutoff]
    W = r.get("max_window_days", 0)
    pairs = [(i, j) for i in range(len(seq)) for j in range(i + 1, len(seq))
             if j == i + 1 or (seq[j]["date"] - seq[i]["date"]).days <= W]
    pairs.sort(key=lambda p: (seq[p[1]]["date"], -seq[p[0]]["date"].toordinal()))   # earliest removal, narrowest interval
    for i, j in pairs:
        a, b = seq[i], seq[j]
        drop = a["ndvi"] - b["ndvi"]
        if drop < r["min_ndvi_drop"] or b["ndvi"] > r["max_ndvi_after"]: continue
        gap = (b["date"] - a["date"]).days
        later = [o for o in looks if b["date"] < o["date"] <= b["date"] + timedelta(days=r["confirm_window_days"])]
        recov = [o for o in looks if b["date"] < o["date"] <= b["date"] + timedelta(days=r["recovery_window_days"])
                 and o["ndvi"] >= a["ndvi"] - r["recovery_margin"]]
        confirmed = bool(later) and all(o["ndvi"] <= b["ndvi"] + r["confirm_tol"] for o in later[:1]) and not recov
        fast = drop >= r["senescence_max_ndvi_rate_per_day"] * gap + r["rate_margin"]
        out.append(dict(a=a, b=b, drop=drop, gap=gap, confirmed=confirmed, fast=fast, recovery=bool(recov),
                        confirm_date=later[0]["date"] if later else None))
    return out


def radar_changes(series, year, cfg):
    rc = cfg["radar"]; W = timedelta(days=rc["window_days"])
    s = sorted([dict(x, date=D(x["date"])) for x in series if ok(x.get("vh")) and ok(x.get("vv"))], key=lambda x: x["date"])
    lo, hi = md(year, rc["search_window"][0]), md(year, rc["search_window"][1])
    out = []
    for j in range(1, len(s)):
        t = s[j]["date"]
        if not (lo <= t <= hi): continue
        pre = [x for x in s if t - W <= x["date"] < t]; post = [x for x in s if t <= x["date"] <= t + W]
        if len(pre) < rc["min_acq_each_side"] or len(post) < rc["min_acq_each_side"]: continue
        m = lambda xs, k: sum(x[k] for x in xs) / len(xs)
        dvh, dvv = m(post, "vh") - m(pre, "vh"), m(post, "vv") - m(pre, "vv")
        dcr = dvh - dvv
        if dvh <= -rc["vh_drop_db"] and dcr <= -rc["cr_drop_db"]:
            out.append(dict(start=s[j - 1]["date"], end=t, dvh=dvh, dvv=dvv, dcr=dcr, wetting=dvv >= rc["vv_wetting_db"]))
    return out


def radar_standing_across(series, a, b, cfg):
    rc = cfg["radar"]; W = timedelta(days=rc["window_days"])
    s = [dict(x, date=D(x["date"])) for x in series if ok(x.get("vh"))]
    pre = [x["vh"] for x in s if a - W <= x["date"] <= a]; post = [x["vh"] for x in s if b <= x["date"] <= b + W]
    if len(pre) < rc["min_acq_each_side"] or len(post) < rc["min_acq_each_side"]: return None
    return abs(sum(post) / len(post) - sum(pre) / len(pre)) < rc["standing_max_change_db"]


def overlaps(i0, i1, j0, j1, tol):
    return i0 - timedelta(days=tol) <= j1 and j0 <= i1 + timedelta(days=tol)


def step4_removal(fy, looks, cyc, optical_ok, radar_ok, cfg):
    y, r, rc = fy["year"], cfg["removal"], cfg["radar"]
    cutoff = md(y, cfg["season"]["reporting_cutoff"])
    radar = radar_changes(fy["radar"]["series"], y, cfg) if radar_ok else []
    radar_valid = [c for c in radar if not c["wetting"]]
    res = dict(source=None, a=None, b=None, radar=None, optical=None, radar_candidates=radar)
    if cyc["peak"] is None:
        if not radar_ok:
            return check(UNK, "INSUFFICIENT DATA", reasons=["insufficient_data"], missing=["optical season peak", "radar"]), res
    peak_date = cyc["peak"]["date"] if cyc["peak"] else date(y, 8, 1)
    cands = optical_candidates(looks, peak_date, cutoff, cfg)
    vals = {"optical_candidates": [{"a": iso(c["a"]["date"]), "b": iso(c["b"]["date"]), "ndvi_a": r3(c["a"]["ndvi"]), "ndvi_b": r3(c["b"]["ndvi"]),
                                    "drop": r3(c["drop"]), "gap_days": c["gap"], "faster_than_senescence": c["fast"], "confirmed": c["confirmed"],
                                    "recovery": c["recovery"]} for c in cands],
            "radar_changes": [{"start": iso(c["start"]), "end": iso(c["end"]), "dVH_db": r3(c["dvh"]), "dVV_db": r3(c["dvv"]),
                               "dCR_db": r3(c["dcr"]), "wetting": c["wetting"]} for c in radar]}
    src = ["Sentinel-2 L2A NDVI (20 m)"] + (["Sentinel-1 RTC VV/VH, orbit %s" % fy["radar"].get("orbit")] if radar_ok else [])
    # 1) optical removal confirmed and faster than senescence
    for c in cands:
        if c["recovery"]: continue
        if c["fast"] and c["confirmed"]:
            ov = [x for x in radar_valid if overlaps(c["a"]["date"], c["b"]["date"], x["start"], x["end"], rc["overlap_tol_days"])]
            res.update(source="optical", a=c["a"], b=c["b"], optical=c, radar=ov[0] if ov else None)
            return check(SUP, "REMOVAL", vals, {"last_standing": iso(c["a"]["date"]), "first_removed": iso(c["b"]["date"]),
                                                "confirmed_by": iso(c["confirm_date"])}, src, ["optical_abrupt_confirmed_removal"]), res
        # 2) optical change not sufficient alone -> needs overlapping radar structural change
        ov = [x for x in radar_valid if overlaps(c["a"]["date"], c["b"]["date"], x["start"], x["end"], rc["overlap_tol_days"])]
        if ov:
            res.update(source="optical+radar", a=c["a"], b=c["b"], optical=c, radar=ov[0])
            return check(SUP, "REMOVAL", vals, {"last_standing": iso(c["a"]["date"]), "first_removed": iso(c["b"]["date"]),
                                                "radar_change": [iso(ov[0]["start"]), iso(ov[0]["end"])]}, src,
                         ["optical_change_with_overlapping_radar_structural_change"]), res
    amb = [c for c in cands if c["recovery"]]
    # 3) radar-only removal: no optical change candidate, canopy already non-green before, not green after
    if not cands and radar_valid:
        for x in radar_valid:
            before = [o for o in looks if o["date"] <= x["start"]]
            after = [o for o in looks if x["end"] <= o["date"] <= min(cutoff, x["end"] + timedelta(days=r["confirm_window_days"]))]
            if before and before[-1]["ndvi"] <= r["senescence_ndvi"] and all(o["ndvi"] < r["senescence_ndvi"] for o in after):
                res.update(source="radar", a=before[-1], b=None, radar=x)
                return check(SUP, "REMOVAL", vals, {"last_standing": iso(before[-1]["date"]), "radar_change": [iso(x["start"]), iso(x["end"])]},
                             src, ["radar_structural_change_after_nongreen_canopy"]), res
    if amb:
        return check(UNK, "AMBIGUOUS REMOVAL VS DAMAGE", vals, {"candidate": [iso(amb[0]["a"]["date"]), iso(amb[0]["b"]["date"])]}, src,
                     ["ndvi_recovered_after_drop"]), res
    late = [o for o in looks if o["date"] >= md(y, r["standing_check_after"])]
    post = [o for o in looks if o["date"] > peak_date]
    declined = bool(post) and min(o["ndvi"] for o in post) <= r["senescence_ndvi"]
    if late and late[-1]["ndvi"] >= r["standing_ndvi_at_cutoff"] and not declined:
        return check(CON, "NOT YET HARVESTED", vals, {"last_look": iso(late[-1]["date"])}, src, ["standing_green_canopy_at_cutoff"]), res
    if declined:
        why = ["decline_not_faster_than_senescence_or_unconfirmed"] if cands else ["decline_without_abrupt_change"]
        if late and late[-1]["ndvi"] >= r["standing_ndvi_at_cutoff"]: why.append("post_decline_regrowth_possible_cover_crop")
        if not radar_ok: why.append("no_radar_to_corroborate")
        elif any(c["wetting"] for c in radar): why.append("radar_change_masked_by_wetting")
        return check(UNK, "SENESCENCE ONLY", vals, {}, src, why, ["supported canopy removal"]), res
    return check(UNK, "INSUFFICIENT DATA", vals, {}, src, ["insufficient_data"], ["late-season observations"]), res


# ------------------------------------------------------------------ step 5: condition before removal
def drydown_days(looks, pk, a, cfg):
    c = cfg["condition"]
    seg = [o for o in looks if pk["date"] < o["date"] <= a["date"]]
    for i, o in enumerate(seg):
        if o["ndvi"] <= pk["ndvi"] - c["drydown_start_drop_from_peak"] and all(x["ndvi"] <= pk["ndvi"] - c["drydown_rebound_tol"] for x in seg[i:]):
            return (a["date"] - o["date"]).days, o["date"]
    return 0, None


def step5_condition(fy, looks, cyc, rem, cfg):
    c, y = cfg["condition"], fy["year"]
    a = rem["a"]
    if a is None: return check(NA, "NO REMOVAL", reasons=["removal_not_established"]), None
    end = rem["b"]["date"] if rem["b"] else rem["radar"]["end"]
    age = (end - a["date"]).days
    s0, s1 = md(y, cfg["season"]["summer_peak_window"][0]), md(y, cfg["season"]["summer_peak_window"][1])
    ndre_peak = max([o["ndre"] for o in looks if s0 <= o["date"] <= s1 and ok(o.get("ndre"))], default=None)
    frac = a["ndre"] / ndre_peak if ok(a.get("ndre")) and ok(ndre_peak) and ndre_peak > 0 else None
    dd, dd_start = drydown_days(looks, cyc["peak"], a, cfg) if cyc["peak"] else (None, None)
    ndvi_sig = "green" if a["ndvi"] >= c["ndvi_green_min"] else "dry" if a["ndvi"] <= c["ndvi_dry_max"] else None
    ndre_sig = None if frac is None else "green" if frac >= c["ndre_green_frac_of_peak"] else "dry" if frac <= c["ndre_senescent_frac_of_peak"] else None
    dd_sig = "dry" if dd is not None and dd >= c["drydown_days_min"] else None
    vals = {"ndvi_last_standing": r3(a["ndvi"]), "ndre_last_standing": r3(a.get("ndre")), "ndre_corn_season_peak": r3(ndre_peak),
            "ndre_frac_of_peak": r3(frac), "drydown_days_lower_bound": dd, "pre_removal_obs_age_max_days": age,
            "signals": {"ndvi": ndvi_sig, "ndre": ndre_sig, "drydown": dd_sig}}
    dates = {"last_standing": iso(a["date"]), "drydown_start": iso(dd_start)}
    src = ["Sentinel-2 NDVI/NDRE at the last standing look"]
    missing = [] if frac is not None else ["NDRE at last standing look"]
    if age > c["max_preharvest_obs_age_days"]:
        return check(UNK, "UNCERTAIN CONDITION", vals, dates, src, ["preharvest_observation_too_distant"], missing), "uncertain"
    others = [ndre_sig, dd_sig]
    if ndvi_sig == "green" and ndre_sig == "green" and dd_sig is None:
        return check(SUP, "GREEN REMOVAL", vals, dates, src), "green"
    if ndvi_sig == "dry" and "dry" in others and ndre_sig != "green":
        return check(SUP, "DRY REMOVAL", vals, dates, src), "dry"
    if ndvi_sig is None: why = "ndvi_between_green_and_dry"
    elif any(s and s != ndvi_sig for s in others): why = "signals_disagree"
    else: why = "missing_supporting_signal"
    return check(UNK, "UNCERTAIN CONDITION", vals, dates, src, [why], missing), "uncertain"


# ------------------------------------------------------------------ step 6: development timing
def step6_timing(fy, cyc, rem, cond, gdd, cfg):
    t = cfg["timing"]
    srcs = [s["url"] for s in t["sources"]] + [fy.get("weather_source", "weather")]
    if rem["a"] is None: return check(NA, "NO REMOVAL", reasons=["removal_not_established"]), None
    if gdd is None: return check(UNK, "TIMING UNKNOWN", reasons=["no_weather"], missing=["daily temperature"]), None
    det = cyc["detect"]
    if det is None:
        return check(UNK, "TIMING UNKNOWN", reasons=["emergence_not_observed"], missing=["planting date or observed canopy emergence"], sources=srcs), None
    p_early = gdd.back_to(det["date"], t["gdd_planting_to_canopy_detect"][1])
    p_late = gdd.back_to(det["date"], t["gdd_planting_to_canopy_detect"][0])
    a = rem["a"]["date"]; b = rem["b"]["date"] if rem["b"] else rem["radar"]["end"]
    if p_early is None or p_late is None:
        return check(UNK, "TIMING UNKNOWN", reasons=["weather_does_not_cover_planting_window"], sources=srcs), None
    gmin, gmax = gdd.cum(p_late, a), gdd.cum(p_early, b)
    if gmin is None or gmax is None:
        return check(UNK, "TIMING UNKNOWN", reasons=["weather_gap"], sources=srcs), None
    vals = {"gdd_at_removal_min": round(gmin), "gdd_at_removal_max": round(gmax), "planting_window": [iso(p_early), iso(p_late)],
            "silage_stage_min_gdd": t["silage_stage_min_gdd"], "black_layer_range_gdd": [t["black_layer_min_gdd"], t["black_layer_max_gdd"]]}
    dates = {"canopy_detect": iso(det["date"]), "removal_interval": [iso(a), iso(b)]}
    flags = []
    if gmax < t["silage_stage_min_gdd"]: flags.append("removal_earlier_than_any_silage_stage")
    standing_dry = cond == "dry"
    if gmax < t["black_layer_min_gdd"] and gmin >= t["silage_stage_min_gdd"]:
        out, st = "SILAGE TIMING", SUP
    elif gmin >= t["black_layer_max_gdd"] and standing_dry:
        out, st = "GRAIN TIMING", SUP
    else:
        out, st = "TIMING UNKNOWN", UNK
        flags.append("gdd_range_overlaps_stage_windows" if not flags else flags[0])
        if gmin >= t["black_layer_max_gdd"] and not standing_dry: flags.append("mature_by_gdd_but_standing_dry_down_not_established")
    if (cond == "green" and out == "GRAIN TIMING") or (cond == "dry" and out == "SILAGE TIMING"): flags.append("timing_condition_conflict")
    return check(st, out, vals, dates, srcs, sorted(set(flags)), ["planting date", "hybrid relative maturity"]), out


# ------------------------------------------------------------------ step 7: post-harvest surface
def step7_surface(fy, looks, rem, gdd, cfg):
    s = cfg["surface"]
    if rem["a"] is None: return check(NA, "NO REMOVAL", reasons=["removal_not_established"]), None
    start = rem["b"]["date"] if rem["b"] else rem["radar"]["end"]
    win = [o for o in looks if start <= o["date"] <= start + timedelta(days=s["max_days_after_removal"])]
    conf = ["manure", "tillage", "residue_removal", "irrigation (unknown)"]
    if not win:
        return check(UNK, "SURFACE UNKNOWN", reasons=["no_clear_look_after_removal"], missing=["clear look within %d d after removal" % s["max_days_after_removal"]]), None
    tried = []
    for o in win:
        rain = gdd.rain(o["date"]) if gdd else None
        tried.append({"date": iso(o["date"]), "ndvi": r3(o["ndvi"]), "ndti": r3(o.get("ndti")), "rain_prior_2d_mm": r3(rain)})
        if o["ndvi"] > s["ndvi_max"]: continue
        if rain is not None and rain >= s["rain_mm_prior_2d_max"]: continue
        if not ok(o.get("ndti")): continue
        v = {"looks": tried, "ndti": r3(o["ndti"]), "assumption": "A2 NDTI thresholds (unvalidated)"}
        dts = {"surface_look": iso(o["date"])}
        src = ["Sentinel-2 B11/B12 NDTI (20 m)", fy.get("weather_source", "weather")]
        if o["ndti"] <= s["ndti_bare_max"]: return check(SUP, "SILAGE SUPPORT", v, dts, src, ["little_residue_proxy"] + (["rain_unknown"] if rain is None else []), [], ), "silage"
        if o["ndti"] >= s["ndti_residue_min"]: return check(SUP, "GRAIN SUPPORT", v, dts, src, ["abundant_residue_proxy"] + (["rain_unknown"] if rain is None else [])), "grain"
        return check(UNK, "SURFACE UNKNOWN", v, dts, src, ["ndti_between_thresholds"] + ["possible_confounders:" + ",".join(conf)]), None
    why = []
    if all(t["ndvi"] is not None and t["ndvi"] > s["ndvi_max"] for t in tried): why.append("green_cover_or_incomplete_removal")
    if any(t["rain_prior_2d_mm"] is not None and t["rain_prior_2d_mm"] >= s["rain_mm_prior_2d_max"] for t in tried): why.append("wet_surface_recent_rain")
    return check(UNK, "SURFACE UNKNOWN", {"looks": tried}, {}, [], why or ["no_suitable_surface_look"]), None


# ------------------------------------------------------------------ step 8: radar corroboration
def step8_radar(fy, looks, cyc, rem, radar_ok, cfg):
    rc, r = cfg["radar"], cfg["removal"]
    if not radar_ok:
        return check(UNK, "RADAR UNKNOWN", reasons=["radar_unavailable:" + (fy["radar"].get("reason") or "no_series")], missing=["full-coverage S1 orbit"]), None
    if rem["a"] is None: return check(NA, "NO REMOVAL", reasons=["removal_not_established"]), None
    src = ["Sentinel-1 RTC orbit %s" % fy["radar"].get("orbit")]
    x = rem["radar"]
    grain = None
    if x is not None and cyc["peak"]:
        sen = next((o for o in looks if o["date"] > cyc["peak"]["date"] and o["ndvi"] <= r["senescence_ndvi"]), None)
        if sen and (x["start"] - sen["date"]).days >= rc["prior_senescence_days"]: grain = sen["date"]
    vals = {} if x is None else {"dVH_db": r3(x["dvh"]), "dVV_db": r3(x["dvv"]), "dCR_db": r3(x["dcr"]), "interval": [iso(x["start"]), iso(x["end"])]}
    if rem["source"] in ("radar", "optical+radar"):
        # radar was needed to establish removal, so it cannot also be the independent corroboration in step 10
        return check(NA, "RADAR USED TO ESTABLISH REMOVAL", vals, {"senescence_onset": iso(grain)} if grain else {}, src,
                     ["radar_cannot_corroborate_itself"] + (["structural_change_after_prolonged_senescence"] if grain else [])), "used_for_removal"
    if x is not None:
        return check(SUP, "REMOVAL CORROBORATED", vals, {"senescence_onset": iso(grain)} if grain else {}, src,
                     ["radar_interval_overlaps_optical_interval"] + (["structural_change_after_prolonged_senescence"] if grain else [])), ("grain" if grain else "corroborated")
    standing = radar_standing_across(fy["radar"]["series"], rem["a"]["date"], rem["b"]["date"], cfg)
    if standing:
        return check(CON, "RADAR SHOWS NO STRUCTURAL CHANGE", {}, {}, src, ["vh_stable_across_optical_interval"]), "contradicted"
    wet = [c for c in rem["radar_candidates"] if c["wetting"]]
    return check(UNK, "RADAR UNKNOWN", {}, {}, src, ["moisture_effect"] if wet else ["no_clear_structural_change"]), None


# ------------------------------------------------------------------ step 9: exceptions
def zone_mixed(fy, looks, cfg):
    z, out = cfg["zones"], {}
    for k in ("z1", "z2", "z3", "z4"):
        zl = [dict(o, ndvi=o[k]) for o in looks if ok(o.get(k))]
        if len(zl) < z["min_zone_looks"]: continue
        pk = [o for o in zl if 6 <= o["date"].month <= 9]
        if not pk: continue
        p = max(pk, key=lambda o: o["ndvi"])
        cands = [c for c in optical_candidates(zl, p["date"], date(fy["year"], 12, 31), cfg) if c["fast"] and c["confirmed"] and not c["recovery"]]
        if cands: out[k] = (cands[0]["a"]["date"], cands[0]["b"]["date"])
    iv = list(out.values())
    mixed = any(not overlaps(i[0], i[1], j[0], j[1], z["non_overlap_days"]) for n, i in enumerate(iv) for j in iv[n + 1:])
    return mixed, {k: [iso(v[0]), iso(v[1])] for k, v in out.items()}


def step9_exceptions(fy, looks, cyc, rem, cond_check, timing_check, gdd, cfg):
    e = cfg["exceptions"]; flags, vals = [], {}
    if rem["a"] is not None and gdd and cyc["peak"]:
        t = gdd.min_tmin(cyc["peak"]["date"], rem["a"]["date"])
        vals["min_tmin_peak_to_removal_f"] = r3(t)
        if t is not None and t <= e["frost_tmin_f"] and cond_check["outcome"] != "DRY REMOVAL": flags.append("possible_frost_before_removal")
    if gdd:
        p = gdd.precip(date(fy["year"], 6, 1), date(fy["year"], 8, 31)); vals["precip_jun_aug_mm"] = r3(p)
        if p is not None and p < e["drought_jun_aug_precip_mm_max"]: flags.append("possible_drought_stress")
    mixed, zv = zone_mixed(fy, looks, cfg); vals["zone_removal_intervals"] = zv
    if mixed: flags.append("possible_mixed_management_zones")
    flags += [f for f in timing_check.get("reasons", []) if f in ("timing_condition_conflict", "removal_earlier_than_any_silage_stage")]
    unverifiable = ["lodging", "disease", "earlage / high-moisture grain / other harvested product", "stress chopping for silage", "post-harvest baling or tillage"]
    st = UNK if flags else NA
    return check(st, "POSSIBLE CONFOUNDERS" if flags else "NONE OBSERVED", vals, {}, [fy.get("weather_source", "weather")], flags,
                 ["unobservable: " + ", ".join(unverifiable)]), flags, mixed


# ------------------------------------------------------------------ step 11: context
def step11_context(fy, registry):
    items = ["field-linked dairy/livestock operating records", "documented actual forage supply relationships",
             "historical satellite harvest trajectories (prior seasons)", "winter-forage -> summer-corn cropping pattern",
             "historically active livestock facilities / hauling routes", "county silage/grain prevalence"]
    rows = [{"item": i, "state": UNK, "reason": "not_available_in_v1" if n < 5 else "context_alone_cannot_decide"} for n, i in enumerate(items)]
    rows[2]["reason"] = "prior-season imagery for this polygon not extracted in v1"
    return check(UNK, "NO CONTEXTUAL EVIDENCE", {"items": rows}, reasons=["contextual_evidence_unavailable"],
                 missing=[r["item"] for r in rows[:5]])


# ------------------------------------------------------------------ decision
def decide(fy, weather, cfg, registry):
    """fy: field-year input (no labels). weather: per-year daily weather dict or None. Returns the full trace."""
    for k in ("label", "Corn_2021", "Corn_2022", "Corn_2023", "reference", "y"):
        assert k not in fy, f"label-like key {k!r} in inference input"
    looks = clear_looks(fy.get("optical", []))
    gdd = GDD(weather) if weather else None
    path, review, reasons = [], False, []
    checks = {}
    checks["1_identity"], checks["1_crop_type_exclusion"], cyc = step1_identity(fy, looks, cfg)
    checks["2_quality"], optical_ok, radar_ok, area_ok = step2_quality(fy, looks, cfg)
    checks["3_documents"] = step3_documents(fy, registry)
    checks["4_removal"], rem = step4_removal(fy, looks, cyc, optical_ok, radar_ok, cfg)
    checks["5_condition"], cond = step5_condition(fy, looks, cyc, rem, cfg)
    checks["6_timing"], timing = step6_timing(fy, cyc, rem, cond, gdd, cfg)
    checks["7_surface"], surf = step7_surface(fy, looks, rem, gdd, cfg)
    checks["8_radar"], rad = step8_radar(fy, looks, cyc, rem, radar_ok, cfg)
    checks["9_exceptions"], flags, mixed = step9_exceptions(fy, looks, cyc, rem, checks["5_condition"], checks["6_timing"], gdd, cfg)
    final = None

    def done(cls, why, rev=False):
        nonlocal final, review
        final = cls; reasons.extend(why); review = review or rev
        path.append(f"=> {cls}")

    # step 3: verified direct records decide first
    d3 = checks["3_documents"]
    path.append(f"3 documents: {d3['outcome']}")
    if d3["state"] == SUP: done(d3["outcome"], ["direct_record"])
    elif d3["state"] == CON: review = True; reasons.append("conflicting_direct_records")
    if final is None:
        idc = checks["1_identity"]; path.append(f"1 identity: {idc['outcome']}")
        if idc["state"] == CON: done("UNRESOLVED", ["not_corn_like_cycle"], True)
        elif idc["state"] == UNK and "uncertain_identity" in idc["reasons"]: done("UNRESOLVED", ["uncertain_identity"], True)
    if final is None:
        q = checks["2_quality"]; path.append(f"2 quality: {q['outcome']}")
        if not area_ok and not (optical_ok or radar_ok): done("UNRESOLVED", ["insufficient_data", "insufficient_interior_area"])
        elif not optical_ok and not radar_ok: done("UNRESOLVED", ["insufficient_data"])
        elif checks["1_identity"]["state"] == UNK: done("UNRESOLVED", ["insufficient_data", "identity_unknown"], True)
    if final is None:
        rm = checks["4_removal"]; path.append(f"4 removal: {rm['outcome']} ({','.join(rm['reasons'])})")
        if rm["outcome"] == "NOT YET HARVESTED": done("NOT YET HARVESTED", ["standing_at_cutoff"])
        elif rm["outcome"] == "AMBIGUOUS REMOVAL VS DAMAGE": done("UNRESOLVED", ["ambiguous_removal_vs_damage"], True)
        elif rm["outcome"] == "SENESCENCE ONLY": done("UNRESOLVED", ["senescence_only"], True)
        elif rm["state"] != SUP: done("UNRESOLVED", ["insufficient_data"])
    if final is None and mixed:
        path.append("9 zones: removal intervals differ between zones"); done("UNRESOLVED", ["possible_mixed_management"], True)
    if final is None:
        path.append(f"5 condition: {checks['5_condition']['outcome']}")
        path.append(f"6 timing: {checks['6_timing']['outcome']}")
        path.append(f"7 surface: {checks['7_surface']['outcome']}")
        path.append(f"8 radar: {checks['8_radar']['outcome']}")
        conflict = []
        if "timing_condition_conflict" in checks["6_timing"]["reasons"]: conflict.append("timing_condition_conflict")
        if rad == "contradicted": conflict.append("radar_contradicts_removal")
        if cond == "green" and surf == "grain": conflict.append("surface_contradicts_condition")
        if cond == "dry" and surf == "silage": conflict.append("surface_contradicts_condition")
        if conflict:
            done("UNRESOLVED", ["conflicting_evidence"] + conflict, True)
        elif cond == "uncertain" or cond is None:
            done("UNRESOLVED", ["uncertain_condition"] + checks["5_condition"]["reasons"])
        else:
            corr_s = surf == "silage" or rad in ("corroborated",)
            corr_g = surf == "grain" or rad in ("corroborated", "grain")
            if cond == "green" and timing == "SILAGE TIMING" and corr_s: done("LIKELY SILAGE", ["green_removal", "silage_timing", "surface_or_radar_corroboration"])
            elif cond == "dry" and timing == "GRAIN TIMING" and corr_g: done("LIKELY GRAIN", ["dry_removal", "grain_timing", "surface_or_radar_corroboration"])
            elif timing in ("SILAGE TIMING", "GRAIN TIMING"):
                done("UNRESOLVED", ["no_surface_or_radar_corroboration"])
            else:
                c11 = step11_context(fy, registry); checks["11_context"] = c11
                path.append("11 context: " + c11["outcome"])
                done("UNRESOLVED", ["removal_condition_established_timing_unknown", "contextual_evidence_unavailable"])
    if "11_context" not in checks: checks["11_context"] = check(NA, "NOT REACHED")
    a = rem["a"]["date"] if rem.get("a") else None
    b = rem["b"]["date"] if rem.get("b") else (rem["radar"]["end"] if rem.get("radar") and rem.get("source") == "radar" else None)
    interval = None
    if a and b:
        interval = {"last_standing": iso(a), "first_removed": iso(b), "width_days": (b - a).days,
                    "midpoint_estimate": iso(a + timedelta(days=(b - a).days // 2)), "source": rem["source"],
                    "note": "harvest occurred within (last_standing, first_removed]; midpoint is an estimate only"}
    missing = sorted({m for c in checks.values() for m in c["missing"]})
    return {"key": fy["key"], "year": fy["year"], "crop_cycle_id": fy.get("crop_cycle_id", fy["key"] + "-c1"),
            "config_version": cfg["version"], "final": final, "review": review, "reason_codes": sorted(set(reasons)),
            "harvest_interval": interval, "checks": checks, "exception_flags": flags, "missing_evidence": missing, "decision_path": path}


def binary(final):
    """Evaluation-only mapping. Everything else is an abstention, never a wrong-class default."""
    return {"CONFIRMED SILAGE": "silage", "LIKELY SILAGE": "silage", "CONFIRMED GRAIN": "grain", "LIKELY GRAIN": "grain"}.get(final)
