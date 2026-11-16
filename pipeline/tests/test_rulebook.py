"""Focused tests for rulebook v1: label isolation, missing-data abstention, decision branching, evaluation rows.
Run: cd pipeline && python -m unittest tests.test_rulebook -v   (standard library only)
"""
import json, copy, pathlib, unittest
from datetime import date, timedelta
import sys
HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rulebook.engine import decide, binary
from rulebook.evaluate import join_rows, summarize

CFG = json.load(open(HERE / "rulebook" / "config_v1.json"))
REG = {"items": []}
Y = 2022


def weather(tmax=80.0, tmin=60.0, rain=2.0):
    """Constant weather: (80 + 60) / 2 - 50 = 20 GDD/day; 2 mm/day (Jun-Aug 184 mm, 4 mm in any 2 days)."""
    d0 = date(Y, 1, 1); days = [d0 + timedelta(days=i) for i in range(365)]
    return {"daily": {"date": [d.isoformat() for d in days], "tmax_f": [tmax] * 365, "tmin_f": [tmin] * 365, "precip_mm": [rain] * 365}}


def look(m, d, ndvi, ndti=0.08):
    return dict(date=date(Y, m, d).isoformat(), clear_frac=1.0, n_valid_px=60, ndvi=ndvi, ndre=None if ndvi is None else round(ndvi * 0.6, 4),
                ndti=ndti, lswi=0.2, psri=0.0, B04=0.05, B11=0.2, B12=0.15, z1=None, z2=None, z3=None, z4=None)


SPRING = [look(4, 1, 0.30), look(4, 20, 0.20), look(5, 20, 0.40), look(6, 1, 0.50), look(6, 20, 0.70), look(7, 1, 0.80), look(7, 15, 0.85)]


def radar(change_day, start=date(Y, 7, 1), end=date(Y, 12, 15)):
    out, d = [], start
    while d <= end:
        out.append(dict(date=d.isoformat(), vv=-9.0, vh=-14.0 if d < change_day else -18.0, n_px=300)); d += timedelta(days=6)
    return out


def fy(optical, rad=None, full=True):
    return dict(key="FYTEST", year=Y, crop_cycle_id="FYTEST-c1", area_ha=5.0, interior_px10=400, interior_ha=4.0, eligible=True,
                weather_source="synthetic", optical=optical,
                radar=dict(orbit=4, coverage_frac=1.0 if full else 0.2, full_cover=full, reason=None if full else "partial_coverage_or_few_dates",
                           series=rad if (rad and full) else []))


SILAGE = SPRING + [look(8, 1, 0.82), look(8, 10, 0.75), look(8, 14, 0.20, ndti=0.02), look(8, 25, 0.22, ndti=0.03), look(9, 10, 0.25), look(10, 1, 0.30)]
GRAIN = SPRING + [look(8, 15, 0.78), look(9, 1, 0.65), look(9, 15, 0.50), look(10, 1, 0.40), look(10, 15, 0.36),
                  look(10, 20, 0.18, ndti=0.15), look(11, 1, 0.20, ndti=0.14)]


class TestBranches(unittest.TestCase):
    def test_likely_silage(self):
        t = decide(fy(SILAGE, radar(date(Y, 8, 12))), weather(), CFG, REG)
        self.assertEqual(t["final"], "LIKELY SILAGE", t["decision_path"])
        self.assertEqual(t["checks"]["5_condition"]["outcome"], "GREEN REMOVAL")
        self.assertEqual(t["checks"]["6_timing"]["outcome"], "SILAGE TIMING")
        self.assertEqual(t["harvest_interval"]["last_standing"], f"{Y}-08-10")
        self.assertEqual(t["harvest_interval"]["first_removed"], f"{Y}-08-14")

    def test_likely_grain(self):
        t = decide(fy(GRAIN, radar(date(Y, 10, 18))), weather(), CFG, REG)
        self.assertEqual(t["final"], "LIKELY GRAIN", t["decision_path"])
        self.assertEqual(t["checks"]["5_condition"]["outcome"], "DRY REMOVAL")
        self.assertEqual(t["checks"]["6_timing"]["outcome"], "GRAIN TIMING")

    def test_senescence_only_is_not_harvest(self):
        opt = SPRING + [look(8, 15, 0.75), look(9, 1, 0.62), look(9, 15, 0.52), look(10, 1, 0.44), look(10, 15, 0.38), look(11, 1, 0.33), look(11, 20, 0.30)]
        t = decide(fy(opt, None, full=False), weather(), CFG, REG)
        self.assertEqual(t["checks"]["4_removal"]["outcome"], "SENESCENCE ONLY")
        self.assertEqual(t["final"], "UNRESOLVED"); self.assertIn("senescence_only", t["reason_codes"]); self.assertTrue(t["review"])

    def test_not_yet_harvested(self):
        opt = SPRING + [look(8, 15, 0.80), look(9, 15, 0.75), look(10, 15, 0.72), look(11, 20, 0.70), look(12, 10, 0.68)]
        t = decide(fy(opt, None, full=False), weather(), CFG, REG)
        self.assertEqual(t["final"], "NOT YET HARVESTED")

    def test_cover_crop_regrowth_is_not_standing_corn(self):
        opt = SPRING + [look(8, 15, 0.80), look(9, 1, 0.66), look(9, 10, 0.55), look(9, 20, 0.45), look(10, 1, 0.35), look(10, 10, 0.27),
                        look(11, 1, 0.45), look(11, 20, 0.55), look(12, 10, 0.58)]
        t = decide(fy(opt, None, full=False), weather(), CFG, REG)
        self.assertNotEqual(t["final"], "NOT YET HARVESTED")
        self.assertIn("post_decline_regrowth_possible_cover_crop", t["checks"]["4_removal"]["reasons"])

    def test_removal_spread_over_consecutive_looks(self):
        # dense revisits: each consecutive fall < 0.15, but 0.72 -> 0.30 within 7 days is faster than senescence
        opt = SPRING + [look(8, 1, 0.80), look(8, 20, 0.74), look(8, 23, 0.72), look(8, 25, 0.60), look(8, 27, 0.48), look(8, 30, 0.30, ndti=0.02),
                        look(9, 10, 0.27, ndti=0.03), look(9, 25, 0.28), look(10, 10, 0.30)]
        t = decide(fy(opt, None, full=False), weather(), CFG, REG)
        self.assertEqual(t["checks"]["4_removal"]["outcome"], "REMOVAL", t["checks"]["4_removal"])
        self.assertEqual(t["harvest_interval"]["first_removed"], f"{Y}-08-30")

    def test_timing_unknown_goes_to_context_then_unresolved(self):
        hot = weather(tmax=90.0, tmin=70.0)   # 28 GDD/day: silage-window removal now overlaps black-layer range
        t = decide(fy(SILAGE, radar(date(Y, 8, 12))), hot, CFG, REG)
        self.assertEqual(t["checks"]["6_timing"]["outcome"], "TIMING UNKNOWN")
        self.assertEqual(t["final"], "UNRESOLVED")
        self.assertIn("removal_condition_established_timing_unknown", t["reason_codes"])
        self.assertNotEqual(t["checks"]["11_context"]["outcome"], "NOT REACHED")

    def test_surface_contradiction_is_review(self):
        opt = [dict(o) for o in SILAGE]
        for o in opt:
            if o["date"] == f"{Y}-08-14": o["ndti"] = 0.15          # abundant-residue proxy after a green removal
        t = decide(fy(opt, None, full=False), weather(), CFG, REG)
        self.assertEqual(t["final"], "UNRESOLVED"); self.assertIn("conflicting_evidence", t["reason_codes"]); self.assertTrue(t["review"])

    def test_radar_cannot_corroborate_itself(self):
        # optical change too slow to be removal on its own; radar establishes removal -> radar not counted as corroboration
        opt = SPRING + [look(8, 15, 0.78), look(9, 1, 0.65), look(9, 15, 0.50), look(10, 1, 0.42), look(10, 25, 0.25, ndti=0.07), look(11, 10, 0.26, ndti=0.07)]
        t = decide(fy(opt, radar(date(Y, 10, 13))), weather(), CFG, REG)
        self.assertEqual(t["checks"]["4_removal"]["reasons"], ["optical_change_with_overlapping_radar_structural_change"])
        self.assertEqual(t["checks"]["8_radar"]["state"], "not_applicable")
        self.assertNotIn(t["final"], ("LIKELY GRAIN", "LIKELY SILAGE"))


class TestMissingData(unittest.TestCase):
    def test_no_data_abstains(self):
        t = decide(fy([], None, full=False), weather(), CFG, REG)
        self.assertEqual(t["final"], "UNRESOLVED"); self.assertIn("insufficient_data", t["reason_codes"])

    def test_cloudy_series_abstains(self):
        cloudy = [dict(look(m, 10, None), clear_frac=0.1) for m in range(4, 13)]
        t = decide(fy(cloudy, None, full=False), weather(), CFG, REG)
        self.assertEqual(t["final"], "UNRESOLVED")

    def test_missing_data_never_becomes_grain(self):
        # grain-like series with weather and radar removed, and the post-removal looks removed
        g = [o for o in GRAIN if o["date"] < f"{Y}-10-20"]
        for w in (None, weather()):
            t = decide(fy(g, None, full=False), w, CFG, REG)
            self.assertNotEqual(binary(t["final"]), "grain", t["decision_path"])

    def test_no_weather_gives_timing_unknown(self):
        t = decide(fy(GRAIN, radar(date(Y, 10, 18))), None, CFG, REG)
        self.assertEqual(t["checks"]["6_timing"]["outcome"], "TIMING UNKNOWN"); self.assertEqual(t["final"], "UNRESOLVED")


class TestLabelIsolation(unittest.TestCase):
    def test_engine_rejects_label_keys(self):
        for k in ("label", "Corn_2022", "reference"):
            x = fy(SILAGE); x[k] = "Silage"
            with self.assertRaises(AssertionError): decide(x, weather(), CFG, REG)

    def test_inference_scripts_never_open_labels(self):
        for name in ("14_rulebook_inputs.py", "15_rulebook_run.py", "rulebook/engine.py"):
            src = (HERE / name).read_text()
            self.assertNotIn("reference_labels", src, name)
            self.assertNotIn("Corn_20", src.replace("\"Corn_2021\", \"Corn_2022\", \"Corn_2023\"", ""), name)

    def test_output_independent_of_key_and_deterministic(self):
        a = decide(fy(SILAGE, radar(date(Y, 8, 12))), weather(), CFG, REG)
        x = fy(SILAGE, radar(date(Y, 8, 12))); x["key"] = "OTHERKEY"
        b = decide(x, weather(), CFG, REG)
        self.assertEqual((a["final"], a["reason_codes"]), (b["final"], b["reason_codes"]))
        self.assertEqual(json.dumps(a, sort_keys=True, default=str), json.dumps(decide(fy(SILAGE, radar(date(Y, 8, 12))), weather(), CFG, REG), sort_keys=True, default=str))


class TestEvaluationRows(unittest.TestCase):
    def test_all_rows_preserved_including_failures(self):
        labels = [dict(OBJECTID=str(i), year="2022", label=("silage" if i % 3 == 0 else "grain")) for i in range(1, 7)]
        traces = [dict(OBJECTID=1, year=2022, final="LIKELY GRAIN", review=False, reason_codes=[]),
                  dict(OBJECTID=2, year=2022, final="UNRESOLVED", review=True, reason_codes=["senescence_only"]),
                  dict(OBJECTID=3, year=2022, final="LIKELY GRAIN", review=False, reason_codes=[]),
                  dict(OBJECTID=4, year=2022, final="NOT YET HARVESTED", review=False, reason_codes=[])]
        rows = join_rows(traces, labels)
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(r["final"] == "NO OUTPUT (failure)" for r in rows), 2)
        s = summarize(rows)
        self.assertEqual((s["decided"], s["correct"]), (2, 1))
        self.assertAlmostEqual(s["correct_over_all"], 1 / 6)
        self.assertEqual(s["binary"]["fp"], 0); self.assertEqual(s["binary"]["fn"], 1)


if __name__ == "__main__":
    unittest.main()
