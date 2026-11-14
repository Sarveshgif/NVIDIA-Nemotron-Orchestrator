# Five-minute presentation — Harvest Console

All numbers below are from `pipeline/results/` (held-out test regions, Netherlands 2025; console model trained on
2024 + 2025, experiment E2). Year experiments: `results/experiments/`. Figures: `results/fig_ndvi_by_class.png`,
`results/fig_model_comparison.png`, `results/fig_review_tradeoff.png`, `results/confusion_matrices.png`.
Field IDs are real test-region parcels in the console.

## Timed outline

| Time | Beat | Show | Say (key line) |
|---|---|---|---|
| 0:00–0:30 | Problem & user | Title + console workspace | "USDA analysts estimate corn acreage, but silage and grain look alike for most of the season. The analyst needs a split by field, with the uncertain fields flagged — not a black box." |
| 0:30–1:15 | One field's season | `fig_ndvi_by_class.png`, then NL-GEN-0155 drawer | "Silage is chopped green in Aug–Sep and a catch crop greens up by December; grain senesces and is combined in Oct–Nov, leaving bare stubble. December NDVI: 0.77 vs 0.52." |
| 1:15–1:50 | Public data & labels | Reports → evaluation header, label audit | "No public US source labels silage vs grain fields, so we validated in the Netherlands, where every parcel is declared: code 259 silage, 316 grain. Two seasons, 2024 and 2025: about 10,600 field-seasons, Sentinel-2 and Sentinel-1, all anonymous public data. Labels are farmer declarations, not verified harvests." |
| 1:50–2:50 | Evaluation & strongest comparison | `fig_model_comparison.png`, confusion matrices, years table | "Split by region; test regions never touched in either year. Our California rules transfer badly — balanced accuracy 0.50, zero grain found. A Random Forest on optical + radar: 0.72, grain recall 73 %. Adding 2024 helped a little (0.70 → 0.72) and a model trained only on 2024 still scores 0.69 on 2025. With two years the Transformer catches up (0.73 ± 0.02) but doesn't beat the RF, so we ship the RF. When we remove six weeks of optical looks (simulated cloud), radar keeps the model at 0.71 vs 0.68 without it." |
| 2:50–4:10 | Live workflow | Console (see demo script) | select → evidence → review → totals → export |
| 4:10–4:40 | Insight | `fig_review_tradeoff.png` | "Cloud gaps are not what makes fields hard here: the post-harvest signal carries the call. Routing the 24 % of fields in the score band to analysts leaves 76 % auto-decided at 0.77 balanced accuracy. And for totals, summing calibrated scores gives grain 699 ha vs 599 declared — hard calls would say 1,853." |
| 4:40–5:00 | Limits & next steps | Limitations slide | "Two years, one region, declared labels, grain precision ~0.24, and accuracy varies by year (0.65 when tested on 2024). Next: more BRP years (2017–2023 are public), a California pilot with a few hundred analyst-verified fields, and radar-first scoring for cloudy autumns." |

## Demo script (console, Netherlands 2025 — opens by default)

1. **Open** `npm run dev` → http://localhost:5173/. Status bar: *Analysis ready · latest observation Dec 30, 2025*;
   badge **Validated (NL)**; prediction toggle on **ML model** (`rf_s2_s1-2025.1`, threshold 0.88 on the silage score).
2. **Clear silage — NL-GEN-0155** (8.9 ac, score 0.98). Search the ID in the queue filter or pan to Gennep.
   Evidence summary: NDVI 0.84 → 0.28 between Aug 19 and Sep 10; Nov–Dec NDVI 0.85 (green cover). Before/after chips:
   green parcel → bare. Radar tab: VH falls 4.2 dB across the cut.
3. **Grain — NL-MAA-0056** ("Strongest grain call", 4.2 ac, score 0.31 → grain, declared grain). NDVI only 0.62
   before a small Oct 1–10 fall, Nov–Dec NDVI 0.28 (bare), VH does not fall (+0.8 dB). Point out the score: even the
   strongest grain call is only p(grain) ≈ 0.69 — that is why grain calls are reviewed.
4. **Uncertain — NL-DOE-0100** ("Uncertain: in review band", 32.6 ac). Reasons: low confidence, 47-day cloud gap,
   large acreage. Score 0.876, just under the 0.88 threshold → grain, yet the evidence leans silage (NDVI 0.82 → 0.23
   in Sep, VH −2.8 dB). **Record a review**: choose a class or *Insufficient evidence* with a note → the review
   counter and totals update; the original model prediction stays in History.
5. **Confident but disagrees with the declaration — NL-GEN-0093** (declared grain, model 0.97 silage). Its series is
   textbook silage (cut by Sep 10, Nov–Dec NDVI 0.84). Say: "declared crop use is not a verified harvest; this is the
   kind of field an analyst should check, not proof the model is wrong."
6. **Totals**: summary bar shows hard-call tiles and the *calibrated estimate for totals* line; open the ⓘ.
7. **Reports** → held-out evaluation table, confusion matrices, the *Does more training data help?* years table;
   scroll to the check line (current predictions vs reference). **Download CSV** → file includes prediction source,
   model version, scores, analyst decisions.
8. **Toggle Rule-based** in the header: the queue fills with abstentions — the live version of "rules don't transfer".
9. (Optional) Area → *Tulare County, CA* 2023: the same console on real California data, rule-based and unvalidated.

## Fallback

If the live console fails: `results/*.png` plus screenshots, and `public/nl/nl2025.json` is the cached real output
(static, no network needed except the optional Esri basemap, which degrades to a schematic). The console loads it in
≈ 1.4 s locally.

## Claims we do not make

No business-savings figures, no user-study results, no California accuracy, no sub-field silage/grain acreage, no
claim that the declared labels are verified harvest outcomes, and no claim that the 2-year gain is statistically
significant (it is inside the bootstrap interval).
