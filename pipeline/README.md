# Silage vs grain maize — Netherlands 2024 + 2025 validation pipeline

End-to-end, real-data pipeline: public parcel labels → Sentinel-2 + Sentinel-1 time series → rules, Random Forest
and CNN→Transformer → held-out spatial evaluation → analyst console (`public/nl/`).

Everything here ran on 2026-11-13/14 (2025 first, then crop year 2024 added and all models retrained). No synthetic
observations, no mock outputs. The California (Tulare 2023) console
dataset is separate, unlabelled and rule-based only; Dutch data is never shown as California.

## Why the Netherlands

No public US source labels individual corn fields as silage vs grain (USDA CDL only says "corn"). The Dutch
parcel register (BRP) does: every parcel is declared each spring with a crop code, including
**259 "Mais, snij-" (silage maize)** and **316 "Mais, korrel-" (grain maize)**. It is anonymous, open and
geometry-complete, so it supports an honest train/test evaluation that California cannot yet.

## What ran (in order)

| Step | Script | What it does | Wall time (lab workstation) |
|---|---|---|---|
| 1 | `01_scan_blocks.py` | PDOK BRP OGC API (anonymous) by ~10×10 km bbox; 36 candidate blocks scanned; all parcels cached | ~3 min |
| 1b | `01b_blocks_from_gpkg.py` | 2024: same 20 block windows read from the PDOK annual GeoPackage (3.1 GB, anonymous) | ~1 min after download |
| 2 | `02_extract_s2.py` | Sentinel-2 L2A (Element84 Earth Search, anonymous AWS COGs), Mar–Dec 2025, one MGRS tile per block, block window read once per date, all parcels summarised | 55–80 s per block (20 blocks, 3 workers ≈ 8 min) |
| 3 | `03_extract_s1.py` | Sentinel-1 RTC (Microsoft Planetary Computer, anonymous SAS signing), all orbits kept and tagged | 100–260 s per block (5 workers ≈ 25 min) |
| 4 | `04_build_features.py` | spatial split, labels table (separate), faithful Python port of `src/lib/rules.ts`, RF features | 70 s |
| 5 | `00_label_audit.py` | label audit (`results/label_audit.md`) | 5 s |
| 6 | `05_models.py` | majority, rules (original + adapted), RF ablations, calibration, review band, bootstrap CIs, simulated gap, console predictions; run once per year experiment (E1–E4, `run_2024.sh`) | ~3 min each |
| 7 | `06_deep.py` | CNN→Transformer variants × 3 seeds on the GPU | 35–180 s per run |
| 8 | `07_export_console.py` | held-out test fields → `public/nl/nl2025.json`, chips, separate reference file | 13 s |
| 9 | `08_figures.py` | `results/fig_*.png` | 5 s |
| 10 | `02b_extract_s2_20m.py` → `09_index_features.py` → `10_index_experiment.py` (`run_indices.sh`) | spectral-index experiment on a common 20 m grid | ~50–150 s per block-year; models ~2 min |

Reproduce (Python 3.10+, `pip install numpy pandas pyarrow rasterio shapely pyproj pystac-client planetary-computer scikit-learn scipy matplotlib pillow tabulate torch`):

```bash
export HTH_DATA=/path/to/data            # cache + outputs (≈ 4 GB)
python 01_scan_blocks.py
python 02_extract_s2.py <block> [<block> ...]     # see REGIONS in 04_build_features.py for the 20 blocks used
python 03_extract_s1.py <block> [<block> ...]
python 04_build_features.py && python 00_label_audit.py && python 05_models.py
HTH_SEED=1 python 06_deep.py s2 s2s1 s2s1_nocnn   # repeat for seeds 2, 3
python 07_export_console.py && python 08_figures.py
./sync_from_gpu.sh                                 # copies console data to public/nl and results to pipeline/results
```
Adding a year: download `brpgewaspercelen_definitief_<year>.gpkg`, run `01b_blocks_from_gpkg.py` with
`HTH_YEAR=<year> HTH_DATA=<new dir>`, then steps 2–5 for that year. Multi-year models read
`HTH_ROOTS="2024:<dir>,2025:<dir>"`, `HTH_TRAIN_YEARS`, `HTH_TEST_YEAR`, `HTH_RESULTS` (see `multiyear.py`, `run_2024.sh`).
`run_all.sh` / `run_s1_queue.sh` on the lab machine chain steps 3–7. All downloads are cached and resumable
(blocks, parquet per block); retries with back-off on HTTP 429/5xx.

**Hardware.** Lab workstation (reached over Tailscale): 64 CPU cores, 125 GB RAM, 1× NVIDIA RTX 4070 Ti (12 GB),
Ubuntu, Python 3.10, PyTorch 2.6 + CUDA 12.4. Data were free; compute was the lab machine; network egress was
anonymous public reads (AWS Open Data, Planetary Computer, PDOK).

## Data and licences

| Source | Access | Licence / terms | Used for |
|---|---|---|---|
| PDOK BRP Gewaspercelen 2025 (RVO), status *Definitief* | `api.pdok.nl/rvo/gewaspercelen/ogc/v1` (anonymous) | open data (PDOK/NGR record b812a145…) | parcel geometry + declared crop |
| PDOK BRP Gewaspercelen 2024, *Definitief* | `service.pdok.nl/rvo/gewaspercelen/atom/downloads/brpgewaspercelen_definitief_2024.gpkg` (anonymous) | same | 2024 parcels, same codes and schema |
| Copernicus Sentinel-2 L2A, collection 1 | Element84 Earth Search, AWS Open Data | Copernicus open licence | optical time series, chips |
| Copernicus Sentinel-1 GRD RTC | Microsoft Planetary Computer | Copernicus open licence (RTC by Microsoft/Catalyst) | radar time series |
| EuroCrops (Zenodo 14094196) | checked, not used | CC-BY-SA-4.0 | — same declarations; HCAT merges classes |
| BonaRes FOR1695 (doi:10.20387/bonares-a0qc-46jc) | checked, not used | CC-BY-4.0 | — too few field-seasons |

Full label audits: [`results/label_audit_2025.md`](results/label_audit_2025.md), [`results/label_audit_2024.md`](results/label_audit_2024.md). Key points:
- 5 maize codes occur; only 259 and 316 are used. 317 corn-cob mix, 1935 ear silage and 814 sweet corn are excluded.
- Labels are **farmer-declared intended use**, not verified harvest outcome.
- 7,967 candidate parcels in 20 blocks → **5,169 eligible fields** (≥ 50 interior 10 m pixels after a 10 m inward
  buffer). Excluded as too small: grain 37.5 %, silage 34.8 %.
- 2024, same blocks: 9,399 candidate parcels → **5,446 eligible fields** (too small: grain 40.1 %, silage 42.3 %).
  Single-part parcels only, as in 2025.
- Sample is complete within blocks but blocks were chosen where grain maize grows (south-east NL): grain is 12 % of
  fields here, far above the national share.

## Observations

**Sentinel-2.** Reflectance = DN × 0.0001 − 0.1 (from asset `raster:bands`; baseline 05.x). Bands B02 B03 B04 B08
(10 m), B05 B8A B11 B12 and SCL (20 m, read onto the 10 m grid by nearest neighbour — no added detail).
Clear pixel = SCL 4 or 5. A field-date is observed when ≥ 60 % of its interior pixels are clear; statistics use clear
pixels only. NDVI = (B08−B04)/(B08+B04); LSWI = (B8A−B11)/(B8A+B11); NDRE = (B8A−B05)/(B8A+B05).
~46 clear looks per field (Mar–Dec 2025). No interpolation: features are medians of observed looks per half-month
and are NaN when a half-month has no clear look.

**Spatial representation.** (a) *Within-field zones*: each field's interior pixels are split at their median row and
median column into a 2×2 grid of zones; a zone counts only with ≥ 10 pixels; every pixel belongs to one zone (no
duplication). (b) *Chips* for the deep model and console: 24 × 24 px = 240 m × 240 m at 10 m, centred on the parcel,
with parcel and interior masks.

**Sentinel-1.** RTC gamma0 in linear power. Field mean in linear power over interior pixels (≥ 20 px; speckle is
handled by aggregation), then dB = 10·log10. Cross-ratio VH/VV computed in linear units = VH_dB − VV_dB. All orbits
kept with relative orbit, pass direction and platform; features use the block's most frequent relative orbit
(consistent geometry). PC RTC has no incidence-angle layer. One HH/HV-only scene was skipped.

## Models

- **Majority**: always silage.
- **Rules v1 (original)**: line-by-line port of `src/lib/rules.ts` (Tulare thresholds, Jun–Sep peak, drop to Nov 15,
  15-day gap limit, 1.5 dB radar veto).
- **Rules adapted**: same structure; calendar, radar veto and thresholds grid-searched on train, selected on val
  (chosen: drop window to Dec 31, no radar veto, NDVI ≥ 0.5 / dry-down ≤ 7 d silage, ≥ 14 d grain, gap ≤ 25 d).
- **Random Forest**: one row per field; 500 trees, class-balanced; median imputation + missing indicators fit on
  train; 12-setting grid chosen by validation ROC-AUC; threshold chosen on validation for balanced accuracy; Platt
  calibration on validation. Acquisition counts, clear fractions and gap lengths are **excluded as predictors**
  (they track tile overlap, not crop). Feature sets: S2 summaries (105), + within-field zones, S1 only, S2 + zones + S1.
- **CNN → temporal Transformer** (`06_deep.py`, 131–135 k parameters): per-date CNN over the 10-channel chip
  (8 bands + valid mask + parcel mask), masked pooling over the parcel interior, real day-of-year sinusoidal time
  encoding, sensor embedding, S1 as a separate token branch (field-mean VV/VH/cross-ratio), 2-layer Transformer with
  padding masks. Train-only normalisation, balanced sampling, flips/rotations, token dropout and random 30–45-day
  optical blackouts. Early stopping on validation AUC (≤ 40 epochs). Variants: S2; S2+S1; S2+S1 without the CNN
  (field means only). Three seeds each.

## Evaluation (held-out regions — never used for fitting or tuning)

Split by whole regions, **identical in both years**: **train** Limburg-mid, Peel, Kempen; **val** NE Brabant, Echt;
**test** Limburg-south, Limburg-north, Gelderland-east. No block spans two splits, nearest train–test parcels ≈ 3 km
apart, and because the regions are fixed a parcel that appears in 2024 and 2025 always stays in the same split.
Positive class = **silage**; FPR = share of declared-grain fields called silage.

**Console model = E2:** trained on 2024 + 2025 train regions (4,243 field-seasons, 619 grain), tuned and calibrated
on 2024 + 2025 validation regions (2,671), tested on the 2025 test regions (1,847 = 1,643 silage, 204 grain).
Model and feature set chosen on validation AUC → **RF · S2 + zones + S1**.

| Model (identical test fields, E2) | Coverage | Bal. acc. | Silage P / R | Grain P / R | FPR | ROC-AUC | PR-AUC (grain) |
|---|---|---|---|---|---|---|---|
| Always silage | 100 % | 0.50 | 0.89 / 1.00 | — / 0.00 | 1.00 | — | — |
| Rules v1, unchanged | 31 % | 0.50 | 0.88 / 0.99 | 0.00 / 0.00 | 1.00 | 0.49 | 0.14 |
| Rules adapted (train/val) | 79 % | 0.54 | 0.91 / 0.98 | 0.39 / 0.11 | 0.89 | 0.59 | 0.18 |
| RF · S2 summaries | 100 % | 0.69 | 0.95 / 0.71 | 0.23 / 0.68 | 0.32 | 0.77 | 0.34 |
| RF · S2 + zones | 100 % | 0.71 | 0.95 / 0.72 | 0.24 / 0.70 | 0.30 | 0.78 | 0.35 |
| RF · S1 only | 100 % | 0.68 | 0.96 / 0.55 | 0.18 / 0.81 | 0.19 | 0.72 | 0.24 |
| **RF · S2 + zones + S1** (console model, chosen on val) | 100 % | **0.72** | 0.95 / 0.71 | 0.24 / 0.73 | 0.27 | **0.80** | 0.40 |
| RF · S2 + S1, data to Oct 31 only | 100 % | 0.70 | 0.95 / 0.69 | 0.22 / 0.70 | 0.30 | 0.77 | 0.37 |
| CNN→Transformer · S2 (3 seeds) | 100 % | 0.69 ± 0.01 | | | | 0.74 | |
| CNN→Transformer · S2+S1 (3 seeds) | 100 % | 0.71 ± 0.02 | | | | 0.77 | |
| Transformer, field means · S2+S1 (3 seeds) | 100 % | 0.73 ± 0.02 | | | | 0.79 | |

95 % block-bootstrap intervals (console model): balanced accuracy 0.61–0.79, ROC-AUC 0.71–0.86, grain recall
0.58–0.86, grain precision 0.13–0.35. Full numbers: `results/metrics.json`, `results/model_comparison_test.csv`,
`results/deep_*.json`, confusion matrices in `results/confusion_matrices.png`; every year experiment in
`results/experiments/`; the original 2025-only run in `results/experiments/original_2025_only/`.

**With the review band** (30 % of validation fields closest to the threshold go to an analyst): on test, 24 % of
fields are routed to review; the 76 % decided automatically reach balanced accuracy 0.77, grain recall 0.81 and
silage precision 0.97. (2025-only model: 44 % routed, 0.80 on the rest. The band is set as a share of validation fields; how many
test fields fall in it depends on the model's threshold and calibration.)

**Acreage** (test, declared: silage 4,929 ha, grain 599 ha): hard calls give grain 1,853 ha (3.1×); the sum of
calibrated scores gives grain **699 ha (+17 %)** and silage 4,829 ha. The console shows both and recommends the
calibrated line for totals.

### Does a second year help? (same regions, RF, `results/experiments/`)

| Exp. | Train years → test year | Train field-seasons (grain) | RF · S2 + zones bal. acc. / AUC | RF · S2 + zones + S1 bal. acc. / AUC | S2+S1 under simulated gap |
|---|---|---|---|---|---|
| E1 | 2025 → 2025 | 2,050 (305) | 0.71 / 0.78 | 0.70 / 0.79 | 0.69 |
| **E2** | **2024 + 2025 → 2025** | 4,243 (619) | 0.71 / 0.78 | **0.72 / 0.80** | 0.71 |
| E3 | 2024 → 2025 (next-year transfer) | 2,193 (314) | 0.68 / 0.73 | 0.69 / 0.75 | 0.67 |
| E4 | 2024 + 2025 → 2024 | 4,243 (619) | 0.60 / 0.66 | 0.65 / 0.71 | 0.65 |

- **Doubling the training data gives a small gain**, mainly for the radar model (AUC 0.79 → 0.80, balanced accuracy
  0.70 → 0.72). This is within the bootstrap interval; read it as "not worse, slightly better", not as a proven gain.
- **A model trained on a previous year still works** (E3: 0.69 on 2025 with no 2025 data), a modest loss from 0.70–0.71.
  Radar alone does not transfer across years (S1-only in E3: 0.50), but radar *combined with* optical does.
- **2024 is a harder test year** (E4: 0.65; we have not diagnosed why), so year-to-year variation is real and a
  single-year number should not be quoted as the expected accuracy.
- **Deep models gain more from the second year** than the RF: field-mean Transformer 0.70 → 0.73, CNN→Transformer
  S2+S1 0.68 → 0.71 (3 seeds each). The best deep variant is now level with the RF (0.73 ± 0.02 vs 0.72); not a
  significant difference, so the console keeps the faster, simpler RF.

### Controlled experiments — what helped and what did not (E2 unless noted)

1. **Rules vs RF.** The Tulare rules do not transfer: they abstain on 69 % of Dutch fields (mostly the 15-day gap
   limit) and never call grain correctly. Adapting thresholds barely helps (0.54). The RF on S2 alone reaches
   0.69–0.71. The dry-down premise is weak in NL: the date NDVI first falls below 0.6 has the same median for both
   classes.
2. **S2 vs S2 + S1 (radar).** Small gain on normal data (0.708 → 0.719 balanced accuracy, AUC 0.776 → 0.800;
   McNemar p = 0.23, not significant). **Under a simulated optical gap** (all clear looks removed Sep 1 – Oct 15 in
   test fields, clearly simulated), radar matters: RF 0.68 → 0.71, AUC 0.74 → 0.79; deep S2 0.65 vs S2+S1 0.70–0.72.
3. **Field summaries vs + within-field zones.** RF on S2: 0.694 → 0.708 (McNemar p = 0.04), AUC 0.771 → 0.776. Zone
   variation did not predict errors on validation, so it is shown to analysts but does not trigger review.
4. **Best RF vs CNN→Transformer.** With two years the neural models catch up (0.69–0.73, AUC 0.74–0.79) but do not
   beat the RF; the CNN over 10 m chips adds nothing over field means.
5. **Timeliness.** Using only data up to Oct 31 gives 0.70 / AUC 0.77 — an in-season estimate is feasible at a small cost.

**Analyst insight (validated on val, confirmed on test).** Dutch autumn cloud gives a median 23-day clear-sky gap
around harvest, so the California 15-day gap rule would queue 91 % of fields — but gap length does not predict
errors, because the model leans on the post-harvest signal: silage fields are cut earlier and carry green
catch-crop cover by Nov–Dec (NDVI ≈ 0.77 vs 0.52 for grain). Review effort should go to the score band and large
uncertain fields, not to every gap.

**Most important features** (RF impurity, console model): NDRE and NDVI in December, NDRE in May, November radar
cross-ratio, VH change across the NDVI fall, LSWI in December, LSWI after the fall.

### Spectral indices (NDRE, PSRI, NDTI) — tested, not adopted

On a common 20 m grid, with the same fields, split, seed and 12-setting tuning budget, adding NDRE, PSRI or NDTI
(alone or together) to an NDVI + LSWI baseline did not give a validated improvement. Test ROC-AUC changes ranged
from −0.015 to +0.004, every balanced-accuracy change sat inside a block-bootstrap interval around zero, and
inference time was unchanged. The validation-selected configuration (S2 + S1 + NDTI) won validation by 0.0004 AUC,
a tie, and was slightly worse on test. The console therefore keeps its 10 m model. Full report:
[`results/experiments/indices_20m/README.md`](results/experiments/indices_20m/README.md).

### Latency

| What | Time | Conditions |
|---|---|---|
| S2 retrieval + preprocessing | 55–80 s per ~10×10 km block (≈ 250–520 fields, ~105 dates) | 12 threads, anonymous AWS COGs |
| S1 retrieval + preprocessing | 100–260 s per block (~90–230 scenes) | Planetary Computer, throttled at 5 workers |
| RF inference, batch | 0.07 s for 1,847 fields | lab CPU, cached features |
| RF inference, single field | ~39 ms | dominated by joblib start-up |
| Console: load analysis | 1.4 s | 9 MB JSON, localhost, headless Chromium |
| Console: switch field | median 320 ms | cached evidence |

Cached console speed is not end-to-end processing speed.

## Console integration

`07_export_console.py` writes the **1,847 2025 test-region fields** (predictions from the E2 model, which never saw
these regions in either year, so every field shown is a genuine held-out prediction): BRP geometry and RD-New area,
provenance, calibrated score, review reasons, observation gaps, NDVI/LSWI and VV/VH series, 2×2 zone series, the
transition interval (never an invented harvest date) and 10 Sentinel-2 RGB/NDVI chips from real dates with the
parcel outline. Reference labels live in `public/nl/nl2025_reference.json` and are read only by the Reports →
evaluation panel, which also shows the E1–E4 table. Review triggers for NL were chosen on validation (score band,
large uncertain fields, radar not registering the change, gaps > 45 d, 5 % audit).

## Limitations and unvalidated claims

- Labels are declarations; a declared-grain field chopped for silage counts as a model error. Some confident
  "errors" (e.g. NL-GEN-0093: green until late Aug, cut by Sep 10, cover crop afterwards) look like that, but this is
  not verified.
- Two years (2024, 2025), one region of one country. Accuracy differs by year (0.65 on 2024 vs 0.72 on 2025); no
  cross-country test; California transfer is untested.
- Grain precision is low (≈ 0.24 at 11 % prevalence); field-level grain calls need review.
- Calibration was fit on validation regions whose prevalence differs from test; the review band sized for 30 % on
  validation routed 24 % on test.
- Within-field zone variation is variation in timing, not proof of a mixed harvest; no sub-field acreage is allocated.
- PDOK GeoPackages exist for 2017–2023 as well; more years were not added.
