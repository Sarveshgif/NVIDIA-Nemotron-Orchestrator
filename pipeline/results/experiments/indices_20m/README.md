# Do NDRE, PSRI and NDTI help? — spectral-index experiment (20 m grid)

**Answer: no.** None of the three indices, alone or together, gives a validated improvement over the baseline.
Every change in test balanced accuracy and ROC-AUC lies inside its 95 % block-bootstrap interval around zero,
except NDTI added to the S2+S1 model, which is slightly *worse* on test AUC (−0.015, CI −0.027 to −0.003).
Inference cost is unchanged (≤ 3 ms per batch of 1,847 fields).

Run 2026-11-14 on the lab workstation. Scripts: `02b_extract_s2_20m.py` → `09_index_features.py` →
`10_index_experiment.py` (chained by `run_indices.sh`). Raw outputs: `index_experiment.json`,
`index_experiment_test.csv`, `index_experiment_val.csv`.

## Setup (held fixed for every configuration)

- **Data.** Same fields and region split as the console model (E2): train 4,243 and validation 2,671 field-seasons
  from the 2024 + 2025 train/val regions; test 1,847 fields in the 2025 test regions (204 grain).
- **Common 20 m grid.** The tile's native 20 m grid. B05, B06, B8A, B11, B12 and SCL are read natively; B02, B04 and B08
  are aggregated by a 2 × 2 mean of the native 10 m pixels. A 20 m pixel is invalid if any contributing DN is nodata (0).
  Reflectance = DN × 0.0001 − 0.1 for every band (from the asset metadata). Check: 20 m NDVI vs the 10 m pipeline,
  r = 0.9999, mean difference 0.002.
- **Cloud masking unchanged.** SCL 4/5 = clear; a field-date counts when ≥ 60 % of interior pixels are clear and
  ≥ 3 valid pixels remain. Zero denominators, and PSRI where B06 < 0.01, become missing for that index only.
  The forest's imputer fills missing values with the training median and adds missing-indicator columns.
- **Indices.** NDRE = (B8A − B05)/(B8A + B05); PSRI = (B04 − B02)/B06; NDTI = (B11 − B12)/(B11 + B12).
- **Temporal features, same template as the existing pipeline.** 18 half-month medians (Apr–Dec) of observed looks
  (never interpolated), plus the index value just before and just after the NDVI transition: 20 features per index.
- **Baseline.** NDVI + LSWI half-months, Aug–Nov B04/B11/B12, transition/phenology features (81), and 2 × 2 zones (5).
  Optional S1 radar block (30). The baseline contains no NDRE, PSRI or NDTI.
- **Model and tuning budget.** Random Forest, 500 trees, seed 20251114, the same 12 hyper-parameter settings chosen by
  validation ROC-AUC, the threshold chosen on validation for balanced accuracy, Platt calibration on validation. This
  is identical to `05_models.py`.
- **Selection.** Validation ROC-AUC, the same rule as the main pipeline. The test set was scored once per
  configuration, after all selection was fixed.

## Validation (used for selection)

| Configuration | Val ROC-AUC (Δ) | Val bal. acc. | Val macro-F1 |
|---|---|---|---|
| S2 · baseline | **0.7065** | 0.664 | 0.540 |
| S2 · +NDRE | 0.6993 (−0.007) | 0.654 | 0.553 |
| S2 · +PSRI | 0.7010 (−0.006) | 0.668 | 0.584 |
| S2 · +NDTI | 0.7064 (−0.000) | 0.665 | 0.569 |
| S2 · +all three | 0.6995 (−0.007) | 0.660 | 0.565 |
| S2+S1 · baseline | 0.7141 | 0.684 | 0.635 |
| S2+S1 · +NDRE | 0.7108 (−0.003) | 0.673 | 0.572 |
| S2+S1 · +PSRI | 0.7098 (−0.004) | 0.689 | 0.586 |
| S2+S1 · +NDTI | **0.7144 (+0.0004)** | 0.681 | 0.633 |
| S2+S1 · +all three | 0.7103 (−0.004) | 0.674 | 0.623 |

Selected on validation: **S2 · baseline** in the S2 family; **S2+S1 · +NDTI** in the S2+S1 family and overall.
The S2+S1 pick is a tie: +0.0004 AUC, while validation balanced accuracy and macro-F1 are slightly lower than the
baseline's.

## Test (final comparison; Δ vs the baseline of the same family)

Silage is the positive class. FPR is the share of declared-grain fields called silage.

| Configuration | Features | Silage P / R / F1 | Grain P / R / F1 | FPR (Δ) | Bal. acc. (Δ) | ROC-AUC (Δ) | Batch, 1,847 fields (Δ) | Per field (Δ) |
|---|---|---|---|---|---|---|---|---|
| S2 · baseline | 86 | 0.96 / 0.63 / 0.76 | 0.20 / 0.77 / 0.32 | 0.23 | 0.699 | 0.779 | 0.072 s | 36.1 ms |
| S2 · +NDRE | 106 | 0.95 / 0.68 / 0.80 | 0.22 / 0.72 / 0.34 | 0.28 (+0.05) | 0.700 (+0.001) | 0.776 (−0.003) | 0.073 s (+0.001) | 35.3 ms (−0.8) |
| S2 · +PSRI | 106 | 0.95 / 0.70 / 0.81 | 0.23 / 0.71 / 0.34 | 0.29 (+0.06) | 0.703 (+0.004) | 0.767 (−0.012) | 0.073 s (+0.001) | 35.7 ms (−0.4) |
| S2 · +NDTI | 106 | 0.95 / 0.69 / 0.80 | 0.22 / 0.72 / 0.34 | 0.28 (+0.05) | 0.702 (+0.003) | 0.771 (−0.008) | 0.073 s (+0.001) | 36.8 ms (+0.6) |
| S2 · +all three | 146 | 0.95 / 0.68 / 0.79 | 0.22 / 0.74 / 0.34 | 0.26 (+0.03) | 0.705 (+0.007) | 0.779 (+0.001) | 0.075 s (+0.003) | 36.5 ms (+0.4) |
| S2+S1 · baseline | 116 | 0.95 / 0.76 / 0.84 | 0.26 / 0.67 / 0.37 | 0.33 | 0.716 | 0.802 | 0.072 s | 35.6 ms |
| S2+S1 · +NDRE | 136 | 0.96 / 0.68 / 0.79 | 0.23 / 0.77 / 0.35 | 0.23 (−0.10) | 0.723 (+0.007) | 0.806 (+0.004) | 0.073 s (+0.001) | 35.4 ms (−0.2) |
| S2+S1 · +PSRI | 136 | 0.96 / 0.66 / 0.78 | 0.22 / 0.76 / 0.34 | 0.24 (−0.09) | 0.709 (−0.008) | 0.806 (+0.003) | 0.073 s (+0.001) | 36.6 ms (+1.0) |
| S2+S1 · +NDTI ★ | 136 | 0.95 / 0.77 / 0.85 | 0.26 / 0.65 / 0.37 | 0.35 (+0.02) | 0.709 (−0.007) | 0.787 (−0.015) | 0.074 s (+0.001) | 35.1 ms (−0.5) |
| S2+S1 · +all three | 176 | 0.95 / 0.76 / 0.85 | 0.26 / 0.68 / 0.38 | 0.32 (−0.01) | 0.723 (+0.006) | 0.800 (−0.003) | 0.075 s (+0.002) | 36.5 ms (+0.9) |

★ = selected on validation. Inference times are on the lab CPU (64 cores) from cached features, the median of 5
batch runs and the mean of 50 single-field calls; the single-field time is dominated by start-up overhead.
Computing three more indices during extraction adds negligible time next to the image reads.

**Paired block-bootstrap 95 % intervals for the change vs baseline** (2,000 resamples of the 8 test blocks' fields):

| Added | S2 Δ bal. acc. | S2 Δ AUC | S2+S1 Δ bal. acc. | S2+S1 Δ AUC |
|---|---|---|---|---|
| NDRE | −0.023 to +0.025 | −0.010 to +0.005 | −0.026 to +0.041 | −0.006 to +0.011 |
| PSRI | −0.028 to +0.035 | −0.033 to +0.005 | −0.043 to +0.027 | −0.005 to +0.008 |
| NDTI | −0.022 to +0.026 | −0.024 to +0.009 | −0.020 to +0.011 | **−0.027 to −0.003** |
| all three | −0.025 to +0.032 | −0.012 to +0.014 | −0.008 to +0.023 | −0.012 to +0.005 |

## How to read it

- **Threshold-free ranking barely moves.** ROC-AUC changes by −0.015 to +0.004. The larger swings in precision,
  recall, F1 and FPR (for example FPR −0.10 with +NDRE in S2+S1) come from each configuration choosing a different
  threshold on validation: the models trade silage recall for grain recall along nearly the same ROC curve.
  They are not better separation.
- **The information is already present.** The baseline already carries the red-edge, senescence and residue signal
  through NDVI/LSWI in December, B04/B11/B12 in Aug–Nov, and the transition features. NDRE, PSRI and NDTI are highly
  correlated with these at field-mean level.
- **McNemar is not used as evidence.** An exact McNemar test on raw accuracy (in the JSON) is dominated by the
  89 % silage class. It mostly measures the threshold shifts above, not grain detection.
- **The 20 m grid costs nothing.** The 20 m S2+S1 baseline (test balanced accuracy 0.716, AUC 0.802) matches the
  10 m console model (0.719 / 0.800).

## Retained configuration

By the pre-declared rule (validation ROC-AUC), the retained configuration is **S2+S1 · +NDTI**
(`models/index_best.joblib` on the lab machine; test predictions in `index_best_test_preds.parquet`). Its validation
lead is 0.0004 AUC, which is a tie, and on test it is slightly worse than its own baseline. **The console is not
switched.** Its 10 m S2 + zones + S1 model is equivalent on validation and test and does not require the 20 m
re-extraction. Recommendation: keep the current pipeline and do not add these indices.

## Limitations

- One test year (2025), three test regions, 204 grain fields. Changes below about ±0.03 balanced accuracy cannot be
  resolved.
- Labels are farmer declarations.
- Indices enter as field means. Pixel-level or chip-level use (for example in the CNN) was not tested.
