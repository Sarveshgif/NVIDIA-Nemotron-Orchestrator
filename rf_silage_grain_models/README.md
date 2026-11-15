# Silage vs grain maize: Random Forest models

Two files, same trained forest (scikit-learn RandomForestClassifier, 500 trees, 135 features):

| File | What it is | Use for |
|---|---|---|
| `rf_nl_console.joblib` | The model as trained and calibrated in the Netherlands (threshold 0.88, review band) | Dutch / NW-European maize |
| `rf_us_barc_recalibrated.joblib` | The same forest + a US recalibration of its probability, fitted on 44 labelled field-years from one US research farm (BARC, Beltsville, Maryland, 2021–2023); call threshold 0.5 | First attempt for the US; see the caveats |

**Requirements:** Python 3.10+, scikit-learn 1.7.x, joblib, numpy, pandas. The pickles are tied to the
scikit-learn version.

```python
import pandas as pd
from predict import load, predict
b = load("rf_nl_console.joblib")
out = predict(b, features)        # features: DataFrame, one row per field-season, columns from feature_columns.json
# out.p_silage (probability of silage), out.call ("silage"/"grain"), out.review (NL bundle: borderline)
```

## Inputs (must be computed the same way)

The 135 columns (`feature_columns.json`) are per-field seasonal features built by
`04_build_features.py` (`s2_features` + `s1_features`):
- Sentinel-2 L2A field-mean NDVI / LSWI / NDRE as half-month medians (Apr–Dec)
- B04 / B11 / B12 (Aug–Nov)
- transition and dry-down timing
- 2×2 within-field zones
- Sentinel-1 RTC VV / VH / cross-ratio (monthly medians in dB, plus changes around the greenness drop)

Field means come from the parcel interior after a 10 m inward buffer, using cloud-free pixels only (SCL 4/5).
NaN means "not observed"; the model imputes it. **Features computed differently will give wrong results.**

## How good is it

| Test | Result |
|---|---|
| NL held-out regions, 2025 (1,847 fields, never used in training) | balanced accuracy 0.72, ROC-AUC 0.80 |
| BARC, Maryland (44 field-years), NL model as-is | balanced accuracy 0.86, AUC 0.95 |
| BARC, US-recalibrated (leave-one-field-out) | balanced accuracy 0.90; silage area share 30% vs 35% declared |
| West Virginia 2025 vs NASS state total (22.5% silage) | NL cut-off 9%; US-recalibrated, probability-weighted 15% |
| Rhode Island 2025 vs NASS (~100% silage) | NL cut-off 41%; US-recalibrated ~47–54% (does **not** fix New England) |

## Caveats

- **Labels:** the NL training labels are farmer declarations (BRP), not verified harvests.
- **Scope of the US recalibration:** it comes from one Maryland farm with 12 silage field-years. It improves
  BARC-like (Mid-Atlantic, grain-majority) settings and does not transfer to all US regions.
- **Probabilities:** NL probabilities are calibrated for NL (about 90% silage). Recalibrate on local labels before
  summing probabilities into acreage.

Data: Contains modified Copernicus Sentinel data 2024–2025; BRP © RVO/PDOK.
