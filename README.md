# Harvest Console (prototype)

Console for reviewing whether corn fields were harvested for silage or grain. It holds two real-data datasets that
are never mixed:

1. **South-east Netherlands 2025 — validated (opens by default).** 1,847 held-out maize parcels from PDOK BRP 2025
   (farmer-declared silage 259 / grain 316) with Sentinel-2 + Sentinel-1 evidence and Random Forest predictions from
   a model trained on *other* regions in 2024 + 2025 (4,243 field-seasons). Held-out balanced accuracy 0.72, ROC-AUC
   0.80 (rules: 0.50); a model trained on 2024 alone scores 0.69 on 2025. Pipeline, evaluation, label
   audit and limitations: [`pipeline/README.md`](pipeline/README.md); demo script: [`pipeline/PRESENTATION.md`](pipeline/PRESENTATION.md).
   Data: `public/nl/` (reference labels in a separate file, read only by Reports → evaluation).
2. **Tulare County, CA 2023 — rule-based, unvalidated.** Described below.

Tulare County, CA, 2023 season:

- **Field outlines:** [DWR 2023 Statewide Crop Mapping](https://data.cnra.ca.gov/dataset/statewide-crop-mapping),
  crop code F16 (corn, sorghum or sudan), 48 fields in four sample areas.
- **Evidence:** Copernicus Sentinel-2 L2A (cloud-masked NDVI / LSWI per clear look, 3 × 2 sub-field grid, true-colour
  and NDVI image chips) and Sentinel-1 RTC (VV / VH backscatter, one relative orbit), via Microsoft Planetary Computer.
- **Predictions:** rule-based (`src/lib/rules.ts`). There is no trained ML model yet, and **no ground truth**: labels
  only become trusted when an analyst reviews them. The review history starts empty.

Contains modified Copernicus Sentinel data 2023.

```bash
npm install
npm run dev              # local dev server
npm run build            # standard static build → dist/
npm run build:artifact   # single-file build → dist-artifact/harvest-console.html
```

Refresh the data (Python 3, standard venv):

```bash
python3 scripts/fetch-dwr-fields.py                      # field outlines → src/data/dwrFields.ts
python3 -m venv .venv && .venv/bin/pip install pystac-client planetary-computer rasterio numpy pyproj shapely pillow
.venv/bin/python scripts/fetch-sentinel.py               # → src/data/sentinel2023.json + public/s2/<field>/<date>_{rgb,ndvi}.jpg
```

## Where things live

| Path | Purpose |
| --- | --- |
| `src/types.ts` | Typed models: fields, predictions, observations, subregions, decisions, history |
| `src/data/dwrFields.ts` | Real corn field outlines (DWR 2023, 4 clusters × 12), with DWR `UniqueID`s |
| `src/data/sentinel2023.json` | Real per-field Sentinel-2 / Sentinel-1 series (generated) |
| `src/data/generate.ts` | Builds fields, evidence, review flags and showcase examples from the real data |
| `public/s2/` | Sentinel-2 image chips per field and clear date (generated) |
| `src/services/analysisService.ts` | Data access layer. Replace these functions with real API calls |
| `src/lib/acreage.ts` | Single source of acreage allocation; summaries, map and CSV all use it |
| `src/lib/rules.ts` | Rule set: time-series features, weak labels (silage / grain / abstain), rule-based predictions |
| `src/components/WeakLabelsView.tsx` | Weak labels tab: tune thresholds, compare with analysts, export a training CSV |
| `src/lib/priority.ts` | Explainable queue ranking |
| `src/state/store.ts` | App state, decisions, undo, browser persistence |
| `src/components/` | UI |

Review decisions are stored in `localStorage` only. There is no backend or multi-user storage.

## Predictions and weak labels

There is no trained model yet, so predictions are **rule-based** (`src/lib/rules.ts`): silage = still green
(NDVI ≥ 0.60) with ≤ 14 days of dry-down before an abrupt drop; grain = ≥ 25 days of dry-down with NDVI ≤ 0.55 before
the drop. Fields with too few clear looks, a cloud gap over the drop or optical/radar timing disagreement abstain.
The **Prediction** toggle in the top bar has an **ML model** option that stays disabled until a trained model's
predictions are served (`ML_AVAILABLE` in `src/lib/format.ts`).

The **Weak labels** tab runs the same rules as a labelling function. Analyst silage/grain decisions override weak labels,
and the export (`training_label`, `label_source`) is the starting training set for the first model. Weak labels are not
ground truth: test any model on analyst-reviewed fields held out by area and year.
