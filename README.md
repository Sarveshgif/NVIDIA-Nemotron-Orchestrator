# NVIDIA Nemotron Orchestrator — Hack the Harvest

Classifying corn fields as **harvested for silage or for grain** from satellite time series, with a calibrated
**Random Forest** as the primary classifier, a **CNN → temporal Transformer** deep model trained on the same data
for comparison and US transfer, and an **NVIDIA Nemotron** agent as a downstream investigator for the fields the RF
is unsure about.

> Built for the Hack the Harvest hackathon. Original team repository: `pavan-nuthi/hack-the-harvest`
> (commit history preserved here). The earlier console README is kept in [`HARVEST_CONSOLE.md`](HARVEST_CONSOLE.md).

---

## 1. Problem statement

Corn (maize) is grown for two very different end uses:

- **Silage** — the whole green plant is chopped while still moist (roughly 60–70 % moisture) and fermented as cattle feed.
- **Grain** — the crop is left to dry down in the field and only the kernels are harvested weeks later.

Knowing which is which, field by field, matters for crop-production statistics, feed and grain supply estimates,
insurance and carbon/residue accounting. But public crop maps (e.g. USDA Cropland Data Layer) only say **"corn"**.
The challenge: **infer silage vs grain from what satellites can see** — how long a field stays green, how it dries
down, and when (and how abruptly) the canopy disappears — and do it honestly, with real labels, held-out
evaluation and a reviewable trail of evidence.

What makes it hard:

| Difficulty | Why it matters |
|---|---|
| No US field-level labels | CDL does not separate silage from grain, so a model needs labels from elsewhere. |
| Clouds in autumn | The harvest window (Sep–Nov) is the cloudiest; median clear-sky gap in NL autumn ≈ 23 days. |
| Similar phenology | Silage is cut green, grain dries first — but timing varies by hybrid, year and weather. |
| Class imbalance | Grain is ~12 % of fields in the study area, so precision on grain is the weak point. |
| Labels are declarations | Farmers declare *intended* use in spring; a field declared grain may still be chopped. |

## 2. Approach at a glance

```
 Field polygons + labels                Satellite evidence                       Decision
 ───────────────────────     ─────────────────────────────────────     ──────────────────────────────
 Dutch BRP parcels  ───────▶ Sentinel-2 L2A (NDVI, LSWI, NDRE, bands) ─▶ Random Forest (135 features)
 (silage 259 / grain 316)    Sentinel-1 RTC (VV, VH, VH/VV, dB)          calibrated p(silage), p(grain)
 BARC corn 2021–2023         2×2 within-field zones                                │
 (US dev set, 44 fields)                                                           ▼
                                                              max class score ≥ 0.80 ──▶ keep RF label
                                                              max class score < 0.80 ──▶ Nemotron investigation
                                                                                          │  Sentinel trajectories
                                                                                          │  Daymet weather + history
                                                                                          │  USGS 3DEP elevation
                                                                                          │  v4 rulebook finding
                                                                                          │  NASS / public reports
                                                                                          ▼
                                                                      cited GRAIN / SILAGE hypothesis
                                                                      (falls back to RF if it fails)
```

## 3. Stage 1 — Random Forest (`pipeline/`, `rf_silage_grain_models/`)

**Why the Netherlands.** No public US source labels individual corn fields as silage vs grain. The Dutch parcel
register (PDOK BRP) does: crop code **259 "Mais, snij-"** (silage) and **316 "Mais, korrel-"** (grain), open and
geometry-complete. That gives a real train/test evaluation.

**Data.** 20 blocks (~10 × 10 km) in south-east NL, 2024 + 2025 → 5,169 (2025) and 5,446 (2024) eligible fields
(≥ 50 interior 10 m pixels). Sentinel-2 L2A from Element84 Earth Search; Sentinel-1 RTC from Microsoft Planetary
Computer. No interpolation: features are half-month medians of real clear looks, NaN when none exist.

**Model.** 500-tree class-balanced Random Forest, one row per field; median imputation + missing indicators;
hyper-parameters picked on validation ROC-AUC; threshold picked on validation for balanced accuracy; Platt
calibration on validation. Observation counts / gap lengths are deliberately **excluded** (they track tile
overlap, not crop). Final feature set: **S2 summaries + within-field zones + S1 radar** (135 features,
contract `rf_s2s1_135_v1`).

**Evaluation — held-out regions, never used for fitting or tuning** (train: Limburg-mid, Peel, Kempen;
val: NE Brabant, Echt; test: Limburg-south, Limburg-north, Gelderland-east; 1,847 test fields in 2025):

| Model (same 1,847 test fields) | Coverage | Bal. acc. | ROC-AUC | Grain P / R |
|---|---|---|---|---|
| Always silage | 100 % | 0.50 | — | — / 0.00 |
| Original rulebook (Tulare thresholds) | 31 % | 0.50 | 0.49 | 0.00 / 0.00 |
| Rulebook, thresholds re-tuned | 79 % | 0.54 | 0.59 | 0.39 / 0.11 |
| RF · S2 only | 100 % | 0.69 | 0.77 | 0.23 / 0.68 |
| **RF · S2 + zones + S1 (deployed)** | 100 % | **0.72** | **0.80** | 0.24 / 0.73 |
| CNN → Transformer · S2 + S1 (3 seeds, RTX 4070 Ti) | 100 % | 0.71 ± 0.02 | 0.77 | |
| Transformer on field means · S2 + S1 (3 seeds) | 100 % | 0.73 ± 0.02 | 0.79 | |

95 % block-bootstrap CI for the deployed RF: balanced accuracy 0.61–0.79, ROC-AUC 0.71–0.86.
With a **review band** (30 % of validation fields nearest the threshold go to an analyst) the 76 % of test fields
decided automatically reach balanced accuracy 0.77 and grain recall 0.81. Calibrated scores also give much better
acreage totals than hard calls (grain 699 ha vs 599 ha declared, +17 %, instead of 3.1× with hard labels).

Key findings: radar helps most when autumn clouds hide the optical signal (simulated Sep 1–Oct 15 gap: AUC
0.74 → 0.79); a model trained on 2024 alone still reaches 0.69 on 2025; deep models tie the RF on NL (section 4),
so the faster RF stays. Full method, ablations, label audit and limitations: [`pipeline/README.md`](pipeline/README.md).

**US transfer.** `rf_us_barc_recalibrated.joblib` re-fits only the probability mapping on 44 BARC (Beltsville, MD)
corn field-years, 2021–2023 (`BARC Corn 2021-2023.gpkg`). Because those labels were used for calibration, BARC
numbers are **development results, not held-out accuracy**.

## 4. Deep model — CNN → temporal Transformer (`pipeline/06_deep.py`, `pipeline/13_barc_deep.py`, `models/deep_e2/`)

A second, independent classifier that learns directly from image chips and irregular time series instead of
hand-built features. Trained on the lab RTX 4070 Ti (PyTorch 2.6, CUDA 12.4); ~131–135 k parameters.

**Architecture.**

```
 per clear S2 date: 24×24 chip (240 m), 10 channels          per S1 date: field-mean VV, VH, VH−VV (dB)
 (8 bands + valid mask + parcel mask)                                     │
          │                                                        MLP 3 → 64 → 64
   3-layer CNN 10→32→64→64 (stride 2)                                      │
   masked mean-pool over parcel interior                                   │
   ⊕ 8 field-mean bands → Linear → 64                                      │
          │                                                                │
          └──────── + sinusoidal day-of-year encoding + sensor embedding ──┘
                                      │
                 [CLS] + S2 tokens (≤ 64) + S1 tokens (≤ 64), padding masked
                                      │
              2-layer Transformer encoder (d = 64, 4 heads, pre-norm)
                                      │
                         LayerNorm → Linear → p(silage)
```

- **No resampling or interpolation:** one token per real acquisition, positioned by its actual day of year, so
  cloudy gaps are simply missing tokens.
- **Variants:** `s2` (optical chips only), `s2s1` (chips + radar tokens), `s2s1_nocnn` (field-mean bands + radar,
  no CNN — tests whether spatial detail helps).
- **Training:** train-region normalisation only, class-weighted BCE, AdamW (lr 1e-3, wd 1e-2), flips/rot90, random
  S2 token dropout, and a random **30–45-day optical blackout** (Aug 15–Nov 15) on 30 % of samples to make the model
  robust to autumn clouds. Early stopping on validation AUC (≤ 40 epochs); threshold and Platt calibration on
  validation. 3 seeds per variant; 80–360 s training per run on the GPU, ~1.5–2 s to score 1,847 test fields.

**NL held-out test (E2, same 1,847 fields as the RF, mean ± sd over 3 seeds):**

| Variant | Bal. acc. | ROC-AUC | Under simulated Sep 1–Oct 15 optical gap |
|---|---|---|---|
| CNN → Transformer · S2 | 0.69 ± 0.01 | 0.74 | 0.65 |
| CNN → Transformer · S2 + S1 | 0.71 ± 0.02 | 0.77 | 0.70–0.72 |
| Transformer on field means · S2 + S1 | **0.73 ± 0.02** | **0.79** | 0.70–0.72 |
| *RF · S2 + zones + S1 (reference)* | *0.72* | *0.80* | *0.71* |

The deep models gain more from the second training year than the RF does (field-mean Transformer 0.70 → 0.73) and
radar makes them markedly more robust to optical gaps. On NL they tie the RF; the CNN over 10 m chips adds nothing
over field means.

**US external check on BARC (44 field-years, Beltsville MD, 2021–2023; 12 silage / 32 grain).**
`13_barc_deep.py` loads the 9 saved E2 checkpoints **without retraining or tuning anything on BARC**. Each
checkpoint's normalisation, calibration and threshold are rebuilt from NL and verified to reproduce its saved NL test
metrics exactly before scoring BARC. Labels are joined only after predictions are written. Chips of 24 px (as in
training) and 64 px (covers whole BARC parcels) were both tested.

| Model (nothing tuned on BARC) | Bal. acc. | ROC-AUC | Silage TP / FN | Grain called silage (FP) |
|---|---|---|---|---|
| RF · S2 + zones + S1 (NL threshold/calibration; [`barc_rf/`](pipeline/results/barc_rf/barc_eval.json)) | 0.859 | 0.948 | 9 / 3 | 1 |
| CNN → Transformer · S2, 3-seed ensemble | 0.833 | 0.938 | 8 / 4 | 0 |
| CNN → Transformer · S2 + S1, 3-seed ensemble | 0.833 | **0.992** | 8 / 4 | 0 |
| **Transformer field means · S2 + S1, 3-seed ensemble** | **0.917** | 0.969 | **10 / 2** | 0 |

Per-seed results (24 px): S2 0.80 ± 0.07, S2 + S1 0.83 ± 0.08, field means 0.84 ± 0.03 balanced accuracy.
Models trained only on Dutch fields transfer to a US research farm, and the radar-fused Transformer ensembles rank
BARC fields better than the RF. **Caveat:** 44 fields with 12 silage — the gap between 0.86 and 0.92 is one or two
fields and well inside the RF's bootstrap interval (0.72–0.99), so this is an encouraging signal, not a proven win.
Full outputs: [`pipeline/results/barc_deep/`](pipeline/results/barc_deep/).

**Reproduce.**

```bash
HTH_SEED=1 python pipeline/06_deep.py s2 s2s1 s2s1_nocnn          # repeat for seeds 2, 3 (GPU recommended)
# BARC: 11_barc_prepare.py → 11_barc_extract.py <year> (HTH_CHIP=24|64) → 12_barc_infer.py (RF) → 13_barc_deep.py
HTH_BARC=<dir> HTH_BARC_CHIPS=<dir> E2_RESULTS=<e2 results dir with models/> python pipeline/13_barc_deep.py
```

The trained checkpoints are in `models/deep_e2/` (9 × ~0.55 MB); `13_barc_deep.py` expects them under
`$E2_RESULTS/models/`.

## 5. Stage 2 — Nemotron downstream investigation (`investigation/`)

The RF is good on average but weakest exactly where the satellite record is ambiguous. Stage 2 spends extra effort
only on those fields, using **`nvidia/nemotron-3.5-super-vl-120b-a12b`** via the NVIDIA API as a tool-using agent.

**Routing.** If the RF's assigned-class score is **≥ 0.80**, the RF label is kept and no model or external call is
made. Below 0.80 (or on missing quality, contradictions, review band or unvalidated geography) the field is
investigated.

**Evidence the backend gathers before calling Nemotron** (`INVESTIGATION_PREFETCH_EVIDENCE=true`):

| Tool / source | What it adds |
|---|---|
| Sentinel-2 / Sentinel-1 (`sentinel.py`) | Field-masked NDVI and sub-field statistics on a 20 m grid; VV/VH radar from one orbit/pass |
| v4 rulebook (`rulebook_v4.py`) | Deterministic dry-down / abrupt-drop finding, or UNRESOLVED with the reason |
| Daymet V4 (`weather.py`, `historical_context.py`) | Daily weather, corn GDD, frost, rain around radar dates, 30/60/90-day anomalies vs 1981–2010 |
| USGS 3DEP (`tools.py`) | Field elevation statistics |
| NASS Quick Stats, curated UMD reports | State crop progress / regional context (never used to label a single field) |

**What Nemotron must return.** A structured `evaluate_evidence` call with a **GRAIN or SILAGE** hypothesis,
qualitative evidence strength, cited evidence IDs and reason codes. It may not invent facts, dates, URLs or
probabilities. The backend (`policy.py`, `assessment.py`) validates every citation and prerequisite.

**Safeguards.**
- Original RF probabilities are never changed; `final_probability` stays `null` (no fusion calibration was trained).
- Hard budgets: ≤ 8 tool attempts, ≤ 6 model rounds, ≤ 30 public HTTP requests, 180 s per field; bounded format corrections.
- If Nemotron times out, fails or exhausts corrections, the field falls back to the RF label (`rf_fallback`) and the failure is logged.
- Reference labels are only joined *after* predictions are frozen (SHA-256 recorded); no inference path reads them.
- Retrieved web text is treated as untrusted data, never as instructions.

### Measured effect so far (BARC development set, `public/barc/evaluation.json`)

| | RF only | RF + Nemotron |
|---|---|---|
| Fields (high-score kept / investigated) | 44 | 35 / 9 |
| Correct | 37 / 44 (84.1 %) | 37 / 44 (84.1 %) |
| Investigated fields correct | 6 / 9 | 6 / 9 |
| Labels changed by Nemotron | — | 0 |
| Mean backend time, investigated field | — | ~46 s |

**Honest read:** on this small development set Nemotron agreed with the RF on every investigated field, so it has
**not yet shown an accuracy gain**. What it adds today is an auditable, source-cited explanation for every uncertain
field (weather, radar, rulebook evidence) instead of a bare score. Demonstrating improvement needs an independent
held-out labelled set — see next steps. The deep models' independent BARC result (section 4) also suggests a second
opinion worth routing alongside the RF for uncertain fields.

## 6. Repository layout

| Path | Contents |
|---|---|
| `pipeline/` | NL data pipeline: block scan → S2/S1 extraction → features → RF/rules/deep models → console export; results and figures |
| `pipeline/06_deep.py` | CNN → temporal Transformer (variants `s2`, `s2s1`, `s2s1_nocnn`) |
| `pipeline/11_barc_*.py`, `12_barc_infer.py`, `13_barc_deep.py` | BARC external check: prepare → extract → RF inference → deep-model inference |
| `models/deep_e2/` | 9 trained deep checkpoints (3 variants × 3 seeds, E2: trained on NL 2024 + 2025) |
| `pipeline/results/barc_deep/` | Deep-model BARC predictions, matched labels, evaluation JSON and run log |
| `rf_silage_grain_models/` | Saved RF bundles (`rf_nl_console.joblib`, `rf_us_barc_recalibrated.joblib`), `predict.py`, 135-feature contract |
| `investigation/` | RF + Nemotron backend ("Cornstellation"): routing, tools, NVIDIA transport, policy, CLI and FastAPI server |
| `tests/investigation/` | Unit tests for the agent protocol and prediction policy |
| `data_barc_rules/` | BARC 2021–2023 observations, v4 rulebook outputs, Nemotron reviews, evaluations |
| `data_alabama2025/` | Alabama 2025 CDL tiles and patch inventory (US expansion work) |
| `public/` | Console data: `nl/` (held-out NL predictions + chips), `barc/` (reports and evaluation) |
| `src/` | Harvest Console — React + TypeScript + Leaflet analyst UI |

## 7. Running it

```bash
# Analyst console
npm install
npm run dev

# RF + Nemotron backend
python3 -m venv .venv-investigation
.venv-investigation/bin/pip install -r investigation/requirements.lock.txt
cp investigation/config.example.env .env.investigation   # add NVIDIA_API_KEY (never commit it)
npm run agent:test                                       # unit tests
npm run agent:server                                     # http://127.0.0.1:8788/docs
```

Reproducing the NL pipeline (≈ 4 GB cache, GPU optional for the deep models) is documented step by step in
[`pipeline/README.md`](pipeline/README.md). Satellite and parcel downloads are anonymous public reads.

## 8. Limitations and next steps

- Labels are farmer-declared intended use, not verified harvest outcome.
- Two years, one region of one country; US transfer is only checked on 44 BARC fields used for calibration.
- Grain precision is low (≈ 0.24 at 11 % prevalence) — grain calls need review.
- Next: an untouched US labelled set to measure whether the Nemotron stage actually changes errors; tune the 0.80
  routing gate on held-out data; add earlier-season (in-season) estimates (data to Oct 31 already gives 0.70).

## Data sources and licences

PDOK BRP Gewaspercelen 2024/2025 (open data) · Copernicus Sentinel-2 L2A and Sentinel-1 RTC (Copernicus open
licence; via Element84 Earth Search / AWS Open Data and Microsoft Planetary Computer) · Daymet V4 (ORNL DAAC) ·
USGS 3DEP · USDA NASS Quick Stats · DWR 2023 Statewide Crop Mapping. Contains modified Copernicus Sentinel data.
