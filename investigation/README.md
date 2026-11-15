## v2.5 historical context

In always_predict mode only RF assigned-class scores below 0.80 enter the weather/Nemotron branch. `INVESTIGATION_HISTORICAL_WEATHER=true` adds 30/60/90-day matching-season comparisons of temperature, precipitation, modified corn GDD and frost exposure against the configured pre-crop-year baseline (default 1981–2010). At least 20 complete historical windows are required for numeric comparisons. Missing measurements remain unknown. Window GDD is never called planting-to-harvest development or maturity without actual planting and hybrid metadata.

`INVESTIGATION_NASS_CONTEXT=true` adds optional USDA NASS state crop progress and previous-year harvested acreage during weather prefetch. It requires the existing NASS key. Requests use documented endpoints; responses are independently filtered by commodity, state, year, units, production/domain scope, observation date and load date. Out-of-range and future rows are rejected even if the API ignores filters. Only latest unambiguous observations are retained; suppression remains explicit. This is regional context, not field truth or a verified planting date. The old get_agricultural_context field tool remains excluded, and aggregate-acreage CLI remains separate. HTTP/schema failures produce UNKNOWN regional context, preserving weather and the RF fallback.

Health reports historical flags and configured baseline. All enriched summaries are included in cited backend findings before Nemotron evaluates. High-score fields still skip retrieval and Nemotron. Predictions stay binary whenever RF exists; independent evidence verdicts and warnings remain. All original RF scores and null final_probability are preserved.

Sources: https://daymet.ornl.gov/single-pixel-tool-guide ; https://quickstats.nass.usda.gov/api . This product uses the NASS API but is not endorsed or certified by NASS. Current source vintages can contain revisions; observation/load-date filtering does not establish an archived, release-time backtest.

---

## v2.4 prediction mode

Server/CLI settings now default to `INVESTIGATION_PREDICTION_MODE=always_predict`.
At an assigned RF class score >= `INVESTIGATION_THRESHOLD` (default 0.80), retain RF without a model request. Below that score, prefetch evidence and require Nemotron's cited hypothesis to be GRAIN or SILAGE. The schema excludes UNRESOLVED and the backend rejects it with bounded corrections. Evidence can still be insufficient. If Nemotron is unavailable, times out or exhausts format corrections, retain the original RF label as `rf_fallback`; its failure status remains visible. This guarantees a binary output only when a valid RF prediction exists. Invalid input or missing RF is not fabricated into a label.

`final_label` is the operational prediction. `evidence_decision` retains the independent evidence verdict (often REVIEW); `prediction_status`, `prediction_is_evidence_supported`, `nemotron_used_for_prediction` and exception flags distinguish tentative predictions, evidence-backed classifications and fallbacks. All evidence, input warnings, calibration provenance and original scores remain. `final_probability` remains null: no fusion calibration was trained. A confidence score is not field-level accuracy. A tentative Nemotron override may be incorrect and is not confirmation.

`INVESTIGATION_PREDICTION_MODE=evidence_only` restores the earlier strict abstention behavior described below. Direct Python `Settings()` retains that strict mode for compatibility; explicitly set `Settings(prediction_mode='always_predict')` for programmatic use. Runtime health reports the effective mode and score threshold. Restart the backend to load v2.4. This changes prediction coverage, not the underlying evidence or demonstrated accuracy.

To improve evidence-backed predictions, supply actual planting intervals and documented local hybrid maturity, already computed source-linked removal/canopy/drydown observations, and feature vectors built with the saved RF feature contract. Do not fill missing values with invented measurements or use BARC reference labels as inference evidence. Evaluate improvements on untouched field-grouped labels; BARC remains development data.

---

# Cornstellation: RF + Nemotron investigation

This is a new, separate backend for the existing Harvest Console. It loads the supplied RF artifacts or accepts saved RF scores, routes uncertain cases to Nemotron, and exposes Daymet, elevation, web research and deterministic evaluation tools; NASS acreage is retrieved separately. The existing dashboard is unchanged. Use the CLI or local API below.

No satellite retrieval or feature extraction, Maps/Places, dairy proximity, training, or deployment is included.

## Setup and start

From the repository root, use Python 3.11 and an isolated environment:

```bash
python3.11 -m venv .venv-investigation
.venv-investigation/bin/python -m pip install -r investigation/requirements.txt
npm run agent:models
npm run agent:test
npm run agent:server
```

The local API listens on **127.0.0.1:8788** by default. Open `http://127.0.0.1:8788/docs` for the request form, or check `http://127.0.0.1:8788/health`. The root `/` is not an application page. Stop an older process or set `INVESTIGATION_PORT` if the port is occupied. This is a local research service, not a deployed multi-user API.

Existing `.env` settings for `NVIDIA_API_KEY` and `NVIDIA_MODEL` are reused. Optional settings are documented in `investigation/config.example.env`; copy it to `.env.investigation`, fill only needed values, and never commit real keys. Existing environment/`.env` values take precedence. Do not replace the current `.env` file.

## Run an existing example

```bash
npm run agent:run -- --input investigation/examples/barc_missing_rf.json --out .agent-runs/investigation/barc.json
npm run agent:run -- --input investigation/examples/cached_rf_nl.json --out .agent-runs/investigation/nl.json
```

The first example is a real BARC polygon with **no RF scores or 135-feature vector supplied**. It deliberately reports that missing input and normally ends in REVIEW. It is not a fabricated RF prediction or a BARC accuracy test.

The second uses a real existing Netherlands field and its already exported RF score. Daymet and USGS 3DEP are outside coverage for the Netherlands, so those tools report UNKNOWN. No new imagery is requested. This example demonstrates routing and source limitations, not a new US classification.

Use `--mode context` to run public-data tools sequentially without Nemotron, or `--mode offline` to perform routing/evaluation without external retrieval. Reports distinguish these modes from a completed live Nemotron run. Batch inputs are JSON arrays; they are validated before any external calls.

## Inputs and RF artifact contract

Obtain the complete input schema using:

```bash
.venv-investigation/bin/python -m investigation schema
```

For supplied predictions, provide `rf.p_silage`, `rf.p_grain`, `rf.model_version`, `rf.calibration_scope`, and the original `decision_threshold`. Preserve `review_band` when applicable. The original NL decision threshold is 0.88, not 0.5; a 0.85 silage score can therefore carry the original GRAIN decision and is routed for investigation.

For saved-model inference, omit `rf` and supply `features` with **exactly the 135 names in `rf_silage_grain_models/feature_columns.json`**, using null for unobserved values, plus `feature_contract: "rf_s2s1_135_v1"`. These must already have been computed according to the supplied feature builder and README. No new features are generated by this backend. All-null vectors and missing/extra feature names are rejected. The author’s `predict.py` is reused, including imputation, calibration, review band and original threshold. Artifact SHA-256 and calibration scope are recorded.

The default is `rf_nl_console.joblib`. `RF_BUNDLE=rf_silage_grain_models/rf_us_barc_recalibrated.joblib` selects the BARC-recalibrated bundle. It fitted its probability mapping using the 44 BARC reference field-years; evaluating that mapping on those labels is not independent validation. The code carries this warning into reports/evaluation. Neither bundle establishes nationwide US calibration.

The initial investigation gate is `max(p_silage,p_grain)<0.80`, with additional gates for low assigned-class score under the original model threshold, original review band, missing/inadequate quality, contradictory evidence and unvalidated geography. The gate is provisional, not a measured accuracy guarantee. Unknown quality does not silently pass the gate.

## Tools and actual source behavior

| Tool | Implementation | Credential or input requirement |
| --- | --- | --- |
| `get_daymet_weather` | Live Daymet V4 single-pixel CSV; daily values, historical rainfall baseline, fixed weather checks | Public access; supplied event dates determine which rules are usable |
| `get_elevation_context` | Live USGS 3DEP DEM export, polygon mask, field median/min/max and valid-cell coverage | Public access; US polygon |
| `search_public_harvest_evidence` | Direct curated public report URLs, bounded page reads, optional reviewed harvest-record registry | No search key; `PUBLIC_REPORT_SOURCES` selects the source manifest |
| `evaluate_evidence` | Fixed backend decision policy | Received evidence only; no LLM confidence adjustment |

Direct report retrieval is enabled without Google or Brave. `PUBLIC_SEARCH_ENABLED=false` is the default. `PUBLIC_REPORT_SOURCES=investigation/public_report_sources.json` points to a curated URL manifest with three dated University of Maryland regional reports for 2021–2023. Reports are selected by year/state and optional county/field selectors, capped at three unique URLs, fetched with the existing bounded cached public HTTPS reader, and logged with content hashes and excerpts. A selector does not prove exact-field association. Regional reports remain context and cannot label fields. Manifest dates are declared metadata, not automatically verified from the page; post-cutoff records are excluded. Failed reads remain explicit, and no matching source is UNKNOWN. PDF tables, Census tables and county-report tables are not automatically parsed. Current NASS revisions may postdate the reporting cutoff; their date limitation is explicit and they cannot decide individual-field labels.

Daymet and USGS 3DEP require no API key for the implemented endpoints. NASS uses `NASS_API_KEY` or an official CSV. Only Nemotron needs `NVIDIA_API_KEY`. Optional legacy Google discovery can be explicitly enabled with `PUBLIC_SEARCH_ENABLED=true` and eligible Google credentials, but no search service is called by default. No Brave integration is installed. Retrieved page text is untrusted material, never instructions.

Set `AGRICULTURAL_CSV` to the path of an official Quick Stats export to reuse the downloaded acreage data. The parser preserves year, state/county, units, definitions, domain and suppression flags. State-level CSV rows are not silently substituted when county data was requested. This is aggregate context; there is no forcing of predicted acreage to reported acreage.

For actual harvest confirmation, an external reviewer must inspect a public source and establish field/season association. `VERIFIED_HARVEST_RECORDS` points to a separately maintained JSON object with `records`, each following the `VerifiedRecord` schema in `schemas.py`: field ID, year, actual harvest use, harvest interval, publication date, public source URL, source SHA-256, verifier, verification method, and `source_type="public_actual_harvest_record"`. This is an explicit trusted-review registry, not automatic web extraction. Do not import BARC reference labels into this registry. No verified records are included by default.

## Weather and elevation behavior

- GDD uses Fahrenheit degree-days: cap Tmax at 86°F, floor Tmin at 50°F, subtract 50°F after averaging, and clip negative daily totals to zero. Dates are inclusive. A planting interval and harvest interval yield a GDD range. Missing daily values prevent a complete sum. Missing local maturity documentation means TIMING UNKNOWN.
- Three-day rain windows are calculated on both supplied radar observation dates and supplied harvest interval endpoints. The provisional threshold is 10 mm. Calendar-day totals cannot identify rain timing within a satellite acquisition day, and rain absence does not rule out irrigation.
- Frost uses Tmin ≤0°C in the seven calendar days ending on the supplied greenness-decline date. It is exposure risk, not confirmed damage or harvest.
- Rainfall deficit compares the matching seasonal 30-day sum to the 20th percentile of the configured 1981–2010 baseline, limited to years before the crop year. At least 20 complete historical windows are required. Irrigation and soil moisture remain unknown.
- Dry-down summaries report temperature, precipitation and approximate VPD. VPD is derived from saturation pressure at Tmin/Tmax minus actual vapor pressure, converted from Pa to kPa. It is not observed daily mean VPD or grain moisture.
- Daymet leap years include February 29 and omit December 31. Missing dates stay missing.
- Daymet response metadata includes the requested location, tile, projected coordinates, grid elevation, product header, and 1 km grid resolution. The single-pixel API does not provide an explicit cell index, so the cache uses the exact requested location/parameters rather than assuming adjacent points share a cell. Annual/baseline responses are reused for identical locations.
- USGS 3DEP is a multi-resolution DEM. Export spacing and source provenance are recorded separately; a 10 m export does not prove every input cell has 10 m native resolution. Terrain products can have been updated after the crop year. Elevation is static context, not a historic harvest observation.
- Field minus Daymet-grid elevation is reported when both are available. No temperature correction, fixed lapse rate, or fixed harvest-delay rule is applied.

## Decisions, audit and limits

Nemotron receives only the fixed field/year scope and approved tools. Tool arguments cannot switch field/year. Invalid calls are rejected; bounded correction attempts are permitted. There are at most eight tool attempts, six model rounds, 30 public HTTP requests, and a 180-second investigation budget. Network failures become explicit UNKNOWN results. Candidate page text is untrusted and cannot execute tools, files or shell commands.

In the optional legacy planning mode (`INVESTIGATION_PREFETCH_EVIDENCE=false`), every model response must call a tool. Once weather, elevation and public-report retrievals have been attempted (including UNKNOWN results), the only offered tool is `evaluate_evidence`, and the NVIDIA request explicitly forces that function. Completed retrievals are not offered again. Agricultural context is excluded from the schemas, tool dispatcher, context mode and Nemotron prompt; it cannot be requested in field investigations. A run is COMPLETED only after a valid evaluation; prose or a model-supplied final label cannot finish it.

Malformed envelopes, truncated completions, missing calls, invalid argument JSON and rejected hypotheses receive at most `INVESTIGATION_MAX_CORRECTIONS` correction requests (default 2, allowed 0–3). These consume the existing round, tool and time budgets. Correction prompts contain the fixed scope, schemas and known findings; they do not invent a hypothesis. Valid tool-call IDs receive matching tool responses, and unusable envelopes are never replayed to the provider. Unknown tool names and attempts to switch fields fail closed. Exhaustion returns REVIEW with an explicit FORMAT_LIMIT, ROUND_LIMIT, TOOL_LIMIT or TIME_LIMIT status. The report's `agent` object records correction attempts, protocol events and `evaluation_completed`. Original RF scores and `final_probability: null` are preserved.

`NVIDIA_MAX_TOKENS` configures the completion budget (default 2048, allowed 256–8192); reasoning may consume part of it. A `finish_reason: length` response is treated as truncated and corrected within the same limits, not accepted as a decision. Increase this setting only when supported by the configured model. No automatic model change or token-budget escalation occurs.

A high-scoring RF decision with acceptable supplied quality/applicability may pass unchanged. Investigated fields require a reviewed exact-field actual-harvest record or adequate already supplied physical evidence plus documented development timing. Rain/frost/rainfall-deficit exceptions block the latter route. Regional context, web mentions, weather or elevation alone cannot overturn uncertain RF output. Conflicting verified records produce REVIEW.

`final_probability` is always null in this implementation. Original RF probabilities, raw forest score when computed, calibration scope, threshold, and model checksum remain separately available. No fusion calibration was trained. Outputs contain tool results/traces, sources, dates, missing evidence, exceptions, decisions and elapsed time. The LLM cannot replace the backend label or invent a probability. Server reports are saved under `.agent-runs/investigation/`.

## Separate evaluation

```bash
.venv-investigation/bin/python -m investigation evaluate \
  --predictions .agent-runs/investigation/batch.json \
  --references data_barc_rules/evaluation/reference_labels.json \
  --out .agent-runs/investigation/evaluation.json
```

Only the offline evaluation command reads reference labels. It joins exact field/year keys after predictions, ignores blank labels, rejects duplicates, and compares both methods on the same RF-scored cohort. It reports class precision/recall, review-aware confusion matrices, coverage, resolved and all-eligible accuracy, by-year metrics, unmatched records and investigation time. BARC results are development results; the US-recalibrated bundle is explicitly flagged as contaminated for BARC evaluation. No BARC accuracy comparison is asserted until genuine RF inputs/predictions are supplied.

## Source documentation

- Daymet: https://daymet.ornl.gov/overview and https://daymet.ornl.gov/single-pixel/api
- USGS elevation: https://www.usgs.gov/3d-elevation-program
- NASS Quick Stats: https://quickstats.nass.usda.gov/api
- NVIDIA tools: https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-super-vl-120b-a12b-infer


## Evidence assessment and Nemotron hypothesis (v2)

The configured score gate (initially 0.80) and existing quality/applicability routing remain in effect. The output now includes `original_rf_prediction`, `evidence_assessment`, `nemotron_hypothesis`, `contradictions`, and `decision_reasons`, in addition to the original `rf` and final decision. Original probabilities and calibration metadata are unchanged. Final probability remains null.

Every finding has a stable ID, source references, relevant dates, field/year association, measurements with source units, limitations, missing inputs, and one of `SUPPORTS`, `CONTRADICTS`, or `DOES_NOT_RESOLVE` relative to the RF hypothesis. Only adequate field evidence or verified actual-harvest records can support/contradict harvest use decisively. Weather hazards flag alternative explanations; their absence is not proof of either product. Related findings share provenance groups. The same public page returned through two tools is deduplicated and no findings are counted as independent votes.

Optionally supply `feature_summary`, a list of existing source-linked measurements: `metric`, numeric `value`, `units`, optional `observed_between` interval, and `source_ids`. Allowed metrics are NDVI, NDRE, LSWI, NDTI, VH_DB, VV_DB, NDVI_DROP, PREHARVEST_NDVI, NDRE_PEAK_FRACTION and DAYS_PEAK_TO_HARVEST. Units must be unitless, dB or days as appropriate. Dates must respect the crop year/reporting cutoff. This accepts already computed evidence; it fetches no imagery and does not reconstruct missing RF columns. Arbitrary label columns or extra summary fields are rejected.

Nemotron receives the supplied season/quality/physical evidence, typed feature summaries and tool-backed finding IDs. Its final `evaluate_evidence` call must include `hypothesis`: a SILAGE/GRAIN/UNRESOLVED proposal, qualitative `evidence_strength`, cited `evidence_ids` and supported `reason_codes`. Numerical confidence, new facts, dates, source URLs and free-form factual claims are not allowed in this proposal schema. The backend checks citations and reason-code prerequisites, then renders its explanation from known facts. Invalid submissions are rejected within the existing round/call budgets.

The stored proposal is distinct from the final decision. Its `proposed_evidence_strength` records what Nemotron submitted; the displayed strength is limited when the backend cannot substantiate a final classification. `supported_for_final_decision` and `validation_flags` make that distinction explicit. A proposal cannot override missing-input flags, RF calibration status or final classification rules. Context/offline mode and skipped/unavailable investigations return a null Nemotron hypothesis rather than fabricating one.

Independent retrieval tool calls may be grouped in one NVIDIA response. Model HTTP requests have at most three attempts, respect the remaining investigation budget and return explicit failures. `NVIDIA_REASONING_EFFORT` is optional; blank preserves the configured model's default. Set `low` only when supported by your model. The existing NVIDIA model selection is unchanged.

BARC remains development-only because its 44 labels were used in recalibration and its results were previously inspected. Evaluation joins reference labels only after predictions and compares both methods on exactly the same eligible records, including REVIEW coverage. A new hypothesis does not establish improved accuracy. An independent held-out comparison requires separate, untouched labels. No new held-out dataset, retraining, satellite retrieval, deployment or interface changes are introduced by v2.


## NVIDIA transport diagnostics (v2.2)

Network retries and malformed-output corrections have separate budgets. NVIDIA requests retry connection/read timeouts, interrupted connections, HTTP 429 and selected 5xx failures with 1-second then 2-second backoff. Defaults are `NVIDIA_MAX_ATTEMPTS=3` (allowed 1–3) and `NVIDIA_READ_TIMEOUT_SECONDS=60` (allowed 5–90). Each attempt uses the remaining investigation budget; the total investigation limit is unchanged. Retry-After is honored when it fits both the remaining budget and a maximum 10-second retry wait; longer delays stop the request explicitly instead of retrying too soon. Authentication, bad-request and TLS errors are not retried.

`agent.transport_attempts` records phase, attempt, elapsed time, safe error category or HTTP status, configured timeouts and retry delay. It excludes keys, request bodies, exception strings and provider error bodies. `agent.transport_http_requests` counts HTTP attempts, while `agent.calls` counts logical completion requests. A success response with invalid JSON still uses the bounded format-correction path. Failed transport never manufactures an evaluation or changes RF scores.

An older generic connection/response error cannot establish the exact cause retrospectively. The new trace distinguishes CONNECT_TIMEOUT, READ_TIMEOUT, CONNECTION_ERROR, TLS_ERROR and HTTP status. Network failures remain possible. Restart the existing backend to load these changes; it is not restarted automatically.


## Separate aggregate acreage comparisons

NASS is excluded from individual-field investigations. Its adapter and CSV handling remain available through a separate command that does not run RF inference, Nemotron or the field decision policy:

```bash
.venv-investigation/bin/python -m investigation aggregate-acreage --input investigation/examples/barc_missing_rf.json --out .agent-runs/aggregate-acreage.json
```

Supply one existing input JSON with the year and state/county scope. The output is explicitly aggregate acreage only, with no final field label. Source failures and suppression flags remain explicit. Aggregate totals are not field truth and never force predicted acreage to match NASS.

## Connection reuse and concurrency

A dedicated Requests Session per investigation reuses the NVIDIA connection across retries and model rounds; it is closed on every exit and never shared between investigations or used for public-data tools. Redirects from the fixed NVIDIA endpoint are rejected. The default `NVIDIA_MAX_CONCURRENCY=2` limits simultaneous model HTTP requests per backend process (allowed 1–8). Keep batch clients at two workers. Multi-process server deployments have one limit per process, not a global service limit.

Waiting for a model slot consumes the existing investigation budget. Network timeouts are recalculated after waiting. CONCURRENCY_WAIT_TIMEOUT and TIME_BUDGET_EXHAUSTED events mark `http_request_sent=false` and do not increment `transport_http_requests`. Connection, read-timeout and selected HTTP retries remain bounded; authentication and TLS errors are not hidden by retries. Queue delay is logged separately from network time.

`/health.runtime_configuration` and `agent.runtime_configuration` expose the same safe effective tool list, model, retry/read timeout/concurrency settings, and `transport_implementation=session-bounded-v1`. Check these fields after restarting your backend; a version number alone cannot prove that new code/settings were loaded. No keys or response bodies are exposed. Existing `.env`/environment precedence remains unchanged.

These changes reduce avoidable connection setup and contention. They cannot establish or guarantee a cure for the old generic connection failures. Preserve typed traces from a fresh run to identify any remaining cause. A timeout is a Requests connection/read timeout, not an absolute wall-clock deadline for DNS, multiple resolved addresses, or a response that continues sending data; the investigation budget prevents starting further attempts after exhaustion.


## RCA recovery changes (v2.3)

The default `INVESTIGATION_PREFETCH_EVIDENCE=true` retrieves the three mandatory evidence sources in the backend before contacting Nemotron. Their traces explicitly identify `requested_by=backend_prefetch`. Their compact outputs and source-backed findings are supplied to the model, whose first available tool is `evaluate_evidence`. This removes the initial planning request and preserves evidence when NVIDIA is unavailable. All evidence acceptance rules and original RF values remain unchanged. Missing planting/maturity inputs still remain unknown. Legacy model-planned retrieval can be explicitly selected with `false`.

Transient attempts use jittered retry waits of 2–4 seconds then 4–8 seconds (or an eligible longer Retry-After, still capped at 10 seconds). After `NVIDIA_CIRCUIT_FAILURE_THRESHOLD=3` consecutive transient attempt failures, a shared per-process circuit waits `NVIDIA_CIRCUIT_COOLDOWN_SECONDS=20` before allowing one half-open probe. A successful connection/response closes the circuit; a failed probe renews the cooldown. No additional probes or extra attempts are created. All waiting consumes the existing 180-second investigation budget; attempts remain capped at three. Authentication and certificate errors do not trigger transient retries.

`NVIDIA_REQUESTS_PER_MINUTE=40` caps local admissions in a rolling minute, counting retries conservatively. Other processes, machines or clients using the account are outside this guard. The two-request concurrency cap remains in force.

Actual requests now record UTC attempt/request times, safe nested exception class names, DNS duration and resolved address families, each TCP attempt and selected peer, connection reuse, and TLS/proxy setup duration. The TLS/proxy duration is connection setup minus measured DNS/TCP time and includes certificate validation; it is not a packet-level TLS-only measurement. Certificate verification and address ordering are preserved. No address is pinned and configured proxies are not bypassed. SOCKS routing is preserved, with unsupported connection details marked unavailable. Keys, request bodies, raw error text and proxy credentials are excluded. `http_request_attempted` accurately identifies a client invocation; the legacy `http_request_sent` has the same compatibility meaning and does not prove provider receipt. `http_response_received` records a returned response. Only UUID-shaped whitelisted provider request IDs are retained.

Restart your backend and verify `/health` reports `investigation-v2.3`, `transport_implementation=rca-recovery-v2`, `evidence_retrieval_mode=backend_prefetch`, and the effective 40-RPM/circuit settings. The connection diagnostics use the installed urllib3 2.x adapter APIs; the requirements file makes that dependency explicit. These mitigations do not establish that an external connection fault is permanently fixed. No full-cohort improvement is claimed without another measured run.
