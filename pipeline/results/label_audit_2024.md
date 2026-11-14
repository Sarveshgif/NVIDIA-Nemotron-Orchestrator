# Label audit — Netherlands BRP 2024

**Source.** PDOK Basisregistratie Gewaspercelen (BRP) 2024, status 'Definitief', annual GeoPackage service.pdok.nl/rvo/gewaspercelen/atom/downloads/brpgewaspercelen_definitief_2024.gpkg (anonymous); same block windows as 2025. Gewaspercelen are the parcels farmers declare each spring for CAP payments (RVO).
The `Definitief` status means RVO finalised the registration; it is **not** an independent check of how the crop
was finally harvested. Labels are therefore *farmer-declared intended crop use*. A field declared `korrelmais`
can still be chopped as silage in autumn (and vice versa); that is irreducible label noise for this study.

**License.** BRP open data from RVO via PDOK (published as open data; verify current terms on the PDOK/NGR metadata record b812a145-b4fe-4331-8dc6-d914327a87ff).

**Class definitions (official BRP codes, read from the 2024 records — not guessed).**

| Code | BRP name | Used as |
|---|---|---|
| 259 | Mais, snij- | **silage** (positive class) |
| 316 | Mais, korrel- | **grain** |
| 317 | Mais, corncob mix | excluded (ear-only harvest; neither class) |
| 1935 | Maiskolvensilage | excluded (ear silage) |
| 814 | Mais, suiker- | excluded (sweet corn) |

Maize categories found in the scanned blocks: 259 Mais, snij-: 8755, 316 Mais, korrel-: 1202, 317 Mais, corncob mix : 513, 1935 Maiskolvensilage: 102, 814 Mais, suiker-: 37.

**EuroCrops.** EuroCrops (Zenodo 14094196, CC-BY-SA-4.0) re-publishes the same RVO declarations (NL only for 2020)
with harmonised HCAT classes. It is *not* an independent second source. Its HCAT mapping merges 317 corncob mix,
814 sweet corn and 2032 energy maize into `grain_maize_corn_popcorn`, and 1935 into `green_silo_maize`, so using
HCAT classes would contaminate both labels. We use original BRP codes only, and only PDOK 2024 (no duplication).

**German FOR1695 / BonaRes (doi:10.20387/bonares-a0qc-46jc, CC-BY-4.0).** Investigated via DataCite metadata:
a handful of eddy-covariance research fields in Kraichgau and the Swabian Jura (2009–2018). Too few field-seasons
overlapping Sentinel-2 (2015+) to contribute a held-out class-balanced test; not used. Not a blocker.

**Not used as labels:** USDA CDL (corn only), our California rule outputs (weak labels), satellite crop maps.

## Counts

Candidate silage/grain parcels in the 20 blocks (centroid-in-block de-duplication): {'grain': 1134, 'silage': 8265}.
Parcels whose geometry touched two block bounding boxes (assigned once, by centroid): 49. Invalid geometries: 0.

Exclusions (field counts), checked **per class** so filtering does not silently remove one class:

| label   |   too_small_interior |
|:--------|---------------------:|
| grain   |                  455 |
| silage  |                 3498 |

Eligibility: inside one Sentinel-2 tile footprint, and >= 50 clear-able interior 10 m pixels after a 10 m inward buffer
(>= 0.5 ha interior). Share excluded as too small: grain 40.1%, silage 42.3% — similar for both classes.

**Unique fields by split (area from polygon in EPSG:28992 RD New):**

|                     |   fields |   area_ha |
|:--------------------|---------:|----------:|
| ('test', 'grain')   |      229 |     618.8 |
| ('test', 'silage')  |     1625 |    4856.5 |
| ('train', 'grain')  |      314 |     954.6 |
| ('train', 'silage') |     1879 |    5815.3 |
| ('val', 'grain')    |      136 |     384.8 |
| ('val', 'silage')   |     1263 |    3852.4 |

**By region:**

|                         |   grain |   silage |
|:------------------------|--------:|---------:|
| ('test', 'GE-east')     |      69 |      936 |
| ('test', 'LI-north')    |      80 |      430 |
| ('test', 'LI-south')    |      80 |      259 |
| ('train', 'LI-mid')     |     164 |      367 |
| ('train', 'NB-kempen')  |      26 |      313 |
| ('train', 'NB-peel')    |     124 |     1199 |
| ('val', 'LI-echt')      |      49 |      105 |
| ('val', 'NB-northeast') |      87 |     1158 |

**Sample design.** Blocks were chosen deliberately where grain maize occurs (south-east NL), and every eligible
silage and grain parcel inside them is used. The sample is therefore *grain-enriched relative to NL as a whole*
(nationally grain maize is a small minority of maize area), but within blocks it is complete, not class-balanced.
Prevalence-dependent numbers (precision, acreage) apply to these regions, not to the whole country.

**Geometry quality.** BRP parcels are digitised on aerial imagery by RVO / farmers; boundaries are parcel-level
(not field-operation level). A parcel may contain field margins; the 10 m inward buffer removes mixed edge pixels.
