"""Rulebook v1, step A: label-free field-year inference inputs for BARC.

Reads only extraction outputs (no reference labels exist in them):
  <HTH_BARC>/<year>/s2/fields_BARC.parquet       (10 m interior pixel count, area, geometry; label columns dropped at extraction)
  <HTH_BARC>/<year>/s2_20m/obs20_BARC.parquet    (native 20 m field means: NDVI, NDRE, NDTI, LSWI, PSRI, B04/B11/B12, 2x2 zones)
  <HTH_BARC>/<year>/s1/s1_BARC.parquet           (all relative orbits, per-field pixel counts)
Writes <HTH_BARC>/rulebook/:
  inputs_<year>.json   one record per field-year: opaque key, year, geometry, interior area, optical series, selected-orbit radar
  weather_<year>.json  Open-Meteo daily Tmax/Tmin (F) and precipitation (mm) at the mean field centroid, with URL + retrieval date
  key_map.csv          opaque key -> OBJECTID, year (read only by exports and evaluation, never by the engine)
Radar orbit rule (config radar_orbit): per year choose the orbit that FULLY covers the most fields, then most Jul-Dec dates.
"""
import os, sys, json, hashlib, pathlib, datetime, urllib.request, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
BARC = pathlib.Path(os.environ["HTH_BARC"])
CFG = json.load(open(HERE / "rulebook" / "config_v1.json"))
OUT = BARC / "rulebook"; OUT.mkdir(parents=True, exist_ok=True)
YEARS = (2021, 2022, 2023)
SALT = "rulebook-v1"
FORBIDDEN = {"label", "code", "crop_name", "Corn_2021", "Corn_2022", "Corn_2023"}
f = lambda v: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4)
key_of = lambda fid, y: "FY" + hashlib.sha1(f"{SALT}|{fid}|{y}".encode()).hexdigest()[:10]


def weather(year, lat, lon):
    p = OUT / f"weather_{year}.json"
    if p.exists(): return json.load(open(p))
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=%.4f&longitude=%.4f&start_date=%d-01-01&end_date=%d-12-31"
           "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&temperature_unit=fahrenheit&timezone=America%%2FNew_York"
           % (lat, lon, year, year))
    d = json.load(urllib.request.urlopen(url, timeout=60))["daily"]
    w = {"source": "Open-Meteo historical weather API (ERA5-based)", "url": url, "retrieved": datetime.date.today().isoformat(),
         "point": [lat, lon], "daily": {"date": d["time"], "tmax_f": d["temperature_2m_max"], "tmin_f": d["temperature_2m_min"],
                                         "precip_mm": d["precipitation_sum"]}}
    json.dump(w, open(p, "w")); return w


keymap, orbit_log = [], {}
cents = []
for y in YEARS:
    fl = pd.read_parquet(BARC / str(y) / "s2" / "fields_BARC.parquet")
    cents += list(zip(fl.lat, fl.lon))
lat, lon = float(np.mean([c[0] for c in cents])), float(np.mean([c[1] for c in cents]))

for y in YEARS:
    R = BARC / str(y)
    fl = pd.read_parquet(R / "s2" / "fields_BARC.parquet")
    assert not (FORBIDDEN & set(fl.columns)), f"label-like columns in fields file: {FORBIDDEN & set(fl.columns)}"
    obs = pd.read_parquet(R / "s2_20m" / "obs20_BARC.parquet")
    assert not (FORBIDDEN & set(obs.columns))
    s1 = pd.read_parquet(R / "s1" / "s1_BARC.parquet")
    w = weather(y, lat, lon)
    # ---- orbit selection: full field coverage first, then temporal coverage ----
    npx = fl.set_index("field_id").n_interior_px
    s1["cov"] = s1.n_px / s1.field_id.map(npx)
    jd = s1[(pd.to_datetime(s1.date) >= f"{y}-07-01")]
    stats = {}
    for o, g in s1.groupby("rel_orbit"):
        cov = g.groupby("field_id")["cov"].median()
        nd = jd[jd.rel_orbit == o].groupby("field_id").date.nunique()
        full = [k for k in cov.index if cov[k] >= CFG["radar_orbit"]["full_cover_frac"] and nd.get(k, 0) >= CFG["radar_orbit"]["min_dates_jul_dec"]]
        stats[int(o)] = dict(fields_full=len(full), dates_jul_dec=int(jd[jd.rel_orbit == o].date.nunique()), full=set(full), cov=cov.to_dict())
    orbit = max(stats, key=lambda o: (stats[o]["fields_full"], stats[o]["dates_jul_dec"]))
    orbit_log[y] = {o: {k: v for k, v in s.items() if k in ("fields_full", "dates_jul_dec")} for o, s in stats.items()}
    orbit_log[y]["selected"] = orbit
    so = s1[s1.rel_orbit == orbit]
    blocks = {ft["id"]: ft["geometry"] for ft in json.load(open(R / "blocks" / "BARC.json"))["features"]}
    recs = []
    for r in fl.itertuples():
        k = key_of(r.field_id, y)
        keymap.append(dict(key=k, OBJECTID=int(r.field_id[3:]), year=y))
        o = obs[obs.field_id == r.field_id].sort_values("date")
        optical = [dict(date=str(x.date)[:10], clear_frac=f(x.clear_frac), n_valid_px=int(x.n_valid_px) if np.isfinite(x.n_valid_px) else 0,
                        ndvi=f(x.ndvi), ndre=f(x.ndre), ndti=f(x.ndti), lswi=f(x.lswi), psri=f(x.psri), B04=f(x.B04), B11=f(x.B11), B12=f(x.B12),
                        z1=f(x.ndvi_z1), z2=f(x.ndvi_z2), z3=f(x.ndvi_z3), z4=f(x.ndvi_z4)) for x in o.itertuples()]
        g = so[so.field_id == r.field_id].sort_values("date")
        full = r.field_id in stats[orbit]["full"]
        cov = stats[orbit]["cov"].get(r.field_id)
        radar = dict(orbit=orbit, coverage_frac=f(cov), full_cover=bool(full),
                     reason=None if full else ("field_not_in_orbit" if cov is None else "partial_coverage_or_few_dates"),
                     series=[dict(date=str(x.date)[:10], vv=f(x.vv_db), vh=f(x.vh_db), n_px=int(x.n_px)) for x in g.itertuples()] if full else [])
        recs.append(dict(key=k, year=y, crop_cycle_id=k + "-c1", geometry=blocks[r.field_id], area_ha=f(r.area_utm_ha),
                         interior_px10=int(r.n_interior_px), interior_ha=f(r.n_interior_px * 100 / 1e4), eligible=bool(r.eligible),
                         weather_source=w["url"], optical=optical, radar=radar))
    for rec in recs: assert not (FORBIDDEN & set(rec)), "label-like key in input"
    json.dump(recs, open(OUT / f"inputs_{y}.json", "w"))
    print(y, len(recs), "field-years; radar orbit", orbit_log[y], "; optical rows", len(obs), flush=True)
pd.DataFrame(keymap).to_csv(OUT / "key_map.csv", index=False)
json.dump(dict(orbit_selection=orbit_log, weather_point=[lat, lon], config=CFG["version"]), open(OUT / "inputs_log.json", "w"), indent=1, default=str)
print("RULEBOOK_INPUTS_DONE")
