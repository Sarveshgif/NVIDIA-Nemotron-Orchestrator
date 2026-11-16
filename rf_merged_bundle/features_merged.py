"""Feature construction for the merged silage/grain RF (160 columns = 135 console + 25 relative/GDD).

Input F: one row per field-season with the standard per-field features from 04_build_features.py
(bf.s2_features + bf.s1_features: half-month NDVI/LSWI/NDRE medians, transition / phenology timing as day of
year (0 = Jan 1), Sentinel-1 monthly dB medians and changes). cums: for each row, a numpy array of cumulative
corn GDD (86/50 F) for that field's year, index 0 = Jan 1 (e.g. from Open-Meteo daily Tmax/Tmin at the field).
"""
import numpy as np, pandas as pd


def relative(F):
    F = F.copy(); pk = F.peak_ndvi
    g = lambda c: F[c] if c in F else pd.Series(np.nan, index=F.index)
    hm = lambda var: [c for c in F.columns if c.startswith(f"{var}_hm") and c[-3:-1] in ("06", "07", "08", "09")]
    R = pd.DataFrame(index=F.index)
    R["r_season_len"] = g("tr_start_doy") - g("greenup_doy")
    R["r_peak_to_tr_days"] = g("tr_start_doy") - g("peak_doy")
    R["r_greenup_to_peak_days"] = g("peak_doy") - g("greenup_doy")
    R["r_ndvi_before"] = g("tr_ndvi_before") / pk; R["r_ndvi_after"] = g("tr_ndvi_after") / pk
    R["r_drop"] = g("tr_fall") / pk; R["r_drop_rate"] = g("tr_rate") / pk; R["tr_gap"] = g("tr_gap")
    R["r_predrop_decline"] = g("predrop_decline") / pk; R["r_predrop_rate"] = g("predrop_decline_rate") / pk
    R["r_slope30_pre"] = g("slope30_pre") / pk
    R["r_post_20_60"] = g("post_ndvi_20_60") / pk; R["r_post_min30"] = g("post_min_ndvi_30") / pk
    R["r_first_below6_days"] = g("first_below_6_doy") - g("peak_doy"); R["r_first_below4_days"] = g("first_below_4_doy") - g("peak_doy")
    R["r_ndre_before"] = g("tr_ndre_before") / F[hm("ndre")].max(axis=1)
    R["r_lswi_change"] = g("tr_lswi_after") - g("tr_lswi_before"); R["r_lswi_before_vs_peak"] = g("tr_lswi_before") - F[hm("lswi")].max(axis=1)
    R["zone_tr_spread"] = g("zone_tr_spread"); R["r_zone_sd_before"] = g("zone_sd_before") / pk; R["r_zone_sd_after"] = g("zone_sd_after") / pk
    R["r_ndvi_sd_before"] = g("ndvi_sd_before") / pk
    for c in ("s1_vh_db_change_at_tr", "s1_vv_db_change_at_tr", "s1_cr_db_change_at_tr", "s1_vh_pre60_slope", "s1_vh_maxdrop"): R[c] = g(c)
    R["r_s1_maxdrop_vs_tr_days"] = g("s1_vh_maxdrop_doy") - g("tr_mid_doy")
    # radar level in the month of the transition relative to the field's own Jul-Aug level (dB difference)
    mo = (pd.to_datetime("2025-01-01") + pd.to_timedelta(g("tr_mid_doy").fillna(0), unit="D")).dt.month.where(g("tr_mid_doy").notna())
    for v in ("vh_db", "cr_db"):
        summer = F[[c for c in (f"{v}_m07", f"{v}_m08") if c in F]].mean(axis=1)
        at = pd.Series([F.at[i, f"{v}_m{int(m):02d}"] if pd.notna(m) and f"{v}_m{int(m):02d}" in F else np.nan for i, m in mo.items()], index=F.index)
        R[f"r_{v}_tr_vs_summer"] = at - summer
    return R



def gdd_day(tx, tn):
    tx, tn = np.clip(tx, 50, 86), np.clip(tn, 50, 86)
    return (tx + tn) / 2 - 50


def add_gdd(F, cums):
    """cums: Series of cumulative-GDD arrays aligned with F (index 0 = Jan 1, matching 0-based doy)."""
    def at(col):
        out = []
        for c, v in zip(cums, F[col] if col in F else [np.nan] * len(F)):
            out.append(c[int(min(max(v, 0), len(c) - 1))] if v == v and c is not None else np.nan)
        return np.array(out, float)
    G = pd.DataFrame(index=F.index)
    gu, pk, tr, b6, b4, mid, md = (at(c) for c in ("greenup_doy", "peak_doy", "tr_start_doy", "first_below_6_doy", "first_below_4_doy", "tr_mid_doy", "s1_vh_maxdrop_doy"))
    G["g_greenup_to_peak"] = pk - gu; G["g_peak_to_tr"] = tr - pk; G["g_season"] = tr - gu
    G["g_peak_to_below6"] = b6 - pk; G["g_peak_to_below4"] = b4 - pk
    G["g_s1_maxdrop_vs_tr"] = md - mid; G["g_at_greenup"] = gu; G["g_at_tr"] = tr
    return G


DAY_COLS = ["r_season_len", "r_peak_to_tr_days", "r_greenup_to_peak_days", "r_first_below6_days", "r_first_below4_days", "r_s1_maxdrop_vs_tr_days"]


def merged_features(F, cums, console_cols, extra_cols):
    """Returns the 160-column design matrix in the model's column order."""
    A = pd.concat([relative(F).drop(columns=DAY_COLS), add_gdd(F, cums)], axis=1)
    X = pd.concat([F.reindex(columns=console_cols).reset_index(drop=True), A[extra_cols].reset_index(drop=True)], axis=1)
    return X


def cum_gdd_from_daily(tmax_f, tmin_f):
    """Cumulative GDD from daily max/min temperature in F (Jan 1 ... Dec 31)."""
    return np.cumsum(gdd_day(np.asarray(tmax_f, float), np.asarray(tmin_f, float)))
