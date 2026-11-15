"""Silage-vs-grain maize Random Forest: load a bundle and score a feature table.

    import pandas as pd
    from predict import load, predict
    b = load("rf_nl_console.joblib")             # or "rf_us_barc_recalibrated.joblib"
    out = predict(b, features_df)                # one row per field-season, columns = b["cols"] (missing -> NaN)

Returns p_silage (calibrated probability of silage) and call ("silage"/"grain"); for the NL bundle also review
(inside the review band). Features must be built exactly as pipeline/04_build_features.py (s2_features +
s1_features) from Sentinel-2 L2A and Sentinel-1 RTC field means; NaN = not observed (the model imputes).
Requires scikit-learn 1.7.x and joblib (pickles are version-specific).
"""
import joblib, numpy as np, pandas as pd


def load(path):
    return joblib.load(path)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def predict(bundle, X):
    X = X.reindex(columns=bundle["cols"])                      # exact column order; absent columns -> NaN
    raw = bundle["model"].predict_proba(X)[:, 1]
    p = bundle["calibration"].predict_proba(_logit(raw)[:, None])[:, 1]          # NL (Platt) calibration
    if "us_platt" in bundle:                                   # BARC-fitted US recalibration on top of the NL probability
        a, b = bundle["us_platt"]
        p = 1 / (1 + np.exp(-(a + b * _logit(p))))
    out = pd.DataFrame({"p_silage": p, "call": np.where(p >= bundle["threshold"], "silage", "grain")}, index=X.index)
    if "review_band" in bundle and "us_platt" not in bundle:
        lo, hi = bundle["review_band"]; out["review"] = (p >= lo) & (p < hi)
    return out
