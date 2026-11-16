"""Score fields with the merged RF: Dutch calibration -> optional US (BARC) recalibration -> optional county prior.

    from predict_merged import load, predict
    b = load("rf_merged.joblib")
    out = predict(b, X)                                  # X = features_merged.merged_features(...), US probabilities
    out = predict(b, X, county_prior=priors)             # priors: per-row county silage share (0-1), for AREA TOTALS ONLY

Columns: p_nl (Dutch-calibrated), call_nl (p_nl >= threshold_nl 0.90), p_us (BARC Platt probability),
call_us (p_us >= 0.5), call_us_cutoff (p_nl >= us_cutoff_on_p_nl 0.87, the BARC-fitted cut-off on the Dutch scale),
p_prior / call_prior if county_prior is given.
Do NOT use the county prior for individual field decisions; see README (BARC: 11/12 silage found without the prior,
0/12 with it, because FSA reports no silage in that county).
"""
import joblib, numpy as np, pandas as pd


def load(path):
    return joblib.load(path)


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))


def predict(b, X, county_prior=None, prior_floor=0.005):
    X = X.reindex(columns=b["cols"])
    raw = b["model"].predict_proba(X)[:, 1]
    p_nl = b["calibration_nl"].predict_proba(_logit(raw)[:, None])[:, 1]
    a, s = b["us_platt"]
    p_us = 1 / (1 + np.exp(-(a + s * _logit(p_nl))))       # same clipping (1e-4) as the BARC fit
    out = pd.DataFrame({"p_nl": p_nl, "call_nl": p_nl >= b["threshold_nl"], "p_us": p_us, "call_us": p_us >= 0.5,
                        "call_us_cutoff": p_nl >= b["us_cutoff_on_p_nl"]}, index=X.index)
    if county_prior is not None:
        pc = np.clip(np.asarray(county_prior, float), prior_floor, 1 - prior_floor); pi0 = b["us_base_rate"]
        r, q = pc / pi0, (1 - pc) / (1 - pi0)
        out["p_prior"] = r * p_us / (r * p_us + q * (1 - p_us)); out["call_prior"] = out.p_prior >= 0.5
    return out
