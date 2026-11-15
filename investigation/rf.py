"""Load existing artifacts only. No training or feature extraction."""
from functools import lru_cache
from hashlib import sha256
import importlib.util
import json

from .schemas import RFResult


@lru_cache(maxsize=4)
def inspect_bundle(path, root):
    import joblib
    import sklearn
    path, root = __import__('pathlib').Path(path), __import__('pathlib').Path(root)
    if not path.is_file():
        raise FileNotFoundError('RF artifact unavailable; supply rf probabilities instead')
    if not sklearn.__version__.startswith('1.7.'):
        raise RuntimeError('Saved artifact requires scikit-learn 1.7.x; install investigation/requirements.txt')
    bundle = joblib.load(path)  # Trusted local artifact supplied by the user, never a URL/upload.
    columns = json.loads((root / 'rf_silage_grain_models' / 'feature_columns.json').read_text())
    if bundle.get('cols') != columns or len(columns) != 135:
        raise ValueError('Artifact feature schema disagrees with the supplied manifest')
    if list(bundle['model'].classes_) != [0, 1] or list(bundle['calibration'].classes_) != [0, 1]:
        raise ValueError('Unsupported class mapping; training code declares 0=grain, 1=silage')
    return bundle, sha256(path.read_bytes()).hexdigest()


def infer(request, settings):
    if request.rf is not None:
        return request.rf
    if request.features is None:
        return None
    import numpy as np
    import pandas as pd
    path = settings.root / ('rf_silage_grain_models/rf_us_barc_recalibrated.joblib' if request.rf_model == 'barc_development' else settings.rf_bundle)
    bundle, checksum = inspect_bundle(str(path), str(settings.root))
    expected = set(bundle['cols'])
    if set(request.features) != expected:
        raise ValueError('Feature vector must contain exactly the 135 manifest keys; use null for unobserved values')
    if not any(v is not None for v in request.features.values()):
        raise ValueError('All RF features are missing; cannot score this field')
    x = pd.DataFrame([request.features], columns=bundle['cols']).astype(float)
    # Reuse the artifact author's preprocessing, probability calibration and threshold.
    spec = importlib.util.spec_from_file_location('cornstellation_saved_rf', settings.root / 'rf_silage_grain_models' / 'predict.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = module.predict(bundle, x).iloc[0]
    raw = float(bundle['model'].predict_proba(x)[0, 1])
    p = float(row.p_silage)
    if not np.isfinite(p):
        raise ValueError('RF returned a non-finite probability')
    barc = 'us_platt' in bundle
    return RFResult(p_silage=p, p_grain=1-p, raw_p_silage=raw,
                    model_version=str(bundle.get('name', path.name)), decision_threshold=bundle['threshold'],
                    review_band=bundle.get('review_band'), provenance='saved_artifact', artifact_sha256=checksum,
                    barc_fitted=barc, calibration_scope=('BARC 44 field-years; fitted on the BARC evaluation labels, not independent validation'
                                                       if barc else 'Netherlands validation regions; US transfer calibration unvalidated'))
