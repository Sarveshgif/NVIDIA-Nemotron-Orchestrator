"""Numerical adapter checks with synthetic features, not field predictions."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import unittest

from investigation.config import Settings, ROOT
from investigation.rf import inspect_bundle, infer
from test_workflow import payload


class SavedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.getenv('INVESTIGATION_TEST_ROOT',str(ROOT)))
        if not (cls.root/'rf_silage_grain_models').is_dir():
            raise unittest.SkipTest('Local user RF artifact folder is unavailable')
        cls.settings=Settings(root=cls.root)

    def test_bundle_mapping_manifest_and_barc_provenance(self):
        for filename in ('rf_nl_console.joblib','rf_us_barc_recalibrated.joblib'):
            b,h=inspect_bundle(str(self.root/'rf_silage_grain_models'/filename),str(self.root))
            self.assertEqual(len(b['cols']),135)
            self.assertEqual(list(b['model'].classes_),[0,1])
            self.assertEqual(len(h),64)

    def test_reuses_authors_calibration_and_feature_order(self):
        import pandas as pd
        cols=json.loads((self.root/'rf_silage_grain_models/feature_columns.json').read_text())
        # Explicitly synthetic: exercise missing-value preprocessing, no real crop claim.
        features={c:None for c in reversed(cols)}; features['peak_ndvi']=.7
        r=payload(rf=None,features=features,feature_contract='rf_s2s1_135_v1')
        spec=importlib.util.spec_from_file_location('author_predict',self.root/'rf_silage_grain_models/predict.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for filename in ('rf_nl_console.joblib','rf_us_barc_recalibrated.joblib'):
            settings=replace(self.settings,rf_bundle='rf_silage_grain_models/'+filename)
            actual=infer(r,settings)
            b,_=inspect_bundle(str(self.root/settings.rf_bundle),str(self.root))
            expected=module.predict(b,pd.DataFrame([features]).astype(float)).iloc[0]
            self.assertAlmostEqual(actual.p_silage,float(expected.p_silage),places=12)
            self.assertEqual(actual.predicted_label,expected['call'].upper())
            self.assertEqual(actual.barc_fitted,'us_platt' in b)

    def test_missing_columns_not_silently_imputed(self):
        r=payload(rf=None,features={'peak_ndvi':.7},feature_contract='rf_s2s1_135_v1')
        with self.assertRaisesRegex(ValueError,'135'):infer(r,self.settings)


if __name__=='__main__':unittest.main()
