"""Binary predictions must retain evidence gaps and model failure status."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from investigation.agent import FUNCTIONS, run, schemas, runtime_configuration
from investigation.config import Settings
from investigation.policy import routing, evaluate, prediction_decision
from investigation.schemas import ToolResult
from test_workflow import payload


def unknown(name):
    def retrieve(request,*_):
        return ToolResult(evidence_id=name,tool=name,field_id=request.field_id,year=request.year,
                          status='UNKNOWN',missing_inputs=['planting date','documented maturity'])
    return retrieve


def completion(label,seen=None):
    def complete(messages,tools,remaining):
        if seen is not None: seen.append((messages.copy(),tools))
        proposal={'label':label,'evidence_strength':'insufficient',
                  'evidence_ids':['rf:prediction'],'reason_codes':['RF_HYPOTHESIS']}
        return {'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[
            {'id':'evaluate','type':'function','function':{'name':'evaluate_evidence',
             'arguments':json.dumps({'hypothesis':proposal})}}]}}]}
    return complete


class PredictionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.s=Settings(prediction_mode='always_predict')
        p=patch.dict(FUNCTIONS,{name:unknown(name) for name in FUNCTIONS})
        p.start(); self.addCleanup(p.stop)

    def test_binary_choice_retains_insufficient_evidence_and_original_scores(self):
        for label in ['GRAIN','SILAGE']:
            seen=[]; result=run(payload(),self.s,complete=completion(label,seen))
            self.assertEqual(result['final_label'],label)
            self.assertEqual(result['decision_source'],'nemotron_best_effort')
            self.assertEqual(result['agent']['status'],'COMPLETED')
            self.assertEqual(result['evidence_decision']['final_label'],'REVIEW')
            self.assertEqual(result['evidence_strength'],'insufficient')
            self.assertFalse(result['prediction_is_evidence_supported'])
            self.assertTrue(result['nemotron_used_for_prediction'])
            self.assertEqual(result['rf'],result['original_rf_prediction'])
            self.assertEqual(result['rf']['p_silage'],.57)
            self.assertIsNone(result['final_probability'])
            self.assertEqual(len(result['evidence']),3)
            self.assertTrue(result['missing_evidence'])
            schema=seen[0][1][0]['function']['parameters']['properties']['hypothesis']['properties']['label']
            self.assertEqual(schema['enum'],['SILAGE','GRAIN'])
            self.assertIn('UNRESOLVED and REVIEW are not permitted',seen[0][0][0]['content'])
            self.assertFalse(result['nemotron_hypothesis']['supported_for_final_decision'])
            if label=='GRAIN': self.assertIn('TENTATIVE_NEMOTRON_OVERRIDE_UNVERIFIED',result['exception_flags'])

    def test_unresolved_rejected_then_rf_fallback(self):
        result=run(payload(),self.s,complete=completion('UNRESOLVED'))
        self.assertEqual(result['agent']['status'],'FORMAT_LIMIT')
        self.assertEqual(result['agent']['correction_attempts'],2)
        self.assertEqual(result['final_label'],'SILAGE')
        self.assertEqual(result['decision_source'],'rf_fallback')
        self.assertFalse(result['nemotron_used_for_prediction'])
        self.assertIn('NEMOTRON_BINARY_PREDICTION_UNAVAILABLE_RF_FALLBACK',result['exception_flags'])

    def test_missing_key_keeps_failure_status_and_prefetched_evidence(self):
        result=run(payload(),replace(self.s,nvidia_key=''))
        self.assertEqual(result['agent']['status'],'UNAVAILABLE')
        self.assertEqual(result['final_label'],'SILAGE')
        self.assertEqual(result['decision_source'],'rf_fallback')
        self.assertEqual(len(result['evidence']),3)
        self.assertTrue(result['errors'])

    def test_transport_failure_does_not_become_model_success(self):
        from investigation.http import DataUnavailable
        def fail(*_): raise DataUnavailable('NVIDIA read timeout')
        result=run(payload(),self.s,complete=fail)
        self.assertEqual(result['agent']['status'],'FAILED')
        self.assertEqual(result['final_label'],'SILAGE')
        self.assertEqual(result['decision_source'],'rf_fallback')
        self.assertFalse(result['nemotron_used_for_prediction'])

    def test_threshold_inclusive_high_score_skips_model_preserves_quality_gaps(self):
        for p in [.8,.95]:
            r=payload(rf={'p_silage':p,'p_grain':1-p,'model_version':'test','calibration_scope':'unknown'},
                      quality={'flags':['MISSING_FEATURES']})
            def forbidden(*_): raise AssertionError('High RF score must skip model')
            result=run(r,self.s,complete=forbidden)
            self.assertEqual(result['final_label'],'SILAGE')
            self.assertEqual(result['agent']['status'],'SKIPPED')
            self.assertEqual(result['evidence_decision']['final_label'],'REVIEW')
            self.assertIn('MISSING_FEATURES',result['exception_flags'])
            self.assertFalse(result['prediction_is_evidence_supported'])

    def test_just_below_threshold_invokes_model(self):
        r=payload(rf={'p_silage':.7999,'p_grain':.2001,'model_version':'test','calibration_scope':'unknown'})
        result=run(r,self.s,complete=completion('GRAIN'))
        self.assertEqual(result['agent']['calls'],1)
        self.assertEqual(result['final_label'],'GRAIN')

    def test_original_label_threshold_preserved(self):
        r=payload(rf={'p_silage':.85,'p_grain':.15,'decision_threshold':.88,
                      'model_version':'NL','calibration_scope':'NL'})
        self.assertTrue(routing(r,r.rf,self.s)['investigate'])
        result=run(r,replace(self.s,nvidia_key=''))
        self.assertEqual(result['rf']['predicted_label'],'GRAIN')
        self.assertEqual(result['final_label'],'GRAIN')
        self.assertAlmostEqual(result['rf_assigned_class_score'],.15)

    def test_verified_evidence_wins_over_tentative_choice(self):
        r=payload(); route=routing(r,r.rf,self.s)
        ev=ToolResult(evidence_id='web',tool='search_public_harvest_evidence',field_id=r.field_id,
                      year=r.year,status='OK',data={'verified_actual_harvest_records':[{'harvest_use':'GRAIN'}]})
        d=evaluate(r,r.rf,route,[ev])
        result=prediction_decision(d,r.rf,route,{'label':'SILAGE','citations_validated':True},'COMPLETED',self.s)
        self.assertEqual(result['final_label'],'GRAIN')
        self.assertTrue(result['prediction_is_evidence_supported'])
        self.assertFalse(result['nemotron_used_for_prediction'])

    def test_no_rf_does_not_invent_label(self):
        result=run(payload(rf=None),replace(self.s,nvidia_key='',rf_bundle='missing-model.joblib'))
        self.assertIsNone(result['rf'])
        self.assertEqual(result['final_label'],'REVIEW')
        self.assertEqual(result['prediction_status'],'unavailable_no_rf')

    def test_strict_mode_retains_abstention(self):
        result=run(payload(),replace(self.s,prediction_mode='evidence_only'),complete=completion('UNRESOLVED'))
        self.assertEqual(result['agent']['status'],'COMPLETED')
        self.assertEqual(result['final_label'],'REVIEW')
        spec=schemas(prediction_mode='evidence_only')[-1]
        self.assertIn('UNRESOLVED',spec['function']['parameters']['properties']['hypothesis']['properties']['label']['enum'])

    def test_env_default_validation_and_health_settings(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
            settings=Settings.from_env(Path(tmp))
            self.assertEqual(settings.prediction_mode,'always_predict')
            self.assertEqual(runtime_configuration(settings)['rf_confidence_threshold'],.8)
            self.assertEqual(runtime_configuration(settings)['prediction_mode'],'always_predict')
            with patch.dict(os.environ,{'INVESTIGATION_PREDICTION_MODE':'invalid'}):
                with self.assertRaisesRegex(ValueError,'PREDICTION_MODE'):Settings.from_env(Path(tmp))
