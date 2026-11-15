from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from investigation.config import Settings
from investigation.schemas import InvestigationInput, RFResult, ToolResult
from investigation.weather import corn_gdd_f, parse_daymet, weather_rules, approximate_vpd_kpa
from investigation.policy import routing, evaluate
from investigation.agent import run, ToolRunner, FUNCTIONS, schemas
from investigation.tools import verified_records, number_or_suppression, get_agricultural_context
from investigation.evaluation import compare
from investigation.http import DataUnavailable, PublicHTTP


def payload(**updates):
    value = {'field_id':'TEST-FIELD', 'year':2023, 'reporting_cutoff':'2023-12-31',
             'geometry':{'type':'Polygon','coordinates':[[[-76.9,39.0],[-76.899,39.0],[-76.899,39.001],[-76.9,39.001],[-76.9,39.0]]]},
             'rf':{'p_silage':.57,'p_grain':.43,'model_version':'supplied-test','calibration_scope':'unvalidated'},
             'quality':{'observations_adequate':True,'geographic_applicability':'validated'}}
    value.update(updates)
    return InvestigationInput.model_validate(value)


def weather_result(request, timing='BEFORE_PLAUSIBLE_GRAIN_MATURITY', flags=None):
    return ToolResult(evidence_id='weather',tool='get_daymet_weather',field_id=request.field_id,year=request.year,status='OK',
                      data={'rules':{'development_timing':{'status':timing}}},quality_flags=flags or [])


class WeatherTests(unittest.TestCase):
    def setUp(self):
        self.s = Settings()

    def test_gdd_units_and_nonnegative(self):
        self.assertAlmostEqual(corn_gdd_f((60-32)*5/9,(90-32)*5/9),23)
        self.assertEqual(corn_gdd_f(-5,0),0)

    def test_daymet_leap_day_retained_dec31_omitted(self):
        rows,meta = parse_daymet('Elevation: 30 meters\nyear,yday,tmin (deg c),tmax (deg c),prcp (mm/day)\n2024,60,0,10,1\n2024,365,0,10,-9999\n')
        self.assertEqual(rows[0]['date'],date(2024,2,29))
        self.assertEqual(rows[1]['date'],date(2024,12,30))
        self.assertIsNone(rows[1]['prcp'])
        self.assertEqual(meta['grid_elevation_m'],30)

    def test_missing_dates_never_become_timing(self):
        rules,flags = weather_rules(payload(),[],[],self.s)
        self.assertEqual(rules['development_timing']['status'],'UNKNOWN')
        self.assertFalse(flags)

    def test_rain_both_radar_dates(self):
        r = payload(radar_dates=['2023-09-01','2023-09-13'])
        rows = [{'date':date(2023,8,30)+timedelta(days=i),'prcp':12 if i==1 else 0} for i in range(15)]
        rules,flags = weather_rules(r,rows,[],self.s)
        self.assertTrue(rules['rain_context']['wet_to_dry_radar_transition'])
        self.assertIn('RAIN_MAY_CONFOUND_RADAR_OR_SURFACE',flags)

    def test_frost_flags_risk_not_harvest(self):
        r = payload(greenness_decline_date='2023-10-15')
        rows = [{'date':date(2023,10,9)+timedelta(days=i),'tmin':-1 if i==3 else 5} for i in range(7)]
        rules,flags = weather_rules(r,rows,[],self.s)
        self.assertEqual(rules['frost_risk']['status'],'FLAG')
        self.assertIn('FROST_COULD_EXPLAIN_BROWNING',flags)
        self.assertNotIn('harvest_use',rules['frost_risk'])

    def test_gdd_interval_and_unknown_hybrid(self):
        r = payload(planting={'start':'2023-05-01','end':'2023-05-02'},harvest={'start':'2023-05-04','end':'2023-05-05'})
        rows = [{'date':date(2023,5,1)+timedelta(days=i),'tmin':10,'tmax':30,'prcp':0,'vp':1500} for i in range(5)]
        rules,_ = weather_rules(r,rows,[],self.s)
        self.assertEqual(rules['development_timing']['status'],'TIMING UNKNOWN')
        self.assertEqual(rules['development_timing']['accumulated_gdd_f_range'],[54,90])

    def test_drought_uses_matching_season_and_not_future(self):
        r = payload(greenness_decline_date='2023-09-30')
        rows = [{'date':date(2023,9,1)+timedelta(days=i),'prcp':0,'tmin':10,'tmax':25,'vp':1000} for i in range(30)]
        baseline = [{'date':date(y,9,1)+timedelta(days=i),'prcp':2} for y in range(1981,2011) for i in range(30)]
        rules,flags = weather_rules(r,rows,baseline,self.s)
        self.assertEqual(rules['drought_context']['status'],'FLAG')
        self.assertEqual(rules['drought_context']['baseline_years_used'],30)

    def test_vpd_pa_to_kpa_and_clips_negative(self):
        self.assertEqual(approximate_vpd_kpa({'tmin':10,'tmax':20,'vp':10000}),0)
        self.assertGreater(approximate_vpd_kpa({'tmin':10,'tmax':20,'vp':1000}),0)


class PolicyTests(unittest.TestCase):
    def test_threshold_inclusive_and_original_cutoff(self):
        r = payload(rf={'p_silage':.8,'p_grain':.2,'model_version':'test','calibration_scope':'unknown'})
        self.assertFalse(routing(r,r.rf,Settings())['investigate'])
        r = payload(rf={'p_silage':.85,'p_grain':.15,'model_version':'NL','calibration_scope':'NL','decision_threshold':.88})
        self.assertTrue(routing(r,r.rf,Settings())['investigate'])
        self.assertEqual(r.rf.predicted_label,'GRAIN')

    def test_unknown_quality_triggers(self):
        r=payload(quality={})
        self.assertIn('OBSERVATION_QUALITY_INADEQUATE_OR_UNKNOWN',routing(r,r.rf,Settings())['reasons'])

    def test_context_alone_never_overrides(self):
        r=payload()
        ev=[weather_result(r),ToolResult(evidence_id='elevation',tool='get_elevation_context',field_id=r.field_id,year=r.year,status='OK',data={'median_elevation_m':2000})]
        decision=evaluate(r,r.rf,routing(r,r.rf,Settings()),ev)
        self.assertEqual(decision['final_label'],'REVIEW')
        self.assertIsNone(decision['final_probability'])
        self.assertEqual(r.rf.p_silage,.57)

    def test_physical_evidence_needs_dates_sources_and_weather(self):
        physical={'removal_confirmed':True,'canopy_condition':'green','corroborated':True,'source_ids':['existing-measurements']}
        r=payload(physical_evidence=physical,harvest={'start':'2023-09-01','end':'2023-09-05'})
        decision=evaluate(r,r.rf,routing(r,r.rf,Settings()),[weather_result(r)])
        self.assertEqual(decision['final_label'],'SILAGE')
        for flag in ('FROST_COULD_EXPLAIN_BROWNING','RAINFALL_DEFICIT_COULD_EXPLAIN_BROWNING','RAIN_MAY_CONFOUND_RADAR_OR_SURFACE'):
            decision=evaluate(r,r.rf,routing(r,r.rf,Settings()),[weather_result(r,flags=[flag])])
            self.assertEqual(decision['final_label'],'REVIEW')

    def test_conflicting_records_require_review(self):
        r=payload()
        ev=ToolResult(evidence_id='web',tool='search_public_harvest_evidence',field_id=r.field_id,year=r.year,status='OK',data={'verified_actual_harvest_records':[{'harvest_use':'GRAIN'},{'harvest_use':'SILAGE'}]})
        self.assertEqual(evaluate(r,r.rf,routing(r,r.rf,Settings()),[ev])['final_label'],'REVIEW')

    def test_confirmed_record_no_probability_invention(self):
        r=payload()
        ev=ToolResult(evidence_id='web',tool='search_public_harvest_evidence',field_id=r.field_id,year=r.year,status='OK',data={'verified_actual_harvest_records':[{'harvest_use':'GRAIN'}]})
        out=evaluate(r,r.rf,routing(r,r.rf,Settings()),[ev])
        self.assertEqual(out['final_label'],'GRAIN')
        self.assertIsNone(out['final_probability'])

    def test_labels_extra_inputs_and_future_dates_rejected(self):
        for updates in ({'reference_label':'grain'}, {'greenness_decline_date':'2024-01-01'}, {'geometry':{'type':'Point','coordinates':[0,0]}}):
            with self.assertRaises(ValidationError):payload(**updates)

    def test_rf_sum_and_class_mapping_rejected(self):
        for rf in ({'p_silage':.7,'p_grain':.7,'model_version':'x','calibration_scope':'unknown'},
                   {'p_silage':.7,'p_grain':.3,'model_version':'x','calibration_scope':'unknown','predicted_label':'GRAIN'}):
            with self.assertRaises(ValidationError):payload(rf=rf)


class AgentTests(unittest.TestCase):
    def fake_function(self,name):
        def call(request,settings,http):
            return ToolResult(evidence_id=name,tool=name,field_id=request.field_id,year=request.year,status='UNKNOWN',missing_inputs=['test fixture'])
        return call

    def test_tool_loop_and_deterministic_final(self):
        queue=['get_daymet_weather','get_elevation_context','search_public_harvest_evidence','evaluate_evidence']
        def complete(messages,tools,remaining):
            name=queue.pop(0)
            args = {'hypothesis': {'label':'UNRESOLVED', 'evidence_strength':'insufficient',
                    'evidence_ids':['rf:prediction'], 'reason_codes':['RF_HYPOTHESIS']}} if name == 'evaluate_evidence' else {}
            return {'choices':[{'message':{'tool_calls':[{'id':name,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}}]}
        replacements={name:self.fake_function(name) for name in FUNCTIONS}
        with patch.dict(FUNCTIONS,replacements):
            report=run(payload(),Settings(prefetch_evidence=False),complete=complete)
        self.assertEqual(report['agent']['status'],'COMPLETED')
        self.assertEqual(report['final_label'],'REVIEW')
        self.assertIsNone(report['final_probability'])
        self.assertEqual(len(report['tool_trace']),4)
        self.assertFalse(report['new_satellite_data_fetched'])

    def test_unauthorized_tool_or_changed_field_fails_closed(self):
        for name,args in [('fetch_satellite','{}'),('get_daymet_weather','{"field_id":"another"}')]:
            def complete(*_):
                return {'choices':[{'message':{'tool_calls':[{'id':'x','type':'function','function':{'name':name,'arguments':args}}]}}]}
            report=run(payload(),Settings(prefetch_evidence=False),complete=complete)
            self.assertIn(report['agent']['status'],('FAILED','ROUND_LIMIT'))
            self.assertEqual(report['final_label'],'REVIEW')

    def test_round_limit_and_missing_key_are_review(self):
        report=run(payload(),Settings(nvidia_key='', prefetch_evidence=False))
        self.assertEqual(report['agent']['status'],'UNAVAILABLE')
        def complete(*_):return {'choices':[{'message':{'tool_calls':[],'content':'{"final_label":"SILAGE","confidence":0.99}'}}]}
        report=run(payload(),Settings(prefetch_evidence=False),complete=complete)
        self.assertEqual(report['final_label'],'REVIEW')
        self.assertIsNone(report['final_probability'])

    def test_no_retrieval_on_good_rf(self):
        r=payload(rf={'p_silage':.9,'p_grain':.1,'model_version':'test','calibration_scope':'supplied'})
        with patch('investigation.agent.nvidia_complete',side_effect=AssertionError('should not call')):
            report=run(r,Settings(prefetch_evidence=False))
        self.assertEqual(report['final_label'],'SILAGE')
        self.assertEqual(report['tool_trace'],[])


class SourceTests(unittest.TestCase):
    def test_registry_exact_field_year_and_cutoff(self):
        r=payload()
        row={'field_id':r.field_id,'year':2023,'harvest_use':'GRAIN','harvest':{'start':'2023-10-01','end':'2023-10-02'},
             'published_on':'2024-01-01','source_url':'https://example.org/actual-record','source_sha256':'a'*64,
             'verified_by':'fixture-reviewer','verification_method':'Fixture verified exact field and actual harvest','source_type':'public_actual_harvest_record'}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'records.json'; path.write_text(json.dumps({'records':[row]}))
            self.assertEqual(verified_records(r,Settings(verified_records=str(path))),[])
            row['published_on']='2023-12-01'; path.write_text(json.dumps({'records':[row]}))
            self.assertEqual(len(verified_records(r,Settings(verified_records=str(path)))),1)
            row['field_id']='OTHER'; path.write_text(json.dumps({'records':[row]}))
            self.assertEqual(verified_records(r,Settings(verified_records=str(path))),[])

    def test_suppression_retained(self):
        self.assertEqual(number_or_suppression('(D)'),(None,'(D)'))
        self.assertEqual(number_or_suppression('330,000'),(330000.,None))

    def test_private_urls_rejected(self):
        for url in ('file:///etc/passwd','http://example.com','https://127.0.0.1/','https://user:password@example.com/'):
            with self.assertRaises(DataUnavailable):PublicHTTP.public_url(url)

    def test_evaluation_same_cohort_and_reviews_counted(self):
        r=payload(); report=run(r,Settings(),mode='offline')
        out=compare([report],[{'field_id':r.field_id,'year':2023,'label':'silage'},{'field_id':'BLANK','year':2023,'label':''}])
        self.assertEqual(out['same_eligible_field_years'],1)
        self.assertEqual(out['rf_only']['coverage'],1)
        self.assertEqual(out['rf_plus_investigation']['coverage'],0)
        self.assertEqual(out['rf_plus_investigation']['per_class']['SILAGE']['recall_all_eligible'],0)

    def test_inference_does_not_import_evaluation_labels(self):
        folder=Path(__file__).parents[2]/'investigation'
        if not folder.exists():folder=Path(__file__).parents[2]/'..'/'investigation'
        for name in ('agent.py','policy.py','rf.py','tools.py','weather.py','server.py'):
            source=(folder/name).read_text()
            self.assertNotIn('reference_labels.json',source)
            self.assertNotIn('from .evaluation',source)


if __name__=='__main__':unittest.main()
