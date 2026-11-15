from dataclasses import replace
from datetime import date,timedelta
from unittest.mock import patch
import json,unittest
from investigation.config import Settings
from investigation.weather import seasonal_comparison,checked_weather_rows,weather_rules,get_daymet_weather
from investigation.historical_context import _matching_rows,nass_regional_context
from investigation.assessment import build_assessment,validate_hypothesis
from investigation.policy import evaluate,routing
from investigation.http import DataUnavailable
from test_workflow import payload


def period(year,days=90,tmin=10,tmax=30,prcp=2):
    end=date(year,9,30)
    return [{'date':end-timedelta(days=i),'tmin':tmin,'tmax':tmax,'prcp':prcp,'vp':1200} for i in range(days)]


def annual(year=2020,value='1,000',**updates):
    r=dict(year=year,load_time=f'{year+1}-02-01 12:00:00.000',source_desc='SURVEY',commodity_desc='CORN',
           state_alpha='MD',agg_level_desc='STATE',domain_desc='TOTAL',prodn_practice_desc='ALL PRODUCTION PRACTICES',
           statisticcat_desc='AREA HARVESTED',freq_desc='ANNUAL',reference_period_desc='YEAR',unit_desc='ACRES',
           short_desc='CORN, GRAIN - ACRES HARVESTED',Value=value)
    r.update(updates);return r


def weekly(year=2023,value='25',**updates):
    r=annual(year,value,load_time=f'{year}-09-25 12:00:00.000',statisticcat_desc='PROGRESS',freq_desc='WEEKLY',
             week_ending=f'{year}-09-24',unit_desc='PCT MATURE',short_desc='CORN - PROGRESS, MEASURED IN PCT MATURE')
    r.update(updates);return r


class HistoricalContextTests(unittest.TestCase):
    def setUp(self):
        self.s=Settings(nass_key='SECRET-TEST-KEY')
        self.r=payload(geography={'country':'US','state_alpha':'MD'},harvest={'start':'2023-09-29','end':'2023-09-30'})
        self.history=[r for y in range(1981,2011) for r in period(y)]

    def test_matching_windows_have_known_anomalies_without_planting(self):
        result=seasonal_comparison(self.r,period(2023,prcp=3),self.history,self.s)
        self.assertEqual(result['status'],'OK')
        for w in result['windows']:
            self.assertEqual(w['historical_year_count'],30)
            self.assertEqual(w['comparison']['precipitation_mm']['anomaly_from_historical_median'],w['window_days'])
            self.assertEqual(w['comparison']['mean_temperature_c']['anomaly_from_historical_median'],0)
            self.assertIn('calendar_window_gdd_f',w['comparison'])
        rules,_=weather_rules(self.r,period(2023),self.history,self.s)
        self.assertEqual(rules['development_timing']['status'],'UNKNOWN')

    def test_future_baseline_years_never_used(self):
        result=seasonal_comparison(self.r,period(2023),self.history+period(2024),replace(self.s,baseline_end=2029))
        self.assertEqual(result['baseline_period'][1],2022)
        self.assertNotIn(2024,result['windows'][0]['complete_historical_years'])

    def test_insufficient_history_and_missing_current_are_unknown(self):
        result=seasonal_comparison(self.r,period(2023),[r for y in range(1981,2000) for r in period(y)],self.s)
        self.assertEqual(result['status'],'UNKNOWN')
        self.assertFalse(result['windows'][0]['comparison'])
        current=period(2023);current[0]['prcp']=None
        result=seasonal_comparison(self.r,current,self.history,self.s)
        self.assertEqual(result['windows'][0]['status'],'UNKNOWN')
        self.assertIsNone(result['windows'][0]['current'])

    def test_invalid_values_are_missing_not_zero(self):
        rows,n=checked_weather_rows([{'date':date(2023,9,30),'tmin':25,'tmax':10,'prcp':-1,'vp':-2}])
        self.assertEqual(n,4)
        self.assertIsNone(rows[0]['prcp']);self.assertIsNone(rows[0]['tmin'])

    def test_nass_filters_future_wrong_state_domain_and_year(self):
        rows=[annual(),annual(2024),annual(state_alpha='VA'),annual(domain_desc='IRRIGATION'),
              annual(load_time='2025-01-01 00:00:00'),annual(2000)]
        got,excluded=_matching_rows(rows,self.r,'acreage',2013,2022,date(2023,9,30))
        self.assertEqual(len(got),1);self.assertEqual(excluded,5)
        self.assertEqual(got[0]['year'],2020)

    def test_nass_preserves_suppression_and_zero(self):
        rows=[annual(2020,'(D)'),annual(2019,'0')]
        got,_=_matching_rows(rows,self.r,'acreage',2013,2022,date(2023,9,30))
        d={r['year']:r for r in got}
        self.assertIsNone(d[2020]['numeric_value']);self.assertEqual(d[2020]['suppression_flag'],'(D)')
        self.assertEqual(d[2019]['numeric_value'],0)

    def test_ambiguous_versions_excluded_and_latest_valid_selected(self):
        got,_=_matching_rows([annual(value='1'),annual(value='2')],self.r,'acreage',2013,2022,date(2023,9,30))
        self.assertEqual(got,[])
        got,_=_matching_rows([annual(value='1'),annual(value='2',load_time='2021-03-01 00:00:00')],self.r,'acreage',2013,2022,date(2023,9,30))
        self.assertEqual(got[0]['numeric_value'],2)

    def test_progress_future_observation_and_release_are_excluded(self):
        got,_=_matching_rows([weekly(),weekly(2023,week_ending='2023-10-01'),weekly(load_time='2023-10-01 00:00:00')],self.r,'progress',2018,2023,date(2023,9,30))
        self.assertEqual(len(got),1)

    def test_regional_summary_ignores_server_out_of_range_rows_and_hides_key(self):
        class HTTP:
            def json(inner,url,params,**kw):
                self.assertEqual(params['key'],'SECRET-TEST-KEY')
                if params['statisticcat_desc']=='PROGRESS':return {'data':[weekly(),weekly(2022,20),weekly(2026,99)]},False
                return {'data':[annual(),annual(short_desc='CORN, SILAGE - ACRES HARVESTED',value='100'),annual(2026)]},False
        context=nass_regional_context(self.r,self.s,HTTP())
        self.assertEqual(context['status'],'OK')
        self.assertEqual(context['crop_progress'][0]['historical_median_pct'],20)
        self.assertAlmostEqual(context['historical_acreage'][0]['grain_share_of_reported_harvest_acres'],1000/1100)
        self.assertNotIn('SECRET-TEST-KEY',json.dumps(context))
        self.assertNotIn('2026',json.dumps(context))

    def test_nass_failure_does_not_throw_or_fabricate_weather(self):
        class HTTP:
            def json(*args,**kwargs):raise DataUnavailable('Public source HTTP 400')
        result=nass_regional_context(self.r,self.s,HTTP())
        self.assertEqual(result['status'],'UNKNOWN');self.assertEqual(len(result['errors']),2)
        self.assertFalse(result['crop_progress'])

    def test_weather_adapter_exposes_citable_history_before_model_evaluation(self):
        def csv(rows):
            text='Elevation: 20 meters\nyear,yday,tmin (deg c),tmax (deg c),prcp (mm/day),vp (Pa)\n'
            return (text+''.join(f"{r['date'].year},{r['date'].timetuple().tm_yday},{r['tmin']},{r['tmax']},{r['prcp']},{r['vp']}\n" for r in rows)).encode()
        current=csv(period(2023));history=csv(self.history)
        class HTTP:
            def get(inner,url,params,**kw):return (history if params['start'].startswith('1981') else current),False
            def json(inner,url,params,**kw):return {'data':[weekly()]},False
        result=get_daymet_weather(self.r,self.s,HTTP())
        self.assertEqual(result.data['historical_comparison']['status'],'OK')
        self.assertEqual(result.data['nass_regional_context']['status'],'OK')
        assessment=build_assessment(self.r,self.r.rf,[result])
        ids={f['finding_id'] for f in assessment['findings']}
        self.assertIn('weather:historical:90day',ids);self.assertIn('nass:regional_history',ids)
        decision=evaluate(self.r,self.r.rf,routing(self.r,self.r.rf,self.s),[result])
        self.assertEqual(decision['final_label'],'REVIEW')
        validated=validate_hypothesis({'label':'GRAIN','evidence_strength':'limited','evidence_ids':['weather:historical:90day','nass:regional_history'],'reason_codes':['CONTEXT_ONLY']},assessment,self.r,self.r.rf,decision)
        self.assertTrue(validated['citations_validated']);self.assertFalse(validated['supported_for_final_decision'])

    def test_historical_flags_validate(self):
        import tempfile,os
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'INVESTIGATION_HISTORICAL_WEATHER':'maybe'},clear=True):
            with self.assertRaisesRegex(ValueError,'HISTORICAL_WEATHER'):Settings.from_env(Path(d))
