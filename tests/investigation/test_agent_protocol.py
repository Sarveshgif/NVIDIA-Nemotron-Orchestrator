"""Regression checks for the live model/tool boundary; no network calls."""
from dataclasses import replace
import json
import os
import unittest
from unittest.mock import Mock, patch

from investigation.agent import FUNCTIONS, ToolRunner, nvidia_complete, run, schemas
from investigation.http import PublicHTTP
from investigation.policy import routing
from investigation.config import Settings
from investigation.schemas import InvestigationInput, ToolResult


RETRIEVALS = ['get_daymet_weather', 'get_elevation_context', 'search_public_harvest_evidence']


def request():
    return InvestigationInput.model_validate({
        'field_id': 'PROTOCOL-FIELD', 'year': 2023, 'reporting_cutoff': '2023-12-31',
        'geometry': {'type': 'Polygon', 'coordinates': [[[-76.9,39],[-76.899,39],
                    [-76.899,39.001],[-76.9,39.001],[-76.9,39]]]},
        'rf': {'p_silage': .57, 'p_grain': .43, 'model_version': 'protocol-fixture',
               'calibration_scope': 'unvalidated'},
        'quality': {'observations_adequate': True, 'geographic_applicability': 'validated'},
    })


def call(name, identifier=None, args=None):
    if args is None:
        args = {'field_id': 'PROTOCOL-FIELD', 'year': 2023}
        if name == 'evaluate_evidence':
            args['hypothesis'] = {'label': 'UNRESOLVED', 'evidence_strength': 'insufficient',
                                  'evidence_ids': ['rf:prediction'], 'reason_codes': ['RF_HYPOTHESIS']}
    return {'id': identifier or name, 'type': 'function', 'function': {
        'name': name, 'arguments': args if isinstance(args, str) else json.dumps(args)}}


def response(*calls, content=None, finish_reason='tool_calls'):
    return {'choices': [{'finish_reason': finish_reason,
                         'message': {'content': content, 'tool_calls': list(calls)}}]}


def unknown(name):
    def retrieve(req, settings, http):
        return ToolResult(evidence_id=name, tool=name, field_id=req.field_id, year=req.year,
                          status='UNKNOWN', missing_inputs=['fixture: no source available'])
    return Mock(side_effect=retrieve)


class ProtocolTests(unittest.TestCase):
    def test_agriculture_excluded_from_schemas_and_execution(self):
        self.assertEqual(set(FUNCTIONS), set(RETRIEVALS))
        self.assertEqual({s['function']['name'] for s in schemas(request())}, set(RETRIEVALS) | {'evaluate_evidence'})
        req, settings = request(), Settings()
        runner = ToolRunner(req, req.rf, routing(req, req.rf, settings), settings, PublicHTTP(settings))
        with self.assertRaisesRegex(ValueError, 'not permitted'):
            runner.call('get_agricultural_context', {})
        report = self.execute([response(call('get_agricultural_context'))])
        self.assertEqual(report['agent']['status'], 'FAILED')
        self.assertEqual(report['tool_trace'], [])

    def test_context_mode_runs_only_three_retrievals_and_evaluation(self):
        with patch.dict(FUNCTIONS, {name: unknown(name) for name in FUNCTIONS}):
            report = run(request(), Settings(), mode='context')
        self.assertEqual([t['tool'] for t in report['tool_trace']], RETRIEVALS + ['evaluate_evidence'])
        self.assertTrue(report['agent']['evaluation_completed'])
        self.assertEqual(report['agent']['transport_http_requests'], 0)

    def execute(self, replies, settings=None):
        self.offered, self.messages = [], []
        replacements = {name: unknown(name) for name in FUNCTIONS}
        iterator = iter(replies)

        def complete(messages, tools, remaining):
            self.offered.append([tool['function']['name'] for tool in tools])
            self.messages.append(json.loads(json.dumps(messages)))
            return next(iterator)

        with patch.dict(FUNCTIONS, replacements):
            report = run(request(), replace(settings or Settings(), prefetch_evidence=False), complete=complete)
        self.retrieval_mocks = replacements
        self.assertEqual(report['rf'], report['original_rf_prediction'])
        self.assertEqual(report['rf']['p_silage'], .57)
        self.assertIsNone(report['final_probability'])
        self.assertFalse(report['new_satellite_data_fetched'])
        return report

    def test_required_evaluation_after_unknown_retrievals(self):
        report = self.execute([response(*(call(name) for name in RETRIEVALS)),
                               response(call('evaluate_evidence'))])
        self.assertEqual(self.offered[1], ['evaluate_evidence'])
        self.assertEqual(report['agent']['status'], 'COMPLETED')
        self.assertTrue(report['agent']['evaluation_completed'])
        self.assertEqual(report['agent']['cited_evidence_ids'], ['rf:prediction'])
        self.assertEqual(report['agent']['correction_attempts'], 0)
        self.assertEqual(report['final_label'], 'REVIEW')
        self.assertEqual(report['errors'], [])
        for name in RETRIEVALS:
            self.assertEqual(self.retrieval_mocks[name].call_count, 1)

    def test_plain_prose_then_missing_final_call_can_be_corrected(self):
        report = self.execute([response(content='I think this is silage.'),
                               response(*(call(name) for name in RETRIEVALS)),
                               response(content='{"final_label":"SILAGE","confidence":0.99}'),
                               response(call('evaluate_evidence'))])
        self.assertEqual(report['agent']['status'], 'COMPLETED')
        self.assertEqual(report['agent']['correction_attempts'], 2)
        self.assertTrue(all(event['code'] == 'REQUIRED_TOOL_CALL_MISSING'
                            for event in report['agent']['protocol_events']))
        self.assertEqual(self.offered[-1], ['evaluate_evidence'])
        self.assertEqual(report['final_label'], 'REVIEW')

    def test_malformed_json_arguments_get_matching_rejection_and_retry(self):
        report = self.execute([response(call('get_daymet_weather', 'bad-json', '{broken')),
                               response(*(call(name) for name in RETRIEVALS)),
                               response(call('evaluate_evidence'))])
        self.assertEqual(report['agent']['status'], 'COMPLETED')
        self.assertEqual(report['agent']['correction_attempts'], 1)
        rejection = next(m for m in self.messages[1] if m.get('tool_call_id') == 'bad-json')
        self.assertEqual(json.loads(rejection['content'])['status'], 'REJECTED')
        self.assertEqual(self.retrieval_mocks['get_daymet_weather'].call_count, 1)

    def test_invalid_citations_corrected_with_same_retrievals(self):
        args = json.loads(call('evaluate_evidence')['function']['arguments'])
        args['hypothesis']['evidence_ids'] = ['invented:record']
        report = self.execute([response(*(call(name) for name in RETRIEVALS)),
                               response(call('evaluate_evidence', 'invalid-citations', args)),
                               response(call('evaluate_evidence', 'corrected'))])
        self.assertEqual(report['agent']['status'], 'COMPLETED')
        self.assertEqual(report['agent']['correction_attempts'], 1)
        self.assertEqual(report['tool_trace'][-2]['result']['status'], 'REJECTED')
        self.assertEqual(self.offered[-1], ['evaluate_evidence'])
        self.assertFalse(any('invented:record' in finding['finding_id']
                             for finding in report['evidence_assessment']['findings']))

    def test_extra_confidence_in_hypothesis_rejected(self):
        args = json.loads(call('evaluate_evidence')['function']['arguments'])
        args['hypothesis']['confidence'] = .99
        report = self.execute([response(*(call(name) for name in RETRIEVALS)),
                               response(call('evaluate_evidence', 'bad-schema', args)),
                               response(call('evaluate_evidence', 'corrected'))])
        self.assertEqual(report['agent']['status'], 'COMPLETED')
        self.assertNotIn('confidence', report['nemotron_hypothesis'])

    def test_exhaustion_never_completes_without_evaluation(self):
        report = self.execute([response(content='SILAGE')] * 3)
        self.assertEqual(report['agent']['status'], 'FORMAT_LIMIT')
        self.assertEqual(report['agent']['calls'], 3)
        self.assertEqual(report['agent']['correction_attempts'], 2)
        self.assertFalse(report['agent']['evaluation_completed'])
        self.assertIsNone(report['nemotron_hypothesis'])
        self.assertEqual(report['final_label'], 'REVIEW')
        self.assertFalse(report['agent']['protocol_events'][-1]['correction_scheduled'])

    def test_disabled_corrections_and_round_limit(self):
        for settings, status in [(Settings(max_corrections=0), 'FORMAT_LIMIT'),
                                 (Settings(max_rounds=1), 'ROUND_LIMIT')]:
            with self.subTest(status=status):
                report = self.execute([response(content='not a tool')], settings)
                self.assertEqual(report['agent']['status'], status)
                self.assertEqual(report['agent']['calls'], 1)

    def test_malformed_envelopes_not_replayed_and_no_partial_calls(self):
        for bad in (None, {}, {'choices': []}, response(call(RETRIEVALS[0]),
                    {'id': 'broken', 'function': {'name': RETRIEVALS[1]}}),
                    response(call(RETRIEVALS[0], 'same'), call(RETRIEVALS[1], 'same'))):
            with self.subTest(bad=bad):
                report = self.execute([bad, response(*(call(name) for name in RETRIEVALS)),
                                       response(call('evaluate_evidence'))])
                self.assertEqual(report['agent']['status'], 'COMPLETED')
                self.assertEqual(len(report['tool_trace']), 4)
                self.assertFalse(any(m['role'] == 'assistant' for m in self.messages[1]))

    def test_length_finish_reason_cannot_be_accepted_as_evaluation(self):
        report = self.execute([response(*(call(name) for name in RETRIEVALS)),
                               response(call('evaluate_evidence', 'truncated'), finish_reason='length'),
                               response(call('evaluate_evidence', 'complete'))])
        self.assertEqual(report['agent']['status'], 'COMPLETED')
        self.assertEqual(report['agent']['protocol_events'][0]['code'], 'TRUNCATED_MODEL_RESPONSE')
        self.assertEqual(len(report['tool_trace']), 4)

    def test_duplicate_retrieval_not_executed_during_correction(self):
        report = self.execute([response(call(RETRIEVALS[0])), response(call(RETRIEVALS[0], 'duplicate')),
                               response(*(call(name) for name in RETRIEVALS[1:])),
                               response(call('evaluate_evidence'))])
        self.assertEqual(report['agent']['status'], 'COMPLETED')
        self.assertNotIn(RETRIEVALS[0], self.offered[1])
        self.assertEqual(self.retrieval_mocks[RETRIEVALS[0]].call_count, 1)

    def test_call_budget_and_time_budget_remain_bounds(self):
        report = self.execute([response(*(call(name) for name in RETRIEVALS))], Settings(max_calls=3))
        self.assertEqual(report['agent']['status'], 'TOOL_LIMIT')
        self.assertEqual(len(report['tool_trace']), 3)
        report = self.execute([], Settings(max_run_seconds=0))
        self.assertEqual(report['agent']['status'], 'TIME_LIMIT')
        self.assertEqual(report['agent']['calls'], 0)

    def test_unknown_tool_and_cross_field_request_fail_closed(self):
        for bad in (call('fetch_satellite'), call(RETRIEVALS[0], args={'field_id': 'OTHER', 'year': 2023})):
            report = self.execute([response(bad)])
            self.assertEqual(report['agent']['status'], 'FAILED')
            self.assertEqual(report['tool_trace'], [])

    def test_oversized_batch_not_executed(self):
        report = self.execute([response(*(call(RETRIEVALS[0], str(i)) for i in range(9))),
                               response(*(call(name) for name in RETRIEVALS)),
                               response(call('evaluate_evidence'))])
        self.assertEqual(report['agent']['status'], 'COMPLETED')
        self.assertEqual(report['agent']['protocol_events'][0]['code'], 'RESPONSE_EXCEEDS_TOOL_BUDGET')
        self.assertEqual(len(report['tool_trace']), 4)

    def test_nvidia_request_forces_evaluation_function(self):
        spec = [tool for tool in schemas(request()) if tool['function']['name'] == 'evaluate_evidence']
        with patch('investigation.agent.requests.Session.post', return_value=Mock(status_code=200, json=lambda: {})) as post:
            nvidia_complete([], Settings(nvidia_key='fixture'), 20, spec)
        data = post.call_args.kwargs['json']
        self.assertEqual(data['tool_choice'], {'type': 'function', 'function': {'name': 'evaluate_evidence'}})
        self.assertFalse(data['parallel_tool_calls'])
        self.assertEqual(data['max_tokens'], 2048)

    def test_nvidia_retrieval_requires_tools_and_keeps_configured_budget(self):
        tools = [tool for tool in schemas(request()) if tool['function']['name'] in RETRIEVALS]
        with patch('investigation.agent.requests.Session.post', return_value=Mock(status_code=200, json=lambda: {})) as post:
            nvidia_complete([], Settings(nvidia_key='fixture', nvidia_max_tokens=4096), 20, tools)
        self.assertEqual(post.call_args.kwargs['json']['tool_choice'], 'required')
        self.assertEqual(post.call_args.kwargs['json']['max_tokens'], 4096)

    def test_http200_invalid_json_uses_shared_format_budget(self):
        provider = Mock(status_code=200)
        provider.json.side_effect = ValueError('untrusted provider body')
        with patch('investigation.agent.requests.Session.post', return_value=provider) as post:
            self.assertIsNone(nvidia_complete([], Settings(nvidia_key='fixture'), 20, schemas(request())))
        self.assertEqual(post.call_count, 1)

    def test_configurable_correction_limits_validated(self):
        with patch('dotenv.load_dotenv'), patch.dict(os.environ, {'INVESTIGATION_MAX_CORRECTIONS': '1', 'NVIDIA_MAX_TOKENS': '4096'}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.max_corrections, 1)
        self.assertEqual(settings.nvidia_max_tokens, 4096)
        for value in ('-1', '4'):
            with patch('dotenv.load_dotenv'), patch.dict(os.environ, {'INVESTIGATION_MAX_CORRECTIONS': value}, clear=True):
                with self.assertRaises(ValueError):
                    Settings.from_env()


if __name__ == '__main__':
    unittest.main()
