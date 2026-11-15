"""Focused outage, rate-window, prefetch and real-adapter regression checks."""
from pathlib import Path
from unittest.mock import patch, Mock
import json
import socket
import unittest
import requests

from investigation.agent import run, FUNCTIONS
from investigation.config import Settings
from investigation.nvidia_transport import (RecoveryGuard, traced_socket, bind_connection_trace,
                                            reset_recovery_state, safe_exception_types)
from test_agent_protocol import request, response, call, unknown, RETRIEVALS


class Clock:
    def __init__(self):self.now = 0.
    def __call__(self):return self.now
    def wait(self, seconds):self.now += seconds


class RecoveryTests(unittest.TestCase):
    def setUp(self):reset_recovery_state()

    def guard(self, rpm=40):
        clock = Clock()
        return RecoveryGuard(rpm, 3, 20, clock=clock, wait=clock.wait), clock

    def test_rate_limit_counts_retries_and_waits_for_rolling_window(self):
        guard, clock = self.guard(rpm=2)
        for _ in range(2):guard.admit(180, {})
        event = {};guard.admit(180, event)
        self.assertEqual(clock.now, 60)
        self.assertEqual(event['rate_wait_ms'], 60000)
        self.assertLessEqual(event['rate_window_attempts'], 2)

    def test_three_failures_open_then_one_probe_closes_on_success(self):
        guard, clock = self.guard()
        for _ in range(3):
            event = {};guard.admit(180, event);event['outcome'] = 'CONNECT_TIMEOUT';guard.record(event)
        self.assertEqual(guard.open_until, 20)
        probe = {};guard.admit(180, probe)
        self.assertEqual(clock.now, 20)
        self.assertTrue(probe['circuit_half_open_probe'])
        self.assertEqual(probe['circuit_wait_ms'], 20000)
        probe['outcome'] = 'RESPONSE_RECEIVED';guard.record(probe)
        self.assertEqual(guard.open_until, 0)
        self.assertFalse(guard.probe_inflight)

    def test_failed_probe_reopens_and_budget_does_not_start_http(self):
        guard, clock = self.guard()
        for _ in range(3):guard.record({'outcome':'CONNECT_TIMEOUT'})
        probe = {};guard.admit(180, probe);probe['outcome']='CONNECT_TIMEOUT';guard.record(probe)
        self.assertEqual(guard.open_until, 40)
        with self.assertRaisesRegex(RuntimeError, 'remaining budget'):
            guard.admit(30, {})
        self.assertEqual(len(guard.requests), 1)

    def test_only_one_half_open_probe_and_cancel_releases_it(self):
        guard, clock = self.guard()
        for _ in range(3):guard.record({'outcome':'CONNECT_TIMEOUT'})
        event = {};guard.admit(180, event)
        self.assertTrue(guard.probe_inflight)
        with self.assertRaises(RuntimeError):guard.admit(22, {})
        guard.cancel(event)
        self.assertFalse(guard.probe_inflight)
        other = {};guard.admit(180, other)
        self.assertTrue(other['circuit_half_open_probe'])

    def test_prefetch_survives_complete_nvidia_outage(self):
        with patch.dict(FUNCTIONS, {n:unknown(n) for n in FUNCTIONS}), \
             patch('investigation.agent.requests.Session.post', side_effect=requests.ConnectTimeout('SECRET URL')), \
             patch('investigation.agent.time.sleep'):
            report = run(request(), Settings(nvidia_key='SECRET'))
        self.assertEqual(report['agent']['status'], 'FAILED')
        self.assertEqual([t['tool'] for t in report['tool_trace']], RETRIEVALS)
        self.assertTrue(all(t['requested_by']=='backend_prefetch' for t in report['tool_trace']))
        self.assertEqual(len(report['evidence']), 3)
        self.assertEqual(report['agent']['calls'], 1)
        self.assertTrue(all(e['phase']=='evaluation' for e in report['agent']['transport_attempts']))
        self.assertIsNone(report['final_probability'])
        self.assertEqual(report['rf'], report['original_rf_prediction'])
        self.assertNotIn('SECRET', json.dumps(report))

    def test_prefetch_needs_only_final_model_request_and_valid_evaluation(self):
        observed = []
        def complete(messages, tools, remaining):
            observed.append(([t['function']['name'] for t in tools], json.loads(messages[1]['content'])))
            return response(call('evaluate_evidence'))
        with patch.dict(FUNCTIONS, {n:unknown(n) for n in FUNCTIONS}):
            report = run(request(), Settings(), complete=complete)
        self.assertEqual(report['agent']['status'], 'COMPLETED')
        self.assertEqual(report['agent']['calls'], 1)
        self.assertEqual(observed[0][0], ['evaluate_evidence'])
        self.assertEqual(len(observed[0][1]['retrieved_evidence']), 3)
        self.assertTrue(report['agent']['evaluation_completed'])
        self.assertEqual(report['final_label'], 'REVIEW')

    def test_missing_key_still_preserves_prefetched_evidence(self):
        with patch.dict(FUNCTIONS, {n:unknown(n) for n in FUNCTIONS}):
            report = run(request(), Settings(nvidia_key=''))
        self.assertEqual(report['agent']['status'], 'UNAVAILABLE')
        self.assertEqual(len(report['evidence']), 3)

    def test_actual_address_fallback_records_tcp_without_error_text(self):
        event = {};connection = Mock()
        connection._dns_host='integrate.api.nvidia.com';connection.host=connection._dns_host
        connection.port=443;connection.timeout=5;connection.source_address=None;connection.socket_options=[]
        bad, good = Mock(), Mock()
        bad.connect.side_effect = socket.timeout('SECRET detail')
        addresses=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('99.83.136.103',443)),
                   (socket.AF_INET,socket.SOCK_STREAM,6,'',('75.2.113.119',443))]
        with patch('investigation.nvidia_transport.socket.getaddrinfo', return_value=addresses), \
             patch('investigation.nvidia_transport.socket.socket', side_effect=[bad,good]):
            import time
            actual=traced_socket(connection,event,time.monotonic()+30)
        self.assertIs(actual,good)
        self.assertEqual([e['outcome'] for e in event['tcp_attempts']], ['TIMEOUT','CONNECTED'])
        self.assertEqual(event['peer_address'],'75.2.113.119')
        self.assertIn('started_at_utc',event['tcp_attempts'][0])
        self.assertNotIn('SECRET',json.dumps(event))
        bad.close.assert_called_once()

    def test_adapter_keeps_tls_verification_and_session_isolation(self):
        with requests.Session() as first, requests.Session() as second:
            a=bind_connection_trace(first,{},999999999);b=bind_connection_trace(second,{},999999999)
            self.assertIsNot(a,b)
            self.assertTrue(first.verify)
            adapter=first.get_adapter('https://integrate.api.nvidia.com/v1/models')
            self.assertEqual(adapter.max_retries.total,0)
            self.assertNotEqual(type(adapter),type(first.get_adapter('https://example.com/')))

    def test_exception_chain_logs_types_only(self):
        try:
            try:raise socket.timeout('SECRET')
            except OSError as inner:raise requests.ConnectTimeout('SECRET KEY') from inner
        except requests.ConnectTimeout as error:
            self.assertEqual(safe_exception_types(error),['ConnectTimeout','TimeoutError'])


if __name__=='__main__':unittest.main()
