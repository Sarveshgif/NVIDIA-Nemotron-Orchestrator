from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import os
from threading import Event, Lock
import unittest
from unittest.mock import Mock, patch

import requests

from investigation.agent import FUNCTIONS, nvidia_complete, post_nvidia, retry_after_seconds, run, runtime_configuration, schemas
from investigation.config import Settings
from investigation.http import DataUnavailable
from test_agent_protocol import RETRIEVALS, call, request, response, unknown


def provider(status=200, headers=None, result=None):
    return Mock(status_code=status, headers=headers or {}, json=lambda: result if result is not None else {})


class NvidiaTransportTests(unittest.TestCase):
    def setUp(self):
        # Legacy transport mechanics stay isolated from the separately tested shared guard.
        self.guard_patch = patch('investigation.agent.recovery_guard')
        self.guard_patch.start()
        self.addCleanup(self.guard_patch.stop)
        jitter = patch('investigation.agent.random.uniform', side_effect=lambda low, high: low)
        jitter.start()
        self.addCleanup(jitter.stop)

    def test_one_session_reused_for_retry_and_final_evaluation(self):
        replies = [requests.ConnectionError('hidden'), provider(result=response(*(call(n) for n in RETRIEVALS))),
                   provider(result=response(call('evaluate_evidence')))]
        with patch.dict(FUNCTIONS, {name: unknown(name) for name in FUNCTIONS}), \
             patch('investigation.agent.requests.Session.post', autospec=True, side_effect=replies) as post, \
             patch('investigation.agent.requests.Session.close', autospec=True) as close, \
             patch('investigation.agent.time.sleep'):
            report = run(request(), Settings(nvidia_key='fixture', prefetch_evidence=False))
        self.assertEqual(report['agent']['status'], 'COMPLETED')
        sessions = [c.args[0] for c in post.call_args_list]
        self.assertEqual(len(sessions), 3)
        self.assertTrue(all(s is sessions[0] for s in sessions))
        self.assertTrue(any(c.args[0] is sessions[0] for c in close.call_args_list))
        self.assertTrue(all(c.kwargs['allow_redirects'] is False for c in post.call_args_list))

    def test_concurrency_wait_failure_is_logged_without_counting_http(self):
        slots = Mock()
        slots.acquire.return_value = False
        with patch('investigation.agent.nvidia_slots', return_value=slots), \
             patch('investigation.agent.requests.Session.post') as post:
            report = run(request(), Settings(nvidia_key='fixture', prefetch_evidence=False))
        post.assert_not_called()
        slots.release.assert_not_called()
        self.assertEqual(report['agent']['transport_http_requests'], 0)
        self.assertEqual(report['agent']['transport_attempts'][0]['outcome'], 'CONCURRENCY_WAIT_TIMEOUT')
        self.assertFalse(report['agent']['transport_attempts'][0]['http_request_sent'])
        self.assertEqual(report['agent']['status'], 'FAILED')
        self.assertEqual(report['final_label'], 'REVIEW')

    def test_queue_time_is_subtracted_before_network_timeouts(self):
        slots = Mock()
        slots.acquire.return_value = True
        session, event = Mock(), {'http_request_sent': False}
        session.post.return_value = provider()
        with patch('investigation.agent.nvidia_slots', return_value=slots), \
             patch('investigation.agent.time.monotonic', side_effect=[0, 6, 6, 6, 7]):
            post_nvidia(session, {}, Settings(), 10, event)
        connect, read = session.post.call_args.kwargs['timeout']
        self.assertLessEqual(connect + read, 4)
        self.assertEqual(event['queue_wait_ms'], 6000)
        slots.release.assert_called_once()

    def test_expired_queue_budget_does_not_send_and_releases_slot(self):
        slots = Mock()
        slots.acquire.return_value = True
        session, event = Mock(), {'http_request_sent': False}
        with patch('investigation.agent.nvidia_slots', return_value=slots), \
             patch('investigation.agent.time.monotonic', side_effect=[0, 9, 9]):
            with self.assertRaisesRegex(DataUnavailable, 'TIME_BUDGET_EXHAUSTED'):
                post_nvidia(session, {}, Settings(), 10, event)
        session.post.assert_not_called()
        slots.release.assert_called_once()
        self.assertEqual(event['outcome'], 'TIME_BUDGET_EXHAUSTED')

    def test_concurrent_calls_never_exceed_two_http_requests(self):
        release, two_started, third_started = Event(), Event(), Event()
        lock = Lock()
        active, peak, entered = 0, 0, 0
        def post(*args, **kwargs):
            nonlocal active, peak, entered
            with lock:
                active += 1
                entered += 1
                peak = max(peak, active)
                if entered == 2:
                    two_started.set()
                if entered == 3:
                    third_started.set()
            try:
                if not release.wait(5):
                    raise AssertionError('Test fixture release timed out')
                return provider()
            finally:
                with lock:
                    active -= 1
        with patch('investigation.agent.requests.Session.post', side_effect=post):
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(nvidia_complete, [], Settings(nvidia_max_concurrency=2), 10) for _ in range(3)]
                try:
                    self.assertTrue(two_started.wait(5))
                    self.assertFalse(third_started.wait(.05))
                finally:
                    release.set()
                self.assertTrue(all(f.result(timeout=5) == {} for f in futures))
        self.assertEqual(peak, 2)

    def test_runtime_configuration_exposes_loaded_limits_without_keys(self):
        config = runtime_configuration(Settings(nvidia_key='SECRET', nass_key='NASS-SECRET', nvidia_max_concurrency=1))
        self.assertEqual(config['enabled_retrieval_tools'], sorted(RETRIEVALS))
        self.assertEqual(config['nvidia_max_concurrency_per_process'], 1)
        self.assertEqual(config['nvidia_read_timeout_seconds'], 60)
        self.assertNotIn('SECRET', json.dumps(config))

    def test_connection_failure_retries_then_succeeds(self):
        trace = []
        with patch('investigation.agent.requests.Session.post', side_effect=[requests.ConnectionError('private detail'), provider()]) as post, patch('investigation.agent.time.sleep') as sleep:
            self.assertEqual(nvidia_complete([], Settings(), 180, transport_trace=trace), {})
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(2.)
        self.assertEqual([e['outcome'] for e in trace], ['CONNECTION_ERROR', 'RESPONSE_RECEIVED'])

    def test_read_timeouts_stop_after_bounded_attempts(self):
        trace = []
        with patch('investigation.agent.requests.Session.post', side_effect=requests.ReadTimeout('private URL')) as post, patch('investigation.agent.time.sleep') as sleep:
            with self.assertRaisesRegex(DataUnavailable, 'READ_TIMEOUT after 3 attempt'):
                nvidia_complete([], Settings(), 180, transport_trace=trace)
        self.assertEqual(post.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2., 4.])
        self.assertEqual(len(trace), 3)
        self.assertNotIn('private URL', json.dumps(trace))

    def test_timeout_types_are_distinguished(self):
        for error, code in [(requests.ConnectTimeout('hidden'), 'CONNECT_TIMEOUT'),
                            (requests.Timeout('hidden'), 'TIMEOUT'),
                            (requests.RequestException('hidden'), 'REQUEST_ERROR'),
                            (requests.exceptions.SSLError('hidden'), 'TLS_ERROR')]:
            with self.subTest(code=code), patch('investigation.agent.requests.Session.post', side_effect=error):
                trace = []
                with self.assertRaisesRegex(DataUnavailable, code):
                    nvidia_complete([], Settings(nvidia_max_attempts=1), 20, transport_trace=trace)
                self.assertEqual(trace[0]['outcome'], code)

    def test_tls_and_authentication_errors_are_not_retried(self):
        for value, code in [(requests.exceptions.SSLError('secret'), 'TLS_ERROR'), (provider(401), 'HTTP_401'), (provider(403), 'HTTP_403')]:
            with self.subTest(code=code):
                kwargs = {'side_effect': value} if isinstance(value, Exception) else {'return_value': value}
                with patch('investigation.agent.requests.Session.post', **kwargs) as post, patch('investigation.agent.time.sleep') as sleep:
                    with self.assertRaisesRegex(DataUnavailable, code):
                        nvidia_complete([], Settings(), 180)
                self.assertEqual(post.call_count, 1)
                sleep.assert_not_called()

    def test_rate_limit_retry_after_is_honored(self):
        with patch('investigation.agent.requests.Session.post', side_effect=[provider(429, {'Retry-After': '3'}), provider()]), patch('investigation.agent.time.sleep') as sleep:
            nvidia_complete([], Settings(), 180)
        sleep.assert_called_once_with(3.)

    def test_large_retry_after_does_not_cause_unbounded_wait(self):
        trace = []
        with patch('investigation.agent.requests.Session.post', return_value=provider(429, {'Retry-After': '120'})) as post, patch('investigation.agent.time.sleep') as sleep:
            with self.assertRaisesRegex(DataUnavailable, 'retry delay exceeds'):
                nvidia_complete([], Settings(), 180, transport_trace=trace)
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(trace[0]['retry_skipped'], 'RETRY_DELAY_EXCEEDS_BUDGET')

    def test_server_errors_retry_but_bad_request_does_not(self):
        for status in (500, 502, 503, 504):
            with self.subTest(status=status), patch('investigation.agent.requests.Session.post', side_effect=[provider(status), provider()]) as post, patch('investigation.agent.time.sleep'):
                nvidia_complete([], Settings(), 180)
                self.assertEqual(post.call_count, 2)
        with patch('investigation.agent.requests.Session.post', return_value=provider(400)) as post:
            with self.assertRaisesRegex(DataUnavailable, 'HTTP_400'):
                nvidia_complete([], Settings(), 180)
        self.assertEqual(post.call_count, 1)

    def test_retry_after_parsing(self):
        self.assertEqual(retry_after_seconds('2'), 2.)
        for value in (None, '-1', 'NaN', 'inf', 'not a date'):
            self.assertIsNone(retry_after_seconds(value))
        with patch('investigation.agent.time.time', return_value=0):
            self.assertEqual(retry_after_seconds('Thu, 01 Jan 1970 00:00:05 GMT'), 5.)

    def test_remaining_budget_caps_connect_and_read_timeouts(self):
        with patch('investigation.agent.requests.Session.post', return_value=provider()) as post:
            nvidia_complete([], Settings(nvidia_read_timeout=90), 10)
        connect, read = post.call_args.kwargs['timeout']
        self.assertLessEqual(connect + read, 10)
        with patch('investigation.agent.requests.Session.post') as post:
            with self.assertRaisesRegex(DataUnavailable, 'TIME_BUDGET_EXHAUSTED'):
                nvidia_complete([], Settings(), 1)
        post.assert_not_called()

    def test_failed_run_logs_attempts_without_secret_exception_text(self):
        with patch('investigation.agent.requests.Session.post', side_effect=requests.ConnectionError('SECRET-KEY https://host/?key=SECRET-KEY')), patch('investigation.agent.time.sleep'):
            report = run(request(), Settings(nvidia_key='SECRET-KEY', prefetch_evidence=False))
        self.assertEqual(report['agent']['status'], 'FAILED')
        self.assertEqual(report['agent']['transport_http_requests'], 3)
        self.assertEqual(report['agent']['correction_attempts'], 0)
        self.assertEqual(report['agent']['calls'], 1)
        self.assertEqual(report['final_label'], 'REVIEW')
        self.assertIsNone(report['final_probability'])
        self.assertEqual(report['rf'], report['original_rf_prediction'])
        self.assertNotIn('SECRET-KEY', json.dumps(report))

    def test_transport_retry_does_not_use_format_correction_or_skip_final_evaluation(self):
        replies = [requests.ReadTimeout('hidden'), provider(result=response(*(call(name) for name in RETRIEVALS))),
                   provider(result=response(call('evaluate_evidence')))]
        with patch.dict(FUNCTIONS, {name: unknown(name) for name in FUNCTIONS}), patch('investigation.agent.requests.Session.post', side_effect=replies), patch('investigation.agent.time.sleep'):
            report = run(request(), Settings(nvidia_key='fixture', prefetch_evidence=False))
        self.assertEqual(report['agent']['status'], 'COMPLETED')
        self.assertEqual(report['agent']['calls'], 2)
        self.assertEqual(report['agent']['transport_http_requests'], 3)
        self.assertEqual(report['agent']['correction_attempts'], 0)
        self.assertTrue(report['agent']['evaluation_completed'])
        self.assertEqual(report['agent']['transport_attempts'][-1]['phase'], 'evaluation')

    def test_transport_settings_are_validated(self):
        with patch('dotenv.load_dotenv'), patch.dict(os.environ, {'NVIDIA_MAX_ATTEMPTS': '2', 'NVIDIA_READ_TIMEOUT_SECONDS': '70'}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.nvidia_max_attempts, 2)
        self.assertEqual(settings.nvidia_read_timeout, 70)
        for invalid in ({'NVIDIA_MAX_ATTEMPTS': '4'}, {'NVIDIA_READ_TIMEOUT_SECONDS': '0'}, {'NVIDIA_READ_TIMEOUT_SECONDS': 'NaN'},
                        {'NVIDIA_MAX_CONCURRENCY': '0'}, {'NVIDIA_MAX_CONCURRENCY': '9'}):
            with patch('dotenv.load_dotenv'), patch.dict(os.environ, invalid, clear=True):
                with self.assertRaises(ValueError):
                    Settings.from_env()


if __name__ == '__main__':
    unittest.main()
