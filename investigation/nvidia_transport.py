"""Per-session connection diagnostics and process-local NVIDIA outage/rate control.

Never log exception text, proxy URLs, request headers or response bodies.
The connection adapter preserves urllib3 address ordering and TLS verification.
"""
from collections import deque
from datetime import datetime, timezone
import ipaddress
import socket
import sys
from threading import Condition, Lock
import time

from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPSConnection
from urllib3.connectionpool import HTTPSConnectionPool
from urllib3.exceptions import ConnectTimeoutError, NameResolutionError, NewConnectionError
from urllib3.util.connection import allowed_gai_family

from .http import DataUnavailable


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_exception_types(error):
    result, seen = [], set()
    while error is not None and id(error) not in seen and len(result) < 6:
        seen.add(id(error))
        result.append(type(error).__name__)
        error = getattr(error, 'reason', None) or error.__cause__ or error.__context__
        if not isinstance(error, BaseException):
            break
    return result


def public_address(value):
    try:
        return value if ipaddress.ip_address(value).is_global else 'PRIVATE_ADDRESS'
    except ValueError:
        return 'ADDRESS_UNAVAILABLE'


def traced_socket(connection, event, deadline):
    """Equivalent address fallback to urllib3, with each TCP attempt measured."""
    event['connection_reused'] = False
    event['connection_phase'] = 'dns'
    started = time.monotonic()
    try:
        addresses = socket.getaddrinfo(connection._dns_host, connection.port,
                                       allowed_gai_family(), socket.SOCK_STREAM)
    except socket.gaierror as error:
        event['dns_exception_types'] = safe_exception_types(error)
        raise NameResolutionError(connection.host, connection, error) from error
    finally:
        event['dns_elapsed_ms'] = round((time.monotonic() - started) * 1000, 3)
    event['resolved_addresses'] = [{'address': public_address(a[4][0]),
                                   'family': 'IPv6' if a[0] == socket.AF_INET6 else 'IPv4'}
                                  for a in addresses]
    attempts = event.setdefault('tcp_attempts', [])
    last_error = None
    for family, kind, protocol, _, sockaddr in addresses:
        budget = deadline - time.monotonic()
        if budget <= 0:
            last_error = socket.timeout('connection deadline exhausted')
            break
        attempt = {'address': public_address(sockaddr[0]),
                   'family': 'IPv6' if family == socket.AF_INET6 else 'IPv4',
                   'started_at_utc': utc_now()}
        attempts.append(attempt)
        raw = None
        tcp_started = time.monotonic()
        event['connection_phase'] = 'tcp_connect'
        try:
            raw = socket.socket(family, kind, protocol)
            for option in connection.socket_options or []:
                raw.setsockopt(*option)
            raw.settimeout(min(float(connection.timeout), budget))
            if connection.source_address:
                raw.bind(connection.source_address)
            raw.connect(sockaddr)
            attempt['outcome'] = 'CONNECTED'
            event['peer_address'] = public_address(sockaddr[0])
            event['peer_family'] = attempt['family']
            event['connection_phase'] = 'tls_or_proxy_setup'
            sys.audit('http.client.connect', connection, connection.host, connection.port)
            return raw
        except OSError as error:
            last_error = error
            attempt.update(outcome='TIMEOUT' if isinstance(error, TimeoutError) else 'SOCKET_ERROR',
                           exception_types=safe_exception_types(error))
            if error.errno is not None:
                attempt['errno'] = error.errno
            if raw is not None:
                raw.close()
        finally:
            attempt['elapsed_ms'] = round((time.monotonic() - tcp_started) * 1000, 3)
            attempt['finished_at_utc'] = utc_now()
    if isinstance(last_error, TimeoutError):
        raise ConnectTimeoutError(connection, 'NVIDIA TCP connection timeout') from last_error
    if last_error is not None:
        raise NewConnectionError(connection, 'NVIDIA TCP connection unavailable') from last_error
    raise NewConnectionError(connection, 'NVIDIA DNS returned no usable addresses')


class ConnectionTrace:
    event = None
    deadline = None


class DiagnosticAdapter(HTTPAdapter):
    def __init__(self, trace):
        self.trace = trace
        # Each adapter owns its connection classes; no global urllib3 mutation.
        class DiagnosticConnection(HTTPSConnection):
            def _new_conn(self):
                if trace.event is None:
                    return super()._new_conn()
                return traced_socket(self, trace.event, trace.deadline)

            def connect(self):
                event = trace.event
                if event is None:
                    return super().connect()
                started = time.monotonic()
                event['proxy_used'] = bool(self.proxy)
                try:
                    super().connect()
                    event['connection_phase'] = 'connected'
                    if self.sock is not None:
                        event['tls_version'] = self.sock.version()
                finally:
                    total = (time.monotonic() - started) * 1000
                    tcp_dns = event.get('dns_elapsed_ms', 0) + sum(a['elapsed_ms'] for a in event.get('tcp_attempts', []))
                    event['connection_setup_elapsed_ms'] = round(total, 3)
                    if any(a.get('outcome') == 'CONNECTED' for a in event.get('tcp_attempts', [])):
                        # Includes certificate verification and proxy CONNECT if configured.
                        event['tls_or_proxy_setup_ms'] = round(max(0, total - tcp_dns), 3)
        class DiagnosticPool(HTTPSConnectionPool):
            ConnectionCls = DiagnosticConnection
        self.diagnostic_pool = DiagnosticPool
        super().__init__(max_retries=0)

    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = dict(self.poolmanager.pool_classes_by_scheme,
                                                       https=self.diagnostic_pool)

    def proxy_manager_for(self, *args, **kwargs):
        manager = super().proxy_manager_for(*args, **kwargs)
        proxy = args[0] if args else kwargs.get('proxy', '')
        if str(proxy).lower().startswith('socks'):
            # Preserve SOCKS routing/DNS semantics; never replace its ConnectionCls.
            if self.trace.event is not None:
                self.trace.event['connection_details_unavailable'] = 'SOCKS_ADAPTER'
                self.trace.event['connection_reused'] = None
            return manager
        # Keep the configured proxy path; never bypass it for diagnostics.
        manager.pool_classes_by_scheme = dict(manager.pool_classes_by_scheme, https=self.diagnostic_pool)
        return manager


def bind_connection_trace(session, event, deadline):
    trace = getattr(session, '_cornstellation_connection_trace', None)
    if trace is None:
        trace = ConnectionTrace()
        session.mount('https://integrate.api.nvidia.com/', DiagnosticAdapter(trace))
        session._cornstellation_connection_trace = trace
    trace.event, trace.deadline = event, deadline
    event['connection_reused'] = True
    return trace


TRANSIENT_OUTCOMES = frozenset({'CONNECT_TIMEOUT', 'READ_TIMEOUT', 'CONNECTION_ERROR',
                               'TIMEOUT', 'REQUEST_ERROR', 'HTTP_429', 'HTTP_500',
                               'HTTP_502', 'HTTP_503', 'HTTP_504'})


class RecoveryGuard:
    def __init__(self, rpm, failure_threshold, cooldown, clock=None, wait=None):
        self.rpm, self.failure_threshold, self.cooldown = rpm, failure_threshold, cooldown
        self.clock = clock or time.monotonic
        self.condition = Condition()
        self.wait = wait or self.condition.wait
        self.requests = deque()
        self.failures = 0
        self.open_until = 0
        self.probe_inflight = False

    def admit(self, deadline, event):
        rate_wait = circuit_wait = 0.
        with self.condition:
            while True:
                now = self.clock()
                while self.requests and self.requests[0] <= now - 60:
                    self.requests.popleft()
                event.update(rate_wait_ms=round(rate_wait * 1000, 3),
                             circuit_wait_ms=round(circuit_wait * 1000, 3))
                if now >= deadline - 2:
                    raise DataUnavailable('NVIDIA recovery wait TIME_BUDGET_EXHAUSTED; no request attempted')
                circuit_delay = max(0, self.open_until - now)
                if self.open_until and not circuit_delay and self.probe_inflight:
                    circuit_delay = min(.25, deadline - now - 2)
                rate_delay = max(0, self.requests[0] + 60 - now) if len(self.requests) >= self.rpm else 0
                delay = max(rate_delay, circuit_delay)
                if delay:
                    if delay >= deadline - now - 2:
                        event['outcome'] = 'RECOVERY_WAIT_EXCEEDS_BUDGET'
                        raise DataUnavailable('NVIDIA recovery wait exceeds remaining budget; no request attempted')
                    started = now
                    self.wait(delay)
                    elapsed = max(0, self.clock() - started)
                    if circuit_delay >= rate_delay:
                        circuit_wait += elapsed
                    else:
                        rate_wait += elapsed
                    continue
                event['circuit_half_open_probe'] = bool(self.open_until)
                if self.open_until:
                    self.probe_inflight = True
                self.requests.append(now)
                event['rate_window_attempts'] = len(self.requests)
                return

    def record(self, event):
        with self.condition:
            if event.get('outcome') in TRANSIENT_OUTCOMES:
                self.failures += 1
                if event.get('circuit_half_open_probe') or self.failures >= self.failure_threshold:
                    self.open_until = self.clock() + self.cooldown
                    event['circuit_opened_seconds'] = self.cooldown
            else:
                # A valid non-transient HTTP result proves this connection path works.
                self.failures, self.open_until = 0, 0
            if event.get('circuit_half_open_probe'):
                self.probe_inflight = False
            event['circuit_state_after'] = 'open' if self.open_until else 'closed'
            self.condition.notify_all()

    def cancel(self, event):
        """Release a half-open reservation when no HTTP invocation took place."""
        with self.condition:
            if event.get('circuit_half_open_probe'):
                self.probe_inflight = False
            self.condition.notify_all()


_guards = {}
_guard_lock = Lock()


def recovery_guard(settings):
    key = (settings.nvidia_requests_per_minute, settings.nvidia_circuit_failures,
           settings.nvidia_circuit_cooldown)
    with _guard_lock:
        if key not in _guards:
            _guards[key] = RecoveryGuard(*key)
        return _guards[key]


def reset_recovery_state():
    """Tests only: each test starts with an independent process-equivalent state."""
    with _guard_lock:
        _guards.clear()
