"""Bounded, cached public-data reads. Secrets never enter cache files or errors."""
from hashlib import sha256
import ipaddress
import json
from pathlib import Path
import socket
import time
from uuid import uuid4
from urllib.parse import urlsplit, urljoin

import requests


class DataUnavailable(RuntimeError):
    pass


class PublicHTTP:
    def __init__(self, settings):
        self.settings = settings
        self.deadline = time.monotonic() + settings.max_run_seconds
        self.count = 0
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'Cornstellation-Investigation/1.0 (public agricultural research)'

    def remaining(self):
        return max(0, self.deadline - time.monotonic())

    @staticmethod
    def public_url(url):
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.port not in (None, 443):
            raise DataUnavailable('Only public HTTPS sources are allowed')
        try:
            addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
        except OSError:
            raise DataUnavailable('Public source DNS unavailable') from None
        if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
            raise DataUnavailable('Private/local source addresses are not allowed')

    def get(self, url, params=None, *, max_bytes=8_000_000, cache=True):
        params = params or {}
        public_params = {k: v for k, v in params.items() if k.lower() not in ('key', 'api_key', 'token')}
        cache_id = sha256(json.dumps([url, public_params], sort_keys=True).encode()).hexdigest()
        target = self.settings.cache_dir / (cache_id + '.bin')
        if cache and target.exists() and time.time() - target.stat().st_mtime < self.settings.cache_days * 86400:
            return target.read_bytes(), True
        for attempt in range(2):
            current_url, current_params = url, params
            try:
                for _ in range(4):
                    if self.count >= self.settings.max_http_requests or self.remaining() <= 0:
                        raise DataUnavailable('Public-data request/time budget exhausted')
                    self.public_url(current_url)
                    self.count += 1
                    with self.session.get(current_url, params=current_params, timeout=min(self.settings.http_timeout, self.remaining()),
                                          allow_redirects=False, stream=True) as response:
                        if response.is_redirect:
                            next_url = urljoin(current_url, response.headers.get('Location', ''))
                            # Do not forward credentials to a different origin.
                            if urlsplit(next_url).netloc != urlsplit(current_url).netloc and any(k in params for k in ('key', 'api_key')):
                                raise DataUnavailable('Credential-bearing redirect rejected')
                            current_url, current_params = next_url, None
                            continue
                        if response.status_code != 200:
                            raise DataUnavailable(f'Public source HTTP {response.status_code}')
                        chunks, length = [], 0
                        for chunk in response.iter_content(64 * 1024):
                            length += len(chunk)
                            if length > max_bytes or self.remaining() <= 0:
                                raise DataUnavailable('Public response size/time budget exceeded')
                            chunks.append(chunk)
                        data = b''.join(chunks)
                        if cache:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            temporary = target.with_name(target.name+'.'+uuid4().hex+'.tmp')
                            temporary.write_bytes(data)
                            temporary.replace(target)
                        return data, False
                raise DataUnavailable('Too many public-source redirects')
            except requests.RequestException:
                if attempt:
                    raise DataUnavailable('Public source connection unavailable') from None
        raise DataUnavailable('Public source unavailable')

    def json(self, url, params=None, **kwargs):
        raw, cached = self.get(url, params, **kwargs)
        try:
            return json.loads(raw), cached
        except (ValueError, UnicodeDecodeError):
            raise DataUnavailable('Public source did not return valid JSON') from None
