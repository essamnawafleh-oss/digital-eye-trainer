"""Strict offline mode.

``offline()`` is used around every step that touches documents or training data
(import, labelling server, training, evaluation, export). Inside it:

* environment flags stop Hugging Face / PaddleX from contacting model hubs;
* any socket connection to a non-loopback address, and any DNS lookup of a
  non-local name, raises ``OfflineViolation`` immediately.

Loopback (127.0.0.1 / ::1) stays allowed because the labelling page is served on
localhost, EXCEPT the ports of any configured proxy (HTTP(S)_PROXY / ALL_PROXY): a local
proxy would otherwise be a way out to the internet. Proxy variables are also removed
from the environment while the guard is active. The guard covers this Python process only; for a system-level check on
Linux, ``scripts/verify_offline.sh`` runs a command in a network namespace with no
network interfaces. The only steps that use the internet are ``install`` and
``download-models``, and they never read your documents.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import threading
from contextlib import contextmanager

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    # paddlex/utils/flags.py (3.7.2): skip the model-hoster connectivity check.
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
}


PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


class OfflineViolation(RuntimeError):
    pass


def apply_offline_env() -> None:
    os.environ.update(OFFLINE_ENV)


_blocked_ports: set[int] = set()  # loopback ports of configured proxies


def _proxy_ports() -> set[int]:
    from urllib.parse import urlparse

    ports = set()
    for var in PROXY_VARS:
        value = os.environ.get(var)
        if value:
            try:
                u = urlparse(value if "://" in value else "http://" + value)
                ports.add(u.port or (443 if u.scheme == "https" else 80))
            except ValueError:
                pass
    return ports


def _is_local(address) -> bool:
    if isinstance(address, (str, bytes)):  # AF_UNIX socket path
        return True
    host = address[0] if isinstance(address, tuple) and address else address
    port = address[1] if isinstance(address, tuple) and len(address) > 1 else None
    if port in _blocked_ports:
        return False
    if host in ("localhost", "", None):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a host name would need DNS: treat as network


_lock = threading.RLock()
_depth = 0
_saved: dict = {}


def _guard_connect(orig):
    def connect(self, address, *args, **kwargs):
        if not _is_local(address):
            raise OfflineViolation(f"blocked a network connection while offline: {address!r}")
        return orig(self, address, *args, **kwargs)
    return connect


def _guard_getaddrinfo(orig):
    def getaddrinfo(host, *args, **kwargs):
        if not _is_local((host,)):
            raise OfflineViolation(f"blocked a DNS lookup while offline: {host!r}")
        return orig(host, *args, **kwargs)
    return getaddrinfo


def _guard_create_connection(orig):
    def create_connection(address, *args, **kwargs):
        if not _is_local(address):
            raise OfflineViolation(f"blocked a network connection while offline: {address!r}")
        return orig(address, *args, **kwargs)
    return create_connection


@contextmanager
def offline():
    global _depth
    apply_offline_env()
    with _lock:
        if _depth == 0:
            _blocked_ports.clear()
            _blocked_ports.update(_proxy_ports())
            _saved["proxy_env"] = {k: os.environ.pop(k) for k in PROXY_VARS if k in os.environ}
            _saved.update(connect=socket.socket.connect, connect_ex=socket.socket.connect_ex,
                          getaddrinfo=socket.getaddrinfo, create_connection=socket.create_connection,
                          sendto=socket.socket.sendto)
            socket.socket.connect = _guard_connect(_saved["connect"])
            socket.socket.connect_ex = _guard_connect(_saved["connect_ex"])
            socket.getaddrinfo = _guard_getaddrinfo(_saved["getaddrinfo"])
            socket.create_connection = _guard_create_connection(_saved["create_connection"])

            orig_sendto = _saved["sendto"]

            def sendto(self, data, *args):
                if not _is_local(args[-1]):
                    raise OfflineViolation(f"blocked a network packet while offline: {args[-1]!r}")
                return orig_sendto(self, data, *args)
            socket.socket.sendto = sendto
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0:
                socket.socket.connect = _saved["connect"]
                socket.socket.connect_ex = _saved["connect_ex"]
                socket.getaddrinfo = _saved["getaddrinfo"]
                socket.create_connection = _saved["create_connection"]
                socket.socket.sendto = _saved["sendto"]
                os.environ.update(_saved.pop("proxy_env", {}))
                _blocked_ports.clear()


def is_active() -> bool:
    return _depth > 0
