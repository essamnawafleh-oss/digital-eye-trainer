"""Offline mode: the in-process guard, and a system-level check in a network namespace."""

import os
import shutil
import socket
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path

import pytest

from eye_trainer.offline import OFFLINE_ENV, OfflineViolation, is_active, offline

from .conftest import requires_models

ROOT = Path(__file__).resolve().parents[1]


def test_guard_blocks_connections_and_dns():
    with offline():
        assert is_active()
        with pytest.raises(OfflineViolation):
            socket.create_connection(("93.184.216.34", 443), timeout=1)
        with pytest.raises(OfflineViolation):
            socket.getaddrinfo("huggingface.co", 443)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        with pytest.raises(OfflineViolation):
            s.sendto(b"x", ("8.8.8.8", 53))
        s.close()
        with pytest.raises(Exception) as exc:
            urllib.request.urlopen("https://huggingface.co", timeout=2)
        assert "offline" in str(exc.value).lower()
        for k, v in OFFLINE_ENV.items():
            assert os.environ[k] == v
    assert not is_active()


def test_guard_blocks_a_local_proxy(monkeypatch):
    """A proxy on localhost would be a way out: its port is blocked and the variables removed."""
    proxy = socket.socket()
    proxy.bind(("127.0.0.1", 0))
    proxy.listen(1)
    port = proxy.getsockname()[1]
    monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{port}")
    with offline():
        assert "HTTPS_PROXY" not in os.environ
        with pytest.raises(OfflineViolation):
            socket.create_connection(("127.0.0.1", port), timeout=1)
    assert os.environ["HTTPS_PROXY"] == f"http://127.0.0.1:{port}"
    proxy.close()


def test_guard_allows_loopback_and_restores():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    with offline():
        with offline():  # nested use is fine
            c = socket.create_connection(srv.getsockname(), timeout=2)
            c.close()
    srv.close()
    assert socket.socket.connect.__name__ == "connect"
    assert socket.getaddrinfo.__module__ == "socket"


def _unshare_ok() -> bool:
    return sys.platform.startswith("linux") and shutil.which("unshare") is not None and \
        subprocess.run(["unshare", "-rn", "true"], capture_output=True).returncode == 0


@requires_models
@pytest.mark.skipif(not _unshare_ok(), reason="needs Linux with unprivileged network namespaces (unshare -rn)")
def test_full_pipeline_runs_without_any_network(tmp_path, base_models):
    """System-level proof: synth -> train 2 steps -> evaluate -> export -> load in PaddleOCR,
    in a network namespace with no interfaces (no process could reach the network)."""
    script = textwrap.dedent(f"""
        import json, shutil, socket, sys
        sys.path.insert(0, {str(ROOT)!r})
        try:
            socket.create_connection(("1.1.1.1", 443), timeout=2)
            print("NETWORK REACHABLE"); sys.exit(5)
        except OSError:
            pass
        from eye_trainer.workspace import Workspace
        from eye_trainer import pipeline
        ws = Workspace.default({str(tmp_path / 'ws')!r})
        for m in ("PP-OCRv5_mobile_det", "arabic_PP-OCRv5_mobile_rec", "en_PP-OCRv5_mobile_rec"):
            shutil.copytree({str(base_models)!r} + "/" + m, ws.model_dir(m), ignore=shutil.ignore_patterns(".cache"))
        pipeline.make_synthetic(ws, {{"train": (24, 1), "val": (8, 2), "test": (12, 3)}}, log=lambda m: None)
        pipeline.run_training(ws, "t", steps=2, batch_size=4, use_real=False, log=lambda m: None)
        res = pipeline.run_evaluation(ws, "t", log=lambda m: None)
        out = pipeline.run_export(ws, "t", force=not res["synthetic"]["verdict"]["passed"], log=lambda m: None)
        print("OFFLINE PIPELINE OK", out)
    """)
    r = subprocess.run(["unshare", "-rn", sys.executable, "-c", script], capture_output=True, text=True, timeout=900)
    assert "OFFLINE PIPELINE OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
