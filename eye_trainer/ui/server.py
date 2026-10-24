"""Local web page for importing, labelling, training, evaluating and exporting.

* Python standard library only (http.server); no internet, no CDN, no external files.
* Listens on 127.0.0.1 only. Requests with any other Host header are refused (DNS
  rebinding), and every API call needs a random per-session token that only the page
  itself knows (so other web sites open in your browser cannot read or change labels).
* Long jobs (import, training, ...) run one at a time in a background thread, inside the
  offline guard.
"""

from __future__ import annotations

import json
import mimetypes
import secrets
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import pipeline
from .. import textutil as tu
from ..messages import say
from ..offline import offline
from ..store import Store
from ..workspace import Workspace

STATIC = Path(__file__).resolve().parent / "static"


class Jobs:
    """One background job at a time, with a short log for the page."""

    def __init__(self):
        self.lock = threading.Lock()
        self.name: str | None = None
        self.status = "idle"  # idle | running | finished | failed | stopping
        self.log: list[str] = []
        self.result = None
        self.stop = threading.Event()

    def add_log(self, msg: str) -> None:
        self.log = (self.log + [f"{time.strftime('%H:%M:%S')}  {msg}"])[-200:]

    def start(self, name: str, fn) -> bool:
        with self.lock:
            if self.status in ("running", "stopping"):
                return False
            self.name, self.status, self.log, self.result = name, "running", [], None
            self.stop.clear()

        def run():
            try:
                self.result = fn(self.add_log, self.stop.is_set)
                self.status = "finished"
            except Exception as exc:
                self.add_log(f"ERROR: {exc}")
                self.add_log(traceback.format_exc(limit=3))
                self.status = "failed"

        threading.Thread(target=run, daemon=True).start()
        return True

    def snapshot(self) -> dict:
        return {"name": self.name, "status": self.status, "log": self.log[-60:]}


def make_handler(ws: Workspace, store: Store, jobs: Jobs, token: str, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "eye-trainer"

        def log_message(self, *args):  # no request logging (URLs can contain line ids only, but keep quiet)
            pass

        # ------------------------------------------------------------ helpers
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; "
                             "connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            return self.headers.get("Host", "") in allowed_hosts

        def _token_ok(self) -> bool:
            sent = self.headers.get("X-Token", "")
            if not sent and self.command == "GET":  # <img> tags cannot send headers
                sent = parse_qs(urlparse(self.path).query).get("t", [""])[0]
            return secrets.compare_digest(sent, token)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        # ------------------------------------------------------------ GET
        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"forbidden host", "text/plain")
            url = urlparse(self.path)
            path = url.path
            if path in ("/", "/index.html"):
                html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", token)
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            if path.startswith("/static/"):
                f = (STATIC / path[len("/static/"):]).resolve()
                if STATIC not in f.parents or not f.is_file():
                    return self._send(404, b"not found", "text/plain")
                return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
            if not self._token_ok():
                return self._send(403, b"missing token", "text/plain")
            q = parse_qs(url.query)
            if path.startswith("/crop/"):
                line = store.get_line(int(path.rsplit("/", 1)[-1]))
                if not line:
                    return self._send(404, b"not found", "text/plain")
                f = (ws.root / line["crop"]).resolve()
                if ws.root.resolve() not in f.parents or not f.is_file():
                    return self._send(404, b"not found", "text/plain")
                return self._send(200, f.read_bytes(), "image/png")
            if path == "/api/status":
                return self._json({"progress": store.progress(), "job": jobs.snapshot(), "training": _training(ws),
                                   "runs": _runs(ws), "exports": _exports(ws)})
            if path == "/api/next":
                status = q.get("status", ["todo"])[0]
                return self._json([_line_view(x) for x in store.next_lines(int(q.get("n", ["1"])[0]), status)])
            if path.startswith("/api/line/"):
                line = store.get_line(int(path.rsplit("/", 1)[-1]))
                return self._json(_line_view(line) if line else None, 200 if line else 404)
            if path == "/api/environment":
                return self._json(pipeline.check_environment(ws))
            if path.startswith("/api/report/"):
                run = path.rsplit("/", 1)[-1]
                out = {}
                for kind in ("real", "synthetic"):
                    f = ws.report_dir(run) / f"evaluation_{kind}.md"
                    if f.is_file() and run == Path(run).name:
                        out[kind] = f.read_text(encoding="utf-8")
                return self._json(out)
            return self._send(404, b"not found", "text/plain")

        # ------------------------------------------------------------ POST
        def do_POST(self):
            if not self._host_ok() or not self._token_ok():
                return self._send(403, b"forbidden", "text/plain")
            path = urlparse(self.path).path
            try:
                body = self._body()
                if path == "/api/label":
                    line = store.save_label(int(body["id"]), body["status"], body.get("label"),
                                            body.get("direction", "auto"), bool(body.get("persian_to_arabic", True)))
                    return self._json(_line_view(line))
                if path == "/api/job":
                    return self._json({"started": _start_job(ws, store, jobs, body)})
                if path == "/api/job/stop":
                    jobs.stop.set()
                    jobs.status = "stopping" if jobs.status == "running" else jobs.status
                    return self._json({"ok": True})
            except (KeyError, ValueError) as exc:
                return self._json({"error": str(exc)}, 400)
            return self._send(404, b"not found", "text/plain")

    return Handler


def _line_view(line: dict) -> dict:
    label = line["label"] if line["label"] is not None else (line["suggestion"] or "")
    direction = line.get("direction") or "auto"
    if direction == "auto":
        direction = "rtl" if tu.has_arabic(label) else "ltr"
    return {"id": line["id"], "suggestion": line["suggestion"], "score": line["score"], "alt_text": line["alt_text"],
            "alt_score": line["alt_score"], "label": label, "status": line["status"], "direction": direction,
            "priority": line["priority"], "page": line["page"]}


def _training(ws: Workspace) -> dict:
    out = {}
    for run in sorted((ws.root / "runs").glob("*/progress.json")):
        try:
            out[run.parent.name] = json.loads(run.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return out


def _runs(ws: Workspace) -> list[str]:
    return sorted(p.parent.parent.name for p in (ws.root / "runs").glob("*/checkpoint/best.pdparams"))


def _exports(ws: Workspace) -> list[str]:
    return sorted(str(p.parent) for p in (ws.root / "exports").glob("*/manifest.json"))


def _clean_run_name(name: str) -> str:
    name = "".join(c for c in (name or "run1") if c.isalnum() or c in "-_")[:40]
    return name or "run1"


def _start_job(ws: Workspace, store: Store, jobs: Jobs, body: dict) -> bool:
    kind = body.get("job")
    run = _clean_run_name(body.get("run", "run1"))
    if kind == "import":
        folder = Path(body.get("folder") or ws.root / "inbox").expanduser()
        if not folder.is_dir():
            raise ValueError(say("المجلد غير موجود", "folder not found"))

        def fn(log, stop):
            from ..prep import Importer

            return Importer(ws).import_folder(folder, store, log=log)
    elif kind == "synth":
        n = int(body.get("lines", 6000))

        def fn(log, stop):
            return pipeline.make_synthetic(ws, {"train": (n, 1101), "val": (max(100, n // 20), 2202),
                                                "test": (max(300, n // 6), 3303)}, log=log)
    elif kind == "train":
        steps, bs = int(body.get("steps", 3000)), int(body.get("batch_size", 16))
        use_syn = bool(body.get("use_synthetic", True))

        def fn(log, stop):
            return pipeline.run_training(ws, run, steps=steps, batch_size=bs, use_synthetic=use_syn,
                                         log=log, stop_flag=stop)
    elif kind == "evaluate":
        def fn(log, stop):
            res = pipeline.run_evaluation(ws, run, log=log)
            return {k: v["verdict"] for k, v in res.items()}
    elif kind == "export":
        def fn(log, stop):
            return str(pipeline.run_export(ws, run, log=log))
    else:
        raise ValueError(f"unknown job {kind!r}")
    return jobs.start(f"{kind}:{run}" if kind in ("train", "evaluate", "export") else kind, fn)


def serve(ws: Workspace, port: int = 8765, open_browser: bool = True) -> None:
    token = secrets.token_urlsafe(24)
    store = Store(ws.root)
    jobs = Jobs()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(ws, store, jobs, token, port))
    url = f"http://127.0.0.1:{port}/"
    print(say(f"الصفحة تعمل على جهازك فقط: {url}\nأغلق هذه النافذة لإيقافها.",
              f"Page running on this computer only: {url}  (close this window to stop)"))
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    with offline():  # loopback stays allowed; everything else is blocked
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
            store.close()
