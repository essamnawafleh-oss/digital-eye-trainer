"""Local labelling server: localhost only, token required, labels saved."""

import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest
from PIL import Image

from eye_trainer.store import Store
from eye_trainer.ui.server import Jobs, make_handler


@pytest.fixture()
def server(empty_workspace):
    ws = empty_workspace
    store = Store(ws.root)
    (ws.root / "crops" / "d").mkdir(parents=True)
    Image.new("RGB", (120, 30), "white").save(ws.root / "crops" / "d" / "a.png")
    store.add_doc("d", "x.pdf", "pdf", 1)
    store.add_lines([{"doc_id": "d", "page": 1, "idx": 0, "crop": "crops/d/a.png", "poly": None,
                      "suggestion": "رقم القطعه ١٢", "score": 0.7, "alt_text": "12", "alt_score": 0.9, "priority": 1}])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), lambda *a: None)
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = make_handler(ws, store, Jobs(), "TOKEN", port)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield port, store
    httpd.shutdown()
    store.close()


def _req(port, method, path, body=None, headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    h = {"Host": host or f"127.0.0.1:{port}", **(headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    if data:
        h["Content-Type"] = "application/json"
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    return r.status, r.read()


def test_page_and_token(server):
    port, _ = server
    status, html = _req(port, "GET", "/")
    assert status == 200 and b'content="TOKEN"' in html and "dir=\"rtl\"".encode() in html
    assert _req(port, "GET", "/api/status")[0] == 403
    assert _req(port, "GET", "/api/status", headers={"X-Token": "wrong"})[0] == 403
    assert _req(port, "GET", "/api/status", headers={"X-Token": "TOKEN"})[0] == 200


def test_foreign_host_is_refused(server):
    port, _ = server
    assert _req(port, "GET", "/", host="evil.example:80")[0] == 403
    assert _req(port, "POST", "/api/label", {"id": 1, "status": "done", "label": "x"},
                {"X-Token": "TOKEN"}, host="evil.example")[0] == 403


def test_label_and_crop(server):
    port, store = server
    status, body = _req(port, "GET", "/api/next?n=1", headers={"X-Token": "TOKEN"})
    line = json.loads(body)[0]
    assert line["direction"] == "rtl" and line["alt_text"] == "12"
    status, img = _req(port, "GET", f"/crop/{line['id']}?t=TOKEN")
    assert status == 200 and img[:4] == b"\x89PNG"
    status, _ = _req(port, "POST", "/api/label", {"id": line["id"], "status": "done", "label": "رقم القطعة ١٢"},
                     {"X-Token": "TOKEN"})
    assert status == 200
    assert store.get_line(line["id"])["label"] == "رقم القطعة ١٢"
    assert _req(port, "POST", "/api/label", {"id": line["id"], "status": "bogus"}, {"X-Token": "TOKEN"})[0] == 400


def test_static_path_traversal_blocked(server):
    port, _ = server
    assert _req(port, "GET", "/static/../server.py")[0] == 404
