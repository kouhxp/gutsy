import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from gutsy_inference.registry import Registry
from gutsy_inference.server import make_handler

from .fakes import FakeBackend
from .test_engine import BODY


def _start(tmp, api_key=None):
    cfg = tmp / "models.json"
    (tmp / "cal.json").write_text(json.dumps({"temperatures": {"yes_no": 1.3, "choice": 1.1}}))
    cfg.write_text(json.dumps({"default": "gutsy-0.8b", "models": {
        "gutsy-0.8b": {"gguf": "a.gguf", "calibration": "cal.json"},
        "gutsy-2b": {"gguf": "b.gguf", "calibration": "cal.json"}}}))
    reg = Registry(cfg, backend_factory=lambda spec: FakeBackend())
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(reg, api_key, quiet=True))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _call(url, body=None, key=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_end_to_end(tmp_path):
    httpd, base = _start(tmp_path)
    try:
        code, r = _call(base + "/api/alpha/decisions", BODY)
        assert code == 200 and r["model"] == "gutsy-0.8b" and set(r["answers"]) == set(BODY["questions"])
        code, r = _call(base + "/v1/decisions", {**BODY, "model": "gutsy-2b"})
        assert code == 200 and r["model"] == "gutsy-2b"
        code, r = _call(base + "/v1/decisions", {**BODY, "model": "nope"})
        assert code == 400 and "unknown model" in r["error"]["message"]
        code, r = _call(base + "/v1/decisions", {"state": "s", "questions": {}})
        assert code == 400
        code, r = _call(base + "/health")
        assert code == 200 and r["models"]["gutsy-0.8b"]["self_check"].startswith("caching OK")
        assert r["models"]["gutsy-0.8b"]["temperatures"]["yes_no"] == 1.3
        code, r = _call(base + "/v1/models")
        assert [m["id"] for m in r["data"]] == ["gutsy-0.8b", "gutsy-2b"]
    finally:
        httpd.shutdown()


def test_http_auth(tmp_path):
    httpd, base = _start(tmp_path, api_key="secret")
    try:
        assert _call(base + "/v1/decisions", BODY)[0] == 401
        assert _call(base + "/v1/decisions", BODY, key="secret")[0] == 200
    finally:
        httpd.shutdown()
