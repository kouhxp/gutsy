"""HTTP server (stdlib only). Endpoints:
  POST /api/alpha/decisions   Jev/OpenRouter-style path, so existing clients only change the base URL
  POST /v1/decisions          same, shorter alias
  POST /v1/systemone          TypeSafe-style path (JevBench `typesafe` adapter); unknown model
                              names fall back to the default model
  GET  /v1/models             available models
  GET  /health                status, self-check results
Inference is serialized per model (one llama.cpp context each); requests queue on a lock.
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .schema import RequestError

MAX_BODY = 8 * 2**20
DECISION_PATHS = {"/api/alpha/decisions", "/v1/decisions", "/v1/systemone"}
# TypeSafe-style clients (e.g. the JevBench `typesafe` adapter) send their own default model name
# ("jev-latest"); on this path an unknown model name falls back to the default model. The
# response's "model" field always says which model actually answered.
LENIENT_MODEL_PATHS = {"/v1/systemone"}


def make_handler(registry, api_key=None, quiet=False):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"gutsy-inference/{__version__}"

        def _send(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code, msg, etype="invalid_request_error"):
            self._send(code, {"error": {"message": msg, "type": etype, "code": code}})

        def log_message(self, fmt, *args):
            if not quiet:
                sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.address_string()} {fmt % args}\n")

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"status": "ok", "version": __version__,
                                        "models": registry.status()})
            if self.path == "/v1/models":
                return self._send(200, {"data": [{"id": n, "default": n == registry.default}
                                                 for n in registry.names()]})
            self._error(404, f"no route for GET {self.path}")

        def do_POST(self):
            if self.path not in DECISION_PATHS:
                return self._error(404, f"no route for POST {self.path}")
            if api_key and self.headers.get("Authorization") != f"Bearer {api_key}":
                return self._error(401, "missing or invalid bearer token", "authentication_error")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._error(400, "bad Content-Length")
            if length > MAX_BODY:
                return self._error(413, f"request body over {MAX_BODY // 2**20} MB")
            try:
                body = json.loads(self.rfile.read(length) or b"null")
            except json.JSONDecodeError as e:
                return self._error(400, f"invalid JSON: {e}")
            name = body.get("model") if isinstance(body, dict) else None
            if self.path in LENIENT_MODEL_PATHS and name not in registry.models:
                name = None
            try:
                engine = registry.get(name)
            except KeyError:
                return self._error(400, f"unknown model {name!r}; available: {registry.names()}")
            except Exception as e:
                return self._error(500, f"failed to load model: {e}", "server_error")
            try:
                return self._send(200, engine.decide(body))
            except RequestError as e:
                return self._error(400, str(e))
            except Exception as e:
                return self._error(500, f"{type(e).__name__}: {e}", "server_error")

    return Handler


def serve(registry, host="127.0.0.1", port=8765, api_key=None, preload=True, quiet=False):
    if preload:
        for n in registry.names():
            e = registry.get(n)
            print(f"loaded {n}: {e.self_check_result}; temperatures {e.calibration.temperatures}",
                  flush=True)
    httpd = ThreadingHTTPServer((host, port), make_handler(registry, api_key, quiet))
    print(f"gutsy-inference {__version__} listening on http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
