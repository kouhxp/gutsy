"""Compatibility with the JevBench `typesafe` adapter (POST /v1/systemone). The checks in
`adapter_parse` mirror jevbench/adapters/typesafe.py's answer handling."""
import json
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from gutsy_inference.engine import Engine
from gutsy_inference.registry import Registry
from gutsy_inference.schema import RequestError, parse_request
from gutsy_inference.server import make_handler

from .fakes import FakeBackend


def adapter_parse(ans, qtype, labels):
    """Mirror of the adapter: raises on anything it would reject."""
    assert isinstance(ans, dict) and ans.get("type") == qtype, "native answer type mismatch"
    if qtype == "noul":
        p = ans["noul"]
        assert not isinstance(p, bool) and isinstance(p, (int, float)) and 0 <= p <= 1
        return {"yes": p, "no": 1 - p}
    if qtype == "choice":
        assert ans.get("choice") in labels, "Invalid native choice"
    assert isinstance(ans.get("probabilities"), dict)
    return ans["probabilities"]


def test_score_questions():
    body = {"state": "Customer is furious and threatening legal action.",
            "questions": {"s": {"type": "score", "instructions": "How severe is this complaint?",
                                "criteria": ["minor", "moderate", "serious", "critical"]}}}
    _, (q,), _ = parse_request(body)
    assert q.qtype == "score" and q.keys == ["0", "1", "2", "3"] and q.options[2] == "serious"
    ans = Engine(FakeBackend()).decide(body)["answers"]["s"]
    probs = adapter_parse(ans, "score", q.keys)
    assert set(probs) == {"0", "1", "2", "3"} and abs(sum(probs.values()) - 1) < 1e-5
    assert ans["level"] == max(probs, key=probs.get)
    assert abs(ans["expected"] - sum(int(k) * v for k, v in probs.items())) < 1e-5


def test_object_levels_and_instructions():
    body = {"state": {"ticket": 1},
            "questions": {"s": {"type": "score",
                                "instructions": {"task": "rate urgency", "scale": "0-2"},
                                "criteria": [{"level": "low", "examples": ["typo"]}, "medium",
                                             ["high", "outage"]]}}}
    _, (q,), _ = parse_request(body)
    assert q.text == '{"task": "rate urgency", "scale": "0-2"}'
    assert q.options == ['{"level": "low", "examples": ["typo"]}', "medium", '["high", "outage"]']


def test_choice_list_criteria_and_errors():
    _, (q,), _ = parse_request({"state": "s", "questions": {"c": {
        "type": "choice", "instructions": "route", "criteria": ["billing", "tech", "sales"]}}})
    assert q.keys == ["billing", "tech", "sales"] and q.options == ["billing", "tech", "sales"]
    for crit in (["only"], ["a", "a"], [""], [1, 2]):
        try:
            parse_request({"state": "s", "questions": {"c": {"type": "choice", "instructions": "x",
                                                             "criteria": crit}}})
        except RequestError:
            continue
        raise AssertionError(crit)
    try:
        parse_request({"state": "s", "questions": {"c": {"type": "score", "instructions": "x",
                                                         "criteria": ["one"]}}})
    except RequestError:
        pass
    else:
        raise AssertionError("single-level score accepted")


def test_systemone_http_with_adapter_checks():
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "models.json"
        cfg.write_text(json.dumps({"default": "gutsy-0.8b", "models": {
            "gutsy-0.8b": {"gguf": "a"}, "gutsy-2b": {"gguf": "b"}}}))
        reg = Registry(cfg, backend_factory=lambda spec: FakeBackend())
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(reg, quiet=True))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            cases = [("noul", {"type": "noul", "instructions": "Is it urgent?"}, ["yes", "no"]),
                     ("choice", {"type": "choice", "instructions": "Route it",
                                 "criteria": {"billing": "money", "tech": "bugs"}}, ["billing", "tech"]),
                     ("score", {"type": "score", "instructions": "Severity",
                                "criteria": ["low", "high"]}, ["0", "1"])]
            for model, expect in (("jev-latest", "gutsy-0.8b"), ("gutsy-2b", "gutsy-2b")):
                for qtype, q, labels in cases:
                    body = {"state": "Server down since 9am.", "model": model, "questions": {"decision": q}}
                    req = urllib.request.Request(base + "/v1/systemone", json.dumps(body).encode(),
                                                 {"Content-Type": "application/json"})
                    with urllib.request.urlopen(req) as r:
                        parsed = json.loads(r.read())
                    assert parsed["model"] == expect           # unknown name -> default, reported
                    adapter_parse(parsed["answers"]["decision"], qtype, labels)
        finally:
            httpd.shutdown()
