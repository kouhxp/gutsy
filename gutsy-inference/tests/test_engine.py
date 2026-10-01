import numpy as np

from gutsy_inference.engine import Calibration, Engine
from gutsy_inference.schema import RequestError, parse_request

from .fakes import FakeBackend

BODY = {"state": "The shipment left on Monday and arrived Thursday.",
        "questions": {
            "late": {"type": "noul", "instructions": "Was it late?",
                     "criteria": {"true": "arrived after Wednesday", "false": "arrived by Wednesday"}},
            "day": {"type": "choice", "instructions": "Arrival day?",
                    "criteria": {"Mon": "Monday", "Thu": "Thursday", "Fri": ""}},
            "left": {"type": "noul", "instructions": "Did it leave Monday?"}}}


def probs_of(resp):
    out = []
    for a in resp["answers"].values():
        out += [a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values())
    return np.array(out)


def test_cached_equals_full_and_saves_work():
    cached = Engine(FakeBackend())
    full = Engine(FakeBackend(), use_cache=False)
    r1, r2 = cached.decide(BODY), full.decide(BODY)
    assert np.allclose(probs_of(r1), probs_of(r2))
    u1, u2 = r1["usage"], r2["usage"]
    assert u1["state_cached"] and not u1["state_cache_hit"] and not u2["state_cached"]
    # cold cached: state once + each question; full: state re-encoded per question
    assert u1["evaluated_tokens"] == u1["prompt_tokens"]
    assert u2["evaluated_tokens"] == u2["prompt_tokens"] + 2 * u2["state_tokens"]


def test_warm_request_skips_state_and_is_deterministic():
    e = Engine(FakeBackend())
    r1 = e.decide(BODY)
    r2 = e.decide(BODY)
    assert r2["usage"]["state_cache_hit"]
    assert r2["usage"]["evaluated_tokens"] == r2["usage"]["prompt_tokens"] - r2["usage"]["state_tokens"]
    assert r1["answers"] == r2["answers"]


def test_misaligned_tokenization_falls_back_correctly():
    e = Engine(FakeBackend(misaligned=True))
    ref = Engine(FakeBackend(misaligned=True), use_cache=False)
    r = e.decide(BODY)
    assert not r["usage"]["state_cached"]
    assert r["answers"] == ref.decide(BODY)["answers"]


def test_broken_restore_disables_cache_and_stays_correct():
    e = Engine(FakeBackend(broken_restore=True))
    ref = Engine(FakeBackend(), use_cache=False)
    r = e.decide(BODY)
    assert not e.cache_enabled and not r["usage"]["state_cached"]
    assert r["answers"] == ref.decide(BODY)["answers"]


def test_self_check():
    e = Engine(FakeBackend())
    assert e.self_check().startswith("caching OK") and e.cache_enabled
    e2 = Engine(FakeBackend(broken_restore=True))
    assert e2.self_check().startswith("caching disabled") and not e2.cache_enabled
    e3 = Engine(FakeBackend(misaligned=True))
    assert e3.self_check().startswith("caching disabled")


def test_context_limit():
    e = Engine(FakeBackend(n_ctx=100))
    try:
        e.decide(BODY)
    except RequestError as err:
        assert "context window" in str(err)
    else:
        raise AssertionError("should reject")


def test_temperature_applied():
    hot = Engine(FakeBackend(), Calibration({"choice": 5.0}))
    cold = Engine(FakeBackend())
    ph = hot.decide(BODY)["answers"]["day"]["probabilities"]
    pc = cold.decide(BODY)["answers"]["day"]["probabilities"]
    assert max(ph.values()) < max(pc.values())                    # flatter
    assert max(ph, key=ph.get) == max(pc, key=pc.get)             # same ranking
    assert abs(sum(ph.values()) - 1) < 1e-5


def test_cache_lru_bounds():
    e = Engine(FakeBackend(), cache_entries=2)
    for i in range(4):
        e.decide({**BODY, "state": f"state {i}"})
    assert len(e.cache.items) == 2


def test_schema_mapping_and_errors():
    state, qs, _ = parse_request(BODY)
    late, day, left = qs
    assert late.options == ["Yes", "No"] and "Yes means: arrived after Wednesday" in late.text
    assert day.options == ["Mon: Monday", "Thu: Thursday", "Fri"] and day.keys == ["Mon", "Thu", "Fri"]
    assert left.text == "Did it leave Monday?"
    bad = [{}, {"state": 1, "questions": {"a": {}}}, {"state": "s", "questions": {}},
           {"state": "s", "questions": {"a": {"type": "score", "instructions": "x"}}},
           {"state": "s", "questions": {"a": {"type": "choice", "instructions": "x",
                                              "criteria": {"only": ""}}}},
           {"state": "s", "questions": {"a": {"type": "choice", "instructions": "x",
                                              "criteria": {str(i): "" for i in range(256)}}}},
           {"state": "s", "questions": {"a": {"type": "score", "instructions": "x",
                                              "criteria": [str(i) for i in range(17)]}}},
           {"state": "s", "questions": {"a": {"type": "noul", "instructions": " "}}}]
    for b in bad:
        try:
            parse_request(b)
        except RequestError:
            continue
        raise AssertionError(f"should reject {b}")


def test_response_shape_matches_jev_style():
    r = Engine(FakeBackend()).decide(BODY)
    assert set(r["answers"]) == {"late", "day", "left"}
    assert r["answers"]["late"]["type"] == "noul" and 0 <= r["answers"]["late"]["noul"] <= 1
    d = r["answers"]["day"]
    assert d["type"] == "choice" and set(d["probabilities"]) == {"Mon", "Thu", "Fri"}
    assert d["choice"] == max(d["probabilities"], key=d["probabilities"].get)
    pr = sorted(d["probabilities"].values())
    n = len(pr)
    assert abs(d["confidence"] - (n * pr[-1] - 1) / (n - 1)) < 1e-5
    assert abs(d["margin"] - (pr[-1] - pr[-2])) < 1e-5
    assert "confidence" not in r["answers"]["late"] and "margin" not in r["answers"]["late"]   # yes/no


def test_long_lists_shortlisted_or_refused():
    four = {"state": "s", "questions": {"c": {"type": "choice", "instructions": "?",
                                              "criteria": {k: "" for k in "abcd"}}}}
    r = Engine(FakeBackend(), max_options=3).decide(four)          # 3 per call: shortlisted
    assert r["answers"]["c"]["shortlist"]["rounds"] == 2 and abs(sum(r["answers"]["c"]["probabilities"].values()) - 1) < 1e-5
    try:
        Engine(FakeBackend(), max_options=3, shortlist=False).decide(four)
    except RequestError as err:
        assert "at most 3 per call" in str(err)
    else:
        raise AssertionError("4 options accepted by a 3-option model with shortlisting off")
    score = {"state": "s", "questions": {"s": {"type": "score", "instructions": "?", "criteria": list("abcd")}}}
    try:
        Engine(FakeBackend(), max_options=3).decide(score)          # score levels are never shortlisted
    except RequestError as err:
        assert "for score questions" in str(err)
    else:
        raise AssertionError("4 score levels accepted by a 3-option model")
    assert "shortlist" not in Engine(FakeBackend()).decide(four)["answers"]["c"]   # fits in one call


def test_reject_slot_answers():
    e = Engine(FakeBackend(), reject_slot=True)
    r = e.decide(BODY)
    day = r["answers"]["day"]
    assert 0 <= day["reject"] <= 1 and abs(sum(day["probabilities"].values()) - 1) < 1e-5
    assert "reject" not in r["answers"]["late"]                       # yes/no questions: no slot
    plain = Engine(FakeBackend()).decide(BODY)["answers"]["day"]
    assert "reject" not in plain and plain["probabilities"] != day["probabilities"]   # prompt differs
    assert e.self_check().startswith("caching OK")                    # cached == full with the slot
    assert Engine(FakeBackend(), reject_slot=True).decide(BODY)["answers"] == r["answers"]   # deterministic


def test_reject_conditioning_is_temperature_consistent():
    """Scaling all columns by one temperature and conditioning on 'not reject' equals scaling
    the option logits alone."""
    import numpy as np
    from gutsy_inference.engine import Calibration
    cal = Calibration({"choice": 1.7})
    z = np.array([2.0, 0.5, -1.0, 1.2])              # three options + reject
    joint = cal.probs("choice", z)
    cond = joint[:3] / (1 - joint[3])
    assert np.allclose(cond, cal.probs("choice", z[:3]))


def test_confidence_matches_published_shape():
    """The worked examples from the published formula: same confidence, different margins."""
    import numpy as np
    from gutsy_inference.schema import concentration, top_margin
    cases = [([0.90, 0.06, 0.04], 0.85, 0.84), ([0.55, 0.44, 0.01], 0.325, 0.11),
             ([0.55, 0.225, 0.225], 0.325, 0.325), ([0.40, 0.33, 0.27], 0.10, 0.07),
             ([1 / 3] * 3, 0.0, 0.0)]
    for probs, conf, margin in cases:
        assert abs(concentration(np.array(probs)) - conf) < 1e-9
        assert abs(top_margin(np.array(probs)) - margin) < 1e-9
    peak_for_06 = {3: 0.7333, 10: 0.64, 20: 0.62, 255: 0.6016}   # top probability giving 0.6
    for n, peak in peak_for_06.items():
        rest = (1 - peak) / (n - 1)
        assert abs(concentration(np.array([peak] + [rest] * (n - 1))) - 0.6) < 1e-3
