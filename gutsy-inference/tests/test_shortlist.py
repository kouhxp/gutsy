"""Shortlisting (choice lists longer than one model call, up to 255 options)."""
import numpy as np

from gutsy_inference import shortlist
from gutsy_inference.engine import Engine

from .fakes import FakeBackend


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def test_exact_for_consistent_preferences():
    """If the model's preferences are consistent (a fixed score per option, softmax over whatever
    is shown), shortlisting returns exactly the single-pass distribution over the whole list."""
    rng = np.random.default_rng(0)
    for n, size in ((17, 16), (40, 16), (100, 16), (254, 16), (255, 5), (30, 3)):
        scores = rng.normal(0, 2, n)
        stats = {"calls": 0, "rounds": 0}
        p, rej = shortlist.rank(list(range(n)), lambda idx: (softmax(scores[idx]), None), size, stats)
        assert np.allclose(p, softmax(scores), atol=1e-12), (n, size)
        assert rej is None and stats["rounds"] >= 2 and stats["calls"] > n // size


def test_matches_single_pass_on_the_fake_scores():
    """With Zurich at logit 10 and 253 others at 0, one pass over all 254 would give
    e^10 / (e^10 + 253); shortlisting must reproduce that."""
    e = Engine(TargetFake("Zurich"))
    a = e.decide(body(["Zurich"] + CITIES[:253]))["answers"]["dest"]
    assert abs(a["probabilities"]["Zurich"] - np.exp(10) / (np.exp(10) + 253)) < 1e-4


def test_chunks_and_keep():
    parts = shortlist.chunks(list(range(254)), 16)
    assert sum(map(len, parts)) == 254 and max(map(len, parts)) <= 16 and max(map(len, parts)) - min(map(len, parts)) <= 1
    assert [x for p in parts for x in p] == list(range(254))            # order preserved
    assert shortlist.keep(np.array([0.05, 0.9, 0.05])) == [1]           # 0.9 reached
    assert shortlist.keep(np.array([0.3, 0.3, 0.2, 0.2])) == [0, 1, 2]  # capped at 3
    assert shortlist.keep(np.array([0.34, 0.33, 0.33])) == [0, 1]        # always drops one


class TargetFake(FakeBackend):
    """A fake model that 'reads': the option line containing the target word gets a high logit;
    the reject slot is high when no shown option contains it."""
    def __init__(self, target, **kw):
        super().__init__(n_ctx=100_000, **kw)
        self.target = target

    def label_logits(self, ids):
        text = "".join(chr(t) for t in self.mem if t < 0x110000)
        block = text[text.rindex("Options:\n") + 9:].split("\n\n")[0].split("\n")
        lines = {ln[0]: ln for ln in block if len(ln) > 2 and ln[1] == "."}
        out = []
        for k, tid in enumerate(ids):
            letter = "Z" if tid == 16 else chr(ord("A") + k)      # FakeBackend: reject label id 16
            line = lines.get(letter, "")
            if letter == "Z":
                out.append(4.0 if not any(self.target in l for l in lines.values() if l[0] != "Z") else -3.0)
            else:
                out.append(10.0 if self.target in line else 0.0)
        return np.array(out)


CITIES = [f"City{i:03d}" for i in range(260)]


def body(opts):
    return {"state": "The customer is flying to Zurich next week.",
            "questions": {"dest": {"type": "choice", "instructions": "Which destination is mentioned?",
                                   "criteria": {c: "" for c in opts}}}}


def test_long_list_finds_target_at_every_position():
    e = Engine(TargetFake("Zurich"))
    for pos in (0, 1, 15, 16, 17, 100, 199, 253):
        opts = CITIES[:253]
        opts = opts[:pos] + ["Zurich"] + opts[pos:]
        a = e.decide(body(opts))["answers"]["dest"]
        assert a["choice"] == "Zurich" and a["probabilities"]["Zurich"] > 0.9, pos
        assert abs(sum(a["probabilities"].values()) - 1) < 1e-4 and len(a["probabilities"]) == 254
        assert a["shortlist"]["rounds"] == 3 and a["shortlist"]["model_calls"] >= 17


def test_long_list_with_reject_slot():
    e = Engine(TargetFake("Zurich"), reject_slot=True)
    a = e.decide(body(CITIES[:40] + ["Zurich"] + CITIES[40:80]))["answers"]["dest"]
    assert a["choice"] == "Zurich" and a["reject"] < 0.1
    miss = e.decide(body(CITIES[:120]))["answers"]["dest"]            # target not in the list at all
    assert miss["reject"] > 0.5


def test_long_list_cached_equals_full_and_deterministic():
    opts = CITIES[:60] + ["Zurich"] + CITIES[60:100]
    cached = Engine(TargetFake("Zurich")).decide(body(opts))
    full = Engine(TargetFake("Zurich"), use_cache=False).decide(body(opts))
    assert cached["answers"] == full["answers"]
    assert Engine(TargetFake("Zurich")).decide(body(opts))["answers"] == cached["answers"]
    u = cached["usage"]
    assert u["model_calls"] == cached["answers"]["dest"]["shortlist"]["model_calls"]
    assert u["evaluated_tokens"] < full["usage"]["evaluated_tokens"]    # the state is encoded once
