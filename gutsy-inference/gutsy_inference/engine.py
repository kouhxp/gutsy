"""Decision engine: prompt -> cached state -> per-question label logits -> calibrated probabilities.

State caching (works for Qwen3.5's hybrid attention + recurrent layers): encode the state prefix
once, snapshot the model's sequence state, and for each question restore the snapshot and encode
only the question suffix. Recurrent state can't be rolled back, so restoring a snapshot is the
only correct way to branch. Snapshots are also kept across requests (LRU), so repeated calls on
the same state skip the prefix entirely.

Determinism: a given request always takes the same computation path (prefix, snapshot, suffix),
and snapshots are exact copies, so identical requests return identical probabilities.
"""
import hashlib
import json
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import shortlist
from .prompt import LABELS, prefix, suffix
from .schema import RequestError, build_answer, parse_request


class StateError(RuntimeError):
    """The backend couldn't snapshot or restore sequence state."""


class Calibration:
    """Temperature per question type, fitted on labeled data (gutsy-bench fit_calibration)."""

    def __init__(self, temperatures=None):
        self.temperatures = {"yes_no": 1.0, "choice": 1.0, **(temperatures or {})}
        # score questions use the choice temperature until a score-specific one is fitted
        self.temperatures.setdefault("score", self.temperatures["choice"])

    @classmethod
    def load(cls, path):
        if not path:
            return cls()
        return cls(json.loads(Path(path).read_text())["temperatures"])

    def probs(self, qtype, logits):
        z = np.asarray(logits, dtype=np.float64) / self.temperatures[qtype]
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()


class StateCache:
    """LRU of state snapshots, bounded by entry count and total bytes."""

    def __init__(self, max_entries=4, max_bytes=512 * 2**20):
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.items, self.bytes = OrderedDict(), 0

    def get(self, key):
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        return None

    def put(self, key, blob, nbytes):
        if nbytes > self.max_bytes:
            return
        self.items[key] = (blob, nbytes)
        self.bytes += nbytes
        while len(self.items) > self.max_entries or self.bytes > self.max_bytes:
            _, (_, n) = self.items.popitem(last=False)
            self.bytes -= n

    def clear(self):
        self.items.clear()
        self.bytes = 0


class Engine:
    def __init__(self, backend, calibration=None, name="gutsy", cache_entries=4,
                 cache_bytes=512 * 2**20, use_cache=True, max_options=None, reject_slot=False,
                 shortlist=True):
        self.backend, self.name = backend, name
        # Options per model call: the largest list this model was trained on (at most 16 letters).
        # Longer choice lists are answered by shortlisting (shortlist.py) unless shortlist=False,
        # in which case they are refused rather than answered at untrained letter positions.
        self.per_call = min(len(LABELS), max_options or len(LABELS))
        self.max_options = max_options
        self.shortlist = shortlist
        # Reject slot: choice and score prompts end with "Z. None of the options fits, or the state
        # does not say"; answers report P(reject) and the options conditioned on "not reject".
        self.reject_slot = reject_slot
        self.calibration = calibration or Calibration()
        ids = backend.label_ids()
        self.label_ids, self.reject_id = ids[: len(LABELS)], (ids[len(LABELS)] if len(ids) > len(LABELS) else None)
        if reject_slot and self.reject_id is None:
            raise ValueError("reject_slot needs a backend that returns the reject label id")
        self.cache = StateCache(cache_entries, cache_bytes)
        self.cache_enabled = use_cache and getattr(backend, "supports_state", False)
        self.lock = threading.Lock()
        self.self_check_result = None

    # ------------------------------------------------------------------ public

    def decide(self, body):
        t0 = time.perf_counter()
        state, questions, _model = parse_request(body)
        for q in questions:
            if len(q.options) > self.per_call and (q.qtype != "choice" or not self.shortlist):
                raise RequestError(
                    f"questions.{q.qid} has {len(q.options)} options, but {self.name} answers at most "
                    f"{self.per_call} per call" + ("" if q.qtype == "choice" else f" for {q.qtype} questions")
                    + "; answers at later positions would be unreliable.")
        with self.lock:
            results, info = self._run(state, questions)
        answers = {q.qid: build_answer(q, p, rej, short) for q, (p, rej, short) in zip(questions, results)}
        info["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return {"model": self.name, "answers": answers, "usage": info}

    def self_check(self, tol=1e-3):
        """Compare the cached path with full re-encoding on a small request. Disables caching if
        they disagree (e.g. the backend's state snapshot doesn't cover every layer type)."""
        if not self.cache_enabled:
            self.self_check_result = "caching unavailable; full re-encoding per question"
            return self.self_check_result
        body = {"state": "The shipment left the warehouse on Monday and arrived on Thursday.",
                "questions": {
                    "a": {"type": "noul", "instructions": "Did the shipment arrive on time?"},
                    "b": {"type": "noul", "instructions": "Did it leave on Monday?"},
                    "c": {"type": "choice", "instructions": "On which day did it arrive?",
                          "criteria": {"Monday": "", "Wednesday": "", "Thursday": ""}}}}
        state, questions, _ = parse_request(body)
        err = None
        with self.lock:
            try:
                self.cache.clear()
                pre = prefix(state)
                pre_toks = self.backend.tokenize(pre, bos=True)
                if not self._aligned(pre, pre_toks, questions[0]):
                    raise StateError("prefix/suffix tokenization did not line up")
                cached, _ = self._run_with(state, pre, pre_toks, questions, True)
                full, _ = self._run_with(state, pre, pre_toks, questions, False)
                diff = max(float(np.max(np.abs(np.append(a[0], a[1] or 0) - np.append(b[0], b[1] or 0))))
                           for a, b in zip(cached, full))
            except StateError as e:
                diff, err = None, e
            finally:
                self.cache.clear()
        if diff is None:
            self.cache_enabled = False
            self.self_check_result = f"caching disabled: {err}"
        elif diff > tol:
            self.cache_enabled = False
            self.self_check_result = f"caching disabled: cached vs full differ by {diff:.2e}"
        else:
            self.self_check_result = f"caching OK (max prob diff {diff:.1e})"
        return self.self_check_result

    # ------------------------------------------------------------------ internals

    def _rej(self, q):
        return self.reject_slot and q.qtype in ("choice", "score")

    def _aligned(self, pre, pre_toks, q):
        """Prefix tokens + suffix tokens must equal the tokens of the whole prompt."""
        opts = q.options[: self.per_call]
        s = suffix(q.text, opts, self._rej(q))
        return pre_toks + self.backend.tokenize(s, bos=False) == self.backend.tokenize(pre + s, bos=True)

    def _readout(self, qtype, n, rej):
        """-> (option probabilities, P(reject) or None). With the reject slot, the options are
        conditioned on "not reject". One temperature scales all columns, so conditioning gives
        exactly the temperature-scaled distribution over the options alone."""
        ids = self.label_ids[:n] + ([self.reject_id] if rej else [])
        p = self.calibration.probs(qtype, self.backend.label_logits(ids))
        if not rej:
            return p, None
        pr = float(p[-1])
        return p[:-1] / max(1e-12, 1.0 - pr), pr

    def _run(self, state, questions):
        pre = prefix(state)
        pre_toks = self.backend.tokenize(pre, bos=True)
        use_cache = self.cache_enabled and self._aligned(pre, pre_toks, questions[0])
        if use_cache:
            try:
                return self._run_with(state, pre, pre_toks, questions, True)
            except StateError:
                self.cache_enabled = False
                self.self_check_result = "caching disabled after a snapshot/restore failure"
                self.cache.clear()
        return self._run_with(state, pre, pre_toks, questions, False)

    def _run_with(self, state, pre, pre_toks, questions, use_cache):
        """-> ([(probs, p_reject, shortlist_stats|None) per question], usage)."""
        b = self.backend
        live = {"fresh": False, "evaluated": 0, "calls": 0, "question_tokens": 0}
        entry, hit = None, False
        if use_cache:
            key = hashlib.sha256(pre.encode()).hexdigest()
            entry = self.cache.get(key)
            hit = entry is not None
            if not hit:
                b.reset()
                b.eval(pre_toks, 0)
                live["evaluated"] += len(pre_toks)
                blob, nbytes = b.save_seq()
                self.cache.put(key, blob, nbytes)
                entry, live["fresh"] = (blob, nbytes), True

        def score(q, idx):
            opts, rej = [q.options[i] for i in idx], self._rej(q)
            s = suffix(q.text, opts, rej)
            if use_cache:
                toks = b.tokenize(s, bos=False)
                if len(pre_toks) + len(toks) > b.n_ctx:
                    raise RequestError(f"state + question is {len(pre_toks) + len(toks)} tokens; this "
                                       f"model's context window is {b.n_ctx}")
                if not live["fresh"]:
                    b.load_seq(entry[0], pre_toks)
                live["fresh"] = False
                b.eval(toks, len(pre_toks))
            else:
                toks = b.tokenize(pre + s, bos=True)
                if len(toks) > b.n_ctx:
                    raise RequestError(f"state + question is {len(toks)} tokens; this model's context "
                                       f"window is {b.n_ctx}")
                b.reset()
                b.eval(toks, 0)
            live["evaluated"] += len(toks)
            live["question_tokens"] += len(toks) - (0 if use_cache else len(pre_toks))
            live["calls"] += 1
            return self._readout(q.qtype, len(idx), rej)

        results = []
        for q in questions:
            idx = list(range(len(q.options)))
            if len(idx) <= self.per_call:
                p, rej = score(q, idx)
                results.append((p, rej, None))
            else:
                stats = {"calls": 0, "rounds": 0}
                p, rej = shortlist.rank(idx, lambda sub, q=q: score(q, sub), self.per_call, stats)
                results.append((p, rej, stats))
        return results, {"prompt_tokens": len(pre_toks) + live["question_tokens"],
                         "state_tokens": len(pre_toks), "evaluated_tokens": live["evaluated"],
                         "model_calls": live["calls"], "state_cached": use_cache, "state_cache_hit": hit}
