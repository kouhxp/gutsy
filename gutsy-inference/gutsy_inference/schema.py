"""Request parsing and response building (Jev-style Decisions format).

Request:
  {"model": "gutsy-0.8b",                     # optional; server default otherwise
   "state": "text" | {...json...},
   "questions": {
     "<id>": {"type": "noul", "instructions": "...",
              "criteria": {"true": "...", "false": "..."}},        # criteria optional
     "<id>": {"type": "choice", "instructions": "...",
              "criteria": {"<option>": "<description>", ...}}}}   # 2-16 options

     "<id>": {"type": "score", "instructions": "...",
              "criteria": ["level 0", "level 1", ...]}}}       # ordered levels (2-16)
Instructions, criteria values and score levels may be strings, JSON objects or arrays; non-strings
are shown to the model as compact JSON. Choice criteria may also be a plain list of option names.

Response:
  {"model": ..., "answers": {"<id>": {"type": "noul", "noul": P(yes)} |
                                     {"type": "choice", "choice": "<option>",
                                      "probabilities": {"<option>": p, ...}, "confidence": c,
                                      "margin": m} |
                                     {"type": "score", "level": "<index>", "expected": E[index],
                                      "probabilities": {"0": p, "1": p, ...}, "confidence": c,
                                      "margin": m}},
  confidence = (n * p_max - 1) / (n - 1): 1 when all probability is on one option, 0 when uniform
  (the shape TypeSafe publishes for Jev's confidence explainer; comparable across option counts).
  margin = p_max - p_second: whether the top two are close (two answers can share a confidence
  while one is nearly split between two options and the other isn't). Yes/no answers carry
  neither: the probability itself is the answer, and 0.5 is "no view".
   "usage": {...}}
"""
import json
from dataclasses import dataclass

import numpy as np

from .prompt import MAX_OPTIONS

MAX_QUESTIONS = 64
MAX_CHOICE_OPTIONS = 255     # longer than one model call: answered by shortlisting (shortlist.py)


class RequestError(ValueError):
    """Client error -> HTTP 400."""


@dataclass
class Question:
    qid: str
    qtype: str            # "yes_no" (API type "noul") | "choice" | "score"
    text: str             # question text shown to the model
    options: list         # option texts shown to the model, in order
    keys: list            # response keys (choice only)


def _req(cond, msg):
    if not cond:
        raise RequestError(msg)


def as_text(x):
    """Strings as-is (stripped); objects and arrays as compact JSON."""
    return x.strip() if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def _options(crit, where):
    """Choice/score criteria -> (keys, option texts). Dict: name -> description. List: items are
    option names (choice) or level descriptions (score, keyed by index)."""
    keys, options = [], []
    for k, v in crit.items():
        _req(isinstance(k, str) and k.strip(), f"{where}.criteria has an empty option name")
        desc = "" if v is None else as_text(v)
        keys.append(k)
        options.append(k.strip() if not desc or desc == k.strip() else f"{k.strip()}: {desc}")
    return keys, options


def parse_request(body):
    _req(isinstance(body, dict), "request body must be a JSON object")
    _req("state" in body, "missing 'state'")
    state = body["state"]
    _req(isinstance(state, (str, dict, list)), "'state' must be a string or JSON object/array")
    qs = body.get("questions")
    _req(isinstance(qs, dict) and qs, "'questions' must be a non-empty object of id -> question")
    _req(len(qs) <= MAX_QUESTIONS, f"at most {MAX_QUESTIONS} questions per request")
    model = body.get("model")
    _req(model is None or isinstance(model, str), "'model' must be a string")

    out = []
    for qid, q in qs.items():
        where = f"questions.{qid}"
        _req(isinstance(q, dict), f"{where} must be an object")
        instr = q.get("instructions")
        _req(isinstance(instr, (str, dict, list)) and as_text(instr) not in ("", "{}", "[]"),
             f"{where}.instructions must be a non-empty string, object or array")
        instr = as_text(instr)
        crit = q.get("criteria")
        qtype = q.get("type")
        if qtype == "noul":
            text = instr
            if crit is not None:
                _req(isinstance(crit, dict) and set(crit) <= {"true", "false"},
                     f"{where}.criteria for noul may only have 'true' and 'false'")
                extra = [f"{label} means: {as_text(crit[k])}"
                         for k, label in (("true", "Yes"), ("false", "No"))
                         if crit.get(k) not in (None, "", {}, [])]
                if extra:
                    text += "\n" + "\n".join(extra)
            out.append(Question(qid, "yes_no", text, ["Yes", "No"], ["yes", "no"]))
        elif qtype == "choice":
            if isinstance(crit, list):
                _req(all(isinstance(c, str) and c.strip() for c in crit),
                     f"{where}.criteria as a list must contain option names")
                _req(len(set(crit)) == len(crit), f"{where}.criteria has duplicate options")
                crit = {c: "" for c in crit}
            _req(isinstance(crit, dict) and 2 <= len(crit) <= MAX_CHOICE_OPTIONS,
                 f"{where}.criteria must map 2-{MAX_CHOICE_OPTIONS} option names to descriptions")
            keys, options = _options(crit, where)
            out.append(Question(qid, "choice", instr, options, keys))
        elif qtype == "score":
            if isinstance(crit, list):
                _req(2 <= len(crit) <= MAX_OPTIONS, f"{where}.criteria must list 2-{MAX_OPTIONS} levels")
                _req(all(as_text(c) for c in crit), f"{where}.criteria has an empty level")
                keys, options = [str(i) for i in range(len(crit))], [as_text(c) for c in crit]
            else:
                _req(isinstance(crit, dict) and 2 <= len(crit) <= MAX_OPTIONS,
                     f"{where}.criteria must be an ordered list (or object) of 2-{MAX_OPTIONS} levels")
                keys, options = _options(crit, where)
            out.append(Question(qid, "score", instr, options, keys))
        else:
            raise RequestError(f"{where}.type {qtype!r} is not supported; "
                               "supported types are 'noul', 'choice' and 'score'")
    return state, out, model


def build_answer(q, probs, p_reject=None, shortlist=None):
    """p_reject: probability of the reject slot (models trained with it; choice and score only);
    probs are then the options conditioned on "not reject". shortlist: {"rounds", "calls"} when a
    long choice list was answered in several model calls."""
    out = _answer(q, np.asarray(probs, dtype=np.float64))
    if p_reject is not None:
        out["reject"] = round(float(p_reject), 6)
    if shortlist:
        out["shortlist"] = {"rounds": shortlist["rounds"], "model_calls": shortlist["calls"]}
    return out


def concentration(probs):
    """(n * p_max - 1) / (n - 1), clipped to [0, 1]."""
    n = len(probs)
    return min(1.0, max(0.0, (n * float(np.max(probs)) - 1.0) / (n - 1))) if n > 1 else 1.0


def top_margin(probs):
    top2 = np.sort(np.asarray(probs, dtype=np.float64))[-2:]
    return float(top2[-1] - top2[0]) if len(top2) == 2 else 1.0


def _answer(q, probs):
    if q.qtype == "yes_no":
        return {"type": "noul", "noul": round(float(probs[0]), 6)}
    best = int(np.argmax(probs))
    if q.qtype == "score":
        out = {"type": "score", "level": q.keys[best],
               "probabilities": {k: round(float(p), 6) for k, p in zip(q.keys, probs)},
               "confidence": round(concentration(probs), 6), "margin": round(top_margin(probs), 6)}
        if all(k.isdigit() for k in q.keys):
            out["expected"] = round(float(sum(int(k) * p for k, p in zip(q.keys, probs))), 6)
        return out
    return {"type": "choice", "choice": q.keys[best],
            "probabilities": {k: round(float(p), 6) for k, p in zip(q.keys, probs)},
            "confidence": round(concentration(probs), 6), "margin": round(top_margin(probs), 6)}
