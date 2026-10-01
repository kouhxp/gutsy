"""CPU latency benchmark: how long does a request take, by state length and question count?

For each state length it measures three things on the same request:
  cold    first request for this state (encodes the state once, then each question)
  warm    same state again (state snapshot reused from cache; only questions are encoded)
  nocache every question re-encodes the full state (what caching saves you)
"""
import statistics
import time

SENTENCES = [
    "Order {i} for customer {c} shipped from warehouse {w} and was marked {s}.",
    "Ticket {i} from customer {c} reports a {s} delivery at site {w}.",
    "Invoice {i} for customer {c} was issued by office {w} and is {s}.",
]
STATUS = ["on time", "late", "damaged", "pending", "complete"]

QUESTIONS = {
    "late": {"type": "noul", "instructions": "Does the state mention any late delivery?"},
    "damaged": {"type": "noul", "instructions": "Was anything reported as damaged?"},
    "topic": {"type": "choice", "instructions": "What is the state mostly about?",
              "criteria": {"logistics": "shipping and deliveries", "hiring": "recruiting",
                           "finance": "budgets and accounting", "legal": "contracts"}},
    "pending": {"type": "noul", "instructions": "Is any invoice still pending?"},
    "warehouse": {"type": "choice", "instructions": "Which warehouse appears first?",
                  "criteria": {"0": "", "1": "", "2": "", "3": "", "4": ""}},
    "complete": {"type": "noul", "instructions": "Is every order complete?"},
}


def make_state(engine, target_tokens):
    parts, i = [], 0
    while True:
        s = SENTENCES[i % 3].format(i=i, c=100 + i % 37, w=i % 5, s=STATUS[i % 5])
        parts.append(s)
        i += 1
        if i % 20 == 0 and len(engine.backend.tokenize(" ".join(parts), bos=False)) >= target_tokens:
            return " ".join(parts)


def _time(engine, body, repeats):
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        engine.decide(body)
        out.append((time.perf_counter() - t0) * 1000)
    return statistics.median(out)


def run(engine, state_tokens=(200, 1000, 4000), n_questions=5, repeats=3):
    qs = dict(list(QUESTIONS.items())[:n_questions])
    engine.decide({"state": "warm up", "questions": {"q": QUESTIONS["late"]}})   # warm-up
    rows = []
    for n in state_tokens:
        body = {"state": make_state(engine, n), "questions": qs}
        cold = []
        for _ in range(repeats):
            engine.cache.clear()
            t0 = time.perf_counter()
            engine.decide(body)
            cold.append((time.perf_counter() - t0) * 1000)
        warm = _time(engine, body, repeats)           # cache now holds this state
        was = engine.cache_enabled
        engine.cache_enabled = False
        try:
            nocache = _time(engine, body, repeats)
        finally:
            engine.cache_enabled = was
        rows.append((n, statistics.median(cold), warm, nocache))

    k = len(qs)
    lines = [f"model {engine.name}, {k} questions per request, median of {repeats}; "
             f"caching: {engine.self_check_result}", "",
             "| state tokens | cold request | warm request | no cache | cold per question | warm per question |",
             "|---|---|---|---|---|---|"]
    for n, c, w, nc in rows:
        lines.append(f"| {n} | {c:.0f} ms | {w:.0f} ms | {nc:.0f} ms | {c / k:.0f} ms | {w / k:.0f} ms |")
    return "\n".join(lines)
