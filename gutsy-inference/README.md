# gutsy-inference

Local runtime for gutsy decision models. Send a state and typed questions; get calibrated
probabilities back. No text generation, no network calls, and your data never leaves the machine.
The request and response format follows the Jev-style Decisions API, so switching an existing
client is mostly a base-URL change.

## Install

    pip install -e .            # builds llama-cpp-python for CPU if you don't have it yet

Needs llama-cpp-python 0.3.35 or newer (older builds don't know the `qwen35` architecture).

## Model

Put the model and its calibration file in `models/` and copy the example config:

    models/gutsy-0.8b-v04-q8_0.gguf
    models/gutsy-0.8b-v04.calibration.json

and, for the smaller Q4_K_M file (optional):

    models/gutsy-0.8b-v04-q4_k_m.gguf
    models/gutsy-0.8b-v04-q4_k_m.calibration.json

    cp models.example.json models.json

All four files are in the Hugging Face repository `kouhxp/gutsy`
(`hf download kouhxp/gutsy --include "gutsy-0.8b-v04*" --local-dir models`). If you keep only one
pair, delete the other entry from `models.json`. Check that the model loads and that
state caching works on your machine:

    gutsy-inference check

Expect `caching OK`. `caching disabled: ...` means answers are still correct, but every question
re-encodes the whole state (slower).

## Run

    gutsy-inference serve                       # http://127.0.0.1:8765

    curl -s http://127.0.0.1:8765/v1/systemone -H 'Content-Type: application/json' -d '{
      "model": "gutsy-0.8b-v04",
      "state": "Order 1182 shipped Monday. The customer says it arrived Thursday with the box crushed.",
      "questions": {
        "damaged": {"type": "noul", "instructions": "Was the order damaged?"},
        "next":    {"type": "choice", "instructions": "What should support do next?",
                    "criteria": {"refund": "issue a refund",
                                 "replace": "send a replacement",
                                 "escalate": "a manager reviews it within 24 hours"}}}}'

Response (numbers illustrative):

    {"model": "gutsy-0.8b-v04",
     "answers": {
       "damaged": {"type": "noul", "noul": 0.97},
       "next": {"type": "choice", "choice": "replace",
                "probabilities": {"refund": 0.04, "replace": 0.91, "escalate": 0.05},
                "confidence": 0.865, "margin": 0.86, "reject": 0.02}},
     "usage": {"prompt_tokens": 231, "state_tokens": 71, "evaluated_tokens": 231,
               "model_calls": 2, "state_cached": true, "state_cache_hit": false, "latency_ms": 1840.2}}

Other commands:

    gutsy-inference decide request.json            # one request, no server
    gutsy-inference bench                          # CPU latency by state length
    gutsy-inference serve --api-key-env GUTSY_KEY  # require a bearer token
    gutsy-inference serve --cors                   # allow browser pages to call it (examples/)

## Examples

[`examples/`](examples/) has four self-contained browser demos (feedback sheet, model router,
fact check, live call radar) that call a local server started with `--cors`.

## API

`POST /v1/systemone` (TypeSafe-style; works with JevBench's `typesafe` adapter) and
`POST /api/alpha/decisions` (alias `POST /v1/decisions`, OpenRouter Decisions style). On
`/v1/systemone`, an unknown model name such as `jev-latest` falls back to the default model; the
response's `model` field always says which model answered.

| field | type | notes |
|---|---|---|
| `model` | string, optional | a name from `models.json`; the default model otherwise |
| `state` | string or JSON | shared context for all questions; encoded once |
| `questions` | object | id -> question, up to 64 per request |

Question types:

- `noul`: yes/no. Optional `criteria: {"true": "...", "false": "..."}`, shown to the model as
  "Yes means: ... / No means: ...". Answer: `{"noul": P(yes)}`.
- `choice`: 2-255 options as `criteria: {"name": "description"}` (or a plain list of names). An
  empty description, or one equal to the name, shows just the name. Answer: `choice` (top option),
  `probabilities` for every option, `confidence`, `margin` and `reject`.
- `score`: 2-16 ordered levels, lowest first, as `criteria: ["low", "medium", "high"]`. Answer:
  `probabilities` keyed by level index (`"0"`, `"1"`, ...), the top `level`, `expected` (the
  probability-weighted level), `confidence`, `margin` and `reject`.

Instructions, criteria values and levels may be strings, JSON objects or arrays; non-strings are
shown to the model as compact JSON. Probabilities are temperature-scaled with the model's
calibration file (one temperature per question type, fitted on held-out labeled data).

`GET /health` shows loaded models, caching self-check results and temperatures.
`GET /v1/models` lists model names.

### Confidence, margin and reject

- `confidence` = (n x top probability - 1) / (n - 1): 1 when all probability is on one option, 0
  when the options are uniform. Comparable across option counts, unlike the top probability, which
  never drops below 1/n. This is the shape TypeSafe documents for Jev's confidence.
- `margin` = top probability - second probability. Two answers can share a confidence while one is
  nearly split between two options and the other isn't; the margin tells them apart. Gate on
  confidence for "should a machine handle this at all", and on margin for "which of these two".
- `reject`: the probability that none of the options fits, or that the state doesn't say. The
  `probabilities` are the options conditioned on "not reject", so they still sum to 1.
- Yes/no answers have none of these: `noul` is itself the probability, and 0.5 means no view.
  Gate them with a two-sided band (e.g. act above 0.85 or below 0.15, escalate in between).
- Thresholds are per question and per action, not per system: derive them from a few hundred
  labeled real inputs (accuracy vs. share of volume automated), and re-measure when the options,
  the wording or the state changes. Log the full `probabilities`, so thresholds can be re-tuned
  later without re-running anything.

### Long option lists

One model call holds at most 16 options. Longer choice lists (up to 255) are answered by
shortlisting: balanced chunks of up to 16 are scored, each chunk passes on its best options (until
90% of its probability, at most 3), the finalists are scored together (recursively if needed), and
the rounds are combined into one distribution over all options. If the model's preferences are
consistent, this equals what a single pass over the whole list would give (a test checks it
exactly). The answer then includes `"shortlist": {"rounds", "model_calls"}`; the state is encoded
once, so each extra call costs only the question and its options. On a shortlisted answer,
`reject` comes from the final round: "none of the finalists fits". Set `"shortlist": false` for a
model in `models.json` to refuse lists longer than 16 instead.

### models.json

| key | meaning |
|---|---|
| `gguf` | path to the model file |
| `calibration` | path to its calibration file (temperatures per question type) |
| `n_ctx` | context window in tokens (8192 for gutsy-0.8b v0.4) |
| `reject_slot` | `true` for models trained with the reject slot (both v0.4 files are) |
| `cache_tol` | largest cached-vs-full probability difference the startup check accepts before turning caching off (default 0.001; 0.02 for the Q4_K_M file) |
| `shortlist` | answer choice lists longer than 16 by shortlisting (default `true`) |
| `max_options` | options per model call, if lower than 16 |

Top-level keys: `default` (model used when a request names none), `n_threads`, `n_gpu_layers`,
`cache_entries`, `cache_mb`.

## How it works

1. The prompt splits into a state prefix and a per-question suffix.
2. The state is encoded once, and the model's sequence state (attention cache plus the recurrent
   state of Qwen3.5's linear-attention layers) is snapshotted.
3. Each question restores the snapshot, encodes only its own tokens, and reads the probabilities
   of the option letters at the answer position.
4. Snapshots are kept in an LRU cache (`cache_entries`, `cache_mb`), so repeat requests on the
   same state skip step 2 entirely.

Recurrent layers can't be rolled back, so restoring a snapshot is the only correct way to reuse
the state. `check` verifies the cached path gives the same answers as full re-encoding; if it
doesn't on some build, caching turns itself off.

Identical requests return identical probabilities on the same machine and build. Different
llama.cpp builds or hardware can differ in the last decimal places.

## Performance tuning

- Threads default to about the number of physical cores; set `--threads` or `n_threads` in
  `models.json` if `bench` suggests otherwise. Using all logical cores is often slower.
- Put several questions about the same state in one request: the state is paid for once.
- Longer states cost more for the first question only; follow-up questions cost roughly the same
  regardless of state length.
- With a GPU, set `n_gpu_layers: -1`.

## Limitations

- Inference is serialized per model; concurrent requests queue.
- Context: 8,192 tokens for state plus question. Latency grows with state length on CPU.
- Score questions: at most 16 levels (no shortlisting).
- Shortlisted answers on easy lookups tend to understate confidence (the chosen option is right;
  the probability is spread too thin over eliminated options).
- Model-level weak spots (arithmetic, new rulebooks, ambiguity, knowledge) are listed in the
  model card.
