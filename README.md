# gutsy

**Open, local decision models with calibrated probabilities.** Send a state (text or JSON) and
typed questions (yes/no, choice, score); get a probability for every option back. No text
generation, no network calls, no per-call cost. The API follows the Jev-style Decisions format,
so existing clients mostly need only a new base URL.

`gutsy-0.8b` v0.3 is a fine-tune of Qwen3.5-0.8B, served as a 775 MB GGUF by llama.cpp on an
ordinary CPU.

| | |
|---|---|
| Model | `gutsy-0.8b` v0.3 (Q8_0 GGUF, 775 MB): [kouhxp/gutsy](https://huggingface.co/kouhxp/gutsy) |
| Runtime | [`gutsy-inference`](gutsy-inference/) (this repository) |
| Model card | [MODEL_CARD.md](MODEL_CARD.md) |
| License | Apache 2.0 (see [Licensing](#licensing)) |

## Why use it

- **Calibrated.** Trained with proper scoring rules (cross-entropy + Brier) on soft labels, then
  temperature-checked on held-out data: validation calibration error is 0.021 *before* any
  scaling. A 0.8 means roughly 80%.
- **Local and private.** Runs on CPU; your data never leaves the machine.
- **Deterministic.** Identical requests return identical probabilities on the same machine and
  build.
- **Robust to option order.** 94-99% of answers are unchanged when options are shuffled
  (BoolQ, SNLI, dev scenarios).
- **State caching.** The state is encoded once; follow-up questions about the same state cost
  only their own tokens, and repeat requests on a cached state skip it entirely.
- **Up to 255 options.** Lists longer than 16 are answered by shortlisting in the runtime
  (99.9% accuracy on our 20-254-option test).
- **Says "none of these".** Choice and score answers include a `reject` probability for "none of
  the options fits, or the state doesn't say".

## Quickstart

```bash
git clone https://github.com/kouhxp/gutsy
pip install -e gutsy/gutsy-inference           # needs llama-cpp-python >= 0.3.35
pip install -U huggingface_hub
hf download kouhxp/gutsy gutsy-0.8b-v03-q8_0.gguf gutsy-0.8b-v03.calibration.json \
  --local-dir gutsy/gutsy-inference/models
cd gutsy/gutsy-inference
cp models.example.json models.json
gutsy-inference check                   # expect "caching OK"
gutsy-inference serve                   # http://127.0.0.1:8765
```

```bash
curl -s http://127.0.0.1:8765/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "gutsy-0.8b-v03",
  "state": "Order 1182 shipped Monday. The customer says it arrived Thursday with the box crushed.",
  "questions": {"decision": {"type": "choice", "instructions": "What should support do next?",
    "criteria": {"refund": "issue a refund", "replace": "send a replacement",
                 "escalate": "a manager reviews it within 24 hours"}}}}'
```

Endpoints: `POST /v1/systemone` (TypeSafe-style; works with JevBench's `typesafe` adapter),
`POST /api/alpha/decisions` and `/v1/decisions` (OpenRouter Decisions style), `GET /health`.
Each answer has `probabilities`; choice and score answers add `confidence` (how concentrated the
distribution is, from 0 for uniform to 1), `margin` (top two apart) and `reject`. Gate risky
actions on thresholds you measure on your own labeled data; see the
[runtime README](gutsy-inference/README.md).

## Results (gutsy-0.8b v0.3)

All numbers below were run by us. v0.1 and v0.2 are earlier internal versions (never released),
shown for context. Comparisons with other systems are indicative: different harness versions and
item samples, and the other systems' numbers are self-reported unless noted.

### JevBench, public items (231)

| tier | v0.1 | v0.2 | **v0.3** |
|---|---|---|---|
| easy (48) | 45 | 48 | **47** |
| original (72) | 46 | 46 | **61** |
| hard (111) | 49 | 41 | **59** |
| **total** | 140 (0.606) | 135 (0.584) | **167 (0.723)** |
| calibration error (hard) | 0.167 | 0.254 | **0.078** |
| ordinal MAE (original / hard) | 0.53 / 0.73 | 0.55 / 0.46 | **0.30 / 0.59** |

Reported by other projects on the same public set: Jev 0.866, decision-4b 0.883, Neriv 0.6B
0.636, XERON-0.4 0.558, tuned Laya 0.537; Jeff-0.8B 47.6% on the hard tier (105 items).
JevBench describes public-set results as preliminary; official results use sealed items.

Hard-tier families for v0.3 (5-19 items each, so single families are noisy): routing_hard 1.00,
trap 0.88, adversarial 0.83, tradeoff 0.67, judge_hard 0.65, multi_hop 0.50, probability 0.50,
long_policy 0.42, temporal_numeric 0.33, ambiguous 0.00.

### Same items, measured side by side (700 per task)

| | Qwen3.5-0.8B (no fine-tune) | gutsy v0.1 | **gutsy v0.3** | Jev (typesafe/jev-1.13, via API) |
|---|---|---|---|---|
| BoolQ | 0.700 | 0.821 | **0.850** | 0.906 |
| SNLI | 0.489 | 0.886 | **0.879** | 0.883 |
| CommonsenseQA (never trained on) | 0.463 | 0.476 | **0.587** | 0.863 |

Jev was queried through its API for benchmarking only; none of its output was used for training.

### Evaluation panel (700 items each; JudgeBench 245)

| | v0.1 | v0.2 | **v0.3** | Jeff-0.8B v1.0 | Jev |
|---|---|---|---|---|---|
| BBH | 0.419 | 0.413 | **0.464** | 64.0 | 94.3 |
| Financial PhraseBank | 0.857 | 0.481 | **0.921** | 96.4 | 77.0 |
| JudgeBench | 0.416 | 0.424 | **0.596** | 62.6 | 78.6 |
| RAGTruth † | 0.469 | 0.646 | **0.790** | 86.1 | 77.3 |
| WinoGrande † | 0.503 | 0.516 | **0.657** | 68.6 | 90.7 |

† **In-domain for v0.3**: the RAGTruth and WinoGrande training splits are in its training data
(test items never are). Financial PhraseBank is close to the financial-tweet sentiment data used
in training. Jeff and Jev figures are from Jeff's README, on a different sample of these
benchmarks.

### Routing and long lists

- BANKING77 test split, all 77 intents as options, **zero-shot** (not in training): 0.609
- Lists of 20-254 options, answer at every position: 0.999

### Where it is weak

Don't automate on these without your own testing:

- **Date and number arithmetic** (JevBench temporal_numeric 0.33; business-day and similar
  held-out tasks near chance). A 0.8B model can't reliably do multi-step calculation in one
  forward pass: compute dates and numbers in code and ask about the result.
- **New multi-step rulebooks** (long_policy 0.42).
- **Deciding whether ambiguity matters** (ambiguous 0 of 7): it abstains where the stated rules
  already settle the case, and commits where it should ask.
- **World knowledge** (CommonsenseQA 0.59 vs Jev 0.86).
- **Long states on CPU**: median 4.2 s on the hard tier, 95th percentile 23 s.
- **Long-list confidence** understates accuracy on easy lookups (the chosen option is right; the
  probabilities are spread too thin).
- English only.

## How v0.3 was trained

- Base: Qwen/Qwen3.5-0.8B. Full fine-tune with frozen token embeddings, learning rate 1e-5,
  256 questions per step, one epoch over about 90,000 questions; the checkpoint was chosen by
  loss on a held-out development set (step 300, the last one evaluated).
- Loss: cross-entropy + 0.5 x Brier on soft labels where available.
- Data (approximate shares): 27% scenarios written and blind-reviewed by a local open teacher
  model (Qwen3.8-27B), only kept where two independent passes agreed; 19% reading and
  verification (BoolQ, SNLI, SQuAD 2.0, HotpotQA); 12% classification and routing (MASSIVE,
  CLINC150, Civil Comments, financial-tweet sentiment); 11% in-domain panel training splits
  (RAGTruth, WinoGrande); 9% code-generated synthetic data with exact labels (policies, numbers,
  records, scores); 7% agent tool-choice decisions (Jev Decisions v1); 5% each judging
  (HelpSteer2), knowledge (ARC, OpenBookQA, HellaSwag) and real contract excerpts (CUAD).
- A LoRA control run on the same data tied it exactly on JevBench (167/231) and stayed within a
  few points everywhere else: the gains over the earlier internal versions come from the data,
  not the training method.

Full details, sources and licenses: [MODEL_CARD.md](MODEL_CARD.md).

## Disclosures

- **No closed-model output** is used anywhere in the training data. The teacher is an open model.
- **Benchmark awareness.** v0.3's data design was shaped by the family-level JevBench results of
  earlier internal versions (date/policy synthetic data, a judging slice, agent-harness
  scenarios). No JevBench item was trained on; training data was decontaminated against JevBench
  public items and the evaluation panel (shared 13-word sequences).
- **In-domain data**: RAGTruth and WinoGrande training splits, CLINC150 and MASSIVE training
  splits. BANKING77 is held out as a clean zero-shot routing test.
- **Self-run results.** All results here are ours, on public items. No official JevBench result
  yet.

## Licensing

Code and model weights: **Apache 2.0** (see `LICENSE`). The base model (Qwen3.5-0.8B) and the
teacher model (Qwen3.8-27B) are Apache 2.0. The training data draws on CC BY 4.0, CC BY-SA, CC0,
MIT and Apache 2.0 sources, credited in MODEL_CARD.md; Jev Decisions v1 is CC BY 4.0 with
upstream NVIDIA terms. Users who redistribute the training data itself must follow each source's
terms.
