# gutsy

**Open, local decision models with calibrated probabilities.** Send a state (text or JSON) and
typed questions (yes/no, choice, score); get a probability for every option back. No text
generation, no network calls, no per-call cost. The API follows the Jev-style Decisions format,
so existing clients mostly need only a new base URL.

`gutsy-0.8b` v0.4 is a fine-tune of Qwen3.5-0.8B, served by llama.cpp on an ordinary CPU. It comes
in two files from the same checkpoint: a 775 MB **Q8_0** GGUF (the reference and the default in
the examples below) and a 505 MB **Q4_K_M** GGUF (about 25-30% faster, same evaluation-panel
scores, a few points lower on JevBench).

| | |
|---|---|
| Model | `gutsy-0.8b` v0.4: Q8_0 (775 MB) and Q4_K_M (505 MB): [kouhxp/gutsy](https://huggingface.co/kouhxp/gutsy). v0.3: `hf download kouhxp/gutsy --revision v0.3` |
| Runtime | [`gutsy-inference`](gutsy-inference/) 0.4.0 (this repository) |
| Model card | [MODEL_CARD.md](MODEL_CARD.md) |
| Release notes | [docs/releases](docs/releases/) |
| License | Apache 2.0 (see [Licensing](#licensing)) |

## Why use it

- **Calibrated.** Trained with proper scoring rules (cross-entropy + Brier) on soft labels, then
  temperature-checked on held-out data: validation calibration error is 0.022 *before* any
  scaling. A 0.8 means roughly 80%.
- **Local and private.** Runs on CPU; your data never leaves the machine.
- **Small.** The Q8_0 file is 775 MB; the Q4_K_M file is 505 MB.
- **Deterministic.** Identical requests return identical probabilities on the same machine and
  build.
- **Robust to option order.** 96-97% of answers are unchanged when options are shuffled
  (BoolQ, SNLI, development scenarios).
- **State caching.** The state is encoded once; follow-up questions about the same state cost
  only their own tokens, and repeat requests on a cached state skip it entirely. Works for both
  the Q8_0 and the Q4_K_M file.
- **Up to 255 options.** Lists longer than 16 are answered by shortlisting in the runtime
  (99% accuracy on our 20-254-option test).
- **Says "none of these".** Choice and score answers include a `reject` probability for "none of
  the options fits, or the state doesn't say".

## Quickstart

```bash
git clone https://github.com/kouhxp/gutsy
pip install -e gutsy/gutsy-inference           # needs llama-cpp-python >= 0.3.35
pip install -U huggingface_hub
hf download kouhxp/gutsy --include "gutsy-0.8b-v04*" --local-dir gutsy/gutsy-inference/models
cd gutsy/gutsy-inference
cp models.example.json models.json
gutsy-inference check                   # expect "caching OK" for both models
gutsy-inference serve                   # http://127.0.0.1:8765
```

`models.example.json` registers `gutsy-0.8b-v04` (Q8_0, the default) and `gutsy-0.8b-v04-q4`
(Q4_K_M). For the smaller, faster file, use `"model": "gutsy-0.8b-v04-q4"` in requests. To keep
only one file, download that pair (`gutsy-0.8b-v04-q8_0.gguf` + `gutsy-0.8b-v04.calibration.json`,
or `gutsy-0.8b-v04-q4_k_m.gguf` + `gutsy-0.8b-v04-q4_k_m.calibration.json`) and delete the other
entry from `models.json`.

```bash
curl -s http://127.0.0.1:8765/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "gutsy-0.8b-v04",
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

## Results (gutsy-0.8b v0.4)

All numbers below were run by us. Results are for the Q8_0 reference file unless a Q4_K_M column
is shown. Comparisons with other systems are indicative: different harness versions and item
samples, and the other systems' numbers are self-reported unless noted.

### JevBench, public items (231)

| tier | v0.3 (Q8_0) | **v0.4 (Q8_0)** | v0.4 (Q4_K_M) |
|---|---|---|---|
| easy (48) | 47 | **48** | 47 |
| original (72) | 61 | **65** | 62 |
| hard (111) | 59 | **56** | 52 |
| **total** | 167 (0.723) | **169 (0.732)** | 161 (0.697) |
| calibration error (easy / original / hard) | 0.023 / 0.122 / 0.078 | **0.022 / 0.107 / 0.137** | 0.019 / 0.137 / 0.125 |
| ordinal MAE (original / hard) | 0.30 / 0.59 | **0.19 / 0.69** | 0.21 / 0.67 |
| paraphrase agreement (original) | 0.81 | **0.86** | 0.78 |

Reported by other projects on the same public set: Jev 0.866, decision-4b 0.883, Neriv 0.6B
0.636, XERON-0.4 0.558, tuned Laya 0.537; Jeff-0.8B 47.6% on the hard tier (105 items).
JevBench describes public-set results as preliminary; official results use sealed items.

Hard-tier families for v0.4 Q8_0 (5-19 items each, so single families are noisy): routing_hard
1.00, trap 0.88, adversarial 0.83, judge_hard 0.59, ambiguous 0.57 (v0.3: 0.00), probability
0.50, multi_hop 0.44, tradeoff 0.33, long_policy 0.32, temporal_numeric 0.27. Item by item, v0.3
and v0.4 disagree on 31 hard items (17 one way, 14 the other): apart from ambiguity, the hard-tier
differences between the two versions are within noise.

### What v0.4 changed, on held-out tests

| | v0.3 | **v0.4** |
|---|---|---|
| Contrast pairs, both versions right (48 held-out pairs) | 0.104 | **0.417** |
| ... abstains when the missing fact decides the outcome | 0.583 | **0.896** |
| ... commits when the stated rules settle it anyway | 0.312 | **0.458** |
| Logic rules of depth 3 (trained on depth 1-2) | 0.617 | **0.842** |
| Orderings stated as comparatives (trained on positions) | 0.840 | **0.947** |
| Conditional probability (never trained on) | 0.305 | **0.511** |
| Calendar-month arithmetic (never trained on) | 0.683 | 0.519 |

A contrast pair is two near-identical states with the same question: in one the missing
information decides the answer (the right answer is to ask or abstain), in the other the stated
rules decide it anyway.

### Same items, measured side by side (700 per task)

| | Qwen3.5-0.8B (no fine-tune) | gutsy v0.3 | **gutsy v0.4** | Jev (typesafe/jev-1.13, via API) |
|---|---|---|---|---|
| BoolQ | 0.700 | 0.850 | **0.837** | 0.906 |
| SNLI | 0.489 | 0.879 | **0.870** | 0.883 |
| CommonsenseQA (never trained on) | 0.463 | 0.587 | **0.577** | 0.863 |

Jev was queried through its API for benchmarking only; none of its output was used for training.

### Evaluation panel (700 items each; JudgeBench 245)

| | v0.3 | **v0.4 Q8_0** | v0.4 Q4_K_M | Jeff-0.8B v1.0 | Jev |
|---|---|---|---|---|---|
| BBH | 0.464 | **0.481** | 0.474 | 64.0 | 94.3 |
| Financial PhraseBank | 0.921 | **0.927** | 0.926 | 96.4 | 77.0 |
| JudgeBench | 0.596 | **0.539** | 0.563 | 62.6 | 78.6 |
| RAGTruth † | 0.790 | **0.813** | 0.807 | 86.1 | 77.3 |
| WinoGrande † | 0.657 | **0.620** | 0.629 | 68.6 | 90.7 |

† **In-domain**: the RAGTruth and WinoGrande training splits are in the training data (test items
never are). Financial PhraseBank is close to the financial-tweet sentiment data used in training.
Jeff and Jev figures are from Jeff's README, on a different sample of these benchmarks.

### Routing and long lists

- BANKING77 test split, all 77 intents as options, **zero-shot** (not in training): 0.613
- Lists of 20-254 options, answer at every position: 0.991

### Where it is weak

Don't automate on these without your own testing:

- **Date and number arithmetic** (JevBench temporal_numeric 0.27; business-day, calendar-month and
  similar held-out tasks near chance). A 0.8B model can't reliably do multi-step calculation in one
  forward pass: compute dates and numbers in code and ask about the result.
- **New multi-step rulebooks** (long_policy 0.32).
- **Ambiguity**, though much better than v0.3: it still hedges in about a third of the cases where
  the stated rules already settle the answer.
- **Judging close pairs of answers** (JudgeBench 0.54 for Q8_0, 0.56 for Q4_K_M, down from 0.60 in
  v0.3).
- **World knowledge** (CommonsenseQA 0.58 vs Jev 0.86).
- **Calibration on hard, unfamiliar items** is weaker than in v0.3 (hard-tier calibration error
  0.137 vs 0.078); on validation data it is unchanged (0.022).
- **Long states on CPU**: median 4.3 s on the hard tier, 95th percentile 27 s (Q8_0, the author's
  machine; Q4_K_M is about 25-30% faster).
- **Long-list confidence** understates accuracy on easy lookups (the chosen option is right; the
  probabilities are spread too thin).
- English only.

## How v0.4 was trained

- Base: Qwen/Qwen3.5-0.8B. Full fine-tune with frozen token embeddings, learning rate 1e-5,
  256 questions per step, one epoch over about 105,000 questions (379 steps); the checkpoint was
  chosen by loss on a held-out development set (step 300; the end of training was evaluated too).
  One 80 GB GPU, about 1 hour.
- Loss: cross-entropy + 0.5 x Brier on soft labels where available.
- Data (shares of the 105,114 training questions): 32% scenarios written and blind-reviewed by an
  open teacher model (Qwen3.8-27B), kept only where two independent passes agreed (92.8% did);
  2% contrast pairs for ambiguity (687 pairs); 15% reading and verification (BoolQ, SNLI,
  SQuAD 2.0, HotpotQA); 11% in-domain panel training splits (RAGTruth, WinoGrande); 10%
  classification and routing (MASSIVE, CLINC150, Civil Comments, financial-tweet sentiment);
  7% code-generated reasoning puzzles with exact labels (object tracking, ordering deduction,
  boolean logic, truth-teller chains, dates, probability); 7% other code-generated data (policies,
  numbers, records, scores); 6% agent tool-choice decisions (Jev Decisions v1); 4% knowledge
  (ARC, OpenBookQA, HellaSwag); 4% judging (HelpSteer2); 3% real contract excerpts (CUAD).
- **Shortcut audit.** The build measures, per source and question template, how well the answer
  can be predicted from surface features alone (option count, option length, position, key style)
  and fails if any template beats chance. v0.4 fixes the shortcuts this found in v0.3's data: "none
  of these" items always had one option more or fewer than normal ones, and in some families the
  longest option was usually right.
- Q4_K_M is quantized from the same checkpoint with llama.cpp's `llama-quantize` and has its own
  temperatures, fitted the same way.

Full details, sources and licenses: [MODEL_CARD.md](MODEL_CARD.md).

## Disclosures

- **No closed-model output** is used anywhere in the training data. The teacher is an open model.
- **Benchmark awareness.** The code-generated puzzles were designed with v0.3's weak spots on
  BIG-Bench Hard and on JevBench's temporal_numeric and probability families in mind, and the
  contrast pairs with JevBench's ambiguous family in mind. No BBH or JevBench item, template or
  wording was used; training data was decontaminated against JevBench public items and the
  evaluation panel (shared 13-word sequences).
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
