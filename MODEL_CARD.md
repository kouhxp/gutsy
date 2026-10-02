---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
language: [en]
library_name: gguf
tags: [decision-model, classification, calibration, jev-compatible, llama.cpp, gguf]
---

# gutsy-0.8b v0.4

A decision model: given a **state** (text or JSON) and a typed **question** (yes/no, choice
among up to 16 options per call, or ordered score), it returns a calibrated probability for every
option from a single forward pass. It does not generate text. Fine-tuned from Qwen3.5-0.8B;
distributed as GGUF files for llama.cpp, served by `gutsy-inference`, which adds state caching,
shortlisting for lists up to 255 options, and calibrated temperatures.

| file | size | notes |
|---|---|---|
| `gutsy-0.8b-v04-q8_0.gguf` | 775 MB | reference and default in the examples; the results below are for this file unless stated; JevBench 169/231 |
| `gutsy-0.8b-v04-q4_k_m.gguf` | 505 MB | same checkpoint, 4-bit; about 25-30% faster on CPU; same evaluation-panel scores as Q8_0, JevBench 161/231 |

v0.3 is still available: `hf download kouhxp/gutsy --revision v0.3`.

## Intended use

Routing, triage, moderation, policy checks, tool selection in agent loops, judging outputs,
grounding checks: decisions about a state you provide, where you act on the probability. Use the
probabilities with thresholds measured on your own labeled data; route low-confidence and
high-stakes cases to a human.

**Out of scope / unreliable:** date and number arithmetic (compute in code first), applying new
multi-step rulebooks, world-knowledge questions, non-English input. See Limitations.

## How to use

Prompt format (must match exactly; the runtime builds it): Qwen ChatML with thinking disabled;
the state, then the question, then options labeled A-P, then for choice and score questions the
line `Z. None of the options fits, or the state does not say`. The answer is read from the
next-token logits of the option letters (and Z), softmaxed over those letters only. The prompt
format is unchanged from v0.3.

Easiest: the `gutsy-inference` runtime (0.4.0 or later) included in this repository:

```bash
hf download kouhxp/gutsy --include "gutsy-0.8b-v04*" "gutsy-inference/*" --local-dir gutsy
pip install -e gutsy/gutsy-inference              # needs llama-cpp-python >= 0.3.35
cd gutsy/gutsy-inference && mkdir -p models && cp ../gutsy-0.8b-v04* models/
cp models.example.json models.json && gutsy-inference serve
```

`models.example.json` registers `gutsy-0.8b-v04` (Q8_0, the default) and `gutsy-0.8b-v04-q4`
(Q4_K_M). To use one file only, download and copy that one and delete the other entry.

The `models.json` entries:

```json
"gutsy-0.8b-v04":    {"gguf": "models/gutsy-0.8b-v04-q8_0.gguf",
                      "calibration": "models/gutsy-0.8b-v04.calibration.json",
                      "n_ctx": 8192, "reject_slot": true},
"gutsy-0.8b-v04-q4": {"gguf": "models/gutsy-0.8b-v04-q4_k_m.gguf",
                      "calibration": "models/gutsy-0.8b-v04-q4_k_m.calibration.json",
                      "n_ctx": 8192, "reject_slot": true, "cache_tol": 0.02}
```

`reject_slot` must be on. `cache_tol` (runtime 0.4.0) keeps state caching on for the Q4_K_M file:
4-bit weights round slightly differently when a prompt is evaluated in two pieces (measured
difference 0.009 in probability, every top answer the same), which the default check (0.001)
would reject, making every question re-encode the state. The Q8_0 file does not need it.

Shipped temperatures (fitted on held-out validation data with exact labels): Q8_0 choice 1.06,
yes/no 0.77, score 1.00; Q4_K_M choice 1.04, yes/no 0.79, score 1.00. Score is fixed at 1.00
because validation has too few score items to fit it.

The runtime also offers `order_ensemble` per model (off by default), which averages choice
answers over several option orders. It helped BBH and JudgeBench by about 4 points in our tests
but not JevBench, and it slows choice questions down.

## Training

| | |
|---|---|
| Base | Qwen/Qwen3.5-0.8B (Apache 2.0) |
| Method | full fine-tune, token embeddings frozen (tied output head therefore fixed); fp32 master weights, bf16 compute |
| Loss | cross-entropy + 0.5 x Brier over the option letters (and Z), against soft labels where available |
| Schedule | learning rate 1e-5 (cosine, 3% warm-up), 256 questions per step, 1 epoch over 105,114 questions (379 steps) |
| Order robustness | options reshuffled every time an example is seen |
| Checkpoint | lowest loss on a held-out development set (step 300; the end of training was also evaluated); never selected on any benchmark |
| Hardware | one 80 GB GPU, about 1 hour |
| Quantization | Q4_K_M made from the same checkpoint with llama.cpp's `llama-quantize`; its own temperatures |

### Data (counts from the build's stats.json)

| slice | questions | sources (license) |
|---|---|---|
| Teacher scenarios | 33,700 | written by Qwen3.8-27B (Apache 2.0) across 41 domains, with a blind second pass; kept only where both passes agreed (92.8% of questions); soft labels |
| Contrast pairs | 1,880 (687 pairs) | written and blind-reviewed by the same teacher; kept only if every question was answered as intended in both versions |
| Reading and verification | 16,074 | BoolQ (CC BY-SA 3.0), SNLI (CC BY-SA 4.0), SQuAD 2.0 (CC BY-SA 4.0), HotpotQA yes/no (CC BY-SA 4.0) |
| In-domain panel splits | 11,864 | RAGTruth train (MIT), WinoGrande train (see dataset card) |
| Classification and routing | 10,297 | MASSIVE (CC BY 4.0), CLINC150 (CC BY 3.0), Civil Comments (CC0, soft toxicity labels), twitter-financial-news-sentiment (MIT) |
| Code-generated reasoning puzzles | 6,892 | object tracking, ordering deduction, boolean logic, truth-teller chains, dates, probability; labels computed by code |
| Other code-generated data | 7,605 | policies, numbers and dates, records, score rubrics; labels computed by rule engines |
| Agent tool choice | 5,860 | Jev Decisions v1 (CC BY 4.0; derived from NVIDIA datasets, upstream terms apply) |
| Knowledge | 4,129 | ARC (CC BY-SA 4.0), OpenBookQA (Apache 2.0, verify), HellaSwag (MIT) |
| Judging | 3,844 | HelpSteer2 (CC BY 4.0) |
| Long documents | 2,969 | CUAD contract excerpts (CC BY 4.0) |

About 1,900 choice items are "reject" items (no listed option is correct; the target is Z). Unlike
v0.3, they no longer differ from normal items in option count: for intent lists the correct intent
is replaced by an unlisted one, and elsewhere normal items of the same template lose a wrong
option too. About 30% of choice questions use short numeric keys. Roughly half of the reading,
knowledge and judging records use the evaluation panel's request layouts (formats only).

**Shortcut audit.** Every build measures, per source and template, how well the answer can be
predicted without reading the state, from option count, option length, position, key style or the
label prior, and fails if a template beats chance. The final v0.4 training set has no flagged
template; one is accepted and reported (agent tool choice from real traces, where the tool with
the shortest description is rarely the one called).

**Never trained on:** JevBench, BBH, Financial PhraseBank, JudgeBench, the RAGTruth and
WinoGrande test/validation items, BFCL, tau-bench, CommonsenseQA, BANKING77. Training data was
decontaminated against the evaluation panel and JevBench public items (shared 13-word
sequences). **No closed-model output** is used anywhere.

## Evaluation

All results are self-run, on the Q8_0 reference file unless Q4_K_M is named. Other systems'
numbers are reported by their authors on possibly different harness versions and samples, so
comparisons are indicative.

**JevBench public set (231 items):** Q8_0 169 correct (0.732): easy 48/48, original 65/72, hard
56/111 (v0.3: 167). Q4_K_M 161 (0.697): easy 47, original 62, hard 52. Calibration error (Q8_0):
easy 0.022, original 0.107, hard 0.137 (Q4_K_M: 0.019 / 0.137 / 0.125). Ordinal MAE (Q8_0):
original 0.19, hard 0.69 (Q4_K_M: 0.21 / 0.67). Paraphrase agreement (original tier): 0.86
(Q4_K_M: 0.78). Reported by others on the same public set: Jev 0.866, decision-4b 0.883, Neriv
0.6B 0.636, XERON-0.4 0.558, tuned Laya 0.537.

**Held-out tests of the v0.4 changes:** contrast pairs (both versions right, 48 pairs) 0.417
(v0.3 0.104); held-out puzzle variants: depth-3 logic 0.842 (0.617), comparative orderings 0.947
(0.840), conditional probability 0.511 (0.305), calendar-month arithmetic 0.519 (0.683), reverse
object tracking 0.283 (0.266).

**Same items, measured side by side** (700 eval items each): BoolQ 0.837 (Jev 0.906), SNLI 0.870
(Jev 0.883), CommonsenseQA 0.577 (Jev 0.863; never trained on).

**Evaluation panel** (700 items; JudgeBench 245), Q8_0 / Q4_K_M: BBH 0.481 / 0.474, Financial
PhraseBank 0.927 / 0.926, JudgeBench 0.539 / 0.563, RAGTruth 0.813 / 0.807 (in-domain),
WinoGrande 0.620 / 0.629 (in-domain).

**Routing:** BANKING77 with all 77 intents, zero-shot: 0.613. Lists of 20-254 options: 0.991.

**Calibration:** validation calibration error 0.022 before temperature scaling; 0.032-0.053 on
BoolQ, SNLI and CommonsenseQA.

## Limitations

- **Arithmetic and dates:** JevBench temporal_numeric 0.27; held-out business-day, calendar-month
  and new-rulebook synthetic tasks are near chance, and the model is overconfident there.
- **Ambiguity:** much improved (JevBench ambiguous 4/7, was 0/7), but it still hedges in about a
  third of cases where the stated rules already decide the answer.
- **Judging close pairs:** JudgeBench 0.54 (Q8_0) and 0.56 (Q4_K_M), down from 0.60 in v0.3.
- **Knowledge** is limited at 0.8B (CommonsenseQA 0.58).
- **Calibration on hard, unfamiliar items** is weaker than v0.3's (hard-tier calibration error
  0.137 vs 0.078); out of distribution (BBH, WinoGrande) the best temperatures are well above 1.
- **Q4_K_M vs Q8_0:** the 4-bit file matches Q8_0 on the evaluation panel but scores 8 fewer
  JevBench items (161 vs 169), mostly on the hard tier. Use Q8_0 if those cases matter to you.
- **Long-list confidence** is too low on easy lookups when shortlisting is used.
- **Latency on CPU** grows with state length: median 4.3 s, 95th percentile 27 s on JevBench's
  hard tier (long policies, Q8_0) on the author's machine; Q4_K_M is about 25-30% faster. Short
  states take under a second, and follow-up questions on a cached state take a fraction of that.
- English only; context up to 8,192 tokens.

## Disclosures

- Benchmark awareness: the code-generated puzzles were designed with v0.3's weak spots on BBH and
  on JevBench's temporal_numeric and probability families in mind, and the contrast pairs with
  JevBench's ambiguous family in mind. No BBH or JevBench item, template or wording was used.
- In-domain data: RAGTruth and WinoGrande training splits; CLINC150 and MASSIVE training splits.
- Jev (typesafe/jev-1.13) was queried through its API only for side-by-side benchmarking; none of
  its output was used for training.

## License

Apache 2.0 for the weights and the included runtime code. Built on Qwen3.5-0.8B (Apache 2.0);
teacher-written data from Qwen3.8-27B (Apache 2.0). Training data sources and their licenses are
credited in the table above; anyone redistributing that data itself must follow each source's
terms.
