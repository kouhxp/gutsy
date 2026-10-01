---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
language: [en]
library_name: gguf
tags: [decision-model, classification, calibration, jev-compatible, llama.cpp, gguf]
---

# gutsy-0.8b v0.3

A decision model: given a **state** (text or JSON) and a typed **question** (yes/no, choice
among up to 16 options per call, or ordered score), it returns a calibrated probability for every
option from a single forward pass. It does not generate text. Fine-tuned from Qwen3.5-0.8B;
distributed as a Q8_0 GGUF (775 MB) for llama.cpp, served by `gutsy-inference`, which adds
state caching, shortlisting for lists up to 255 options, and calibrated temperatures.

## Intended use

Routing, triage, moderation, policy checks, tool selection in agent loops, judging outputs,
grounding checks: decisions about a state you provide, where you act on the probability. Use the
probabilities with thresholds measured on your own labeled data; route low-confidence and
high-stakes cases to a human.

**Out of scope / unreliable:** date and number arithmetic (compute in code first), applying new
multi-step rulebooks, deciding whether an ambiguity is decisive, world-knowledge questions,
non-English input. See Limitations.

## How to use

Prompt format (must match exactly; the runtime builds it): Qwen ChatML with thinking disabled;
the state, then the question, then options labeled A-P, then for choice and score questions the
line `Z. None of the options fits, or the state does not say`. The answer is read from the
next-token logits of the option letters (and Z), softmaxed over those letters only.

Easiest: the `gutsy-inference` runtime included in this repository:

```bash
hf download kouhxp/gutsy --local-dir gutsy
pip install -e gutsy/gutsy-inference              # needs llama-cpp-python >= 0.3.35
cd gutsy/gutsy-inference && mkdir -p models
cp ../gutsy-0.8b-v03-q8_0.gguf ../gutsy-0.8b-v03.calibration.json models/
cp models.example.json models.json && gutsy-inference serve
```

The `models.json` entry for this model:

```json
"gutsy-0.8b-v03": {"gguf": "models/gutsy-0.8b-v03-q8_0.gguf",
                      "calibration": "models/gutsy-0.8b-v03.calibration.json",
                      "n_ctx": 8192, "reject_slot": true}
```

`reject_slot` must be on for this model. Shipped temperatures (fitted on held-out validation data
with exact labels): choice 1.01, yes/no 0.86, score 1.00.

## Training

| | |
|---|---|
| Base | Qwen/Qwen3.5-0.8B (Apache 2.0) |
| Method | full fine-tune, token embeddings frozen (tied output head therefore fixed); fp32 master weights, bf16 compute |
| Loss | cross-entropy + 0.5 x Brier over the option letters (and Z), against soft labels where available |
| Schedule | learning rate 1e-5 (cosine, 3% warm-up), 256 questions per step, 1 epoch over ~90k questions |
| Order robustness | options reshuffled every time an example is seen |
| Checkpoint | lowest loss on a held-out development set (step 300, the last one evaluated); never selected on any benchmark |
| Hardware | one 48 GB GPU, about 2 hours |

A LoRA run on the same data (rank 32) tied this model exactly on JevBench's public set and stayed
within a few points on every other evaluation.

### Data (approximate counts; exact counts in the build's stats.json)

| slice | questions | sources (license) |
|---|---|---|
| Teacher scenarios | ~24,300 | written by Qwen3.8-27B (Apache 2.0) across 41 domains, with a blind second pass; kept only where both passes agreed; soft labels |
| Reading and verification | ~17,400 | BoolQ (CC BY-SA 3.0), SNLI (CC BY-SA 4.0), SQuAD 2.0 (CC BY-SA 4.0), HotpotQA yes/no (CC BY-SA 4.0) |
| Classification and routing | ~10,700 | MASSIVE (CC BY 4.0), CLINC150 (CC BY 3.0), Civil Comments (CC0, soft toxicity labels), twitter-financial-news-sentiment (MIT) |
| In-domain panel splits | ~10,300 | RAGTruth train (MIT), WinoGrande train (see dataset card) |
| Code-generated synthetic | ~8,300 | policies, numbers and dates, records, score rubrics; labels computed by rule engines |
| Agent tool choice | ~6,600 | Jev Decisions v1 (CC BY 4.0; derived from NVIDIA datasets, upstream terms apply) |
| Judging | ~4,900 | HelpSteer2 (CC BY 4.0) |
| Knowledge | ~4,100 | ARC (CC BY-SA 4.0), OpenBookQA (Apache 2.0, verify), HellaSwag (MIT) |
| Long documents | ~4,100 | CUAD contract excerpts (CC BY 4.0) |

About 8% of eligible choice items are "reject" items (the correct option removed; the target is
Z). About 30% of choice questions use short numeric keys. Roughly half of the reading,
knowledge and judging records use the evaluation panel's request layouts (formats only).

**Never trained on:** JevBench, BBH, Financial PhraseBank, JudgeBench, the RAGTruth and
WinoGrande test/validation items, BFCL, tau-bench, CommonsenseQA, BANKING77. Training data was
decontaminated against the evaluation panel and JevBench public items (shared 13-word
sequences). **No closed-model output** is used anywhere.

## Evaluation

All results are self-run. Other systems' numbers are reported by their authors on possibly
different harness versions and samples, so comparisons are indicative.

**JevBench public set (231 items):** 167 correct (0.723): easy 47/48, original 61/72, hard
59/111. Calibration error: easy 0.023, original 0.122, hard 0.078. Ordinal MAE: original 0.30,
hard 0.59. Paraphrase agreement (original tier): 0.81. Reported by others on the same public
set: Jev 0.866, decision-4b 0.883, Neriv 0.6B 0.636, XERON-0.4 0.558, tuned Laya 0.537.

**Same items, measured side by side** (700 eval items each): BoolQ 0.850 (Jev 0.906), SNLI 0.879
(Jev 0.883), CommonsenseQA 0.587 (Jev 0.863; never trained on).

**Evaluation panel** (700 items; JudgeBench 245): BBH 0.464, Financial PhraseBank 0.921,
JudgeBench 0.596, RAGTruth 0.790 (in-domain), WinoGrande 0.657 (in-domain).

**Routing:** BANKING77 with all 77 intents, zero-shot: 0.609. Lists of 20-254 options: 0.999.

**Calibration:** validation calibration error 0.021 before temperature scaling; calibration
error 0.010-0.052 on BoolQ, SNLI and CommonsenseQA.

## Limitations

- **Arithmetic and dates:** JevBench temporal_numeric 0.33; held-out business-day and new-rulebook
  synthetic tasks are near chance, and the model is overconfident there.
- **Ambiguity:** JevBench ambiguous 0/7: it abstains where the stated rules decide the case and
  commits where it should ask or abstain.
- **Knowledge** is limited at 0.8B (CommonsenseQA 0.59).
- **Long-list confidence** is too low on easy lookups when shortlisting is used.
- **Latency on CPU** grows with state length: median 4.2 s, 95th percentile 23 s on JevBench's
  hard tier (long policies) on the author's machine; short states take under a second, and
  follow-up questions on a cached state take a fraction of that.
- **Calibration out of distribution** is weaker than on validation (BBH and WinoGrande want
  temperatures near 2).
- English only; context up to 8,192 tokens.

## Disclosures

- Benchmark awareness: the data design for v0.3 was informed by the family-level JevBench public
  results of earlier internal versions. No JevBench item was trained on.
- In-domain data: RAGTruth and WinoGrande training splits; CLINC150 and MASSIVE training splits.
- Jev (typesafe/jev-1.13) was queried through its API only for side-by-side benchmarking; none of
  its output was used for training.

## License

Apache 2.0 for the weights and the included runtime code. Built on Qwen3.5-0.8B (Apache 2.0);
teacher-written data from Qwen3.8-27B (Apache 2.0). Training data sources and their licenses are
credited in the table above; anyone redistributing that data itself must follow each source's
terms.
