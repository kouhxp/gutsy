"""Prompt format. MUST stay byte-identical to the format the models were trained with
(gutsy-bench: gutsy_bench/backends/llamacpp.py::build_prompt). tests/test_prompt.py pins it.

The prompt is split into a state PREFIX (encoded once per state, then cached) and a per-question
SUFFIX. The split sits right before "Question:", a point where Qwen's tokenizer always starts a
new token, so prefix tokens + suffix tokens == tokens of the whole prompt. The engine still
verifies that on every request and falls back to full re-encoding if it ever doesn't hold.
"""
import json

SYSTEM = ("You are a decision model. Read the state, then answer the question by choosing exactly "
          "one of the options. Reply with only the letter of your choice.")
LABELS = [chr(ord("A") + i) for i in range(16)]
MAX_OPTIONS = len(LABELS)
# Reject slot, for models trained with it (models.json "reject_slot": true): one extra answer after
# the options of choice and score questions. Must match gutsy-bench's REJECT_LINE exactly.
REJECT_LABEL = "Z"
REJECT_LINE = "Z. None of the options fits, or the state does not say"


def state_text(state):
    s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    s = s.strip()
    return s if s else "(none)"


def prefix(state):
    return ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
            "<|im_start|>user\nState:\n" + state_text(state) + "\n\n")


def suffix(question, options, reject=False):
    opts = "\n".join(f"{LABELS[i]}. {o}" for i, o in enumerate(options))
    if reject:
        opts += "\n" + REJECT_LINE
    return ("Question: " + question.strip() +
            "\n\nOptions:\n" + opts +
            "\n\nAnswer with a single letter.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def build_prompt(state, question, options, reject=False):
    return prefix(state) + suffix(question, options, reject)
