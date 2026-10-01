import json

from gutsy_inference.prompt import build_prompt, prefix, suffix


def training_build_prompt(state, question, options, reject=False):
    """VERBATIM copy of gutsy-bench gutsy_bench/backends/llamacpp.py::build_prompt, the
    format the models were trained on. If this test fails, the runtime no longer matches training."""
    LABELS = [chr(ord("A") + i) for i in range(16)]
    REJECT_LINE = "Z. None of the options fits, or the state does not say"
    SYSTEM = ("You are a decision model. Read the state, then answer the question by choosing exactly "
              "one of the options. Reply with only the letter of your choice.")
    state_txt = state.strip() if state and state.strip() else "(none)"
    opts = "\n".join(f"{LABELS[i]}. {o}" for i, o in enumerate(options))
    if reject:
        opts += "\n" + REJECT_LINE
    return ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
            "<|im_start|>user\nState:\n" + state_txt +
            "\n\nQuestion: " + question.strip() +
            "\n\nOptions:\n" + opts +
            "\n\nAnswer with a single letter.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_matches_training_format():
    cases = [("A dog runs.", "Is it moving?", ["Yes", "No"]),
             ("", "Pick one", ["red", "green", "blue"]),
             ("   ", "Q", ["x", "y"]),
             ("  padded state \n", "  padded question  ", ["a", "b"])]
    for s, q, o in cases:
        assert build_prompt(s, q, o) == training_build_prompt(s, q, o)


def test_json_state_matches_training_serialization():
    state = {"document": "Café prices rose 5%.", "n": [1, 2]}
    # training data serialized JSON states with json.dumps(..., ensure_ascii=False)
    assert build_prompt(state, "Q?", ["Yes", "No"]) == \
        training_build_prompt(json.dumps(state, ensure_ascii=False), "Q?", ["Yes", "No"])


def test_prefix_suffix_split():
    assert prefix("S") + suffix("Q", ["a", "b"]) == build_prompt("S", "Q", ["a", "b"])
    assert prefix("S").endswith("\n\n") and suffix("Q", ["a"]).startswith("Question: ")


def test_reject_variant_matches_training_format():
    for s, q, o in (("A dog runs.", "Which animal?", ["dog", "cat", "cow"]), ("", "Pick", ["a", "b"])):
        assert build_prompt(s, q, o, reject=True) == training_build_prompt(s, q, o, reject=True)
    p = build_prompt("S", "Q", ["x", "y"], reject=True)
    assert "B. y\nZ. None of the options fits, or the state does not say\n\nAnswer" in p
