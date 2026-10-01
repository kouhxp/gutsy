import hashlib

import numpy as np

from gutsy_inference.engine import StateError


class FakeBackend:
    """Character-level fake with recurrent-model semantics: eval() must start exactly where the
    context ends (no rollback), so the engine has to use snapshots to branch."""
    supports_state = True

    def __init__(self, n_ctx=4096, misaligned=False, broken_restore=False):
        self.n_ctx, self.misaligned, self.broken_restore = n_ctx, misaligned, broken_restore
        self.mem, self.evaluated = [], 0

    def label_ids(self):
        return list(range(17))          # A..P, then the reject label Z

    def tokenize(self, text, bos):
        toks = [ord(c) for c in text]
        if self.misaligned and "\n\nQuestion" in text:   # simulate a merge across the boundary
            i = text.index("\n\nQuestion")
            toks = toks[:i] + [900000] + toks[i + 2:]
        return toks

    def reset(self):
        self.mem = []

    def eval(self, tokens, start):
        if start == 0:
            self.mem = []
        assert start == len(self.mem), "eval would require rolling back recurrent state"
        self.mem = self.mem + list(tokens)
        self.evaluated += len(tokens)

    def label_logits(self, ids):
        h = hashlib.sha256(repr(self.mem).encode()).digest()
        return (np.frombuffer(h, dtype=np.uint8)[:17].astype(np.float64) / 40.0)[ids]

    def save_seq(self):
        return list(self.mem), 4 * len(self.mem)

    def load_seq(self, blob, tokens):
        if self.broken_restore:
            raise StateError("restore not supported")
        assert blob == list(tokens)
        self.mem = list(blob)
