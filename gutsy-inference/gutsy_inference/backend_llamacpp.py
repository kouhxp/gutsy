"""llama.cpp backend (via llama-cpp-python): tokenize, evaluate, read label logits, and
snapshot/restore one sequence's state (KV cache + recurrent state) through llama.cpp's
llama_state_seq_* API."""
import ctypes
import os

import numpy as np

from .engine import StateError
from .prompt import LABELS, REJECT_LABEL


def default_threads():
    # Physical cores are usually best for llama.cpp; on SMT machines that's ~half the logical count.
    return max(1, (os.cpu_count() or 2) // 2)


class LlamaCppBackend:
    supports_state = True

    def __init__(self, gguf, n_ctx=8192, n_threads=None, n_gpu_layers=0, n_batch=512, seed=0):
        import llama_cpp
        from llama_cpp import Llama
        self._lib = llama_cpp
        threads = n_threads or default_threads()
        self.llm = Llama(model_path=str(gguf), n_ctx=n_ctx, n_threads=threads,
                         n_threads_batch=threads, n_gpu_layers=n_gpu_layers, n_batch=n_batch,
                         seed=seed, logits_all=False, verbose=False)
        self.n_ctx = self.llm.n_ctx()
        self.n_vocab = self.llm.n_vocab()
        self.n_threads = threads

    def _ctx(self):
        ctx = getattr(self.llm, "ctx", None)
        return ctx if ctx is not None else self.llm._ctx.ctx

    def label_ids(self):
        """Token ids for A..P, then the reject label Z (always returned; used only by models
        configured with reject_slot)."""
        ids = []
        for lab in LABELS + [REJECT_LABEL]:
            t = self.llm.tokenize(lab.encode(), add_bos=False, special=False)
            if len(t) != 1:
                raise ValueError(f"option label {lab!r} is not a single token: {t}")
            ids.append(t[0])
        return ids

    def tokenize(self, text, bos):
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=bos, special=True))

    def reset(self):
        self.llm.reset()

    def eval(self, tokens, start):
        """Evaluate `tokens` at positions start..; the context must hold exactly `start` tokens."""
        if start == 0:
            self.llm.reset()
        else:
            self.llm.n_tokens = start
        self.llm.eval(tokens)

    def label_logits(self, ids):
        ptr = self._lib.llama_get_logits_ith(self._ctx(), -1)
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))[ids].astype(np.float64)

    def save_seq(self):
        lib, ctx = self._lib, self._ctx()
        try:
            size = lib.llama_state_seq_get_size(ctx, 0)
            buf = (ctypes.c_uint8 * size)()
            n = lib.llama_state_seq_get_data(ctx, buf, size, 0)
        except Exception as e:
            raise StateError(f"state snapshot failed: {e}") from e
        if not n:
            raise StateError("state snapshot returned 0 bytes")
        return (buf, n), n

    def load_seq(self, blob, tokens):
        buf, n = blob
        try:
            ok = self._lib.llama_state_seq_set_data(self._ctx(), buf, n, 0)
        except Exception as e:
            raise StateError(f"state restore failed: {e}") from e
        if not ok:
            raise StateError("state restore failed")
        self.llm.input_ids[: len(tokens)] = tokens
        self.llm.n_tokens = len(tokens)

    def close(self):
        try:
            self.llm.close()
        except Exception:
            pass
