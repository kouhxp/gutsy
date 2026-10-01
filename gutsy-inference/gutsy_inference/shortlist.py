"""Answer choice questions with more options than one model call can hold (up to 255).

Every model call sees at most `size` options (the trained per-call limit, 16), so the option
letters and the reject slot keep working unchanged. For a longer list:

  1. split it into balanced chunks of at most `size`, in the order given, and score each chunk;
  2. keep each chunk's best options until they cover KEEP_MASS of its probability (at most
     KEEP_MAX), and score the finalists together, recursively if there are still too many;
  3. combine: chunk c gets weight w_c proportional to (final-round mass of its finalists) / (the
     share of the chunk they covered, R_c). Inside the chunk, finalists split w_c * R_c in
     proportion to their final-round probabilities, and eliminated options split w_c * (1 - R_c)
     by their first-round probabilities.

Consequences: finalists' probabilities are exactly proportional to the final round (the head-to-
head comparison), every chunk's total equals its weight, and the result is one distribution over
all options that sums to 1. The reject probability, if any, comes from the deciding (last) round.
"""
import math

import numpy as np

KEEP_MASS = 0.9
KEEP_MAX = 3


def chunks(idx, size):
    k = math.ceil(len(idx) / size)
    base, extra = divmod(len(idx), k)
    out, s = [], 0
    for c in range(k):
        n = base + (1 if c < extra else 0)
        out.append(idx[s:s + n])
        s += n
    return out


def keep(p, keep_mass=KEEP_MASS, keep_max=KEEP_MAX):
    """Indices of the options a chunk passes on. Always drops at least one option (for chunks of
    two or more), so every round shrinks the list and the recursion ends."""
    keep_max = min(keep_max, max(1, len(p) - 1))
    order = np.argsort(-np.asarray(p), kind="stable")
    kept, mass = [], 0.0
    for j in order:
        kept.append(int(j))
        mass += float(p[j])
        if mass >= keep_mass or len(kept) >= keep_max:
            break
    return kept


def rank(idx, score, size, stats=None, depth=1):
    """idx: option indices to rank. score(indices) -> (probs aligned with indices, p_reject|None).
    Returns (probs aligned with idx, p_reject of the deciding round)."""
    stats = stats if stats is not None else {"calls": 0, "rounds": 0}
    stats["rounds"] = max(stats["rounds"], depth)
    if len(idx) <= size:
        stats["calls"] += 1
        p, rej = score(list(idx))
        return np.asarray(p, dtype=np.float64), rej
    groups = chunks(list(idx), size)
    first, finals, covered = {}, [], []
    for g in groups:
        p, _ = score(g)
        stats["calls"] += 1
        p = np.asarray(p, dtype=np.float64)
        first.update(zip(g, p))
        k = keep(p)
        finals.append([g[j] for j in k])
        covered.append(max(1e-12, float(sum(p[j] for j in k))))
    finalists = [i for f in finals for i in f]
    if len(finalists) >= len(idx):          # cannot happen with keep()'s rule; guard against loops
        finals = [f[:1] for f in finals]
        covered = [max(1e-12, float(first[f[0]])) for f in finals]
        finalists = [f[0] for f in finals]
    pf_vec, rej = rank(finalists, score, size, stats, depth + 1)
    pf = dict(zip(finalists, pf_vec))
    w = np.array([sum(pf[i] for i in f) / r for f, r in zip(finals, covered)])
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
    out = {}
    for g, f, r, wc in zip(groups, finals, covered, w):
        mf = sum(pf[i] for i in f)
        fset = set(f)
        for i in g:
            out[i] = (wc * r * (pf[i] / mf if mf > 0 else 1.0 / len(f)) if i in fset
                      else wc * first[i])
    vec = np.array([out[i] for i in idx])
    return vec / vec.sum(), rej
