"""Bounded subset-sum used to pick the items a payment settles (owner: refs matcher).

    sols = subset_solutions(cents, target_cents, limit=64)           # list of index tuples, exact sums, signed values allowed
    pick = choose(items_sorted, target_cents, forced=(...))          # canonical / unique selection with a status

`cents` are integers (credit notes negative).  The search is meet-in-the-middle (2^(n/2) per half) so n <= MAXN (36) is cheap;
larger pools must be reduced by the caller (`reduce_pool`).
"""
from __future__ import annotations

import numpy as np

MAXN = 28


def _all_sums(vals: np.ndarray) -> np.ndarray:
    s = np.zeros(1, dtype=np.int64)
    for v in vals:
        s = np.concatenate([s, s + v])
    return s


def subset_solutions(cents, target: int, limit: int = 64, tol: int = 0) -> list[tuple]:
    """All index subsets whose values sum to `target` (+- tol cents), up to `limit` (empty subset excluded)."""
    v = np.asarray(cents, dtype=np.int64)
    n = len(v)
    if n == 0 or n > MAXN:
        return []
    h = n // 2
    L = _all_sums(v[:h])
    R = _all_sums(v[h:])
    order = np.argsort(R, kind="stable")
    Rs = R[order]
    out: list[tuple] = []
    lo_t, hi_t = target - tol, target + tol
    lo = np.searchsorted(Rs, lo_t - L, side="left")
    hi = np.searchsorted(Rs, hi_t - L, side="right")
    cnt = hi - lo
    for li in np.nonzero(cnt > 0)[0]:
        for k in range(lo[li], hi[li]):
            ri = order[k]
            if li == 0 and ri == 0:
                continue
            idx = tuple([i for i in range(h) if (li >> i) & 1] + [h + i for i in range(n - h) if (ri >> i) & 1])
            out.append(idx)
            if len(out) >= limit:
                return out
    return out
