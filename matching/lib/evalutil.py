"""Score a matcher's output against golden, per segment. Used by every matcher to measure itself on `val` (or `sample`).

    from matching.lib import data, evalutil
    evalutil.report(alloc, "val", name="batches")      # alloc: DataFrame[bank_txn_id,item_id,amount,(exception)]

Definitions (same tolerance as the official scorer: max(0.01, 0.5% of golden amount)):
  pair precision  = correct pairs / predicted pairs               (only over the lines the matcher claims)
  pair recall     = correct pairs / golden pairs of ALL lines in the segment (so unclaimed lines count as missed)
  line exact      = claimed lines whose predicted pair set equals golden's, over claimed lines
  exc agree       = among correct pairs, share whose exception equals golden's (null == empty)
Never tune on the val golden: use it to measure. Learn only from train before 2025-01-01 (CLAUDE.md 2b).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data


def _tol(g):
    return np.maximum(0.01, 0.005 * g.abs())


def report(alloc: pd.DataFrame, phase: str, name: str = "", top: int = 12, quiet: bool = False) -> dict:
    g = data.golden(phase)
    gi = g[g.item_id.notna()][["bank_txn_id", "item_id", "amount", "exception", "segment"]].rename(columns={"amount": "gamt", "exception": "gexc"})
    seg_of = g.groupby("bank_txn_id").segment.first()
    ng_all = gi.groupby("segment").size()
    a = alloc.drop_duplicates(["bank_txn_id", "item_id"]).copy()
    if "exception" not in a:
        a["exception"] = None
    m = a.merge(gi, on=["bank_txn_id", "item_id"], how="left")
    m["ok"] = m.gamt.notna() & ((m.amount - m.gamt).abs() <= _tol(m.gamt.fillna(0)))
    m["segment"] = m.bank_txn_id.map(seg_of)
    m["excok"] = m.ok & (m.exception.fillna("") == m.gexc.fillna(""))
    ln = m.groupby("bank_txn_id").agg(n=("ok", "size"), tp=("ok", "sum"), seg=("segment", "first"))
    gl = gi.groupby("bank_txn_id").size()
    ln["ng"] = ln.index.map(gl)
    ln["exact"] = (ln.tp == ln.n) & (ln.n == ln.ng)
    tot_g = int(len(gi))
    tp = int(m.ok.sum()); n = int(len(m))
    P = tp / n if n else float("nan"); R = tp / tot_g
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    out = {"pairs_pred": n, "pairs_tp": tp, "pairs_gold": tot_g, "precision": P, "recall": R, "f1": F1,
           "lines": len(ln), "exact_lines": int(ln.exact.sum()), "exc_agree": float(m[m.ok].excok.mean()) if tp else float("nan")}
    if not quiet:
        print(f"== {name or 'matcher'} on {phase}: claims {len(ln)} lines / {n} pairs | precision {P:.2%} recall(all gold pairs) {R:.2%} F1 {F1:.3f} "
              f"| exact lines {out['exact_lines']}/{len(ln)} ({ln.exact.mean():.1%}) | exception agree on correct pairs {out['exc_agree']:.1%}")
        s = m.groupby("segment").agg(pred=("ok", "size"), tp=("ok", "sum"), excok=("excok", "sum"))
        s["gold"] = ng_all
        s["prec"] = (s.tp / s.pred).round(3); s["rec"] = (s.tp / s.gold).round(3)
        s["lines_exact"] = ln.groupby("seg").exact.mean().round(3)
        print(s.sort_values("gold", ascending=False).head(top).to_string())
    return out
