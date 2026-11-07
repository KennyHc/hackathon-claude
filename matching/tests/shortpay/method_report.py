"""Pair precision per method (and the golden segment / exception of the claimed lines). Measurement only, same tolerance
as evalutil / the official scorer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from matching.lib import data


def by_method(alloc: pd.DataFrame, phase: str, show_seg: bool = True) -> pd.DataFrame:
    g = data.golden(phase)
    gi = g[g.item_id.notna()][["bank_txn_id", "item_id", "amount", "exception"]].rename(columns={"amount": "gamt", "exception": "gexc"})
    seg = g.groupby("bank_txn_id").segment.first()
    m = alloc.merge(gi, on=["bank_txn_id", "item_id"], how="left")
    m["ok"] = m.gamt.notna() & ((m.amount - m.gamt).abs() <= np.maximum(0.01, 0.005 * m.gamt.abs().fillna(0)))
    m["seg"] = m.bank_txn_id.map(seg)
    t = m.groupby("method").agg(lines=("bank_txn_id", "nunique"), pairs=("ok", "size"), tp=("ok", "sum"), conf=("confidence", "mean"))
    t["prec"] = (t.tp / t.pairs).round(4)
    print(t.sort_values("pairs", ascending=False).to_string())
    if show_seg:
        print(m.groupby(["method", "seg"]).ok.agg(["size", "mean"]).round(3).to_string())
    return t
