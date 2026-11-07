"""Quick eval of a batch allocation frame against the val golden: per segment (golden segment of the line)
lines in golden, lines claimed, line-exact (items + amounts within tolerance), with exceptions, pair precision."""
import sys

import numpy as np
import pandas as pd

from matching.batch.common import phase_dir

_G = {}


def golden(phase):
    if phase not in _G:
        g = pd.read_csv(f"{phase_dir(phase)}/golden_allocations.csv.gz", dtype=str, keep_default_na=False)
        g = g[g.item_id != ""].copy()
        g["amount"] = g.amount.astype(float)
        _G[phase] = g
    return _G[phase]


def evaluate(alloc, phase="val", by="method", show=25):
    g = golden(phase)
    gl = pd.read_csv(f"{phase_dir(phase)}/golden_allocations.csv.gz", dtype=str, keep_default_na=False,
                     usecols=["bank_txn_id", "segment"]).drop_duplicates("bank_txn_id")
    seg = dict(zip(gl.bank_txn_id, gl.segment))
    a = alloc.copy()
    a["amount"] = a.amount.astype(float)
    m = a.merge(g[["bank_txn_id", "item_id", "amount", "exception"]], on=["bank_txn_id", "item_id"], how="left",
                suffixes=("", "_g"))
    m["tp"] = (m.amount_g.notna() & ((m.amount - m.amount_g).abs() <= np.maximum(0.01, 0.005 * m.amount_g.abs())))
    m["tpx"] = m.tp & (m.exception.fillna("") == m.exception_g.fillna(""))
    gn = g.groupby("bank_txn_id").size()
    per = m.groupby("bank_txn_id").agg(n=("tp", "size"), tp=("tp", "sum"), tpx=("tpx", "sum"),
                                       meth=(by, "first")).reset_index()
    per["gn"] = per.bank_txn_id.map(gn).fillna(0)
    per["exact"] = (per.tp == per.n) & (per.n == per.gn)
    per["exactx"] = (per.tpx == per.n) & (per.n == per.gn)
    per["seg"] = per.bank_txn_id.map(seg).fillna("?")
    allseg = pd.Series(seg).value_counts()
    t = per.groupby("seg").agg(claimed=("n", "size"), exact=("exact", "sum"), exactx=("exactx", "sum"),
                               pairs=("n", "sum"), tp=("tp", "sum"))
    t["gold_lines"] = allseg.reindex(t.index)
    t["prec"] = (t.tp / t.pairs).round(3)
    t["acc_of_seg"] = (t.exactx / t.gold_lines).round(3)
    t = t.sort_values("pairs", ascending=False)
    print(t.head(show).to_string())
    mt = per.groupby("meth").agg(claimed=("n", "size"), exact=("exact", "sum"), exactx=("exactx", "sum"),
                                 pairs=("n", "sum"), tp=("tp", "sum"))
    mt["prec"] = (mt.tp / mt.pairs).round(4)
    print(mt.to_string())
    P = m.tp.sum() / max(1, len(m))
    print(f"TOTAL lines {len(per):,}  exact {per.exact.sum():,}  exact+exc {per.exactx.sum():,}  pairs {len(m):,}  "
          f"pair precision {P:.4f}  pair recall {m.tp.sum() / len(g):.4f}")
    return per
