"""Per-segment line exactness (pairs + amounts, like evalutil 'exact') for the PDF segments.
usage: python matching/tests/pdf/seg_report.py <phase> <allocations.parquet> [more.parquet ...]"""
import sys
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from matching.lib import data

SEGS = ["ar_instalment", "ar_retention", "ar_retention_release", "ar_advance_deduction", "ar_replaced_invoice", "ar_new_iban"]


def seg_table(alloc: pd.DataFrame, phase: str, segs=SEGS) -> pd.DataFrame:
    g = data.golden(phase)
    g = g[g.segment.isin(segs)]
    gi = g[g.item_id.notna()]
    a = alloc[alloc.bank_txn_id.isin(g.bank_txn_id)].drop_duplicates(["bank_txn_id", "item_id"])
    m = gi[["bank_txn_id", "item_id", "amount", "exception", "segment"]].merge(
        a[["bank_txn_id", "item_id", "amount", "exception"]], on=["bank_txn_id", "item_id"], how="outer", suffixes=("_g", "_p"), indicator=True)
    m["segment"] = m.bank_txn_id.map(g.groupby("bank_txn_id").segment.first())
    tol = np.maximum(0.01, 0.005 * m.amount_g.abs().fillna(0))
    m["ok"] = (m._merge == "both") & ((m.amount_p - m.amount_g).abs() <= tol)
    m["eok"] = m.ok & (m.exception_g.fillna("") == m.exception_p.fillna(""))
    ln = m.groupby("bank_txn_id").agg(seg=("segment", "first"), ok=("ok", "all"), eok=("eok", "all"),
                                       claimed=("amount_p", lambda s: s.notna().any()))
    t = ln.groupby("seg").agg(lines=("ok", "size"), claimed=("claimed", "mean"), exact=("ok", "mean"), exact_exc=("eok", "mean"))
    return t.round(3)


if __name__ == "__main__":
    ph = sys.argv[1]
    a = pd.concat([pd.read_parquet(p) for p in sys.argv[2:]], ignore_index=True)
    print(seg_table(a, ph).to_string())
