"""Pair precision of every shortpay method INCLUDING the weak ones (below CONF_MIN), over all val lines, and what the
golden says the wrongly claimed lines really are.

    python matching/tests/shortpay/weak_precision.py [--alloc matching/out/val/allocations.parquet]

Diagnostic only (val golden); nothing here feeds the matchers.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from matching.lib import data  # noqa: E402
from matching.matchers import shortpay as sp  # noqa: E402

PAT = r"^(fx_)?(ref_short_wht|ref_short_disc|ref_over|ref_chg|wht_|disc_|over_|chg_|cn_paid_out|dup_)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", default=os.path.join("matching", "out", "val", "allocations.parquet"))
    ap.add_argument("--methods", default="over|chg", help="regex of methods to break down")
    a = ap.parse_args()
    phase = "val"
    al = pd.read_parquet(a.alloc)
    # everything claimed by shortpay/duplicates is given back, so shortpay sees the same lines as in the pipeline
    keep = al[~al.method.fillna("").str.match(PAT) & ~al.method.fillna("").str.startswith("fx_")]
    pi = pd.read_parquet(os.path.join("classification", "out", phase, "line_labels.parquet"),
                         columns=["bank_txn_id", "p_items"]).set_index("bank_txn_id").p_items
    out = sp.match(phase, set(keep.bank_txn_id), prior=keep, p_items=pi, keep_weak=True)
    out.to_parquet(os.path.join("matching", "out", phase, "shortpay_weak.parquet"), index=False)

    g = data.golden(phase)
    gi = g[g.item_id.notna()][["bank_txn_id", "item_id", "amount", "exception"]].rename(
        columns={"amount": "gamt", "exception": "gexc"})
    line = g.groupby("bank_txn_id").agg(seg=("segment", "first"), gcat=("category", "first"),
                                        gexc_line=("exception", lambda s: ",".join(sorted({str(x) for x in s if x == x}))))
    m = out.merge(gi, on=["bank_txn_id", "item_id"], how="left")
    m["item_ok"] = m.gamt.notna()
    m["ok"] = m.item_ok & ((m.amount - m.gamt).abs() <= np.maximum(0.01, 0.005 * m.gamt.abs().fillna(0)))
    m = m.join(line, on="bank_txn_id")
    t = m.groupby("method").agg(lines=("bank_txn_id", "nunique"), pairs=("ok", "size"), item_ok=("item_ok", "mean"),
                                prec=("ok", "mean"), conf=("confidence", "first"))
    pd.set_option("display.width", 220)
    print(t.round(3).sort_values("pairs", ascending=False).to_string())
    sel = m[m.method.str.contains(a.methods) & ~m.ok]
    print(f"\nwrong pairs of methods ~ /{a.methods}/: what the golden says those lines are")
    print(sel.groupby(["method", "seg", "gexc"], dropna=False).size().sort_values(ascending=False).head(40).to_string())


if __name__ == "__main__":
    main()
