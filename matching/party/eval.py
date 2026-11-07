#!/usr/bin/env python3
"""Per-segment diagnostics of alloc_party.parquet against the val golden (scoring only, never for learning).

    .venv/bin/python matching/party/eval.py --phase val [--file matching/out/val/alloc_party.parquet]

Per segment: lines, claimed, line exact (items+amounts, ignoring exceptions), line exact incl. exception, pair P/R.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K3 = ["ar_ref_noise", "ap_paid_by_holding", "ar_short_discount", "ar_same_amount_fifo", "ap_partial",
      "ar_same_amount_month", "ar_partial", "ar_short_withholding", "ar_discount_late", "ar_third_party_confirming",
      "ar_third_party_agency", "ar_third_party_parent", "ar_new_iban", "ar_overpayment", "ar_duplicate_payment",
      "unidentified_lookalike"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="val")
    ap.add_argument("--file", default=None)
    ap.add_argument("--all", action="store_true", help="all segments, not only K3's")
    a = ap.parse_args()
    f = a.file or os.path.join(ROOT, "matching", "out", a.phase, "alloc_party.parquet")
    s = pd.read_parquet(f)
    g = pd.read_csv(os.path.join(ROOT, "data", a.phase, "golden_allocations.csv.gz"), dtype=str,
                    keep_default_na=False)
    seg = g.drop_duplicates("bank_txn_id").set_index("bank_txn_id").segment
    gi = g[g.item_id != ""].copy()
    gi["amount"] = gi.amount.astype(float)
    s = s.copy()
    s["amount"] = s.amount.astype(float)
    s["exception"] = s.exception.fillna("")
    m = s.merge(gi[["bank_txn_id", "item_id", "amount", "exception"]], on=["bank_txn_id", "item_id"], how="left",
                suffixes=("", "_g"))
    m["ok"] = (m.amount - m.amount_g).abs() <= np.maximum(0.01, 0.005 * m.amount_g.abs())
    m["ok_e"] = m.ok & (m.exception == m.exception_g)
    tp = m.groupby("bank_txn_id").ok.sum()
    tpe = m.groupby("bank_txn_id").ok_e.sum()
    ns = s.groupby("bank_txn_id").size()
    ng = gi.groupby("bank_txn_id").size()
    lines = pd.DataFrame({"segment": seg})
    lines["ng"] = ng.reindex(lines.index).fillna(0)
    lines["ns"] = ns.reindex(lines.index).fillna(0)
    lines["tp"] = tp.reindex(lines.index).fillna(0)
    lines["tpe"] = tpe.reindex(lines.index).fillna(0)
    lines["claimed"] = lines.ns > 0
    lines["exact"] = lines.claimed & (lines.tp == lines.ng) & (lines.ns == lines.ng)
    lines["exact_e"] = lines.exact & (lines.tpe == lines.ng)
    agg = lines.groupby("segment").agg(n=("ng", "size"), claimed=("claimed", "mean"), exact=("exact", "mean"),
                                       exact_e=("exact_e", "mean"), ns=("ns", "sum"), tp=("tp", "sum"),
                                       ng=("ng", "sum"))
    agg["prec_line"] = lines[lines.claimed].groupby("segment").exact.mean()
    agg["pairP"] = agg.tp / agg.ns.replace(0, np.nan)
    agg["pairR"] = agg.tp / agg.ng.replace(0, np.nan)
    agg = agg.drop(columns=["ns", "tp", "ng"])
    show = agg if a.all else agg[agg.index.isin(K3)]
    other = agg[~agg.index.isin(K3)]
    pd.set_option("display.width", 200)
    print(show.sort_values("n", ascending=False).round(3).to_string())
    cl = lines[lines.claimed]
    print(f"\nclaimed {len(cl):,} lines: exact {cl.exact.mean():.3f} (exact+exc {cl.exact_e.mean():.3f}); "
          f"pairs P {lines.tp.sum() / max(lines.ns.sum(), 1):.3f}; claimed outside K3 segments: "
          f"{int(cl[~cl.segment.isin(K3)].shape[0]):,} (exact {cl[~cl.segment.isin(K3)].exact.mean():.3f})")
    if not a.all:
        o = cl[~cl.segment.isin(K3)].groupby("segment").agg(n=("exact", "size"), exact=("exact", "mean"))
        print("top non-K3 segments claimed:", o.sort_values("n", ascending=False).head(12).round(2).to_dict("index"))


if __name__ == "__main__":
    main()


def by_method(phase="val", f=None):
    f = f or os.path.join(ROOT, "matching", "out", phase, "alloc_party.parquet")
    s = pd.read_parquet(f)
    g = pd.read_csv(os.path.join(ROOT, "data", phase, "golden_allocations.csv.gz"), dtype=str, keep_default_na=False)
    gi = g[g.item_id != ""].copy()
    gi["amount"] = gi.amount.astype(float)
    m = s.merge(gi[["bank_txn_id", "item_id", "amount", "exception"]], on=["bank_txn_id", "item_id"], how="left",
                suffixes=("", "_g"))
    m["ok"] = (m.amount - m.amount_g).abs() <= np.maximum(0.01, 0.005 * m.amount_g.abs())
    ng = gi.groupby("bank_txn_id").size()
    ln = m.groupby("bank_txn_id").agg(method=("method", "first"), ok=("ok", "sum"), ns=("ok", "size"))
    ln["ng"] = ng.reindex(ln.index).fillna(0)
    ln["exact"] = (ln.ok == ln.ng) & (ln.ns == ln.ng)
    print(ln.groupby("method").agg(lines=("exact", "size"), exact=("exact", "mean"), pairs=("ns", "sum"),
                                   pair_tp=("ok", "sum")).round(3).to_string())
