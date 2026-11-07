#!/usr/bin/env python3
"""Leave-one-entity-out report on val (golden used for scoring only).

    .venv/bin/python matching/party/loeo_eval.py --entity PL01 --combo matching/party/out/ab_loeo/combo.parquet [--unid]

Prints, on the entity's val lines: allocation P/R/F1 (pairs within amount tolerance), line accuracy of item lines, and
UNIDENTIFIED precision/recall for two rules on receipts no stage claimed:
  old: every unclaimed receipt → UNIDENTIFIED (integration/run.py rule for unseen entities)
  new: unclaimed receipt → UNIDENTIFIED only when unid_evidence says no ref-like text and no known payer.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--combo", required=True)
    ap.add_argument("--unid", action="store_true")
    a = ap.parse_args()
    lines = pd.read_csv(os.path.join(ROOT, "data/val/bank_transactions.csv.gz"), dtype=str, keep_default_na=False)
    E = lines[lines.entity == a.entity]
    ids = set(E.bank_txn_id)
    g = pd.read_csv(os.path.join(ROOT, "data/val/golden_allocations.csv.gz"), dtype=str, keep_default_na=False)
    g = g[g.bank_txn_id.isin(ids)]
    s = pd.read_parquet(a.combo)
    s = s[s.bank_txn_id.isin(ids)].copy()
    gi = g[g.item_id != ""].copy()
    gi["amount"] = gi.amount.astype(float)
    s["amount"] = s.amount.astype(float)
    m = s.merge(gi[["bank_txn_id", "item_id", "amount"]], on=["bank_txn_id", "item_id"], how="left", suffixes=("", "_g"))
    tp = int(((m.amount - m.amount_g).abs() <= np.maximum(0.01, 0.005 * m.amount_g.abs())).sum())
    P, R = tp / max(len(s), 1), tp / max(len(gi), 1)
    print(f"{a.entity}: {len(E):,} lines, {gi.bank_txn_id.nunique():,} item lines, {len(gi):,} gold pairs")
    print(f"  allocation  P {P:.4f}  R {R:.4f}  F1 {2 * P * R / max(P + R, 1e-9):.4f}  (claimed lines "
          f"{s.bank_txn_id.nunique():,})")
    rec = E[pd.to_numeric(E.amount) > 0]
    unclaimed = rec[~rec.bank_txn_id.isin(set(s.bank_txn_id))]
    gold_unid = set(g[g.exception == "UNIDENTIFIED"].bank_txn_id) & set(rec.bank_txn_id)

    def pr(pred):
        pred = set(pred)
        t = len(pred & gold_unid)
        p, r = t / max(len(pred), 1), t / max(len(gold_unid), 1)
        return f"pred {len(pred):,} P {p:.3f} R {r:.3f} F1 {2 * p * r / max(p + r, 1e-9):.3f}"

    print(f"  UNIDENTIFIED gold {len(gold_unid):,} receipts; unclaimed receipts {len(unclaimed):,}")
    print(f"  rule old (all unclaimed receipts)   {pr(unclaimed.bank_txn_id)}")
    if a.unid:
        from unid import unid_evidence
        from integration.lib.ledger import load_items
        ev = unid_evidence(unclaimed, load_items("val"), "val", loeo=a.entity)
        keep = ev[~ev.has_ref_like & ~ev.payer_known].bank_txn_id
        print(f"  rule new (no ref, no known payer)  {pr(keep)}")
        print("  evidence:", ev.evidence.replace("", "none").value_counts().to_dict(),
              "ref_like", int(ev.has_ref_like.sum()))


if __name__ == "__main__":
    main()
