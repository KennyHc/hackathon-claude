#!/usr/bin/env python3
"""K1 batch stage entry point.

    .venv/bin/python matching/batch/run.py --phase val [--eval]
    → matching/out/<phase>/alloc_batch.parquet (bank_txn_id,item_id,amount,exception,confidence,method)
Then score alone:
    .venv/bin/python integration/run.py --phase val --b matching/out/val/alloc_batch.parquet \
        --out matching/out/val/sub_batch.csv.gz
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from integration.lib.ledger import load_items  # noqa: E402
from integration.lib.masters import masters_dir  # noqa: E402
from matching.batch.common import load_lines  # noqa: E402
from matching.batch.matcher import match  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--eval", action="store_true", help="quick per-segment eval against the golden (val/sample)")
    a = ap.parse_args()
    t0 = time.time()
    lines = load_lines(a.phase)
    items = load_items(a.phase)
    print(f"[batch] {len(lines):,} lines, {len(items):,} items loaded in {time.time() - t0:.1f}s", flush=True)
    claimed = pd.DataFrame(columns=["bank_txn_id", "item_id", "amount", "exception", "confidence", "method",
                                    "booking_date"])
    lab = os.path.join(ROOT, "classification", "out", a.phase, "line_labels.parquet")
    res = match(a.phase, lines, items, claimed, line_labels=pd.read_parquet(lab) if os.path.exists(lab) else None,
                masters_dir=masters_dir(a.phase))
    out = os.path.join(ROOT, "matching", "out", a.phase, "alloc_batch.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res.to_parquet(out, index=False)
    print(f"[batch] wrote {len(res):,} rows / {res.bank_txn_id.nunique():,} lines → {os.path.relpath(out, ROOT)} "
          f"in {time.time() - t0:.1f}s", flush=True)
    if a.eval and os.path.exists(os.path.join(ROOT, "data", a.phase, "golden_allocations.csv.gz")):
        from matching.batch.evalb import evaluate
        evaluate(res, a.phase, show=40)


if __name__ == "__main__":
    main()
