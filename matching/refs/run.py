#!/usr/bin/env python3
"""K2 refs stage runner: .venv/bin/python matching/refs/run.py --phase {sample,val,live,final}
Reads data/<phase>/bank_transactions, items (integration.lib.ledger.load_items), matching/out/<phase>/alloc_batch.parquet
(claimed, optional), matching/out/<phase>/pdf_clauses.parquet (optional) → matching/out/<phase>/alloc_refs.parquet."""
import argparse
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from integration.lib.ledger import load_items  # noqa: E402
from integration.lib.masters import masters_dir  # noqa: E402
from matching.refs.matcher import match  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--no-claimed", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    out_dir = os.path.join(ROOT, "matching", "out", a.phase)
    os.makedirs(out_dir, exist_ok=True)
    lines = pd.read_csv(os.path.join(ROOT, "data", a.phase, "bank_transactions.csv.gz"), dtype=str,
                        keep_default_na=False)
    items = load_items(a.phase)
    print(f"[{time.time()-t0:5.1f}s] {len(lines):,} lines, {len(items):,} items", flush=True)
    claimed = pd.DataFrame(columns=["bank_txn_id", "item_id", "amount", "exception", "booking_date"])
    cp = os.path.join(out_dir, "alloc_batch.parquet")
    if os.path.exists(cp) and not a.no_claimed:
        claimed = pd.read_parquet(cp)
        if "booking_date" not in claimed:
            claimed = claimed.merge(lines[["bank_txn_id", "booking_date"]], on="bank_txn_id", how="left")
        lines = lines[~lines.bank_txn_id.isin(set(claimed.bank_txn_id))]
        print(f"[{time.time()-t0:5.1f}s] claimed by batch: {claimed.bank_txn_id.nunique():,} lines", flush=True)
    pdf = None
    pp = os.path.join(out_dir, "pdf_clauses.parquet")
    try:
        if not os.path.exists(pp) and os.path.exists(os.path.join(ROOT, "data", a.phase, "pdfs", "index.csv")):
            from matching.refs import pdf_clauses
            extra = {"val": ["train"], "live": ["train"], "final": ["train", "live"]}.get(a.phase)
            cwd = os.getcwd()
            os.chdir(ROOT)
            try:
                pdf_clauses.load(a.phase, extra_phases=extra)
            finally:
                os.chdir(cwd)
        if os.path.exists(pp):
            pdf = pd.read_parquet(pp)
            print(f"[{time.time()-t0:5.1f}s] pdf clauses: {len(pdf):,}", flush=True)
    except Exception as e:  # PDFs are a bonus: never block the stage
        print(f"pdf clauses unavailable: {e}", flush=True)
    lab = os.path.join(ROOT, "classification", "out", a.phase, "line_labels.parquet")
    labels = pd.read_parquet(lab) if os.path.exists(lab) else None
    res = match(a.phase, lines, items, claimed, line_labels=labels, masters_dir=masters_dir(a.phase),
                pdf_clauses=pdf)
    res.to_parquet(os.path.join(out_dir, "alloc_refs.parquet"), index=False)
    print(f"[{time.time()-t0:5.1f}s] alloc_refs: {len(res):,} rows, {res.bank_txn_id.nunique():,} lines", flush=True)
    print(res.method.value_counts().to_string())


if __name__ == "__main__":
    main()
