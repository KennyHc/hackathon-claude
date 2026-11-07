"""Run all matchers in priority order -> matching/out/<phase>/allocations.parquet

    python matching/run.py --phase sample                 # all matchers in ORDER
    python matching/run.py --phase sample --only ref_exact  # just one (for developing it)
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ingestion import lib  # noqa: E402
from matching.common import ALLOC_COLS, Ctx  # noqa: E402

# Most certain first. A line claimed by an earlier matcher is never touched by a later one.
ORDER = [
    "payment_runs",      # supplier payments via payment_runs/ files (#16)
    "dd_batches",        # direct-debit batches + returns (#14)
    "cards_cheques",     # card settlements, cheque deposits (#15)
    "remittance",        # remittance advices (#18)
    "pdf_cases",         # instalments, retentions, advances, replaced invoices from PDF clauses (#24)
    "refs",              # invoice numbers in the bank text (#10)
    "subset_sum",        # payer's open items summing to the amount (#11)
    "same_amount",       # same-amount invoices: month in text / FIFO (#12)
    "ref_exact",         # baseline: exact amount + exact document number (kept as a safety net)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="sample", choices=list(lib.PHASES))
    ap.add_argument("--only", default="", help="comma-separated matcher names")
    a = ap.parse_args()
    ctx = Ctx(a.phase)
    names = [n for n in a.only.split(",") if n] or ORDER
    parts = []
    for name in names:
        try:
            mod = importlib.import_module(f"matching.matchers.{name}")
        except ModuleNotFoundError as e:
            if e.name == f"matching.matchers.{name}":
                print(f"[matching] {name}: not implemented yet, skipped")
                continue
            raise
        t = time.time()
        # allocations made so far by earlier matchers (item_id -> used amount available via this frame)
        ctx.allocations = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=ALLOC_COLS)
        try:
            df = mod.run(ctx)
        except Exception:
            print(f"[matching] {name}: CRASHED, skipped\n{traceback.format_exc()}")
            continue
        if df is None or df.empty:
            print(f"[matching] {name}: 0 lines ({time.time() - t:.1f}s)")
            continue
        df = df[~df.bank_txn_id.isin(ctx.claimed)][ALLOC_COLS]
        df = df.drop_duplicates(["bank_txn_id", "item_id"])
        ctx.claimed |= set(df.bank_txn_id)
        parts.append(df)
        print(f"[matching] {name}: {df.bank_txn_id.nunique():,} lines, {len(df):,} rows ({time.time() - t:.1f}s)")
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=ALLOC_COLS)
    od = lib.out_dir("matching", a.phase)
    out.to_parquet(os.path.join(od, "allocations.parquet"), index=False)
    print(f"[matching] total {out.bank_txn_id.nunique():,} lines -> {od}/allocations.parquet")


if __name__ == "__main__":
    main()
