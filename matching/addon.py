"""Run extra matchers ON TOP of an existing full matching run, without re-running the pipeline.

    python matching/addon.py --phase final --stages inv_ranker,sd_noadv
    ALLOC_FILE=allocations_plus.parquet python integration/run.py --phase final     # merge + exceptions + submission

Reads matching/out/<phase>/allocations.parquet (the full run) as the prior state: ctx.allocations = those rows and
ctx.claimed = their lines, exactly what the stages would see if they were appended to ORDER. Each stage only adds
rows on lines nobody claimed. Writes matching/out/<phase>/allocations_plus.parquet (the full run's rows + the new ones).
A stage that crashes is skipped (the output then equals the full run).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=list(lib.PHASES))
    ap.add_argument("--stages", required=True, help="comma-separated matcher module names (matching/matchers/<name>.py)")
    ap.add_argument("--base", default="allocations.parquet")
    ap.add_argument("--out", default="allocations_plus.parquet")
    a = ap.parse_args()
    od = lib.out_dir("matching", a.phase)
    base = pd.read_parquet(os.path.join(od, a.base))
    ctx = Ctx(a.phase)
    ctx.claimed = set(base.bank_txn_id)
    parts = [base[ALLOC_COLS]]
    for name in [s for s in a.stages.split(",") if s]:
        t = time.time()
        ctx.allocations = pd.concat(parts, ignore_index=True)
        try:
            df = importlib.import_module(f"matching.matchers.{name}").run(ctx)
        except Exception:
            print(f"[addon] {name}: CRASHED, skipped\n{traceback.format_exc()}")
            continue
        if df is None or df.empty:
            print(f"[addon] {name}: 0 lines ({time.time() - t:.1f}s)")
            continue
        df = df[~df.bank_txn_id.isin(ctx.claimed)][ALLOC_COLS].drop_duplicates(["bank_txn_id", "item_id"])
        ctx.claimed |= set(df.bank_txn_id)
        parts.append(df)
        print(f"[addon] {name}: {df.bank_txn_id.nunique():,} lines, {len(df):,} rows ({time.time() - t:.1f}s)")
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(os.path.join(od, a.out), index=False)
    print(f"[addon] {a.phase}: {out.bank_txn_id.nunique():,} lines ({out.bank_txn_id.nunique() - base.bank_txn_id.nunique():+,}) -> {od}/{a.out}")


if __name__ == "__main__":
    main()
