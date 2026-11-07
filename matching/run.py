"""Matching entry point (piece B): python matching/run.py --phase val [--min-conf 0.0] [--no-exceptions]

Runs the matchers in priority order (most certain evidence first). A bank line claimed by an earlier matcher is never given to a
later one. Writes matching/out/<phase>/allocations.parquet with the CONTRACTS section B columns and, when the phase has a golden
(sample/train/val), prints the per-segment report.

Priority = how deterministic the evidence is: batch files > payment-run ids > remittance advices > explicit references.
A matcher that is missing or crashes is skipped (and counted): the pipeline must always produce a valid file.
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

from matching.lib import data, evalutil  # noqa: E402

ORDER = ["batches", "ap", "remittance", "refs"]   # matcher modules under matching/matchers/, in priority order
COLS = ["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"]


def run(phase: str, min_conf: float = 0.0, only: list[str] | None = None, exceptions: bool = False, quiet: bool = False) -> pd.DataFrame:
    t0 = time.time()
    claimed: set[str] = set()
    parts, stats = [], []
    for name in (only or ORDER):
        try:
            mod = importlib.import_module(f"matching.matchers.{name}")
        except ImportError as e:
            stats.append((name, "missing", 0, 0, f"{e}"))
            continue
        try:
            t1 = time.time()
            df = mod.match(phase, set(claimed))
            if df is None or df.empty:
                stats.append((name, "ok", 0, 0, f"{time.time() - t1:.0f}s"))
                continue
            for c in COLS:
                if c not in df:
                    df[c] = None
            df = df[~df.bank_txn_id.isin(claimed)]
            if min_conf > 0:
                df = df[df.confidence.fillna(1.0) >= min_conf]
            claimed |= set(df.bank_txn_id)
            parts.append(df[COLS])
            stats.append((name, "ok", df.bank_txn_id.nunique(), len(df), f"{time.time() - t1:.0f}s"))
        except Exception:  # never let one matcher kill the run
            stats.append((name, "CRASHED", 0, 0, traceback.format_exc().strip().splitlines()[-1]))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
    if exceptions and len(out):
        try:
            from matching.matchers import exception_rules
            out = exception_rules.assign(out, phase)[COLS]
        except ImportError:
            pass
        except Exception:
            stats.append(("exception_rules", "CRASHED", 0, 0, traceback.format_exc().strip().splitlines()[-1]))
    out["amount"] = out.amount.astype(float).round(2)
    out = out.drop_duplicates(["bank_txn_id", "item_id"])
    p = os.path.join(ROOT, "matching", "out", phase)
    os.makedirs(p, exist_ok=True)
    out.to_parquet(os.path.join(p, "allocations.parquet"), index=False)
    if not quiet:
        print(f"[{time.time() - t0:.0f}s] allocations: {out.bank_txn_id.nunique()} lines / {len(out)} rows -> {p}")
        for s in stats:
            print("  %-16s %-8s lines=%-7s rows=%-8s %s" % s)
        if phase in ("sample", "train", "train_lite", "val") and len(out):
            evalutil.report(out, phase, name="matching (all)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="val")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--exceptions", action="store_true", help="also apply matching.matchers.exception_rules (integration already applies classification/exceptions.py)")
    a = ap.parse_args()
    run(a.phase, a.min_conf, a.only, a.exceptions)
