"""Validate a submission for any phase with the official scorer's checks (format, ids, duplicates, every line present).

    python integration/validate.py --phase live [--file path.csv.gz]
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ingestion import lib  # noqa: E402
from integration.score import S, official_golden  # noqa: E402


def validate(phase: str, path: str) -> bool:
    bank = pd.read_parquet(os.path.join(ROOT, "ingestion", "out", phase, "bank_lines.parquet"))
    res = S.score_submission(official_golden(phase, bank.bank_txn_id, None), open(path, "rb").read(), path)
    raw = gzip.open(path).read() if path.endswith(".gz") else open(path, "rb").read()
    covered = res["metrics"].get("txns_covered", 0)
    ok = res["ok"] and covered == len(bank) and len(raw) <= S.MAX_UNCOMPRESSED
    print(f"[validate] {phase}: {'OK' if ok else 'FAILED'}  lines {covered:,}/{len(bank):,}  rows {res['metrics'].get('rows', 0):,}  "
          f"{len(raw) / 1e6:.1f} MB uncompressed")
    for e in res["errors"][:10]:
        print("   ", e)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="live", choices=list(lib.PHASES))
    ap.add_argument("--file", default="")
    a = ap.parse_args()
    sys.exit(0 if validate(a.phase, a.file or os.path.join(lib.out_dir("integration", a.phase), "submission.csv.gz")) else 1)
