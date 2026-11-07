"""classification -> classification/out/<phase>/line_labels.parquet

One row per bank line: bank_txn_id, p_items, category, exception, confidence, method.
v0 baseline: receipts -> UNIDENTIFIED, payments -> BANK_FEES (replaced by the category model, issue #20).

    python classification/run.py --phase sample
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ingestion import lib  # noqa: E402

LABEL_COLS = ["bank_txn_id", "p_items", "category", "exception", "confidence", "method"]


def baseline(bank: pd.DataFrame) -> pd.DataFrame:
    rec = bank.amount > 0
    return pd.DataFrame({
        "bank_txn_id": bank.bank_txn_id,
        "p_items": 0.5,
        "category": np.where(rec, "UNIDENTIFIED", "BANK_FEES"),
        "exception": np.where(rec, "UNIDENTIFIED", ""),
        "confidence": 0.1,
        "method": "baseline",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="sample", choices=list(lib.PHASES))
    a = ap.parse_args()
    bank = pd.read_parquet(os.path.join(ROOT, "ingestion", "out", a.phase, "bank_lines.parquet"))
    try:
        model = importlib.import_module("classification.categories")
        labels = model.label(bank, a.phase)
    except ModuleNotFoundError as e:
        if e.name != "classification.categories":
            raise
        labels = baseline(bank)
    labels = labels[LABEL_COLS]
    od = lib.out_dir("classification", a.phase)
    labels.to_parquet(os.path.join(od, "line_labels.parquet"), index=False)
    print(f"[classification] {a.phase}: {len(labels):,} lines -> {od}/line_labels.parquet "
          f"({labels.method.value_counts().to_dict()})")


if __name__ == "__main__":
    main()
