"""Merge matching + classification -> integration/out/<phase>/submission.csv.gz (+ methods.parquet for scoring).

    python integration/run.py --phase sample
    python integration/run.py --phase sample --all     # run ingestion? no: runs matching + classification first

Merge rule (CONTRACTS §D): a line gets matching's item rows if matching claimed it and classification's p_items is
not below P_MIN (unless classification has no confident category); else classification's single category row.
Every bank line of the phase is present.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from classification.exceptions import assign_exceptions  # noqa: E402
from ingestion import lib  # noqa: E402

P_MIN = 0.2      # classification must be fairly sure the line settles NO item to override a matcher
CONF_OVERRIDE = 0.9
UNID_P = float(os.environ.get("UNID_P", "0.9"))   # flag UNIDENTIFIED only when p_items is below this


def merge(phase: str) -> tuple[pd.DataFrame, pd.Series]:
    bank = pd.read_parquet(os.path.join(ROOT, "ingestion", "out", phase, "bank_lines.parquet"))
    items = pd.read_parquet(os.path.join(ROOT, "ingestion", "out", phase, "open_items.parquet"))
    # ALLOC_FILE=allocations_only_<name>.parquet scores a partial `matching/run.py --only <name>` run
    alloc = pd.read_parquet(os.path.join(ROOT, "matching", "out", phase, os.environ.get("ALLOC_FILE", "allocations.parquet")))
    labels = pd.read_parquet(os.path.join(ROOT, "classification", "out", phase, "line_labels.parquet")).set_index("bank_txn_id")

    # drop matcher lines that classification is confident settle no item
    lab = labels.reindex(alloc.bank_txn_id)
    veto = (lab.p_items.values < P_MIN) & (lab.confidence.values >= CONF_OVERRIDE)
    alloc = alloc[~veto].copy()

    # exceptions: rules library first, matcher's guess otherwise
    if len(alloc):
        try:
            rule = assign_exceptions(alloc, bank, items)
        except Exception:   # FINAL safety net: keep the matchers' own exceptions
            import traceback
            print(f"[integration] assign_exceptions FAILED, keeping matcher exceptions\n{traceback.format_exc()}")
            rule = pd.Series([None] * len(alloc), index=alloc.index, dtype=object)
        # matchers that decide exceptions from evidence (PDF clauses, advices, batch files) keep their own label
        keep = alloc.method.fillna("").str.match(r"^(pdf_cases|remittance|dd_batches|cards_cheques|payment_runs|refs:explain)") \
            | (alloc.exception.fillna("") != "")
        alloc["exception"] = rule.where(rule.notna() & ~keep, alloc.exception).fillna("")
    item_rows = pd.DataFrame({"bank_txn_id": alloc.bank_txn_id, "item_id": alloc.item_id,
                              "amount": alloc.amount.map(lambda x: f"{x:.2f}"), "category": "",
                              "exception": alloc.exception.fillna("")})
    claimed = set(alloc.bank_txn_id)
    rest = labels[~labels.index.isin(claimed)].reindex(bank.bank_txn_id[~bank.bank_txn_id.isin(claimed)])
    exc = rest.exception.fillna("")
    # an unclaimed receipt that probably DOES settle items (a matcher miss) is wrong either way; flagging it
    # UNIDENTIFIED would only add a false positive to that exception's F1
    exc = exc.where(~((exc == "UNIDENTIFIED") & (rest.p_items.fillna(0) >= UNID_P)), "")
    cat_rows = pd.DataFrame({"bank_txn_id": rest.index, "item_id": "", "amount": "",
                             "category": rest.category.fillna("UNIDENTIFIED").values,
                             "exception": exc.values})
    sub = pd.concat([item_rows, cat_rows], ignore_index=True)
    methods = pd.concat([alloc.drop_duplicates("bank_txn_id").set_index("bank_txn_id").method,
                         "cls:" + rest.method.fillna("baseline")])
    return sub, methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="sample", choices=list(lib.PHASES))
    ap.add_argument("--all", action="store_true", help="run matching and classification first")
    a = ap.parse_args()
    t = time.time()
    if a.all:
        for piece in ("matching", "classification"):
            subprocess.run([sys.executable, os.path.join(ROOT, piece, "run.py"), "--phase", a.phase], check=True)
    sub, methods = merge(a.phase)
    od = lib.out_dir("integration", a.phase)
    path = os.path.join(od, "submission.csv.gz")
    sub.to_csv(path, index=False, compression="gzip")
    methods.rename("method").rename_axis("bank_txn_id").reset_index().to_parquet(os.path.join(od, "methods.parquet"), index=False)
    print(f"[integration] {a.phase}: {len(sub):,} rows, {sub.bank_txn_id.nunique():,} lines -> {path} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
