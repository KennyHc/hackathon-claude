"""Shared context for matchers.

A matcher is a module `matching/matchers/<name>.py` with

    NAME = "<name>"
    def run(ctx: Ctx) -> pd.DataFrame   # columns ALLOC_COLS, only for lines it is confident about

Rules
- Claim a WHOLE bank line (all its item rows) or nothing. One row per (bank_txn_id, item_id).
- `amount` = amount of the item cleared, in ITEM currency, positive (POLICIES §2). Round to 2 decimals.
- Set `exception` when you know it (PARTIAL, SHORT_DISCOUNT, RETURNED, ...), else "" — classification/exceptions.py
  fills the rest later.
- `confidence` 0–1 (same on every row of a line), `method` = short tag for debugging (e.g. "ref_exact").
- Skip lines in `ctx.claimed` (claimed by an earlier matcher in matching/run.py ORDER) to save time; you may ignore it
  when running stand-alone.
- Never write files outside matching/out/.
"""
from __future__ import annotations

import os
import sys
from functools import cached_property

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ingestion import lib  # noqa: E402

ALLOC_COLS = ["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"]
TOL = lambda g: max(0.01, 0.005 * abs(g))  # noqa: E731  scorer tolerance


class Ctx:
    def __init__(self, phase: str):
        self.phase = phase
        self.folder = lib.PHASES[phase][0]
        d = os.path.join(ROOT, "ingestion", "out", phase)
        self.bank: pd.DataFrame = pd.read_parquet(os.path.join(d, "bank_lines.parquet"))
        self.items: pd.DataFrame = pd.read_parquet(os.path.join(d, "open_items.parquet"))
        self.claimed: set[str] = set()

    # --- raw inputs --------------------------------------------------------------------------------------------------
    def data_path(self, *p: str) -> str:
        """Path inside the phase's raw data folder (side files: payment_runs/, direct_debit_batches/, pdfs/, ...)."""
        return os.path.join(lib.DATA, self.folder, *p)

    @cached_property
    def masters(self) -> dict[str, pd.DataFrame]:
        return lib.masters(self.folder)

    # --- handy indexes -----------------------------------------------------------------------------------------------
    @cached_property
    def items_by_id(self) -> pd.DataFrame:
        return self.items.set_index("item_id", drop=False)

    @cached_property
    def open_items(self) -> pd.DataFrame:
        return self.items[self.items.open_amount > 0.005]

    @cached_property
    def by_docnorm(self) -> dict[str, list[str]]:
        """document_number_norm (digits, no leading zeros) -> item_ids"""
        s = self.items[self.items.document_number_norm != ""]
        return s.groupby("document_number_norm").item_id.agg(list).to_dict()

    @cached_property
    def by_alnum(self) -> dict[str, list[str]]:
        s = self.items[self.items.document_alnum != ""]
        return s.groupby("document_alnum").item_id.agg(list).to_dict()

    @cached_property
    def by_party(self) -> dict[str, pd.DataFrame]:
        """party_id -> its open items sorted oldest first"""
        o = self.open_items.sort_values(["issue_date", "item_id"])
        return {k: v for k, v in o.groupby("party_id")}

    def unclaimed(self) -> pd.DataFrame:
        return self.bank[~self.bank.bank_txn_id.isin(self.claimed)]


def alloc_frame(rows: list[tuple]) -> pd.DataFrame:
    """rows of (bank_txn_id, item_id, amount, exception, confidence, method)"""
    df = pd.DataFrame(rows, columns=ALLOC_COLS)
    df["amount"] = df.amount.astype(float).round(2)
    return df
