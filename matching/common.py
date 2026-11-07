"""Helpers shared by the matching cascade (matching/run.py) and the stages (matching/{batch,refs,party}/matcher.py).

    import sys; sys.path.insert(0, ROOT)
    from matching.common import load_lines, load_items, load_line_labels, open_at, Balances, ALLOC_COLS

Everything here is read-only on data/ and import-safe (no work at import time). Stages must not import each other;
they may import this module.
"""
from __future__ import annotations

import os
import sys
from typing import Iterable, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from integration.lib import ledger as _ledger  # noqa: E402
from integration.lib import masters as _masters  # noqa: E402

# Base data dir. BANKREC_DATA overrides it (used to test a mock FINAL phase without touching data/).
DATA = os.environ.get("BANKREC_DATA", os.path.join(ROOT, "data"))

ALLOC_COLS = ["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"]
CLAIMED_COLS = ALLOC_COLS + ["booking_date", "stage"]
NO_BALANCE = {"DUPLICATE_PAYMENT"}  # rows that do not reduce the item's balance
REOPENS = {"RETURNED"}  # rows that re-open (add back) the item's balance


def out_dir(phase: str) -> str:
    return os.path.join(ROOT, "matching", "out", phase)


def load_lines(phase: str, data_dir: str = DATA) -> pd.DataFrame:
    """All bank lines of the phase, every column as str (missing = ''), plus amount_f (float)."""
    p = os.path.join(data_dir, phase, "bank_transactions.csv.gz")
    if not os.path.exists(p):
        p = os.path.join(data_dir, phase, "bank_transactions.csv")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df["amount_f"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df


def load_items(phase: str, data_dir: str = DATA) -> pd.DataFrame:
    """integration.lib.ledger.load_items: ITEM_COLS + EXTRA_COLS, open_amount = balance at phase start."""
    return _ledger.load_items(phase, data_dir=data_dir)


def all_item_ids(phase: str, data_dir: str = DATA) -> frozenset:
    """Every item id known in train/live/final/`phase` (incl. items closed before the phase starts)."""
    ids = set()
    for ph in dict.fromkeys(["train", "live", "final", phase]):
        for name in ("ar_invoices", "ap_invoices", "credit_notes", "open_items_at_start"):
            for ext in (".csv.gz", ".csv"):
                f = os.path.join(data_dir, ph, name + ext)
                if os.path.exists(f):
                    ids.update(pd.read_csv(f, dtype=str, keep_default_na=False, usecols=["item_id"]).item_id)
                    break
    return frozenset(ids)


def load_line_labels(phase: str) -> Optional[pd.DataFrame]:
    p = os.path.join(ROOT, "classification", "out", phase, "line_labels.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None


def masters_dir(phase: str, data_dir: str = DATA) -> str:
    return _masters.masters_dir(phase, data_dir)


def empty_alloc() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=float if c in ("amount", "confidence") else object) for c in ALLOC_COLS})


def _signed(claimed: pd.DataFrame) -> np.ndarray:
    """Effect of each claimed row on its item's balance (negative = reduces it)."""
    exc = claimed["exception"].fillna("").astype(str).values if "exception" in claimed else np.array([""] * len(claimed))
    amt = pd.to_numeric(claimed["amount"], errors="coerce").fillna(0.0).values
    sign = np.where(np.isin(exc, list(REOPENS)), 1.0, np.where(np.isin(exc, list(NO_BALANCE)), 0.0, -1.0))
    return sign * amt


def open_at(items: pd.DataFrame, claimed: Optional[pd.DataFrame], date: Optional[str] = None,
            inclusive: bool = False) -> pd.Series:
    """Open balance per item_id after the claimed allocations booked before `date` (all of them if date is None;
    `inclusive` also counts those booked on `date`). Simple sums (no floor at 0 per step). Index = item_id."""
    base = pd.Series(pd.to_numeric(items["open_amount"], errors="coerce").fillna(0.0).values,
                     index=items["item_id"].values)
    if claimed is None or not len(claimed):
        return base.copy()
    c = claimed
    if date is not None:
        d = c["booking_date"].astype(str)
        c = c[(d <= date) if inclusive else (d < date)]
    if not len(c):
        return base.copy()
    delta = pd.Series(_signed(c), index=c["item_id"].values).groupby(level=0).sum()
    out = base.add(delta.reindex(base.index).fillna(0.0))
    return out.clip(lower=0.0).round(2)


class Balances:
    """Running open balance per item for stages that walk lines in booking order.

        bal = Balances(items, claimed)       # claimed rows applied (all dates: the cascade is budget-based)
        bal.get("ARI-…")  → float           # current open
        bal.take("ARI-…", 120.0)            # reduce (returns the amount actually available, ≤ requested)
    """

    def __init__(self, items: pd.DataFrame, claimed: Optional[pd.DataFrame] = None):
        self.open = open_at(items, claimed).to_dict()

    def get(self, item_id: str) -> float:
        return self.open.get(item_id, 0.0)

    def take(self, item_id: str, amount: float) -> float:
        cur = self.open.get(item_id, 0.0)
        got = min(cur, amount)
        self.open[item_id] = round(cur - got, 2)
        return got

    def give(self, item_id: str, amount: float) -> None:
        self.open[item_id] = round(self.open.get(item_id, 0.0) + amount, 2)


def tol(amount: float) -> float:
    """Scorer amount tolerance: max(0.01, 0.5 % of the amount)."""
    return max(0.01, 0.005 * abs(amount))


def normalize_stage_output(df: Optional[pd.DataFrame], stage: str) -> pd.DataFrame:
    """Coerce a stage's return value to ALLOC_COLS (missing columns filled, dtypes fixed, empty item rows dropped)."""
    if df is None or not isinstance(df, pd.DataFrame) or not len(df):
        return empty_alloc()
    missing = [c for c in ("bank_txn_id", "item_id", "amount") if c not in df.columns]
    if missing:
        raise ValueError(f"stage {stage} output lacks {missing}")
    out = df.copy()
    if "exception" not in out:
        out["exception"] = None
    if "confidence" not in out:
        out["confidence"] = 1.0
    if "method" not in out:
        out["method"] = stage
    out = out[ALLOC_COLS].copy()
    out["bank_txn_id"] = out["bank_txn_id"].astype(str)
    out["item_id"] = out["item_id"].fillna("").astype(str)
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").abs().round(2)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(1.0).astype(float)
    out["method"] = out["method"].fillna(stage).astype(str)
    ex = out["exception"].astype(object)
    out["exception"] = ex.where(ex.notna() & (ex.astype(str).str.strip() != "") & (ex.astype(str) != "nan")
                                & (ex.astype(str) != "None"), None)
    out = out[(out.item_id != "") & out.amount.notna()]
    return out.reset_index(drop=True)


def stages_present(names: Iterable[str]) -> list:
    return [n for n in names if os.path.exists(os.path.join(ROOT, "matching", n, "matcher.py"))]
