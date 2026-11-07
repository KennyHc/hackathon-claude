"""Item ledger and open balance tracking through time for Grupo Valtermia.

Provides item loading across all dataset phases and the Ledger class to track
open amounts as allocations are applied chronologically.

How to run tests:
    .venv/bin/python integration/lib/test_lib.py
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")

ITEM_COLS = [
    "item_id",
    "item_type",
    "entity",
    "party_id",
    "document_number",
    "issue_date",
    "due_date",
    "currency",
    "total_amount",
    "open_amount",
]


EXTRA_COLS = ["party_name", "payment_terms", "po_number", "related_document", "payment_method", "cn_reason"]
_DETAIL_PHASES = ["train", "live", "final"]  # where an item's original document can live (val is a subset of train)


def item_details(phase: str, data_dir: str = DATA) -> pd.DataFrame:
    """EXTRA_COLS per item_id, from the original invoice / credit-note files of train, live, final and `phase`
    (open_items_at_start has only the 10 ITEM_COLS). AR: customer_name, payment_terms, customer_po, payment_method;
    AP: supplier_name, payment_terms, our_po, payment_method; CN: party_name, related_document, reason."""
    out = []
    for ph in dict.fromkeys(_DETAIL_PHASES + [phase]):
        d = os.path.join(data_dir, ph)
        for name, m in [("ar_invoices", {"customer_name": "party_name", "payment_terms": "payment_terms",
                                         "customer_po": "po_number", "payment_method": "payment_method"}),
                        ("ap_invoices", {"supplier_name": "party_name", "payment_terms": "payment_terms",
                                         "our_po": "po_number", "payment_method": "payment_method"}),
                        ("credit_notes", {"party_name": "party_name", "related_document": "related_document",
                                          "reason": "cn_reason"})]:
            f = _find_csv(d, name)
            if f is None:
                continue
            df = pd.read_csv(f, dtype=str, keep_default_na=False, usecols=["item_id"] + list(m))
            out.append(df.rename(columns=m))
    if not out:
        return pd.DataFrame(columns=["item_id"] + EXTRA_COLS)
    det = pd.concat(out, ignore_index=True).drop_duplicates("item_id", keep="last")
    for c in EXTRA_COLS:
        det[c] = det[c].fillna("") if c in det else ""
    return det[["item_id"] + EXTRA_COLS]


def _find_csv(dir_path: str, filename: str) -> Optional[str]:
    """Find a CSV file in directory, checking .csv.gz then .csv."""
    for ext in [".csv.gz", ".csv"]:
        candidate = os.path.join(dir_path, filename + ext)
        if os.path.exists(candidate):
            return candidate
    return None


_ITEMS_CACHE: dict = {}


def load_items(phase: str, data_dir: str = DATA, details: bool = True) -> pd.DataFrame:
    """Cached wrapper (returns a copy): see _load_items."""
    key = (phase, os.path.abspath(data_dir), details)
    if key not in _ITEMS_CACHE:
        _ITEMS_CACHE[key] = _load_items(phase, data_dir, details)
    return _ITEMS_CACHE[key].copy()


def _load_items(phase: str, data_dir: str = DATA, details: bool = True) -> pd.DataFrame:
    """Load all items available in a phase with standard schema ITEM_COLS.

    Amounts (total_amount, open_amount) are parsed as float.

    For val/live/final:
        Loads open_items_at_start (open_amount as given) + ar_invoices,
        ap_invoices, credit_notes of the phase (open_amount = total_amount).
    For train/sample:
        Loads phase invoice and credit note files with open_amount = total_amount.

    Column mappings:
        item_type: item_id[:3]
        AR: party_id = customer_id, document_number = document_number
        AP: party_id = supplier_id, document_number = supplier_invoice_number
        CN: party_id = party_id, document_number = document_number, due_date = issue_date

    Args:
        phase: Phase name ('sample', 'train', 'val', 'live', 'final').
        data_dir: Base data directory path.

    Returns:
        pd.DataFrame with columns ITEM_COLS.
    """
    phase_dir = os.path.join(data_dir, phase)
    dfs: list[pd.DataFrame] = []

    # 1. open_items_at_start (present in val, live, final)
    oi_path = _find_csv(phase_dir, "open_items_at_start")
    if oi_path is not None:
        oi = pd.read_csv(oi_path, dtype=str, keep_default_na=False)
        oi["total_amount"] = pd.to_numeric(oi["total_amount"], errors="coerce").fillna(0.0).astype(float)
        oi["open_amount"] = pd.to_numeric(oi["open_amount"], errors="coerce").fillna(0.0).astype(float)
        if "item_type" not in oi.columns or (oi["item_type"] == "").all():
            oi["item_type"] = oi["item_id"].str[:3]
        dfs.append(oi[ITEM_COLS])

    # 2. ar_invoices
    ar_path = _find_csv(phase_dir, "ar_invoices")
    if ar_path is not None:
        ar = pd.read_csv(ar_path, dtype=str, keep_default_na=False)
        tot = pd.to_numeric(ar["total_amount"], errors="coerce").fillna(0.0).astype(float)
        df_ar = pd.DataFrame(
            {
                "item_id": ar["item_id"],
                "item_type": ar["item_id"].str[:3],
                "entity": ar["entity"],
                "party_id": ar["customer_id"],
                "document_number": ar["document_number"],
                "issue_date": ar["issue_date"],
                "due_date": ar["due_date"],
                "currency": ar["currency"],
                "total_amount": tot,
                "open_amount": tot.copy(),
            }
        )
        dfs.append(df_ar[ITEM_COLS])

    # 3. ap_invoices
    ap_path = _find_csv(phase_dir, "ap_invoices")
    if ap_path is not None:
        ap = pd.read_csv(ap_path, dtype=str, keep_default_na=False)
        tot = pd.to_numeric(ap["total_amount"], errors="coerce").fillna(0.0).astype(float)
        df_ap = pd.DataFrame(
            {
                "item_id": ap["item_id"],
                "item_type": ap["item_id"].str[:3],
                "entity": ap["entity"],
                "party_id": ap["supplier_id"],
                "document_number": ap["supplier_invoice_number"],
                "issue_date": ap["issue_date"],
                "due_date": ap["due_date"],
                "currency": ap["currency"],
                "total_amount": tot,
                "open_amount": tot.copy(),
            }
        )
        dfs.append(df_ap[ITEM_COLS])

    # 4. credit_notes
    cn_path = _find_csv(phase_dir, "credit_notes")
    if cn_path is not None:
        cn = pd.read_csv(cn_path, dtype=str, keep_default_na=False)
        tot = pd.to_numeric(cn["total_amount"], errors="coerce").fillna(0.0).astype(float)
        df_cn = pd.DataFrame(
            {
                "item_id": cn["item_id"],
                "item_type": cn["item_id"].str[:3],
                "entity": cn["entity"],
                "party_id": cn["party_id"],
                "document_number": cn["document_number"],
                "issue_date": cn["issue_date"],
                "due_date": cn["issue_date"],
                "currency": cn["currency"],
                "total_amount": tot,
                "open_amount": tot.copy(),
            }
        )
        dfs.append(df_cn[ITEM_COLS])

    if not dfs:
        return pd.DataFrame(columns=ITEM_COLS)

    items = pd.concat(dfs, ignore_index=True)
    # Deduplicate keeping earlier entries (e.g. open_items_at_start precedence)
    items = items.drop_duplicates(subset=["item_id"], keep="first").reset_index(drop=True)
    if not details:
        return items[ITEM_COLS]
    # ITEM_COLS + EXTRA_COLS (payment terms, PO, related document… for every item, incl. open_items_at_start)
    items = items[ITEM_COLS].merge(item_details(phase, data_dir), on="item_id", how="left")
    items[EXTRA_COLS] = items[EXTRA_COLS].fillna("")
    return items


class Ledger:
    """Stateful temporal ledger tracking open balances of invoices and credit notes."""

    def __init__(self, items: pd.DataFrame):
        """Initialize ledger with open items table.

        Args:
            items: pd.DataFrame with columns ITEM_COLS.
        """
        self.items = items[ITEM_COLS].copy().reset_index(drop=True)
        self.items["total_amount"] = (
            pd.to_numeric(self.items["total_amount"], errors="coerce").fillna(0.0).astype(float)
        )
        self.items["open_amount"] = (
            pd.to_numeric(self.items["open_amount"], errors="coerce").fillna(0.0).astype(float)
        )

        # Internal state of open amounts
        self._item_open: dict[str, float] = dict(zip(self.items["item_id"], self.items["open_amount"]))

        # Item metadata for fast enrichment
        self._item_total_amount: dict[str, float] = dict(
            zip(self.items["item_id"], self.items["total_amount"])
        )
        self._item_currency: dict[str, str] = dict(zip(self.items["item_id"], self.items["currency"]))
        self._item_type: dict[str, str] = dict(zip(self.items["item_id"], self.items["item_type"]))
        self._item_issue_date: dict[str, str] = dict(zip(self.items["item_id"], self.items["issue_date"]))
        self._item_due_date: dict[str, str] = dict(zip(self.items["item_id"], self.items["due_date"]))
        self._item_party_id: dict[str, str] = dict(zip(self.items["item_id"], self.items["party_id"]))
        self._known_items: set[str] = set(self._item_open.keys())

    def apply(self, allocations: pd.DataFrame, bank_lines: pd.DataFrame) -> pd.DataFrame:
        """Apply allocations against open items in chronological order.

        Stateful: balances carry over between calls, so build a fresh Ledger per run (apply twice = double count).

        Processes allocations in (booking_date, bank_txn_id) order.
        Effect on item open balance:
            RETURNED -> open += amount
            DUPLICATE_PAYMENT -> no change
            anything else -> open -= amount (floored at 0.0)

        Args:
            allocations: DataFrame with bank_txn_id, item_id, amount, [exception].
            bank_lines: DataFrame with bank_txn_id, booking_date.

        Returns:
            A copy of allocations (same row index and order) enriched with:
                open_before, open_after, total_amount, item_currency,
                item_type, issue_date, due_date, party_id (NaN/None if item unknown),
                and known (bool).
        """
        n = len(allocations)
        res = allocations.copy()
        if n == 0:
            for col in [
                "open_before",
                "open_after",
                "total_amount",
                "item_currency",
                "item_type",
                "issue_date",
                "due_date",
                "party_id",
            ]:
                res[col] = pd.Series(dtype=float if "amount" in col or "open" in col else object)
            res["known"] = pd.Series(dtype=bool)
            return res

        # Map booking_date from bank_lines
        bank_dates = dict(zip(bank_lines["bank_txn_id"], bank_lines["booking_date"]))
        txn_ids = allocations["bank_txn_id"].astype(str).values
        b_dates = np.array([bank_dates.get(tid, "") for tid in txn_ids], dtype=object)

        # Stable sort indices by (booking_date, bank_txn_id)
        orig_idx = np.arange(n)
        order = np.lexsort((orig_idx, txn_ids, b_dates))

        item_ids = allocations["item_id"].fillna("").astype(str).values
        amounts = pd.to_numeric(allocations["amount"], errors="coerce").fillna(0.0).values
        if "exception" in allocations.columns:
            exceptions = allocations["exception"].fillna("").astype(str).values
        else:
            exceptions = np.array([""] * n, dtype=object)

        sorted_items = item_ids[order]
        sorted_amounts = amounts[order]
        sorted_exceptions = exceptions[order]

        open_before_sorted = np.empty(n, dtype=float)
        open_after_sorted = np.empty(n, dtype=float)

        item_open = self._item_open
        known_items = self._known_items

        for idx in range(n):
            it = sorted_items[idx]
            if it in known_items:
                bal = item_open[it]
                open_before_sorted[idx] = bal
                exc = sorted_exceptions[idx]
                amt = sorted_amounts[idx]
                if exc == "RETURNED":
                    bal += amt
                elif exc == "DUPLICATE_PAYMENT":
                    pass
                else:
                    bal = max(0.0, bal - amt)
                open_after_sorted[idx] = bal
                item_open[it] = bal
            else:
                open_before_sorted[idx] = np.nan
                open_after_sorted[idx] = np.nan

        # Invert permutation to restore exact original order
        open_before = np.empty(n, dtype=float)
        open_after = np.empty(n, dtype=float)
        open_before[order] = open_before_sorted
        open_after[order] = open_after_sorted

        res["open_before"] = open_before
        res["open_after"] = open_after
        res["total_amount"] = [self._item_total_amount.get(it, np.nan) for it in item_ids]
        res["item_currency"] = [self._item_currency.get(it, None) for it in item_ids]
        res["item_type"] = [self._item_type.get(it, None) for it in item_ids]
        res["issue_date"] = [self._item_issue_date.get(it, None) for it in item_ids]
        res["due_date"] = [self._item_due_date.get(it, None) for it in item_ids]
        res["party_id"] = [self._item_party_id.get(it, None) for it in item_ids]
        res["known"] = [it in known_items for it in item_ids]

        return res

    def open_on(self) -> pd.DataFrame:
        """Get current balances of all items after the last apply call."""
        res = self.items.copy()
        res["open_amount"] = [self._item_open[it] for it in res["item_id"]]
        return res[ITEM_COLS]
