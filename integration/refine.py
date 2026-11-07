"""Two exception refinements applied after the rules library (POLICIES §3), on rows the matchers already decided.

P1c  refs:explain rows the matcher left '' while the rules say PARTIAL: label PARTIAL only when the amount is exactly
     a usual fraction of the item (open or total) cut to the cent, and not the remainder after such a fraction.
P2   single-invoice lines labelled PARTIAL where POLICIES explains the shortfall as intermediary bank charges
     (non-SEPA cross-border payer, e.g. GB/CH/PL): SHORT_BANK_CHARGES at the item's full open amount.

Validation 96.41 -> 96.46, holdout 96.50 -> 96.59, no line made worse on either (scratchpad agent a4_exceptions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from classification import exceptions as EX

FR = EX.PARTIAL_FRACTIONS


def _is_fraction(a: float, op: float, tot: float) -> bool:
    c = EX._cents(a)
    isp = any(EX._cents(EX.part_of(base, f)) == c for f in FR for base in (op, tot))
    isr = any(EX._cents(op) - EX._cents(EX.part_of(base, f)) == c for f in FR for base in (op, tot))
    return isp and not isr


def refine(alloc: pd.DataFrame, rule: pd.Series, bank: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    al = alloc.copy()
    ro = EX.running_open(al, bank, items)
    it = items.drop_duplicates("item_id").set_index("item_id")
    # P1c
    m = (al.method.fillna("") == "refs:explain") & (al.exception.fillna("") == "") & (rule.reindex(al.index) == "PARTIAL")
    idx = al.index[m]
    if len(idx):
        tot = al.loc[idx].item_id.map(it.total_amount)
        ok = np.array([_is_fraction(float(a), float(o), float(t)) if t == t else False
                       for a, o, t in zip(al.loc[idx].amount, ro[idx], tot)], dtype=bool)
        al.loc[idx[ok], "exception"] = "PARTIAL"
    # P2
    n = al.groupby("bank_txn_id").item_id.transform("size")
    cand = al[(n == 1) & (al.exception == "PARTIAL")]
    if len(cand):
        masters = EX._load_masters(bank)
        b = bank.drop_duplicates("bank_txn_id").set_index("bank_txn_id")
        for i, r in cand.iterrows():
            if r.item_id not in it.index or r.bank_txn_id not in b.index:
                continue
            item = dict(it.loc[r.item_id]); item["item_id"] = r.item_id; item["open_amount"] = float(ro[i])
            line = dict(b.loc[r.bank_txn_id]); line["bank_txn_id"] = r.bank_txn_id
            try:
                ex = EX.explain([item], abs(line["amount"]), line, masters)
            except Exception:
                ex = None
            if ex and ex[0][2] == "SHORT_BANK_CHARGES":
                al.at[i, "exception"] = "SHORT_BANK_CHARGES"
                al.at[i, "amount"] = ex[0][1]
    return al
