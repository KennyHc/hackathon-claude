"""Exception classification engine implementing POLICIES §3 rules.

Contract (docs/CONTRACTS.md §C):
    def assign_exceptions(allocations, bank_lines, open_items, **ctx) -> pd.Series
"""
from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from integration.lib import fx, ledger  # noqa: E402

Ledger = ledger.Ledger
MASTERS = os.path.join(ROOT, "data", "masters")

# Cache master tables in memory
_CACHED_MASTERS_DIR: Optional[str] = None
_CACHED_PAYMENT_TERMS: Optional[dict[str, tuple[float, int]]] = None
_CACHED_CUSTOMERS: Optional[dict[str, tuple[str, str]]] = None  # cid -> (country, payment_terms)
_CACHED_SUPPLIERS: Optional[dict[str, tuple[str, str]]] = None  # sid -> (country, payment_terms)

EURO_SEPA_COUNTRIES = {"ES", "FR", "DE", "IT", "PT", "NL", "BE", "AT", "LU", "IE", "FI"}
WITHHOLDING_COUNTRIES = {"MX", "BR", "MA"}
WITHHOLDING_RATES = (0.0125, 0.015, 0.10)
MATCHING_PASSTHROUGH = {"RETURNED", "DUPLICATE_PAYMENT", "SHORT_DISPUTE", "UNIDENTIFIED"}


def _load_masters(masters_dir: str):
    global _CACHED_MASTERS_DIR, _CACHED_PAYMENT_TERMS, _CACHED_CUSTOMERS, _CACHED_SUPPLIERS
    if _CACHED_MASTERS_DIR == masters_dir and _CACHED_PAYMENT_TERMS is not None:
        return

    _CACHED_MASTERS_DIR = masters_dir

    # Payment terms
    pt_path = os.path.join(masters_dir, "payment_terms.csv")
    pt_map: dict[str, tuple[float, int]] = {}
    if os.path.exists(pt_path):
        df_pt = pd.read_csv(pt_path, dtype=str, keep_default_na=False)
        for _, r in df_pt.iterrows():
            code = r["code"]
            pct_s = r["early_payment_discount_pct"]
            days_s = r["discount_days"]
            if pct_s and days_s:
                try:
                    pt_map[code] = (float(pct_s), int(days_s))
                except ValueError:
                    pass
    _CACHED_PAYMENT_TERMS = pt_map

    # Customers
    cust_path = os.path.join(masters_dir, "customers.csv")
    cust_map: dict[str, tuple[str, str]] = {}
    if os.path.exists(cust_path):
        df_c = pd.read_csv(cust_path, dtype=str, keep_default_na=False)
        for _, r in df_c.iterrows():
            cust_map[r["customer_id"]] = (r["country"], r["payment_terms"])
    _CACHED_CUSTOMERS = cust_map

    # Suppliers
    supp_path = os.path.join(masters_dir, "suppliers.csv")
    supp_map: dict[str, tuple[str, str]] = {}
    if os.path.exists(supp_path):
        df_s = pd.read_csv(supp_path, dtype=str, keep_default_na=False)
        for _, r in df_s.iterrows():
            supp_map[r["supplier_id"]] = (r["country"], r["payment_terms"])
    _CACHED_SUPPLIERS = supp_map


def _tol(amount: float) -> float:
    return max(0.01, 0.005 * abs(amount))


def assign_exceptions(
    allocations: pd.DataFrame,
    bank_lines: pd.DataFrame,
    open_items: pd.DataFrame,
    **ctx: Any
) -> pd.Series:
    """Exception per allocation row per POLICIES §3; index aligned with `allocations`.

    Precedence:
      1. keep matching's RETURNED / DUPLICATE_PAYMENT / SHORT_DISPUTE / UNIDENTIFIED guesses as given;
      2. compute open_before/open_after via Ledger(open_items).apply(allocations, bank_lines);
         DUPLICATE_PAYMENT if open_before <= 0.01 and settled earlier or known paid;
      3. compare line cash with items;
      4. FX_DIFFERENCE if line currency != item currency and fully cleared (invoices only);
      5. PARTIAL if open_after > tolerance;
         SHORT_DISCOUNT within terms and cash ≈ total - pct*total;
         SHORT_WITHHOLDING if payer MX/BR/MA and shortfall ≈ 1.25%, 1.5% or 10%;
         SHORT_BANK_CHARGES if non-SEPA, shortfall <= 60 EUR eq & <= 2%;
         OVERPAYMENT if single invoice and cash > open amount.
    """
    if allocations is None or len(allocations) == 0:
        return pd.Series([], dtype=object, index=allocations.index if allocations is not None else None)

    masters_dir = ctx.get("masters_dir", MASTERS)
    _load_masters(masters_dir)
    assert _CACHED_PAYMENT_TERMS is not None
    assert _CACHED_CUSTOMERS is not None
    assert _CACHED_SUPPLIERS is not None

    n = len(allocations)
    results: list[Optional[str]] = [None] * n

    # Step 1: Matching's pass-through guesses
    has_matching_exc = "exception" in allocations.columns
    if has_matching_exc:
        for i, val in enumerate(allocations["exception"]):
            if val in MATCHING_PASSTHROUGH:
                results[i] = val

    # Step 2: Ledger open_before / open_after computation
    open_items = open_items.copy()
    for c in ledger.ITEM_COLS:  # tolerate partial item tables (tests, matching mocks)
        if c not in open_items.columns:
            open_items[c] = open_items["item_id"].str[:3] if c == "item_type" else ""
    ledger_instance = Ledger(open_items)
    ledger_res = ledger_instance.apply(allocations, bank_lines)
    open_befores = ledger_res["open_before"].values
    open_afters = ledger_res["open_after"].values
    # already settled earlier in this run: the item is known and its balance is already ~0 (items closed before the
    # phase are not in open_items at all, so they are never flagged here; matching passes those as DUPLICATE_PAYMENT)
    known = ledger_res["known"].values.astype(bool)
    settled_earlier = known & (np.nan_to_num(open_befores, nan=1.0) <= 0.01)
    is_known_paid = settled_earlier

    item_ids = allocations["item_id"].tolist()
    alloc_amts = allocations["amount"].tolist()

    for i in range(n):
        if results[i] is not None:
            continue
        iid = item_ids[i]
        if not iid:
            continue
        # DUPLICATE_PAYMENT rule: open_before <= 0.01 and (settled earlier or known paid)
        if open_befores[i] <= 0.01 and (settled_earlier[i] or is_known_paid[i]):
            if iid.startswith(("ARI", "API")):
                try:
                    amt_val = float(alloc_amts[i]) if alloc_amts[i] != "" else 0.0
                except ValueError:
                    amt_val = 0.0
                if amt_val > 0.01:
                    results[i] = "DUPLICATE_PAYMENT"

    # Pre-build lookup for bank lines
    bl_map: dict[str, dict[str, Any]] = {}
    if bank_lines is not None and len(bank_lines) > 0:
        for r in bank_lines.itertuples(index=False):
            t_id = getattr(r, "bank_txn_id", "")
            if t_id:
                bl_map[t_id] = {
                    "booking_date": getattr(r, "booking_date", ""),
                    "amount": float(getattr(r, "amount", 0.0) or 0.0),
                    "currency": getattr(r, "currency", ""),
                    "orig_amount": getattr(r, "orig_amount", ""),
                    "orig_currency": getattr(r, "orig_currency", ""),
                    "text": getattr(r, "text", "")
                }

    # Pre-build lookup for open items
    item_map: dict[str, dict[str, Any]] = {}
    if open_items is not None and len(open_items) > 0:
        has_pt = "payment_terms" in open_items.columns
        for r in open_items.itertuples(index=False):
            iid = getattr(r, "item_id", "")
            if iid:
                party_id = getattr(r, "party_id", "")
                pt_term = getattr(r, "payment_terms", "") if has_pt else ""
                tot_amt_str = getattr(r, "total_amount", "0.0")
                try:
                    tot_amt = float(tot_amt_str) if tot_amt_str != "" else 0.0
                except ValueError:
                    tot_amt = 0.0
                item_map[iid] = {
                    "currency": getattr(r, "currency", ""),
                    "total_amount": tot_amt,
                    "issue_date": getattr(r, "issue_date", ""),
                    "party_id": party_id,
                    "payment_terms": pt_term,
                    "item_type": getattr(r, "item_type", iid[:3])
                }

    # Group allocations by bank_txn_id
    txn_alloc_indices: dict[str, list[int]] = {}
    bank_txn_ids = allocations["bank_txn_id"].tolist()
    for idx, t_id in enumerate(bank_txn_ids):
        if t_id not in txn_alloc_indices:
            txn_alloc_indices[t_id] = []
        txn_alloc_indices[t_id].append(idx)

    # Process per bank line
    for t_id, indices in txn_alloc_indices.items():
        bl_info = bl_map.get(t_id)
        if not bl_info:
            continue

        b_date = bl_info["booking_date"]
        b_amt = bl_info["amount"]
        b_curr = bl_info["currency"]
        cash = abs(b_amt)

        # Retrieve items for this line
        line_items: list[dict[str, Any]] = []
        for idx in indices:
            iid = item_ids[idx]
            amt_s = alloc_amts[idx]
            try:
                amt = float(amt_s) if amt_s != "" else 0.0
            except ValueError:
                amt = 0.0

            it_info = item_map.get(iid, {})
            i_curr = it_info.get("currency", "") or b_curr
            tot_amt = it_info.get("total_amount", amt)
            issue_date = it_info.get("issue_date", "")
            party_id = it_info.get("party_id", "")
            p_terms = it_info.get("payment_terms", "")
            if pd.isna(p_terms) or str(p_terms) == "nan":
                p_terms = ""

            # Look up payment terms and country if not present
            country = ""
            if iid.startswith(("ARI", "ARC")):
                cust_meta = _CACHED_CUSTOMERS.get(party_id)
                if cust_meta:
                    country = cust_meta[0]
                    if not p_terms:
                        p_terms = cust_meta[1]
            elif iid.startswith(("API", "APC")):
                supp_meta = _CACHED_SUPPLIERS.get(party_id)
                if supp_meta:
                    country = supp_meta[0]
                    if not p_terms:
                        p_terms = supp_meta[1]

            itype = it_info.get("item_type", iid[:3])

            line_items.append({
                "idx": idx,
                "item_id": iid,
                "amount": amt,
                "total_amount": tot_amt,
                "currency": i_curr,
                "issue_date": issue_date,
                "party_id": party_id,
                "payment_terms": p_terms,
                "country": country,
                "item_type": itype,
                "open_before": open_befores[idx],
                "open_after": open_afters[idx]
            })

        # Step 4 & 5: Check each item
        # Rule 4: FX_DIFFERENCE per item where line currency != item currency
        for it in line_items:
            idx = it["idx"]
            if results[idx] is not None:
                continue
            if not it["item_id"]:
                continue

            tol = _tol(it["open_before"])
            if it["currency"] != b_curr:
                # FX row
                if it["open_after"] > tol:
                    results[idx] = "PARTIAL"
                else:
                    # Item fully cleared
                    if it["item_type"] in ("ARI", "API"):
                        results[idx] = "FX_DIFFERENCE"
                    else:
                        results[idx] = None
            else:
                # Non-FX row: if open_after > tolerance, it is PARTIAL
                if it["open_after"] > tol:
                    results[idx] = "PARTIAL"

        # Now examine remaining fully cleared items against line cash
        # Single invoice overpayment: cash > alloc amount + 0.01
        if len(line_items) == 1 and line_items[0]["item_id"]:
            it = line_items[0]
            idx = it["idx"]
            if results[idx] is None and it["item_type"] in ("ARI", "API") and it["currency"] == b_curr:
                if cash > it["amount"] + 0.01:
                    results[idx] = "OVERPAYMENT"
                    continue

        # Remaining undecided items (must be non-FX invoices or credit notes)
        rem_invoices = [
            it for it in line_items
            if it["item_id"] and results[it["idx"]] is None and it["item_type"] in ("ARI", "API")
        ]
        credit_notes = [
            it for it in line_items
            if it["item_id"] and it["item_type"] in ("ARC", "APC")
        ]
        handled_invoices = [
            it for it in line_items
            if it["item_id"] and results[it["idx"]] is not None and it["item_type"] in ("ARI", "API")
        ]

        if not rem_invoices:
            # Credit notes stay empty
            for cn in credit_notes:
                if results[cn["idx"]] is None:
                    results[cn["idx"]] = None
            continue

        # Cash allocated to remaining items:
        # cash_rem = cash - cash_applied_to_handled_invoices + credit_notes_netted
        cn_cash = sum(cn["amount"] for cn in credit_notes)
        handled_cash = sum(hi["amount"] for hi in handled_invoices)
        cash_rem = cash - handled_cash + cn_cash
        gross_rem = sum(it["amount"] for it in rem_invoices)
        diff_rem = gross_rem - cash_rem
        rem_tol = _tol(gross_rem)

        # Check 1: SHORT_WITHHOLDING
        # Payer country in MX/BR/MA and shortfall ≈ 1.25%, 1.5% or 10%
        withholding_matched = False
        first_country = rem_invoices[0]["country"] if rem_invoices else ""
        if first_country in WITHHOLDING_COUNTRIES:
            for rate in WITHHOLDING_RATES:
                exp_wh = sum(round(rate * it["total_amount"], 2) for it in rem_invoices)
                if abs(diff_rem - exp_wh) <= max(0.05, 0.005 * exp_wh) or abs(diff_rem - round(rate * gross_rem, 2)) <= max(0.05, 0.005 * diff_rem):
                    withholding_matched = True
                    for it in rem_invoices:
                        results[it["idx"]] = "SHORT_WITHHOLDING"
                    for cn in credit_notes:
                        if results[cn["idx"]] is None:
                            results[cn["idx"]] = None
                    break

        if withholding_matched:
            continue

        # Check 2: SHORT_DISCOUNT
        # payment_terms has early discount, booking_date <= issue_date + discount_days + 3
        discount_matched = False
        exp_discount = 0.0
        eligible_indices: list[int] = []

        for it in rem_invoices:
            terms = it["payment_terms"]
            pt_info = _CACHED_PAYMENT_TERMS.get(terms)
            if not pt_info:
                continue
            pct, d_days = pt_info
            iss_date_str = it["issue_date"]
            if not (iss_date_str and b_date):
                continue

            try:
                iss_dt = datetime.date.fromisoformat(iss_date_str)
                b_dt = datetime.date.fromisoformat(b_date)
                deadline = iss_dt + datetime.timedelta(days=d_days + 3)
                if b_dt <= deadline:
                    disc_amt = round(it["total_amount"] * pct / 100.0, 2)
                    exp_discount += disc_amt
                    eligible_indices.append(it["idx"])
            except ValueError:
                pass

        if eligible_indices and (abs(diff_rem - exp_discount) <= max(0.05, 0.005 * exp_discount)):
            discount_matched = True
            for it in rem_invoices:
                if it["idx"] in eligible_indices:
                    results[it["idx"]] = "SHORT_DISCOUNT"
                else:
                    results[it["idx"]] = None
            for cn in credit_notes:
                if results[cn["idx"]] is None:
                    results[cn["idx"]] = None

        if discount_matched:
            continue

        # Check 3: SHORT_BANK_CHARGES
        # Non-SEPA payer, single invoice, shortfall <= 60 EUR eq & <= 2% of the item
        payer_party = rem_invoices[0]["party_id"] if rem_invoices else ""
        is_non_sepa = first_country not in EURO_SEPA_COUNTRIES and bool(first_country)
        is_card = "CARD SETTLEMENT" in bl_info["text"] or payer_party.startswith("CWEB")

        if is_non_sepa and not is_card and len(rem_invoices) == 1 and diff_rem > 0.01:
            it = rem_invoices[0]
            diff_eur = fx.convert(diff_rem, b_curr, "EUR", b_date)
            pct_short = diff_rem / it["total_amount"] if it["total_amount"] else 1.0
            if diff_eur <= 60.01 and pct_short <= 0.0201:
                results[it["idx"]] = "SHORT_BANK_CHARGES"
                for cn in credit_notes:
                    if results[cn["idx"]] is None:
                        results[cn["idx"]] = None
                continue

        if abs(diff_rem) <= rem_tol:
            # Paid in full as documented
            for it in rem_invoices:
                results[it["idx"]] = None
            for cn in credit_notes:
                if results[cn["idx"]] is None:
                    results[cn["idx"]] = None
            continue

        # Fallback for remaining items
        for it in rem_invoices:
            results[it["idx"]] = None
        for cn in credit_notes:
            if results[cn["idx"]] is None:
                results[cn["idx"]] = None

    return pd.Series(results, index=allocations.index, dtype=object)
