"""Synthetic unit tests for classification/exceptions.py covering POLICIES §3 rules.

Run with:
    .venv/bin/python classification/test_exceptions.py
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from classification.exceptions import assign_exceptions


def test_matching_passthrough():
    """Rule 1: keep matching's RETURNED / DUPLICATE_PAYMENT / SHORT_DISPUTE / UNIDENTIFIED guesses."""
    allocations = pd.DataFrame({
        "bank_txn_id": ["BT-1", "BT-2", "BT-3", "BT-4"],
        "item_id": ["ARI-2025-001", "ARI-2025-002", "ARI-2025-003", ""],
        "amount": ["1000.00", "1000.00", "500.00", ""],
        "exception": ["RETURNED", "DUPLICATE_PAYMENT", "SHORT_DISPUTE", "UNIDENTIFIED"]
    })
    bank_lines = pd.DataFrame({
        "bank_txn_id": ["BT-1", "BT-2", "BT-3", "BT-4"],
        "booking_date": ["2025-01-10"] * 4,
        "amount": ["-1000.00", "1000.00", "500.00", "100.00"],
        "currency": ["EUR"] * 4
    })
    open_items = pd.DataFrame({
        "item_id": ["ARI-2025-001", "ARI-2025-002", "ARI-2025-003"],
        "currency": ["EUR"] * 3,
        "total_amount": ["1000.00", "1000.00", "1000.00"],
        "open_amount": ["1000.00", "1000.00", "1000.00"]
    })
    res = assign_exceptions(allocations, bank_lines, open_items)
    assert res.iloc[0] == "RETURNED", f"Expected RETURNED, got {res.iloc[0]}"
    assert res.iloc[1] == "DUPLICATE_PAYMENT", f"Expected DUPLICATE_PAYMENT, got {res.iloc[1]}"
    assert res.iloc[2] == "SHORT_DISPUTE", f"Expected SHORT_DISPUTE, got {res.iloc[2]}"
    assert res.iloc[3] == "UNIDENTIFIED", f"Expected UNIDENTIFIED, got {res.iloc[3]}"
    print("✓ test_matching_passthrough passed")


def test_duplicate_payment_ledger():
    """Rule 2: item settled earlier in phase paid again -> DUPLICATE_PAYMENT."""
    allocations = pd.DataFrame({
        "bank_txn_id": ["BT-1", "BT-2"],
        "item_id": ["ARI-2025-010", "ARI-2025-010"],
        "amount": ["500.00", "500.00"],
        "exception": ["", ""]
    })
    bank_lines = pd.DataFrame({
        "bank_txn_id": ["BT-1", "BT-2"],
        "booking_date": ["2025-01-05", "2025-01-15"],
        "amount": ["500.00", "500.00"],
        "currency": ["EUR", "EUR"]
    })
    open_items = pd.DataFrame({
        "item_id": ["ARI-2025-010"],
        "currency": ["EUR"],
        "total_amount": ["500.00"],
        "open_amount": ["500.00"]
    })
    res = assign_exceptions(allocations, bank_lines, open_items)
    assert res.iloc[0] is None, f"First payment should be clean, got {res.iloc[0]}"
    assert res.iloc[1] == "DUPLICATE_PAYMENT", f"Second payment should be DUPLICATE_PAYMENT, got {res.iloc[1]}"
    print("✓ test_duplicate_payment_ledger passed")


def test_fx_difference():
    """Rule 4: line currency != item currency and fully cleared -> FX_DIFFERENCE on invoice, empty on CN."""
    allocations = pd.DataFrame({
        "bank_txn_id": ["BT-1", "BT-1", "BT-2"],
        "item_id": ["ARI-2025-020", "ARC-2025-001", "ARI-2025-021"],
        "amount": ["1000.00", "100.00", "400.00"],
        "exception": ["", "", ""]
    })
    bank_lines = pd.DataFrame({
        "bank_txn_id": ["BT-1", "BT-2"],
        "booking_date": ["2025-01-10", "2025-01-12"],
        "amount": ["850.00", "300.00"],
        "currency": ["EUR", "EUR"]
    })
    open_items = pd.DataFrame({
        "item_id": ["ARI-2025-020", "ARC-2025-001", "ARI-2025-021"],
        "currency": ["USD", "USD", "USD"],
        "total_amount": ["1000.00", "100.00", "1000.00"],
        "open_amount": ["1000.00", "100.00", "1000.00"]
    })
    res = assign_exceptions(allocations, bank_lines, open_items)
    # Fully cleared USD invoice in EUR line -> FX_DIFFERENCE
    assert res.iloc[0] == "FX_DIFFERENCE", f"Expected FX_DIFFERENCE, got {res.iloc[0]}"
    # Credit note row stays empty
    assert res.iloc[1] is None, f"Expected None for ARC, got {res.iloc[1]}"
    # Partial USD invoice (400 of 1000 cleared) -> PARTIAL
    assert res.iloc[2] == "PARTIAL", f"Expected PARTIAL, got {res.iloc[2]}"
    print("✓ test_fx_difference passed")


def test_partial():
    """Rule 5: open_after > tolerance -> PARTIAL."""
    allocations = pd.DataFrame({
        "bank_txn_id": ["BT-1"],
        "item_id": ["ARI-2025-030"],
        "amount": ["400.00"],
        "exception": [""]
    })
    bank_lines = pd.DataFrame({
        "bank_txn_id": ["BT-1"],
        "booking_date": ["2025-01-10"],
        "amount": ["400.00"],
        "currency": ["EUR"]
    })
    open_items = pd.DataFrame({
        "item_id": ["ARI-2025-030"],
        "currency": ["EUR"],
        "total_amount": ["1000.00"],
        "open_amount": ["1000.00"]
    })
    res = assign_exceptions(allocations, bank_lines, open_items)
    assert res.iloc[0] == "PARTIAL", f"Expected PARTIAL, got {res.iloc[0]}"
    print("✓ test_partial passed")


def test_overpayment():
    """Rule 5: single invoice, cash > open amount -> OVERPAYMENT."""
    allocations = pd.DataFrame({
        "bank_txn_id": ["BT-1"],
        "item_id": ["ARI-2025-040"],
        "amount": ["1000.00"],
        "exception": [""]
    })
    bank_lines = pd.DataFrame({
        "bank_txn_id": ["BT-1"],
        "booking_date": ["2025-01-10"],
        "amount": ["1050.00"],
        "currency": ["EUR"]
    })
    open_items = pd.DataFrame({
        "item_id": ["ARI-2025-040"],
        "currency": ["EUR"],
        "total_amount": ["1000.00"],
        "open_amount": ["1000.00"]
    })
    res = assign_exceptions(allocations, bank_lines, open_items)
    assert res.iloc[0] == "OVERPAYMENT", f"Expected OVERPAYMENT, got {res.iloc[0]}"
    print("✓ test_overpayment passed")


def test_short_discount():
    """Rule 5: early discount within terms -> SHORT_DISCOUNT."""
    allocations = pd.DataFrame({
        "bank_txn_id": ["BT-1"],
        "item_id": ["ARI-2025-050"],
        "amount": ["1000.00"],
        "exception": [""]
    })
    # D2N30 has 2% discount within 10 days + 3 days grace = 13 days
    # Issue date: 2025-01-01, booking: 2025-01-10 -> within deadline
    # Total: 1000.00, discount 2% = 20.00, cash = 980.00
    bank_lines = pd.DataFrame({
        "bank_txn_id": ["BT-1"],
        "booking_date": ["2025-01-10"],
        "amount": ["980.00"],
        "currency": ["EUR"]
    })
    open_items = pd.DataFrame({
        "item_id": ["ARI-2025-050"],
        "currency": ["EUR"],
        "total_amount": ["1000.00"],
        "open_amount": ["1000.00"],
        "issue_date": ["2025-01-01"],
        "payment_terms": ["D2N30"],
        "party_id": ["CES01000001"]
    })
    res = assign_exceptions(allocations, bank_lines, open_items)
    assert res.iloc[0] == "SHORT_DISCOUNT", f"Expected SHORT_DISCOUNT, got {res.iloc[0]}"
    print("✓ test_short_discount passed")


def test_short_withholding():
    """Rule 5: payer country in MX/BR/MA with 1.25%, 1.5% or 10% shortfall -> SHORT_WITHHOLDING."""
    allocations = pd.DataFrame({
        "bank_txn_id": ["BT-1"],
        "item_id": ["ARI-2025-060"],
        "amount": ["10000.00"],
        "exception": [""]
    })
    # 10% withholding on 10,000.00 BRL = 1,000.00, cash = 9,000.00
    bank_lines = pd.DataFrame({
        "bank_txn_id": ["BT-1"],
        "booking_date": ["2025-01-10"],
        "amount": ["9000.00"],
        "currency": ["BRL"]
    })
    open_items = pd.DataFrame({
        "item_id": ["ARI-2025-060"],
        "currency": ["BRL"],
        "total_amount": ["10000.00"],
        "open_amount": ["10000.00"],
        "party_id": ["CBR01000001"]
    })
    # In data/masters/customers.csv, CBR01000001 is country BR
    res = assign_exceptions(allocations, bank_lines, open_items)
    assert res.iloc[0] == "SHORT_WITHHOLDING", f"Expected SHORT_WITHHOLDING, got {res.iloc[0]}"
    print("✓ test_short_withholding passed")


def test_short_bank_charges():
    """Rule 5: non-SEPA payer, shortfall <= 60 EUR eq & <= 2% -> SHORT_BANK_CHARGES."""
    allocations = pd.DataFrame({
        "bank_txn_id": ["BT-1"],
        "item_id": ["ARI-2025-070"],
        "amount": ["5000.00"],
        "exception": [""]
    })
    # Shortfall: 5000.00 - 4975.00 = 25.00 CAD (<= 60 EUR eq and <= 2% of 5000)
    bank_lines = pd.DataFrame({
        "bank_txn_id": ["BT-1"],
        "booking_date": ["2025-01-10"],
        "amount": ["4975.00"],
        "currency": ["CAD"],
        "text": ["WIRE IN /ORG/POOLS ROBERTS LTD. /RFB/PAY-001"]
    })
    open_items = pd.DataFrame({
        "item_id": ["ARI-2025-070"],
        "currency": ["CAD"],
        "total_amount": ["5000.00"],
        "open_amount": ["5000.00"],
        "party_id": ["CUS01002538"]  # CA country in customers.csv
    })
    res = assign_exceptions(allocations, bank_lines, open_items)
    assert res.iloc[0] == "SHORT_BANK_CHARGES", f"Expected SHORT_BANK_CHARGES, got {res.iloc[0]}"
    print("✓ test_short_bank_charges passed")


def test_normal_exact_clearing():
    """Rule 5: normal exact clearing without exception -> None."""
    allocations = pd.DataFrame({
        "bank_txn_id": ["BT-1"],
        "item_id": ["ARI-2025-080"],
        "amount": ["1200.00"],
        "exception": [""]
    })
    bank_lines = pd.DataFrame({
        "bank_txn_id": ["BT-1"],
        "booking_date": ["2025-01-10"],
        "amount": ["1200.00"],
        "currency": ["EUR"]
    })
    open_items = pd.DataFrame({
        "item_id": ["ARI-2025-080"],
        "currency": ["EUR"],
        "total_amount": ["1200.00"],
        "open_amount": ["1200.00"]
    })
    res = assign_exceptions(allocations, bank_lines, open_items)
    assert res.iloc[0] is None, f"Expected None for exact clearing, got {res.iloc[0]}"
    print("✓ test_normal_exact_clearing passed")


def main():
    print("=== Running synthetic tests for exceptions.py ===")
    test_matching_passthrough()
    test_duplicate_payment_ledger()
    test_fx_difference()
    test_partial()
    test_overpayment()
    test_short_discount()
    test_short_withholding()
    test_short_bank_charges()
    test_normal_exact_clearing()
    print("\nAll synthetic tests passed successfully!")


if __name__ == "__main__":
    main()
