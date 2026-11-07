"""Test suite for integration.lib (fx and ledger modules).

Tests:
  1. FX library:
     - EUR = 1.0 rate and conversion
     - Saturday / Sunday uses Friday's rate
     - Convert round-trip ≈ identity (EUR and cross-currencies)
     - Vectorized vs scalar equality on 1000 random rows
     - Relative difference and is_fx
  2. Ledger library (synthetic case):
     - Partial payment reduces open balance
     - Remainder payment reduces balance to zero
     - RETURNED payment re-opens balance
     - DUPLICATE_PAYMENT leaves balance unchanged
     - open_on reflects current balances
  3. Ledger on real data (val phase):
     - Loads val items
     - Applies val golden item rows against bank transactions
     - Reports share of golden items known, PARTIAL open_after > 0.01,
       empty-exception open_after <= 0.01, and runtime.

Run:
    .venv/bin/python integration/lib/test_lib.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Add repo root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from integration.lib import fx, ledger


def test_fx():
    print("=" * 60)
    print("1. Testing FX Library (integration.lib.fx)")
    print("=" * 60)

    # Load rates
    t0 = time.time()
    rates_df = fx.load_rates()
    load_time = time.time() - t0
    print(f"Loaded {len(rates_df):,} fx rates in {load_time:.4f}s")
    assert len(rates_df) > 0, "fx_rates DataFrame is empty"

    # EUR = 1.0
    assert fx.rate("EUR", "2021-07-01") == 1.0, "EUR rate must be 1.0"
    assert fx.rate("EUR", "2023-12-25") == 1.0, "EUR rate on holiday must be 1.0"
    assert fx.to_eur(500.0, "EUR", "2021-07-01") == 500.0, "EUR to EUR must be identity"
    assert fx.convert(250.0, "EUR", "EUR", "2022-01-15") == 250.0, "EUR to EUR convert must be identity"
    print("  [PASS] EUR = 1.0 checks")

    # A Saturday uses Friday's rate
    # 2021-07-02 is Friday, 2021-07-03 is Saturday, 2021-07-04 is Sunday
    fri_rate = fx.rate("USD", "2021-07-02")
    sat_rate = fx.rate("USD", "2021-07-03")
    sun_rate = fx.rate("USD", "2021-07-04")
    assert sat_rate == fri_rate, f"Saturday rate ({sat_rate}) != Friday rate ({fri_rate})"
    assert sun_rate == fri_rate, f"Sunday rate ({sun_rate}) != Friday rate ({fri_rate})"
    print(f"  [PASS] Weekend rate propagation: Fri/Sat/Sun USD = {fri_rate}")

    # Convert round-trip ≈ identity
    test_amounts = [1.0, 100.0, 12345.67, 987654.32]
    currencies = ["USD", "GBP", "CHF", "AUD", "BRL", "CAD", "CNY", "MAD", "MXN", "PLN", "ZAR"]
    test_date = "2022-06-15"
    for amt in test_amounts:
        for c in currencies:
            # Round-trip via EUR
            conv_eur = fx.convert(amt, c, "EUR", test_date)
            back_ccy = fx.convert(conv_eur, "EUR", c, test_date)
            assert np.isclose(amt, back_ccy, rtol=1e-7, atol=1e-7), f"Round-trip failed for {c}: {amt} != {back_ccy}"
            # Cross-currency round-trip
            conv_cross = fx.convert(amt, c, "GBP", test_date)
            back_cross = fx.convert(conv_cross, "GBP", c, test_date)
            assert np.isclose(amt, back_cross, rtol=1e-7, atol=1e-7), f"Cross round-trip failed: {amt} != {back_cross}"
    print("  [PASS] Convert round-trip ≈ identity for all currencies")

    # Vectorized == scalar on 1000 random rows
    np.random.seed(42)
    n = 1000
    all_ccys = ["EUR"] + currencies
    rand_from = np.random.choice(all_ccys, size=n)
    rand_to = np.random.choice(all_ccys, size=n)
    rand_dates = pd.date_range("2021-07-01", "2025-06-30", periods=n).strftime("%Y-%m-%d").tolist()
    rand_amts = np.random.uniform(10.0, 100000.0, size=n).round(2)

    s_from = pd.Series(rand_from)
    s_to = pd.Series(rand_to)
    s_dates = pd.Series(rand_dates)
    s_amts = pd.Series(rand_amts)

    t_vec = time.time()
    vec_rates = fx.rate_v(s_from, s_dates)
    vec_converted = fx.convert_v(s_amts, s_from, s_to, s_dates)
    vec_duration = time.time() - t_vec

    max_rate_diff = 0.0
    max_conv_diff = 0.0
    for i in range(n):
        sc_r = fx.rate(rand_from[i], rand_dates[i])
        vr_r = vec_rates.iloc[i]
        diff_r = abs(sc_r - vr_r)
        if diff_r > max_rate_diff:
            max_rate_diff = diff_r

        sc_c = fx.convert(rand_amts[i], rand_from[i], rand_to[i], rand_dates[i])
        vr_c = vec_converted.iloc[i]
        diff_c = abs(sc_c - vr_c)
        if diff_c > max_conv_diff:
            max_conv_diff = diff_c

    assert max_rate_diff < 1e-11, f"Vectorized rate mismatch: {max_rate_diff}"
    assert max_conv_diff < 1e-8, f"Vectorized convert mismatch: {max_conv_diff}"
    print(f"  [PASS] Vectorized == scalar on {n} rows (vec runtime: {vec_duration:.4f}s, max diff: {max_conv_diff:.2e})")

    # Helper checks: rel_diff and is_fx
    assert fx.is_fx("EUR", "USD") is True
    assert fx.is_fx("EUR", "EUR") is False
    assert fx.is_fx("USD", "usd") is False
    diff_same = fx.rel_diff(950.0, "EUR", 1000.0, "EUR", "2022-01-01")
    assert np.isclose(diff_same, -0.05), f"rel_diff expected -0.05, got {diff_same}"
    print("  [PASS] rel_diff and is_fx checks passed\n")


def test_ledger_synthetic():
    print("=" * 60)
    print("2. Testing Ledger Synthetic Case (tiny case)")
    print("=" * 60)

    # 1 synthetic item with open balance 100.0
    item = pd.DataFrame(
        [
            {
                "item_id": "ARI-2025-0000001",
                "item_type": "ARI",
                "entity": "ES01",
                "party_id": "CES010001",
                "document_number": "INV-001",
                "issue_date": "2025-01-01",
                "due_date": "2025-01-31",
                "currency": "EUR",
                "total_amount": 100.0,
                "open_amount": 100.0,
            }
        ]
    )

    ledger_inst = ledger.Ledger(item)

    bank_lines = pd.DataFrame(
        [
            {"bank_txn_id": "BT-1", "booking_date": "2025-01-10"},
            {"bank_txn_id": "BT-2", "booking_date": "2025-01-15"},
            {"bank_txn_id": "BT-3", "booking_date": "2025-01-20"},
            {"bank_txn_id": "BT-4", "booking_date": "2025-01-25"},
        ]
    )

    # 4 allocations:
    # 1. Partial: 40.0 -> balance 100 -> 60
    # 2. Remainder: 60.0 -> balance 60 -> 0
    # 3. RETURNED: 60.0 -> balance 0 -> 60 (re-opens)
    # 4. DUPLICATE_PAYMENT: 60.0 -> leaves it at 60
    allocations = pd.DataFrame(
        [
            {"bank_txn_id": "BT-1", "item_id": "ARI-2025-0000001", "amount": 40.0, "exception": "PARTIAL"},
            {"bank_txn_id": "BT-2", "item_id": "ARI-2025-0000001", "amount": 60.0, "exception": ""},
            {"bank_txn_id": "BT-3", "item_id": "ARI-2025-0000001", "amount": 60.0, "exception": "RETURNED"},
            {"bank_txn_id": "BT-4", "item_id": "ARI-2025-0000001", "amount": 60.0, "exception": "DUPLICATE_PAYMENT"},
        ]
    )

    out = ledger_inst.apply(allocations, bank_lines)

    print("Synthetic test output:")
    print(out[["bank_txn_id", "amount", "exception", "open_before", "open_after", "known"]])

    assert out.loc[0, "known"] is True or out.loc[0, "known"] == 1
    assert out.loc[0, "open_before"] == 100.0 and out.loc[0, "open_after"] == 60.0, "PARTIAL step failed"
    assert out.loc[1, "open_before"] == 60.0 and out.loc[1, "open_after"] == 0.0, "Remainder step failed"
    assert out.loc[2, "open_before"] == 0.0 and out.loc[2, "open_after"] == 60.0, "RETURNED step failed"
    assert out.loc[3, "open_before"] == 60.0 and out.loc[3, "open_after"] == 60.0, "DUPLICATE step failed"

    # Verify open_on reflects current balances
    curr_items = ledger_inst.open_on()
    assert len(curr_items) == 1
    assert curr_items.loc[0, "open_amount"] == 60.0, f"open_on balance expected 60.0, got {curr_items.loc[0, 'open_amount']}"
    print("  [PASS] Synthetic case assertions passed\n")


def test_ledger_real_data():
    print("=" * 60)
    print("3. Testing Ledger on Real Data (phase: val)")
    print("=" * 60)

    # 1. Load val items
    t0_load = time.time()
    items = ledger.load_items("val")
    load_time = time.time() - t0_load
    print(f"Loaded {len(items):,} items for phase 'val' in {load_time:.2f}s")
    assert len(items) > 0, "No items loaded for phase val"
    for col in ledger.ITEM_COLS:
        assert col in items.columns, f"Missing item column: {col}"
    assert items["open_amount"].dtype == np.float64, "open_amount must be float64"
    assert items["total_amount"].dtype == np.float64, "total_amount must be float64"

    # 2. Read bank transactions and golden allocations
    t0_read = time.time()
    bt_path = os.path.join(ROOT, "data/val/bank_transactions.csv.gz")
    golden_path = os.path.join(ROOT, "data/val/golden_allocations.csv.gz")

    bank_lines = pd.read_csv(bt_path, dtype=str, keep_default_na=False)
    golden = pd.read_csv(golden_path, dtype=str, keep_default_na=False)

    # Filter golden rows with item_id
    golden_items = golden[golden["item_id"] != ""].copy()
    print(f"Read {len(bank_lines):,} bank lines and {len(golden_items):,} golden item allocations in {time.time() - t0_read:.2f}s")

    # 3. Apply golden allocations to ledger
    ledger_inst = ledger.Ledger(items)

    t0_apply = time.time()
    applied = ledger_inst.apply(golden_items, bank_lines)
    apply_time = time.time() - t0_apply

    # 4. Compute metrics
    # Share of golden item ids known (should be ~99.9%)
    share_known = applied["known"].mean()

    # Share of rows with exception PARTIAL whose open_after > 0.01 (should be high)
    known_applied = applied[applied["known"]]
    partial_rows = known_applied[known_applied["exception"] == "PARTIAL"]
    share_partial_positive = (partial_rows["open_after"] > 0.01).mean() if len(partial_rows) > 0 else 0.0

    # Share of empty-exception rows with open_after <= 0.01 (should be high)
    empty_exc_rows = known_applied[known_applied["exception"] == ""]
    share_empty_cleared = (empty_exc_rows["open_after"] <= 0.01).mean() if len(empty_exc_rows) > 0 else 0.0

    print("-" * 60)
    print("MEASURED RESULTS ON data/val:")
    print(f"  Share of golden item ids known:               {share_known * 100:.3f}% (target: ~99.9%)")
    print(f"  Share of PARTIAL with open_after > 0.01:      {share_partial_positive * 100:.3f}% (target: high)")
    print(f"  Share of empty-exception with open_after <= 0.01: {share_empty_cleared * 100:.3f}% (target: high)")
    print(f"  Ledger apply runtime ({len(golden_items):,} rows):      {apply_time:.3f}s (budget: <30s)")
    print("-" * 60)

    # Assertions
    assert share_known >= 0.998, f"Share of golden items known too low: {share_known:.4f}"
    assert share_partial_positive >= 0.95, f"Share of PARTIAL open_after > 0.01 too low: {share_partial_positive:.4f}"
    assert share_empty_cleared >= 0.99, f"Share of empty exception cleared too low: {share_empty_cleared:.4f}"
    assert apply_time < 30.0, f"Ledger apply too slow: {apply_time:.2f}s"
    print("  [PASS] All real-data ledger tests passed!\n")


def main():
    test_fx()
    test_ledger_synthetic()
    test_ledger_real_data()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
