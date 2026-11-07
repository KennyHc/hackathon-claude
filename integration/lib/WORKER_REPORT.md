# Worker Report: W3 Shared Libraries (`fx.py` & `ledger.py`)

## 1. What was built

Implemented the shared core libraries in `integration/lib/` for use across all reconciliation workers (ingestion, matching, classification, integration):

1. **`integration/lib/__init__.py`**:
   - Exposes `fx` and `ledger` modules for direct imports:
     ```python
     sys.path.insert(0, ROOT)
     from integration.lib import fx, ledger
     ```

2. **`integration/lib/fx.py`**:
   - `load_rates(masters_dir="data/masters") -> pd.DataFrame`: Reads `fx_rates.csv`, parses `units_per_eur` as float, sets DatetimeIndex without naming conflicts, caches in memory, and builds lookup structures.
   - `rate(ccy: str, date: str) -> float`: Returns units of `ccy` per 1 EUR on `date` (EUR = 1.0; weekend/holiday uses last available rate via O(1) daily calendar cache).
   - `convert(amount: float, from_ccy: str, to_ccy: str, date: str) -> float`: Cross-currency conversion via EUR.
   - `to_eur(amount: float, ccy: str, date: str) -> float`: Direct conversion to EUR.
   - `rel_diff(bank_amount, bank_ccy, item_amount, item_ccy, date) -> float`: Relative discrepancy `(converted_bank - item_amt) / item_amt`.
   - `is_fx(bank_ccy: str, item_ccy: str) -> bool`: Currency disparity check.
   - `rate_v(ccy: pd.Series, date: pd.Series) -> pd.Series`: Vectorized rate lookup using per-currency `pd.merge_asof(direction="backward")`. Handles 1M rows in ~0.34s.
   - `convert_v(amount, from_ccy, to_ccy, date) -> pd.Series`: Vectorized conversion via EUR. Handles identical currencies with fast-path; handles 1M rows in ~0.83s.
   - Helpers: `to_eur_v`, `rel_diff_v`.

3. **`integration/lib/ledger.py`**:
   - `ITEM_COLS = ["item_id", "item_type", "entity", "party_id", "document_number", "issue_date", "due_date", "currency", "total_amount", "open_amount"]`
   - `load_items(phase: str, data_dir="data") -> pd.DataFrame`: Loads all items available in a phase. For `val`/`live`/`final`: `open_items_at_start` (open_amount as given) + `ar_invoices`, `ap_invoices`, `credit_notes` (open_amount = total_amount). For `train`/`sample`: phase files with open_amount = total_amount.
   - `class Ledger`:
     - `__init__(items: pd.DataFrame)`: Stores items with float balances and builds O(1) attribute lookup structures.
     - `apply(allocations: pd.DataFrame, bank_lines: pd.DataFrame) -> pd.DataFrame`: Maps `booking_date`, sorts in `(booking_date, bank_txn_id)` order using stable sorting, applies balance updates:
       - `RETURNED` -> `open += amount`
       - `DUPLICATE_PAYMENT` -> no change
       - anything else -> `open -= amount` (floored at 0.0)
       Inverts permutation to return a copy in the exact original index and order, enriched with `open_before`, `open_after`, `total_amount`, `item_currency`, `item_type`, `issue_date`, `due_date`, `party_id` (NaN/None if unknown), and `known` (bool).
     - `open_on() -> pd.DataFrame`: Returns current open balances after the last apply.

4. **`integration/lib/test_lib.py`**:
   - Plain assert test suite covering FX properties, synthetic ledger scenarios, and real data on phase `val`.

---

## 2. How to run

Run the test suite using the workspace Python virtual environment:

```bash
.venv/bin/python integration/lib/test_lib.py
```

---

## 3. Measured Results

### A. FX Library Benchmarks & Validation
- **Rates Loaded**: 13,706 rates in 0.026s.
- **Weekend / Holiday Propagation**: Verified Friday 2021-07-02 rate propagates through Saturday 2021-07-03 and Sunday 2021-07-04.
- **Round-trip Conversion**: Verified `convert(convert(amt, c1, c2), c2, c1) ≈ amt` across all 11 currencies; maximum discrepancy `< 1e-12`.
- **Vectorized vs Scalar Agreement (1,000 random rows)**:
  - Max rate discrepancy: `0.0`
  - Max conversion discrepancy: `2.33e-10`
- **1,000,000 Rows Throughput**:
  - `rate_v`: **0.336s** (~3M rows/sec)
  - `convert_v`: **0.831s** (~1.2M rows/sec)

### B. Ledger Synthetic Validation
- `PARTIAL` payment: balance `100.0 -> 60.0` (`open_before=100.0, open_after=60.0`).
- Remainder payment: balance `60.0 -> 0.0` (`open_before=60.0, open_after=0.0`).
- `RETURNED` reversal: balance `0.0 -> 60.0` (`open_before=0.0, open_after=60.0`).
- `DUPLICATE_PAYMENT`: balance unchanged at `60.0` (`open_before=60.0, open_after=60.0`).
- `open_on()`: balance correctly retained as `60.0`.

### C. Ledger Real Data on `data/val`
- **Items Loaded**: 1,040,151 items loaded in 1.59s.
- **Bank Lines**: 199,987 lines.
- **Golden Item Allocations**: 735,474 rows.
- **Share of Golden Item IDs Known**: **99.940%** (target: ~99.9%).
- **Share of PARTIAL Rows with `open_after > 0.01`**: **99.129%** (target: high).
- **Share of Empty-Exception Rows with `open_after <= 0.01`**: **99.984%** (target: high).
- **Ledger Apply Runtime (735,474 rows)**: **1.444s** (budget: < 30s).

---

## 4. Known Gaps and Boundary Handling
- **Missing Items / Unidentified Allocations**: Allocations with unknown or blank `item_id` are safely assigned `known=False` with NaN `open_before`, `open_after`, and `total_amount`, and `None` for string metadata columns without affecting ledger state.
- **Pre-window Val Residual**: The 0.06% of golden item IDs not in `val` items correspond to untouched legacy items from before 2021-07 that were excluded when `data/val` was sliced.
- **Floating Point Stability**: Amounts are kept as standard IEEE-754 floats and non-negative balances are strictly floored at `0.0` to prevent negative balance drift on overpayments.
