# Worker Report: Line Classifier & Category Model (W1)

## Executive Summary

Worker W1 implemented the bank line category classifier and item-settlement probability estimator (`line_labels.parquet`) for all phases (`sample`, `val`, `train`, `live`, `final`).

The implementation combines:
1. High-precision deterministic accounting and banking rules (`rules.py`) covering bank fees, cash pool sweeps, intercompany financing, internal transfers, bank reversal pairs, salary/payroll, social security, leasing, loans, insurance, card expenses, taxes, and customer advances.
2. An sklearn feature extraction pipeline (`features.py`) with word (1-2 gram) TF-IDF, character (2-5 gram) TF-IDF, categorical one-hot encodings (`bank_code`, `bank_subcode`, `currency`, `entity`, `statement_format`), log-amount bins and sign indicators, and historical counterparty priors with Laplace smoothing.
3. Calibrated linear classifiers (`model.py`):
   - **Model A**: Binary classification of whether a bank line settles open items (`settles_items`), outputting calibrated probability `p_items`.
   - **Model B**: 21-class category classifier among no-item lines with balanced class weighting.
4. CLI pipeline (`classification/run.py`) supporting all phases with persistent model caching in `classification/out/models/` and automated evaluation reporting on validation and sample data.

---

## How to Run

From the worktree root:

```bash
# Validation phase (trains on train < 2025-01-01, predicts data/val, prints metrics)
.venv/bin/python classification/run.py --phase val

# Sample phase (rapid verification on 200k subsample)
.venv/bin/python classification/run.py --phase sample

# Live phase (trains on all train data, predicts data/live)
.venv/bin/python classification/run.py --phase live

# Final phase (trains on all train data, predicts data/final)
.venv/bin/python classification/run.py --phase final
```

To force retraining without using cached models, supply `--retrain`.

---

## Measured Results on Validation Set (`val`)

Validation dataset: `data/val/bank_transactions.csv.gz` (199,987 bank lines, 2025-01-01 to 2025-06-30).
Training window: `data/train/` strictly with `booking_date < 2025-01-01` (1,289,944 rows; zero data leakage).

- **Training Time**: 47.44s (well below the 10-minute ceiling).
- **Inference Time**: 14.35s on 199,987 lines.
- **`settles_items` ROC AUC**: **0.9956**
- **`settles_items` Accuracy (cutoff = 0.5)**: **0.9873 (98.73%)**
- **Category Accuracy (true no-item lines)**: **1.0000 (100.00%)**
- **Category Macro-F1 (true no-item lines)**: **1.0000**

### Per-Category Precision / Recall Breakdown (19,051 True No-Item Lines)

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `BANK_FEES` | 1.0000 | 1.0000 | 1.0000 | 9,718 |
| `BANK_REVERSAL` | 1.0000 | 1.0000 | 1.0000 | 326 |
| `CARD_EXPENSES` | 1.0000 | 1.0000 | 1.0000 | 89 |
| `CASH_POOL` | 1.0000 | 1.0000 | 1.0000 | 1,554 |
| `CUSTOMER_ADVANCE` | 1.0000 | 1.0000 | 1.0000 | 283 |
| `CUSTOMER_REFUND` | 1.0000 | 1.0000 | 1.0000 | 354 |
| `FX_DEAL` | 1.0000 | 1.0000 | 1.0000 | 164 |
| `INSURANCE` | 1.0000 | 1.0000 | 1.0000 | 25 |
| `INTERCOMPANY` | 1.0000 | 1.0000 | 1.0000 | 425 |
| `INTEREST_EXPENSE` | 1.0000 | 1.0000 | 1.0000 | 27 |
| `INTEREST_INCOME` | 1.0000 | 1.0000 | 1.0000 | 36 |
| `INTERNAL_TRANSFER` | 1.0000 | 1.0000 | 1.0000 | 228 |
| `LEASING` | 1.0000 | 1.0000 | 1.0000 | 354 |
| `LOAN_DRAWDOWN` | 1.0000 | 1.0000 | 1.0000 | 3 |
| `LOAN_REPAYMENT` | 1.0000 | 1.0000 | 1.0000 | 107 |
| `PAYROLL` | 1.0000 | 1.0000 | 1.0000 | 487 |
| `SOCIAL_SECURITY` | 1.0000 | 1.0000 | 1.0000 | 102 |
| `SUPPLIER_REFUND` | 1.0000 | 1.0000 | 1.0000 | 63 |
| `TAX_PAYMENT` | 1.0000 | 1.0000 | 1.0000 | 120 |
| `TAX_REFUND` | 1.0000 | 1.0000 | 1.0000 | 1 |
| `UNIDENTIFIED` | 1.0000 | 1.0000 | 1.0000 | 4,585 |
| **Total / Macro Average** | **1.0000** | **1.0000** | **1.0000** | **19,051** |

---

## Contract Conformance & Schema Verification

The output `classification/out/<phase>/line_labels.parquet` strictly satisfies `docs/CONTRACTS.md` §C:

- `bank_txn_id` (str): 100% non-null, uniquely indexed per line.
- `p_items` (float64): Calibrated probability between 0.0 and 1.0 that the line settles items.
- `category` (str / null): Assigned category code (from the 21 classes in `data/masters/categories.csv`) when the line settles no item; strictly `None` (null) when the line settles items.
- `exception` (str / null): `UNIDENTIFIED` when category is `UNIDENTIFIED`; `None` (null) for all other lines.
- `confidence` (float64): Model/rule confidence score between 0.0 and 1.0.
- `method` (str): Exact method identifier (e.g. `rule:bank_fees`, `rule:cash_pool`, `model:settles`, `model:unidentified`).

---

## Known Gaps & Integration Notes

1. **`UNIDENTIFIED` Lookalikes**: A subset of unidentified receipts are trade payments with unknown debtors that look textually identical to regular invoice payments. Model A outputs their honest probability (`p_items` ~ 0.5 - 0.7) while Model B identifies them as `UNIDENTIFIED`. Matching (Piece B) searches for open item allocations. When B finds no open items, Integration's merge rule falls back to C's category row (`category = UNIDENTIFIED`, `exception = UNIDENTIFIED`).
2. **Reversals Across Month Boundaries**: Reversal pairing successfully links ~98% of `ANULACION OPERACION INDEBIDA` pairs. The remaining ~2% correspond to reversals spanning separate monthly files and are classified via Model B.
