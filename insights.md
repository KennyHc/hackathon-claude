# Insights from branch `Luigi`: what worked, with numbers

Branch `Luigi` scores **LIVE 91.11** with local **val 92.36** (val ran about 1.2 points high on every upload).
This file lists the ideas worth copying into your pipeline. All code is on `Luigi`. Paths are given so you can read the
implementation directly.

| Version | Val | LIVE | What changed |
|---|---|---|---|
| v1 | 13.08 | 13.38 | Starter matching + line classifier |
| v2 | 90.00 | 88.89 | Cascade batch → refs → party, exception engine, PDF clauses |
| v3 | 92.36 | 91.11 | Hardening, learned bank charges, payer matching without history, duplicates, supplier refunds |

---

## 1. Architecture: a precision-first cascade over one shared ledger

- **Stages run in a fixed order: batch → refs → party → k4_sbc** (`matching/run.py`). Each bank line can be claimed
  by only one stage. Each item is capped at its open balance, and over-allocations are dropped.
  - Feeding the golden through the cascade changes only 1 of 735k rows, so the cascade itself loses nothing.
- **Precise stages go first.** Allocation precision is 0.99 and recall 0.86. A wrong allocation costs more than a
  missing one, because the line also falls out of transaction accuracy and the exception gets attached to the wrong item.
- **Every matcher returns the same columns:** `bank_txn_id, item_id, amount, exception, confidence, method`.
  - The `method` tag drives everything after it: which exceptions to trust, the per-method precision table, and the
    A/B tests.
- **Shared ledger** (`integration/lib/ledger.py`) adds `open_before` and `open_after` to each row in booking order.
  - RETURNED re-opens the item; DUPLICATE_PAYMENT doesn't reduce it.
  - Most exception rules are worked out from these two columns.

Order of the gains: batch alone 64.5 → +refs 60.2 → +party 89.1 → full pipeline (classifier + exceptions) 90 → round 2: 92.4.
Batch alone scores higher than batch+refs because, on its own, the integration step's fallback fills its leftover lines.

## 2. Batch stage (`matching/batch/`): about 40 points, pair precision ≈ 1.000

- **End-to-end ids and run ids repeat across entities.** Key them by (account, e2e) and (entity, run_id).
  - Pair precision went from 0.98 to 1.000.
- **N43/MT940 files cut e2e ids to 16 characters.** Match on prefix + account + amount.
  - For foreign-currency lines, take the amount from `orig_amount` or `/OCMT/`.
- **Aggregated payment runs mix currencies.** Convert them, then accept on transfer count plus a total within 1 %.
- **Confirming:** one debit per maturity date ("N PROVEEDORES").
  - Weekend and holiday maturities get pooled into one date range.
  - Allow maturity groups with 1–2 extra transfers.
  - Confirming went from 0.75 to 0.92 on LIVE.
- **DD returns:** match by mandate + amount to the latest collection at least 2 days earlier (median lag 7 days).
  - A return links to the collection that is re-presented within 45 days.
  - Look up the returned items in *all* AR files, not just the open ones.
- **Supplier refunds** (`aprefund.py`): a supplier receipt goes to that supplier's unique open APC credit note with the
  same amount. Exclude duplicate-payment refunds by keyword.
  - 2514/2515 on val; LIVE went from 0.39 to 0.999 (2,600 lines, the biggest single fix in round 2).
- **AP duplicate payments:** an end-to-end id ending in "…D", or a transfer prefix already used at least 7 days earlier.
  - LIVE went from 0 to 0.81.
- **Card chargebacks:** look back 6 months for the settlement and widen the keywords. LIVE went from 0.61 to 1.00.
- **Remittance advices:**
  - Reason codes:
    - a stated reason → SHORT_DISPUTE, amount = cash paid;
    - RET / 1.25 / 1.5 / 10 % → withholding, amount = full open;
    - an in-terms discount → full open amount;
    - anything else → PARTIAL.
  - Make header detection fuzzy: synonyms in EN/ES/FR/DE/IT/PT/PL, accents stripped, any column order.
  - Don't drop a whole advice for one unresolved row. Keep the resolved rows when they still reconcile with the bank amount.

## 3. Refs stage (`matching/refs/`): invoice references in the bank text

- **Strict rules run over all candidate payers before any weak rule.**
- **Recovering damaged references:** complete numbers that lost their prefix, and keep an index of the last 6 digits
  per entity.
  - Rejoin N43 references that were split across 38-character lines.
  - Fill in lists cut off by truncation (4.3k lines at 99.6 %).
  - Allow a one-typo fuzzy match (97 %).
- **Invoice PDFs** (`pdf_clauses.py`): instalment k/N, guarantee retention, advance (proforma) deduction and replaced
  invoices.
  - On val these all decide the exception at 100 %.
  - "Anulación" credit notes block the invoice they cancel.
- **What failed:** matching FX within 3 % at the reference rate without `orig_amount` (0/18 correct). Use the original
  amount instead.

## 4. Party stage (`matching/party/`): who paid, then the amount decides

- **Learn payer aliases from history** (name, IBAN) → customer.
  - **Keep several candidate payers per alias and let the amount decide.** ar_ref_noise went from 0.705 to 0.858.
- **Bounded subset-sum over that customer's open items** (at most 26 candidates).
  - Check single items first when a payer has more than 26 open items. Without that, discount matches were missed.
- **Payer matching without history** (needed for new FINAL entities and customers):
  - Find master names, trade names and IBANs anywhere in the text.
  - Accept only when exactly one candidate fits the amount strongly.
- **Learn aliases during the run** from lines that batch and refs already claimed, then run a second pass.
  - This handles new IBANs and the Moncayo camt.053 format change from 2025-10-01.
- **Test with one entity left out:** remove PL01 from alias history to imitate an unknown entity. Allocation F1 was:
  - 0.882 without history,
  - 0.955 with the two points above,
  - 0.958 with full history.
- **Match on `orig_amount`/`orig_currency` for FX lines.** ar_fx went from 0.47 to 0.82 on val (0.43 to 0.77 on LIVE).
- **Window of open items:** nearest to the booking date beat latest due by 0.1 points.
- **Disabled because imprecise:**
  - paying every member of a customer group (0 %);
  - PARTIAL on a single open item with no citation (55 %).

## 5. Exceptions (`classification/exceptions.py`, `integration/run.py`)

- **Amount convention, POLICIES §2.** This matters a lot.
  - Full clearing with an explained shortfall → the item's **full open amount**.
  - PARTIAL or SHORT_DISPUTE → the cash applied.
  - `normalize_amounts` enforces it on single-invoice lines.
- **SHORT_BANK_CHARGES is per customer, not per country.** Do NOT use a SEPA country list: the golden labels it for
  customers in CH, GB, PL and MT.
  - About 100 customers account for almost all of it. Learn them from the train golden, using only data before
    2025-01-01 for val.
  - Claim leftover receipts that pay one open invoice minus at most 2 % and at most 60 EUR, when the payer is confirmed.
  - F1 went from 0.22 to 0.79, about +1.2 points (`matching/common.py`, stage `k4_sbc`).
- **Trust the matcher's exception for methods that know it**: DD batches, card settlements, late discount, retention
  release, returns, cited duplicates.
  - The generic engine was wrong on all 32 such overrides.
  - Measure trust per method group; for remit_ref, trusting the matcher would have cost 11k rows.
- **Instalment k/N in the text** → PARTIAL except the last one (1,865/1,865 correct).
- **Overpayment guard:** if cash is above 1.35 × the open amount, the item is probably wrong, so drop the row.
  - The first threshold, 1.1, was too strict.
- **UNIDENTIFIED:** the classifier says a line settles nothing (p_items < 0.9) and no stage claimed it.
  - For entities never seen in train (ZA01), use evidence instead: no invoice-like reference and no known payer
    → UNIDENTIFIED; otherwise CUSTOMER_ADVANCE.
  - On the PL01 test: F1 0.87, against 0.50 for "every unmatched receipt".
- Exception macro-F1 on val went from 0.80 to 0.88. Every code now has precision ≥ 0.89; what's left is recall.

## 6. Line classifier (`classification/run.py`)

- **Gradient-boosted model on the bank text, amount sign and counterparty.**
  - "Settles items" AUC 0.996.
  - Category accuracy is 100 % on lines that settle nothing, across all 21 classes.
- **Train on lines before 2025 for val, and on all of train for live/final.**

## 7. Process lessons

- **Build a LIVE-shaped val** (`integration/make_val.py`): train 2025-01 to 06, with `open_items_at_start` rebuilt from
  the golden.
  - It ran 0.3 points high on v1, 1.1 on v2 and 1.25 on v3, so we used it for every decision and spent few uploads.
- **Leakage:** for val, learn only from data before 2025-01-01. We found and closed several leaks:
  - item lookups that read live/final files,
  - DD documents read from the live folder.
- **Stale caches are a hidden FINAL risk.** Alias, PDF-clause and remittance caches were reused without a freshness
  check. `integration/pipeline.sh` now deletes all `final` caches before running.
- **Build a mock FINAL** (`integration/make_mock_final.py`): a new entity, a new currency, the masters_final changes.
  Run the whole pipeline on it before 18:00. Ours: VALID in 371 s.
- **Signs a rule fits val too closely:** advance deductions scored 0.975 on val but only 0.684 on LIVE. Compare the
  per-segment LIVE output with val after each upload.

## 8. Still weak on LIVE: where the next points are

| Segment | LIVE accuracy | Lines | Notes |
|---|---|---|---|
| ar_1toN_credit_note | 0.63 | 3,300 | Biggest gap. Invoice + credit-note combinations are often not unique within 26 candidates. |
| ar_discount_late | 0.61 | 830 | Late discount → PARTIAL, but multi-invoice lines get one invoice net of discount. |
| ar_advance_deduction | 0.68 | 260 | Fits val too closely (val 0.975). |
| ar_fx | 0.77 | 760 | |
| ar_dd_represented_transfer | 0.89 | 980 | |
| OVERPAYMENT recall | 0.51 | | |
| PARTIAL recall | 0.71 | | |
