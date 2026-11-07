# Scenario catalogue: how the dataset's case types are generated

Reverse-engineered from the TRAIN answer key (`golden_allocations.csv.gz`, `segment` column) on the LIVE-shaped
window **TRAIN 2025-01-01 → 2025-06-30** (199,987 bank lines, 754k golden rows). Every number below is measured.
Columns used:
- **lines**: bank lines in the segment.
- **items/line**: golden item rows per line.
- **cash = Σ**: the bank amount equals Σ golden amounts (credit notes negative), same currency.
- **ref hit**: one of the item's document numbers (full, or its last 6 digits) appears in the bank text.
- **days**: median days from the oldest item's issue date to booking.

Use it as a spec: each scenario says **what the generator did** and **how to recognise it**.

---

## 1. Customer transfers (AR receipts, by transfer)

| segment | lines | diff | items/line | cash = Σ | ref hit | days | exceptions | what it is |
|---|---|---|---|---|---|---|---|---|
| `ar_ref_noise` | 42,462 | 2 | 2.2 | 100 % | **15 %** | 52 | none | Customer pays 1–N of **its own** open invoices. Text has no usable number: `PAYMENT`, `ACCOUNT PAYMENT`, `/REMI/PAYMENT`, its own reference, or a mangled number. Amounts are exact. → **identify payer + subset-sum of its open invoices (oldest first)**. |
| `ar_1to1` | 19,143 | 1 | 1.0 | 100 % | 96 % | 42 | none | One invoice, number in text (sometimes without prefix), exact amount. |
| `ar_1toN_text` | 6,124 | 2 | 5.9 | 100 % | 74 % | 80 | 0.4 % SHORT_DISCOUNT / PARTIAL | Several numbers listed in text (`RE-NR RE25… RE25…`), often **cut at 140 chars** (camt Ustrd) → missing ones complete the sum. |
| `ar_1toN_credit_note` | 3,184 | 2 | 10.3 | 99 % | 29 % | 85 | 0.8 % | Invoices minus credit notes (ARC rows with full positive amount), usually only a few numbers in text. |
| `ar_n_to_m` | 1,857 | 3 | 3.7 | 100 % | 38 % | 104 | 13 % PARTIAL | A line pays some invoices fully and **one partially** (e.g. 7,619.95 of 15,239.89 = exactly 50 %); another line pays the rest. Text: `REGLEMENT FACT FA240130337 110012` (second number without prefix). |
| `ar_same_amount_fifo` | 6,151 | 3 | 1.0 | 100 % | **0 %** | 54 | (FX 0.4 %) | Customer with several open invoices of **identical amount** (monthly fees), no reference → **oldest open** wins. |
| `ar_same_amount_month` | 4,242 | 3 | 1.0 | 100 % | 0 % | 46 | (FX 0.6 %) | Same, but text names the service month (`SERVICE MAY 2025`) → invoice issued at the **start of the following month**. |
| `ar_overpayment` | 400 | 2 | 1.0 | 2 % | 45 % | 42 | OVERPAYMENT | One invoice, cash **> open amount** (median ×1.010; e.g. +1.00 or +73.77). Row amount = open amount. |
| `ar_duplicate_payment` | 641 | 2 | 1.0 | 100 % | 44 % | 50 | DUPLICATE_PAYMENT | Exact repeat payment of an invoice **already settled earlier** (no longer in open items). |
| `ar_partial` | 3,679 | 2 | 1.0 | 100 % | 54 % | 68 | 51 % PARTIAL | `A CUENTA` style: cash < open amount. The first payment is PARTIAL; the **remainder is a separate line with no exception** (the other ~49 %). |
| `ar_new_iban` | 278 | 3 | 1.7 | 100 % | 0 % | 52 | none | Payer uses an IBAN not in the master (a PDF notice announced it); text has only the name. Name + amount. |

## 2. Customer receipts with an explained difference

| segment | lines | cash/Σ (median) | rule observed |
|---|---|---|---|
| `ar_short_discount` | 6,317 | **0.9800** | `cash = total − pct·total` (2 % terms), paid **≈ 6 days** after issue (inside discount days + 3 grace). Row amount = full total, `SHORT_DISCOUNT`. 13 % of rows are other items of the same line with no exception. |
| `ar_discount_late` | 788 | – | Same deduction but **after** the deadline (median 48 days) → `PARTIAL`, amount = cash for that invoice; the line's other invoices are normal. |
| `ar_short_withholding` | 1,401 | **0.985** | MX/BR/MA customers, 1.25 % / 1.5 % / 10 % of the total withheld. 3.2 items/line, amount = full total. 5 % of rows SHORT_DISPUTE. |
| `ar_short_bank_charges` | 169 | 0.9963 | Non-SEPA wire (`WIRE IN /ORG/…`, e.g. CAD into US01) short by a small fee (e.g. 29.60 or 37.00), ≤ 60 EUR and ≤ 2 %. The payer name may differ from the invoiced customer. |
| `ar_fx` | 725 | – | **100 % cross-currency**: 19 invoices per line, all `FX_DIFFERENCE` except credit notes (empty) and disputes. |
| `ar_dispute_settlement` | 1,103 | 1.000 | Later payment of the **remaining open amount** of a disputed invoice (amount ≠ total in 100 %, ref hit 93 %, ~123 days). No exception. |

## 3. Remittance-advice payments (big customers)

| segment | lines | items/line | ref hit | exceptions |
|---|---|---|---|---|
| `ar_short_dispute` | 1,067 | **98** | 2 % | 2.7 % SHORT_DISPUTE, 1 % SHORT_DISCOUNT, 0.5 % PARTIAL |
| `ar_1toN_remittance` | 1,206 | **45** | 0 % | **17 % SHORT_DISCOUNT**, 0.8 % PARTIAL |
| `ar_1to1_remittance` | 7 | 1 | 0 % | – |

The bank text carries only the payer's payment reference (`REMITTANCE 9053060237`, `PAYMENT 2071000409`,
`AVISO 2029002386`); the list of documents exists **only** in `remittance_advices/`. The payer is often a group entity
paying for the invoiced customer (`ABWA+<customer>`). Items are old (median 130–182 days).

## 4. Batches (one line = many receipts)

| segment | lines | items/line | how |
|---|---|---|---|
| `ar_dd_batch` | 1,530 | **61.5** | credit = `batch_total` of a `direct_debit_batches/` file; each receipt pays one invoice exactly. |
| `ar_dd_returned` | 3,512 | 1.0 | debit returning one receipt (R-code, mandate `VTM-…`) → `RETURNED`, amount = original allocation; item already settled. |
| `ar_dd_represented_transfer` | 956 | 1.0 | customer re-pays a returned receipt by transfer (69 % ref hit). No exception. |
| `ar_card_settlement` | 1,252 | 17.2 | cash = Σ net (**ratio 0.986** = acquirer fee); rows at **gross**, no exception. Booked ~2 days after the orders. |
| `ar_cheque_deposit` | 1,950 | 3.2 | `cheque_deposits/` total; stub reference often incomplete. |
| `ar_card_chargeback` / `ar_cheque_returned` | 36 / 66 | 1.0 | debit → `RETURNED` on the original order/invoice. |

## 5. Supplier payments (AP debits)

| segment | lines | items/line | ref hit | what it is |
|---|---|---|---|---|
| `ap_payment_multi` | 24,058 | 3.6 | 75 % | one transfer per supplier from a `payment_runs/` file (`FRAS INV… INV…`, e2e `PR…`). |
| `ap_payment_single` | 17,368 | 1.0 | 87 % | same, one invoice. |
| `ap_paid_by_holding` | 4,798 | 2.6 | 12 % | **ES00** pays ES01/ES02/PT01 invoices (`PAGO TRANSFERENCIAS \| 4799-56-…`, no names) → run file only. |
| `ap_direct_debit` | 5,444 | 1.0 | 35 % | supplier pulls money (`SEPA-LASTSCHRIFT MREF+… CRED+…`); no run file; exact invoice amount, ~30 days after issue. |
| `ap_credit_note_netted` | 2,591 | 5.2 | 62 % | run nets supplier credit notes (APC rows, positive). |
| `ap_partial` | 2,536 | 4.9 | 58 % | one document paid partially in the run → `PARTIAL` (21 % of rows). |
| `ap_confirming` | 788 | 10.3 | 0 % | confirming remittance: one debit per maturity date. |
| `ap_factored` | 1,376 | 2.6 | 71 % | invoices assigned to a factor: beneficiary ≠ supplier (factoring PDF). 2 % PARTIAL. |
| `ap_fx` | 1,217 | **37** | 4 % | USD/other invoices paid from a EUR account (97 % cross-currency) → `FX_DIFFERENCE` on fully paid invoices, PARTIAL kept, CNs empty. |
| `ap_supplier_refund` | 2,515 | 1.0 | 56 % | **incoming** money: supplier refunds one of its credit notes → settles the `APC` item (amount = CN total). |
| `ap_returned_transfer` | 171 | 2.8 | 0 % | our transfer bounced (credit) → `RETURNED` on each item of the original transfer. |
| `ap_duplicate_payment` | 66 | 1.0 | 86 % | we paid a supplier invoice twice → `DUPLICATE_PAYMENT`. |

## 6. PDF small print (clauses only in `pdfs/`)

| segment | lines | rule observed |
|---|---|---|
| `ar_instalment` | 4,275 | amount ≠ total in **100 %**; 70 % PARTIAL (not the last instalment), 29 % empty (last). Text often `RATE 2/3`. |
| `ar_retention` | 928 | first payment = total − 5/10 % → **always PARTIAL**. |
| `ar_retention_release` | 710 | paid **~413 days** after issue (≈ 12 months + terms), the retained remainder, no exception. |
| `ar_advance_deduction` | 275 | total − advance (proforma paid earlier) → **always PARTIAL**. |
| `ar_replaced_invoice` | 253 | text cites the **cancelled** number; golden settles the replacement (93 % exact sums). |
| `ar_credit_note_paid_out` | 43 | **outgoing** transfer paying a rappel credit note → settles the `ARC` item; ref hit 100 %. |
| `ar_third_party_parent` / `_confirming` / `_agency` | 175 / 747 / 356 | parent pays the group's invoices (9.5/line); confirming bank `ORD. <customer>`; collection agency `EXPTE` (old invoices, mixed exceptions). |

## 7. Lines with no item (category only)

| segment | lines | category | in/out |
|---|---|---|---|
| `nonitem_bank_fees` / `_intl` | 6,625 / 3,093 | BANK_FEES | out (the intl fee is a separate line the same day as a wire) |
| `unidentified` | 3,221 | UNIDENTIFIED + exception | in |
| `unidentified_lookalike` | 1,364 | UNIDENTIFIED + exception | in, **initials-only payer** (`A.R.`, `B.M.`), amount equals some unrelated open invoice. Never match on amount alone. |
| `nonitem_cash_pool` | 1,554 | CASH_POOL | both (`ZBA SWEEP …`) |
| `nonitem_payroll`, `_social_security`, `_tax_payment`, `_leasing`, `_loan_repayment`, `_insurance`, `_card_expenses`, `_interest_expense` | 487 … 25 | as named | out |
| `nonitem_intercompany`, `_internal_transfer`, `_bank_reversal` | 425 / 228 / 326 | as named | both legs (50/50) |
| `nonitem_fx_deal` | 164 | FX_DEAL | both legs |
| `customer_advance` | 283 | CUSTOMER_ADVANCE | in, text cites a **proforma** `PF25-…` |
| `customer_refund` / `ap_duplicate_refund` | 354 / 63 | CUSTOMER_REFUND / SUPPLIER_REFUND | out / in |
| `nonitem_interest_income`, `_loan_drawdown`, `_tax_refund` | 36 / 3 / 1 | as named | in |

---

## Takeaways for the matchers

1. **Amounts are exact unless an exception says otherwise.** In non-exception segments, cash = Σ items in 99–100 % of
   lines. A non-exact sum always maps to one of: discount (×0.98), withholding (×0.9875 / 0.985 / 0.90), bank charges
   (small absolute fee), card fee (×0.986, no exception), FX (different currency), partial / instalment / retention /
   advance (PDF or text), overpayment (> open).
2. **References are unreliable.** Only `ar_1to1` (96 %) and AP single/multi (75–87 %) can lean on them. The biggest
   segment (`ar_ref_noise`, 21 % of lines) has a usable number in only 15 % → the **payer** is the key.
3. **Several segments are solvable only through side files** (0 % ref hit): remittances, DD/card/cheque batches,
   confirming, holding-paid supplier runs.
4. **Settled items come back**: returns, duplicates, represented transfers and supplier refunds point at items that are
   no longer open → matchers need the full item history, not just open items.

Reproduce: group `ingestion/out/val/golden.parquet` (branch `kenny/full-pipeline`) by `segment`, join items
(`ingestion.lib.items_of("train")`) and bank lines (`ingestion/out/val/bank_lines.parquet`).
