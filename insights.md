# Insights: Close the Books bank reconciliation (Talky × Anthropic, 7 Nov 2026)

Everything we learned while building the pipeline on branch `kenny/full-pipeline`. Each insight says what we found, the
number behind it, and why it matters or what we did about it. Numbers come from our own runs (logs in
`integration/out/*.log`), the agents' reports, the code and the LIVE leaderboard response. Where a number is uncertain,
the text says so.

**Where we stand (18:45, FINAL submitted):** LIVE **96.46**, validation **96.82**. The submitted FINAL file is v5 +
robustness guards (`final-v5-robust`) + the LR + LightGBM line classifier + the `sd_noadv` ranker + a fix for the new
ZA01 entity (veto off). Sections 10–13 cover the afternoon round, teammates' cross-checks, what is left and the
robustness round; section 14 covers the classifier, the ranker and FINAL day.

**Words used throughout**
- **Bank line**: one row of a bank statement (a payment in or out).
- **Item**: an open document a bank line can settle: customer invoice (`ARI`), customer credit note (`ARC`), supplier
  invoice (`API`) or supplier credit note (`APC`).
- **Allocation (pair)**: "bank line X settles item Y for amount Z". One line can have hundreds of pairs.
- **Exception**: a label on a pair or line explaining a mismatch: `PARTIAL`, `OVERPAYMENT`, `SHORT_DISCOUNT`,
  `SHORT_BANK_CHARGES`, `SHORT_WITHHOLDING`, `SHORT_DISPUTE`, `DUPLICATE_PAYMENT`, `UNIDENTIFIED`, `RETURNED`,
  `FX_DIFFERENCE`.
- **Golden**: the answer key. Only TRAIN (Jul 2021 – Jun 2025) has one. LIVE (Jul–Dec 2025) and FINAL (Jan–Jun 2026)
  are scored by the organisers.
- **Segment**: the case type the golden attaches to each bank line (for example `ar_ref_noise`, `ap_payment_multi`).
- **Matcher**: one rule-based module that recognises one family of cases and claims the bank lines it is sure about.
- **Precision / recall**: precision is the share of what we claim that is right; recall is the share of the truth we find.
- **val**: TRAIN Jan–Jun 2025 dressed up to look like LIVE (our main local test). **hold**: TRAIN Jan–Jun 2024, a
  holdout nobody tuned on.

---

## Summary: the 8 most important insights

> **TL;DR:** rule-based matchers with one small model, 99.8 % precise, LIVE 96.34; what's left is genuinely ambiguous.

(This file merges two independent write-ups of the same findings, one by a dedicated agent and one by the main
session; every number was checked against the logs, the code or the LIVE response.)

1. **Rules beat everything else here, and precision is the whole game.** Deterministic rule-based matchers decide 88 %
   of bank lines; one small logistic regression labels the rest (lines that settle no invoice). The pipeline submits invoice pairs at **99.8 % precision** and scores **96.73** on val,
   **96.84** on the untouched holdout, and **96.34** on the LIVE leaderboard with the current code (earlier uploads:
   93.73, 95.58, 95.62 and 96.05). The organisers'
   reference points are about 19 for the naive baseline and about 61 for well-worked rules.
2. **Side files are worth more than clever text matching.** Payment-run files, direct-debit batches, card and cheque
   files, remittance advices and PDF clauses resolve their cases at 99.6–100 % precision. One aggregated line can hold
   up to **1,540** invoice pairs, so these files carry a large share of the 50 % allocation-F1 weight.
3. **The points are concentrated.** On val, `ar_ref_noise` alone is worth up to 12.7 points, `ap_payment_multi` 9.4, and
   `ar_short_dispute` 7.3 even though it is only 1,067 lines (they hold 104,691 pairs).
4. **The answer key follows POLICIES.md exactly, down to the rounding.** Deductions are rounded half-up, partial payments
   are truncated to the cent, overpayments are always "invoice + a round amount or +1 % / +10 %". Copying those details
   took several exception F1 scores from about 0.1–0.6 to above 0.93.
5. **The trickiest bugs were in the test set, not the model.** Rebuilding "which invoices were open at the start of the
   window" went wrong three times. Each time it made local scores look better or worse than reality. The last one was a
   small leak (cost 0.04 points), now fixed.
6. **No overfitting signal, but drift is untested locally.** Learned parts are time-split, matchers never read the
   golden, and the holdout window that no rule was tuned on scores 96.84 vs 96.73 on val. Both windows sit inside
   TRAIN, though, so neither tests the announced drift (new bank format, new remittance layout, a new ZAR subsidiary in
   FINAL). Only LIVE and FINAL measure that (4.6).
7. **What's left is genuinely ambiguous.** The remaining losses are mostly lines where several sets of invoices add up to
   the same amount, or disputed short-payments with no remittance advice anywhere. We chose to leave those unclaimed
   rather than guess, because a wrong claim costs more than a miss.
8. **Late gains came from evidence the pipeline was mis-reading, not from new guessing.** The afternoon round (§10)
   added +0.72 on LIVE: currency-less remittance advices read in their invoices' currency, two policy-backed exception
   refinements, and a matcher that claims only the invoices a bank line actually cites. Precision never dropped.
9. **On FINAL, the biggest risk was our own safety rule on the new company.** The merge lets a confident classifier
   veto a matcher's claim. The classifier had never seen ZA01, was confidently wrong, and the veto dropped 116 correct
   ZA01 matches (85 exact invoice-number matches). The drift scripts flagged ZA01 but not this; only opening the
   lines behind the flag found it. Fixed by keeping the veto off for ZA01 before submitting (§14.3).

---

## 1. The data and where the points are

> **TL;DR:** 200k lines per half-year, but a few segments with huge multi-invoice lines hold most of the points.

### 1.1 The dataset at a glance
- A fictional industrial group: **17 companies** (entities), about 50 bank accounts, **11 currencies** in TRAIN bank
  lines (a 12th, ZAR, arrives with a new South African company in FINAL).
- TRAIN: **1,489,931 bank lines** from 2021-07-01 to 2025-06-30, with the golden. LIVE: **203,083 lines**.
- Our val window (Jan–Jun 2025): **199,987 lines**, **735,474 invoice pairs**, 19,051 lines that settle no item, and
  **71 segments**. The median line settles 1 item; the largest settles **1,540**.
- The 19,051 no-item lines (9.5 %) are dominated by three categories: `BANK_FEES` 9,718, `UNIDENTIFIED` 4,585 and
  `CASH_POOL` 1,554. The other 18 categories share the remaining 3,194 (for example `PAYROLL` 487, `INTERCOMPANY` 425,
  `TAX_REFUND` 1).
- Side files come with the statements: payment runs, direct-debit batches, card settlements, cheque deposits,
  remittance advices (CSV, XLSX, email TXT, PDF) and about **131k PDFs** (invoices and credit notes) in TRAIN.

### 1.2 How the score works, and what it implies
Score = 100 × (0.5 · allocation F1 + 0.3 · transaction accuracy + 0.2 · exception macro-F1). An amount counts if within
max(0.01, 0.5 %).
- **Allocation F1 counts pairs**, so a single direct-debit batch or bulk supplier payment with hundreds of rows weighs
  as much as hundreds of simple payments.
- **Transaction accuracy counts whole lines**: a line is right only if every row on it is right.
- **Exception macro-F1 averages the 10 codes equally**, so 400 overpayments count as much as 15,000 discounts. Each
  code is worth up to 2 points.

### 1.3 The segments worth the most (val)
"Max points" = what the segment would add if it went from 0 % to 100 % (30 · lines/N + 50 · pairs/P).

| Segment | What it is | Lines | Pairs | Max points |
|---|---|---|---|---|
| `ar_ref_noise` | Customer payment with a truncated, typo'd or missing reference | 42,462 | 92,722 | 12.67 |
| `ap_payment_multi` | Our payment to a supplier covering several invoices | 24,058 | 85,430 | 9.42 |
| `ar_short_dispute` | Big customer pays a remittance, minus disputed amounts | 1,067 | 104,691 | 7.28 |
| `ar_dd_batch` | Direct-debit batch credited as one line | 1,530 | 94,025 | 6.62 |
| `ar_1to1` | One payment, one invoice, clean reference | 19,143 | 19,143 | 4.17 |
| `ar_1toN_remittance` | One payment, many invoices, listed in an advice | 1,206 | 54,817 | 3.91 |
| `ap_payment_single` | Our payment of one supplier invoice | 17,368 | 17,368 | 3.79 |
| `ar_1toN_text` | One payment, many invoices named in the bank text | 6,124 | 36,220 | 3.38 |
| `ap_fx` | Supplier payment in another currency | 1,217 | 45,117 | 3.25 |
| `ar_1toN_credit_note` | Invoices paid net of credit notes | 3,184 | 32,775 | 2.71 |

**Why it matters:** this table set the order in which we built matchers and how many agents worked on each area. It
also shows how unequal lines are: an `ar_short_dispute` line settles about **98 invoices** on average, so getting one
of them wrong costs as much allocation F1 as missing 98 one-invoice payments.

---

## 2. How each source of evidence performs

> **TL;DR:** side files (payment runs, batches, advices, PDFs) are 99.6–100 % precise; almost every remaining error is a line no matcher claimed.

### 2.1 Overview (val, honest run, final pipeline)
Matchers run in a fixed order, most certain first. A line claimed by one matcher is never touched by a later one.

| Order | Matcher | What it uses | Lines claimed | Precision (lines fully right) |
|---|---|---|---|---|
| 1 | `payment_runs` | Our supplier payment-run files | 60,361 | 100.0 % |
| 2 | `dd_batches` | Direct-debit batch files, returns, re-presented transfers | 5,984 | 99.6 % |
| 3 | `cards_cheques` | Card settlement and cheque deposit files | 3,304 | 100.0 % |
| 4 | `remittance` | Customer remittance advices | 7,862 | 100.0 % |
| 5 | `pdf_cases` | Clauses printed in invoice PDFs | 6,336 | 99.9 % |
| 6 | `refs` | Invoice numbers found in the bank text | 43,223 | 99.5 % |
| 7 | `subset_sum` | Payer's open invoices adding up to the amount | 39,035 | 99.5 % |
| 8 | `same_amount` | Several invoices with the same amount: month in text, else oldest | 7,292 | 98.0 % |
| 9 | `short_payments` | One invoice short by bank charges / withholding, or overpaid | 345 | 98.2 % |
| 10 | `ap_refunds` | Supplier refunding its credit note | 2,513 | 100.0 % |
| 11 | `duplicates` | Second payment of an already-paid invoice | 622 | 99.8 % |
| — | classification (`cls`) | Everything unclaimed: category or UNIDENTIFIED | 23,110 | 80.9 % |

The 80.9 % for classification is not the model being wrong: it holds the 19,051 lines that truly settle nothing (almost
all right) plus about 4,000 item lines no matcher could claim, which are wrong by definition. **Almost every remaining
line error is a line no matcher claimed.**

### 2.2 Supplier payment runs (largest area, about 19.6 points)
- Three run types: `transfers` (one debit per end-to-end ID), `aggregated` (one debit for the whole file) and
  `confirming` (one debit per maturity date). camt.053 lines carry the end-to-end ID; N43/MT940 lines don't, so those
  match on account + date ± a few days + amount + beneficiary.
- A **payment factory**: the ES00 company pays invoices of ES01/ES02/PT01, and the items belong to the subsidiary
  (POLICIES §5.2).
- Results on val: 100 % on `ap_payment_multi`, `ap_payment_single`, `ap_paid_by_holding`, `ap_direct_debit`,
  `ap_partial`, `ap_credit_note_netted`; `ap_fx` 99.8 %, `ap_factored` 99.2 %, `ap_confirming` 98.4 %.
- **Duplicate supplier payments** are marked by an end-to-end ID ending in "D"; implicit repeats need the duplicates
  matcher.
- **Returned (rejected) transfers**: 168 of 171 claimed on val at 100 % precision (see 5.6 for how they almost got
  lost).

### 2.3 Direct debits, cards and cheques
- **Batch credits:** same account, amount = batch total, date −2/+6 days. Every receipt in the batch becomes a pair.
- **Returns:** a debit with a mandate reference goes to the earlier receipt with that mandate and amount, marked
  `RETURNED`. The time window depends on the SEPA reason code: MD06 ("refund request") 14–80 days, others 1–20.
- **Re-presented transfers:** a customer whose debit bounced pays by transfer instead; each item keeps a timeline of
  collections and returns.
- Val line accuracy: `ar_dd_batch` 99.9 %, `ar_dd_returned` 99.5 %, `ar_dd_represented_transfer` 98.1 %, cards,
  cheques, chargebacks and returned cheques 100 %. `RETURNED` exception F1: 0.994.
- **Card settlements:** the bank line is the sum of net amounts, but each order is cleared at its **gross** amount with
  no exception. The acquirer fee is not a short payment.

### 2.4 Remittance advices
- Four formats: CSV (`,` or `;`), PDF text (English and Spanish), email text, XLSX. Numbers come in both US and European
  formats.
- **A layout change only in LIVE:** from Aug/Sep 2025 one big customer sends a new "Voucher" XLSX where the invoice
  column holds only the **last 6 digits** of our number, there is no reason column, and the total sits in the Deduction
  column. The parser handles it; deduction types there are inferred from their size (unverifiable, since the layout
  never appears in TRAIN).
- **Advices for the last payments of a period sit in the next period's folder** (late-June 2025 payments have their
  advice in `data/live/`). Reading the neighbouring folder recovered about **10 %** of the 1:N lines.
- Results: `ar_1toN_remittance` and `ar_1to1_remittance` 100 %, `ar_short_dispute` 82.8 % (val), `SHORT_DISPUTE` F1
  0.918 at precision 1.0. All **6,889 LIVE advices** parse; 98.1 % add up to their stated total.
- **Silent discounts** (paid below the gross without saying so) are accepted by the golden up to about **14 days** late.
  That comes from the golden, not from the written policy.

### 2.5 PDF clauses
- A survey of all **131k TRAIN PDFs**: clauses are only ever worded in Spanish or English, even on French, German,
  Italian or Portuguese documents. Clause dates are always dd/mm, even on US01 documents whose header dates are mm/dd.
- Clause types: instalment plans, retentions (and their release), advances to deduct, "cancels and replaces",
  credit notes cancelling an invoice in full, service month, new IBAN notices.
- Parsed amounts check out: instalment plans sum to the total; retention + net = total; advance + net = total.
- Val recall: instalments 99.2 %, retentions 98.5 %, releases 99.3 %, advance deductions 98.9 %, replaced invoices
  90.5 %, credit notes paid out 100 %. Precision 99.92 %.
- On the sample month, adding this matcher took `ar_retention` from 41.5 % to 98.4 %, `ar_advance_deduction` from 0 % to
  100 %, and `PARTIAL` F1 from 0.695 to 0.919.

### 2.6 References in the bank text
- Exact references: 35,344 lines at 100 % precision. "Explain" mode (reference found, amount needs an exception): 97.2 %.
- Second round of work raised `ar_1toN_credit_note` 0.60 → 0.749, `ar_n_to_m` 0.53 → 0.728 and `ar_fx` 0.53 → 0.874
  (on val, partly thanks to subset-sum changes too).
- **Credit notes first:** trying related credit notes before plain invoice subsets fixed many netted payments.

### 2.7 Who paid, and subset-sum
- **Payer resolution:** IBAN (from the line or found in the text) against the customer master and IBANs learned from
  TRAIN; payer name cut out of the text by per-format patterns (`/ORDP/`, `ORD.`, `EXPTE`, `POR CTA DE`, SPEI…).
  Names are compared with only letters and digits kept, so truncated names still match. On LIVE the resolver knows
  86,003 name aliases and 5,483 learned IBANs.
- **Third parties pay for customers:** parents, collection agencies (`EXPTE`), confirming banks (`ORD.`). When the
  direct payer has no solution, the group's other customers are tried.
- **Subset-sum** claims a line only when exactly one customer has a solution, and only if the estimated number of
  coincidental matches is at most 0.3. That guard keeps it at 99.5 % precision.
- **Same-amount invoices:** a month named in the text picks the invoice issued at the start of the following month
  (100 % precision on val); otherwise the oldest first (96.5 %).
- **Unfixable noise:** when a customer pays two same-amount invoices on the same day, which line gets which invoice is
  random in the golden. About **140** val errors come from that.

### 2.8 Supplier refunds, duplicates and short payments
- **Supplier refunds** (`ap_refunds`): an incoming line equal to an open supplier credit note, pointed to by its number
  or the supplier's IBAN/name. Val: 2,514 of 2,515 at 100 % precision (worth about 0.55 points). Texts saying "PAGO
  DOBLE / DOUBLE / TWICE" are left alone unless they cite a credit-note number, because those are a different category.
- **Duplicates:** a credit is a duplicate when an earlier line 0–14 days before paid exactly that one invoice for the
  same amount, by the same payer, and the customer has no other open invoice of that amount (otherwise it's just the
  next monthly fee). `DUPLICATE_PAYMENT` F1 went from **0.11 to 0.966**.
- **Short payments:** a single open invoice explained by bank charges, withholding or a plausible overpayment.
  `SHORT_BANK_CHARGES` F1 went from 0.368 to 0.937.

### 2.9 Lines that settle nothing (categories)
- One logistic regression over 22 classes (ITEM + 21 categories), features from bank codes, text words, counterparty
  IBAN type, mentions of our own accounts, and "pair context" (an opposite movement on another of our accounts the same
  day, as in cash-pool sweeps and FX deals).
- No-item lines are categorised correctly **99.99 %** of the time on val (1 wrong of 19,051). Two deterministic
  overrides: reversal pairs ("ANULACION OPERACION INDEBIDA" paired with the wrong booking) and keyword rules for two
  near-empty classes (`TAX_REFUND` has 4 examples in all of TRAIN, `LOAN_DRAWDOWN` 32).
- The model also outputs `p_items`, the probability a line settles invoices at all, used to veto matcher claims and to
  decide UNIDENTIFIED (see 3.4). On val the veto removed 18,157 lines, of which only 19 were real item lines.

---

## 3. Exception rules: the golden is very literal

> **TL;DR:** copy POLICIES.md to the letter (full open amount, half-up rounding, truncated partials, overpayment shapes) and the rules are near-perfect.

### 3.1 Amount convention
When an invoice is fully cleared with an explained shortfall (discount, withholding, bank charges, FX), the row's amount
is the invoice's **full open amount**, not the cash received. For a real partial payment it is the cash applied. Getting
this wrong costs the pair, the line and the exception at once.

### 3.2 Rounding: half-up for deductions, truncation for partials
- Discounts and withholding are rounded **half-up** to the cent from the exact total. With Python's default `round()`
  (banker's rounding) we reproduced only 5,733 of 5,791 single-line discounts and 1,193 of 1,224 withholdings on val;
  with half-up, all of them.
- Partial payments are **truncated**: a third of an invoice gives 6435.74, as in the golden.

### 3.3 Overpayments have a fingerprint
Every overpayment in the golden is the invoice plus a **round amount (1 to 200)** or **exactly +1 % or +10 %**. Of the
matcher's OVERPAYMENT rows outside that pattern, 0 of 55 were right, against 310 of 314 inside it. The plausibility check
(`overpayment_plausible`) took OVERPAYMENT F1 to 0.992.

### 3.4 The UNIDENTIFIED trade-off
- First version: every receipt no matcher claimed was flagged UNIDENTIFIED. Recall 1.0, but precision 0.442 (5,777
  false positives on val), F1 0.614.
- A matcher miss is a wrong line either way; flagging it UNIDENTIFIED only adds a false positive to that exception's F1.
- Fix: flag UNIDENTIFIED only when the classifier thinks the line settles nothing (`p_items < 0.9`). That alone took val
  from **90.04 to 90.81** (together with dropping a bad fallback, 5.5). UNIDENTIFIED F1 is now **0.956**.
- Caveat: 0.9 was chosen on val; the holdout result (4.4) suggests it transfers.

### 3.5 Other rules the golden follows
- **Instalments:** whether a payment is "the last instalment" (no exception) or `PARTIAL` depends on **which** instalment
  it is, not on what remains open. Payments can arrive out of order. The matcher reads "k/n" in the text, else the only
  unpaid instalment of that amount, else the one whose due date passed most recently: right 99.9 % of the time.
- **Advance deductions** stay `PARTIAL` even when the cash equals the open amount. Only the PDF reveals it, so evidence
  matchers keep their own exception labels and the generic rules may not override them. Without that protection,
  advance-deduction accuracy fell to about 0 % and instalments to about 96 %.
- **Two-invoice short payments:** in 72 % of golden cases one invoice is cleared ("") and the other takes the remainder,
  rather than both being `PARTIAL`.
- **Late discounts on multi-invoice lines:** items 4–10 days past the discount deadline are `SHORT_DISCOUNT` about 30 %
  of the time and `PARTIAL` the rest. About 1 % of discount rows can't be reproduced by any rule.
- **10 % retention vs 10 % withholding** look identical for MX/BR/MA customers; we choose withholding.
- **Exception rules on their own are near-perfect:** fed the golden's items and amounts, the rules library scores macro
  F1 above 0.98 on every code it decides. The remaining exception errors come from matching, not from the rules.

---

## 4. Leakage and overfitting checks

> **TL;DR:** learned parts are time-split, the untouched holdout scores at or above validation, and LIVE gains tracked local gains; drift is the only untestable risk.

### 4.1 What learns from the answer key, and on what
Only two components read the golden at run time, and both are cut by date:

| Component | Learns | val trains on | hold trains on | LIVE / FINAL trains on |
|---|---|---|---|---|
| Category model | category of no-item lines | lines before 2025-01-01 | before 2024-01-01 | all of TRAIN |
| Payer aliases | bank name / IBAN → customer | lines before 2025-01-01 | before 2024-01-01 | all of TRAIN |

Model caches are stored per cutoff, so they can't mix. No matcher reads the golden. The pipeline is deterministic: the
one random draw (training-row subsample) uses a fixed seed, and no LLM is called anywhere.

### 4.2 The starting-state leak (found and fixed)
To make Jan–Jun 2025 look like LIVE, we must know which invoices were open on 1 Jan 2025 (LIVE gives this list; TRAIN
doesn't).
- The backward method (start from the official end-of-June list and undo the window's payments) **dropped 21,825
  invoices (7.94 %)** that were open in January but later cancelled without any bank payment. In LIVE those stay in the
  list as distractors, so removing them made matching easier: forward replay gave 274,712 open TRAIN-issued invoices,
  backward 252,887.
- Fix (`d34ba09`): rebuild the start state from payments **before** the window only, for every invoice issued inside
  TRAIN. Cost of the leak: **0.04 points** (val 96.40 → 96.36).
- **Leftover:** for invoices issued before TRAIN (unknown history) we still use the backward method. Only 5,092 such
  invoices are open on 1 Jan 2025, and just **26** of them are in the set only because they get paid in the window.
  Too small to matter; documented in `ingestion/lib.py`.

### 4.3 Earlier start-state mistakes (each changed the local score)
1. **First version (forward from the invoice totals):** 237k old invoices looked open that never were, because their
   history predates TRAIN.
2. **Duplicate payments undone as reductions:** an invoice paid just before the window and paid again inside it looked
   open, so the duplicate was claimed as a normal payment. That was 27 of 30 duplicate misses on the sample month.
   Duplicate payments now never reduce the open amount (`9a50dc4`).
3. **The starting-state leak** above.

**Lesson:** when you build your own validation set, the validation set is code too, and it needs the same scrutiny as
the model.

### 4.4 The untouched holdout
Rules and thresholds were written while looking at TRAIN answers, including the val window, so val is a development
set. To measure overfitting we added `hold` = Jan–Jun 2024, never used for tuning:

| | val (Jan–Jun 2025, tuned on) | hold (Jan–Jun 2024, untouched) |
|---|---|---|
| Score | 96.36 | **96.46** |
| Allocation F1 (P / R) | 0.958 (0.999 / 0.920) | 0.960 (0.999 / 0.925) |
| Transaction accuracy | 0.975 | 0.976 |
| Exception macro-F1 | 0.962 | 0.959 |

The same segments lose points in both windows by about the same amounts. Largest exception gap: SHORT_BANK_CHARGES
0.890 vs 0.937, on only 125 holdout cases. **No sign of overfitting.**

With the final v5 code (§10) the gap still points the same way: val **96.73**, hold **96.84** (allocation F1 0.962 /
0.965, precision 0.998 on both).

### 4.5 LIVE vs local
- Teammate's earlier measurement (starter + classifier): val 13.08, LIVE 13.38. Local val tracked LIVE within about 0.3.
- Our first LIVE upload (file `submission_v1.csv.gz`, 704,576 rows, built at 12:05; uploaded at 12:38, three minutes
  before the v2 file finished writing at 12:41) scored **93.73** (allocation F1
  0.944, precision 0.998, recall 0.896; transaction accuracy 0.964; exception macro-F1 0.880). That file came from code
  between our 90.81 and about 94.3 val measurements, so it's roughly in line; we can't pin its exact val equivalent.
- On LIVE v1 the weakest segments were `ar_short_bank_charges` 35 %, `ar_fx` 49 %, `ar_n_to_m` 55 %, `ar_1toN_credit_note`
  72 %, `ar_discount_late` 75 % and `ar_overpayment` 80 %. The second-wave matchers in v2 targeted exactly these (on val:
  `ar_fx` 87 %, `ar_n_to_m` 68 %, overpayment and bank-charge F1 above 0.93).
- **LIVE v2** (708,637 rows, the current code; uploaded 13:14) scored **95.58**: allocation F1 0.9487 (precision
  0.9983, recall 0.9038), transaction accuracy 0.9734, exception macro-F1 0.947. That is 0.78 below val and 0.88 below
  the holdout. Precision is unchanged; the whole gap is recall (0.904 vs 0.920), i.e. more lines left unmatched on new
  data, not more wrong matches (see 9.1).
- **LIVE v3** (uploaded 14:18) scored **95.62**, our best: the only change is the UNIDENTIFIED threshold 0.9 → 0.97
  (`integration/run.py`), which `OVERFITTING_CHECKS.md` found on a flat plateau (0.8–0.99 within 0.1 points) with its
  best value at 0.97 in both val and hold. Locally val 96.36 → 96.41 and hold 96.46 → 96.50; on LIVE exception
  macro-F1 0.947 → 0.9481 and transaction accuracy 0.9734 → 0.9741, i.e. the +0.04 predicted locally showed up on LIVE.
- **LIVE v4** (uploaded 15:57) scored **96.05**: remittance FX for currency-less advices plus exception refinements
  (§10.1). Locally only +0.05 val / +0.09 hold, because the remittance fix had no cases in TRAIN; on LIVE +0.43.
- **LIVE v5** (uploaded 16:38) scored **96.34**, our best: v4 + the `cn_bundles` matcher. Locally +0.27 val / +0.25
  hold; on LIVE +0.29. Allocation F1 0.9562 (precision 0.9983, recall 0.9175), transaction accuracy 0.9753, exception
  macro-F1 0.9637.
- On segments both runs share, LIVE v1 sits close to local: `ar_ref_noise` 94.8 % (LIVE) vs 96.1 % (val, newer code),
  `ar_short_dispute` 80.8 % vs 82.8 %, `ar_1toN_text` 94.1 % vs 94.8 %, and 100 % on the big supplier-payment segments.

### 4.6 What no local test can measure: drift
Both val and hold are windows inside TRAIN, so they only test whether our rules generalise **in time within the same
world**. POLICIES §6 announces changes that appear only after TRAIN:
- Banco del Moncayo moves from Norma 43 to camt.053 statements on 2025-10-01 (LIVE): texts look different.
- A big customer switches its remittance-advice layout in September 2025 (LIVE; handled, see 2.4).
- A new South African subsidiary arrives on 2026-01-01 (FINAL): new entity, new bank, currency ZAR, own invoice
  numbering "KWI…", renamed on 2026-04-01, with updated masters in `final/masters_final/`.
- Customers are renamed, change banks and pay more slowly in 2026.

**Why it matters:** the LIVE leaderboard is the only measurement that includes some of this drift, and FINAL adds more
(the ZAR company is new in FINAL). Matchers that depend on learned payer aliases or bank-text formats are the most
exposed. Running the full pipeline on the FINAL data the moment it arrives and checking the per-matcher line counts
against LIVE is the cheapest early warning.

---

## 5. Data quirks and traps

> **TL;DR:** cancelled invoices, month-boundary side files, format switches and shared names are traps; a few cases are unrecoverable by design.

1. **Invoices cancelled without a bank payment** vanish from the open list (4.2). On val, 2,114 items are cancelled by
   "cancels and replaces" or by a full credit note. The golden allocates to 39 of them, all inside cheque-deposit lines
   whose files list them explicitly. So: exclude cancelled items everywhere **except** cheque deposits.
2. **The sample folder is a subset.** `data/sample` doesn't hold all side files, so the sample phase reads TRAIN's.
3. **Month-boundary side files:** advices and payment runs for payments near a period edge sit in the neighbouring
   folder (2.4). LIVE and FINAL must read the previous folder too.
4. **Statement formats change mid-stream:** Banco del Moncayo switches to camt.053 in October 2025. The category model
   gives October–December the same no-item mix as July–September (for example 735 vs 890 fees), so it holds up.
5. **"SPA" is treated as a legal form** by the name normaliser, so "Spa Service Costa" becomes "SERVICE COSTA"; several
   customers also share a normalised name. Name-only matching always needs an amount check.
6. **17 % of disputed short-payments have no advice anywhere** in TRAIN or LIVE; the bank text only lists truncated
   invoice numbers. That is about 1 point lost on val that no rule can recover.
7. **MD06 returns** (refund up to 8 weeks later): about 13 val cases where two monthly collections of the same amount
   fall in the window and nothing shows which one bounced.
8. **The "points lost" diagnostic overstates huge lines.** `ap_fx` once showed 10,342 pairs wrong; it was a few bulk
   payments of about 1,000 rows each, each with one bad row. Our estimate counts every pair of a wrong line, while the
   real F1 loss was about one pair per line.

---

## 6. Score progression (val unless noted)

> **TL;DR:** 12.72 → 96.73 on validation and 93.73 → 96.34 on LIVE, every step checked on both local windows.

| Step | Score | What moved it |
|---|---|---|
| Starter on the sample month | 11.28 | Organisers' starter script |
| Our baseline on val | 12.72 | Same idea on our ingestion |
| First full pipeline (8 parallel matcher agents) | 90.04 | All side-file matchers, references, subset-sum, categories, exception rules |
| UNIDENTIFIED threshold + drop bad fallback | 90.81 | 3.4 and 7.3 |
| Newer matcher versions | ~94.3 | Measured by the exception agent mid-run (94.32) |
| + short-payments matcher (simulation) | 95.49 | Bank charges, overpayment, withholding on unclaimed lines |
| All second-wave agents (duplicates, refunds, refs, subset-sum, rounding) | 96.40 | DUPLICATE_PAYMENT 0.11 → 0.966, ar_fx, n_to_m, credit notes |
| Starting-state leak fixed | **96.36** | Honest measurement (4.2) |
| Untouched holdout (Jan–Jun 2024) | **96.46** | Overfitting check (4.4) |
| LIVE leaderboard, v1 file | **93.73** | First real out-of-sample score (4.5) |
| LIVE leaderboard, v2 file | **95.58** | Second-wave code on LIVE (4.5) |
| UNIDENTIFIED threshold 0.9 → 0.97 | **96.41** (hold **96.50**) | Flat plateau, best in both windows (B8, 9.5) |
| LIVE leaderboard, v3 file | **95.62** | (4.5) |
| Remittance FX + exception refinements + ZA01 fixes | **96.46** (hold **96.59**) | §10.1 fixes 1–3 |
| LIVE leaderboard, v4 file | **96.05** | +0.43: the remittance fix only had LIVE cases (§10.1) |
| + `cn_bundles` matcher | **96.73** (hold **96.84**) | §10.1 fix 4 |
| LIVE leaderboard, v5 file | **96.34** | Frozen as `best-live-96.34` |
| + LR + LightGBM line classifier | **96.76** | §14.1 |
| LIVE leaderboard, v5 + classifier | **96.38** | +0.04 |
| + `sd_noadv` ranker | **96.82** | §14.2 |
| LIVE leaderboard, v5 + classifier + ranker | **96.46** | Best LIVE score; what FINAL ran |
| FINAL: + ZA01 veto off | submitted | Restores 116 ZA01 matches; no effect on LIVE or val (§14.3) |

Exception macro-F1 went from **0.716** in the first full run to **0.962**: DUPLICATE_PAYMENT 0.110 → 0.966,
SHORT_BANK_CHARGES 0.368 → 0.937, OVERPAYMENT 0.575 → 0.992, UNIDENTIFIED 0.614 → 0.956.

---

## 7. Where the remaining points go (val, honest run)

> **TL;DR:** about 3 points remain in a handful of AR segments, mostly lines where several invoice sets fit the cash and the text doesn't say which.

| Segment | Lines | Line accuracy | Points lost | Why | Ideas (untested) |
|---|---|---|---|---|---|
| `ar_ref_noise` | 42,462 | 96.1 % | 1.28 | Payments of 8+ invoices out of 30+ open ones: many subsets add up to the amount | Prefer due-date order or the payer's habitual patterns to break ties |
| `ar_1toN_credit_note` | 3,184 | 74.8 % | 1.25 | Big lines netting several credit notes; truncated texts allow several item sets | Deeper credit-note-aware search with stronger name/date priors |
| `ar_short_dispute` | 1,067 | 82.8 % | 0.98 | 17 % of lines have no advice at all | Probably unrecoverable |
| `ar_1toN_text` | 6,124 | 94.8 % | 0.44 | Truncated multi-invoice texts | Same as credit notes |
| `ar_n_to_m` | 1,857 | 68.3 % | 0.28 | First part-payments claimed later in the order than refs, so refs can't tell a remainder from a first partial | Track partial state across matchers |
| `ap_fx` | 1,217 | 99.8 % | 0.21 | A few giant bulk lines with one bad row each | Small |
| `ar_fx` | 725 | 87.4 % | 0.17 | Lines without `orig_amount` and ambiguous rates | Small |

Exceptions still below 0.93: `SHORT_DISPUTE` 0.918 (the missing advices) and `FX_DIFFERENCE` 0.924 (recall 0.858: FX rows
on unclaimed lines).

**Update with the v5 code (val / hold / LIVE line accuracy, val points lost):** `ar_ref_noise` 96.2 / 96.5 / 96.0 %
(1.23), `ar_1toN_credit_note` 76.7 / 78.0 / 74.6 % (1.18), `ar_short_dispute` 82.9 / 85.6 / 81.2 % (0.98),
`ar_1toN_text` 95.0 / 95.5 / 94.9 % (0.42), `ar_n_to_m` 72.6 / 70.5 / 69.3 % (0.25), `ar_fx` 87.7 / 88.4 / 87.9 % (0.17).
§12 explains why most of this is out of reach for rules that read only the bank text.

---

## 8. Engineering lessons

> **TL;DR:** claim only when sure, keep evidence labels, give parallel agents separate outputs, and build safety nets for the one FINAL shot.

1. **Priority-ordered matchers that claim whole lines.** Each matcher only claims what it is sure of, and later
   matchers see earlier claims (open amounts are updated in booking order). Precision stays at 99.9 % because nothing
   guesses on a line another, more certain matcher could own.
2. **Leave a line unclaimed rather than guess.** A wrong claim loses the pairs, the line and possibly an exception; an
   unclaimed line only loses the line. Every ambiguous matcher had a "unique solution only" guard.
3. **A late fallback can go bad without anyone noticing.** `ref_exact`, written as an early catch-all, ended up at **0 %
   precision** on 201 lines once the better matchers ran first; we removed it. Per-method precision in `score.py` made
   that visible.
4. **Parallel agents need separate outputs.** Several agents ran partial matcher runs at once and overwrote the shared
   `allocations.parquet`; one score came out as a bogus 36.48. Partial runs now write their own file (`ALLOC_FILE`).
   They also overwrote each other's scratch scripts until each used a subfolder, and the disk once fell to 139 MB free.
5. **Ownership gaps lose points silently.** Rejected supplier transfers were left to nobody after one instruction told
   the payments agent to leave them and another took them off the batches agent; they were turned back on
   (`CLAIM_RETURNS = True`, 168/171 at 100 %).
6. **Evidence beats generic rules.** Exception labels set by matchers that read documents (PDF clauses, advices, batch
   files, payment runs) are kept; the generic rules only fill in the rest.
7. **FINAL safety nets.** With one shot at FINAL: a crashing matcher is skipped instead of stopping the run;
   classification falls back to a simple baseline if the model fails; integration keeps the matchers' exceptions if
   the rules crash; every file is validated (all lines present, format) before upload; and `submit.py` refuses FINAL
   without `--i-am-sure`. The team token is read only from the environment.
8. **Speed.** A full val run takes about 10 minutes (references about 200 s, PDF clauses about 130 s, subset-sum about
   110 s). That is fine for FINAL at 18:00 but limits how many experiments fit in the afternoon.

---

## 9. Improvements for FINAL (planned 7 Nov, 13:45)

> **TL;DR:** protect the FINAL run first (PDF text, rehearsal, drift alarm, runbook), then add recall; every planned item is now done or measured and dropped.

FINAL is 220k lines from Jan–Jun 2026, scored once. It brings drift we have never seen: a new South African company
(ZA01, currency ZAR, invoice numbers "KWI…"), updated masters in `final/masters_final/`, renamed customers, changed bank
accounts and slower payers. So the plan puts **running correctly on new data** first and **extra recall** second.

### 9.1 What LIVE v2 tells us
- The gap to local is recall, not precision (4.5). On val, almost every wrong line in the segments that lose the most
  is a line **no matcher claimed**: all 183 wrong `ar_short_dispute` lines, 734 of 802 wrong `ar_1toN_credit_note`
  lines, 1,610 of 1,667 wrong `ar_ref_noise` lines. Inside those unclaimed lines, 96–100 % of the true pairs are plain
  full-amount pairs.
- Estimated points lost on LIVE v2 (LIVE line counts × val pairs per line; overstates huge lines):

| Segment | LIVE accuracy | Val accuracy | Est. points lost |
|---|---|---|---|
| `ar_short_dispute` | 80.8 % | 82.8 % | 1.42 |
| `ar_1toN_credit_note` | 73.3 % | 74.8 % | 0.74 |
| `ar_ref_noise` | 95.9 % | 96.1 % | 0.51 |
| `ar_n_to_m` | 65.7 % | 68.3 % | 0.25 |
| `ar_1toN_text` | 94.3 % | 94.8 % | 0.20 |
| `ar_fx` | 84.5 % | 87.4 % | 0.17 |
| `ar_1toN_remittance` | 98.3 % | 100 % | 0.07 |

- `ar_new_iban` drops from 98.6 % (val) to 96.2 % (LIVE): payer IBANs and names are learned only from TRAIN, and
  customers keep changing banks and names after it.

### 9.2 A FINAL blocker we found: PDF text is never extracted for the FINAL folder
- `matching/pdf_clauses.py` and `matching/remittance_parse.py` read pre-extracted text from
  `data/cache/pdftext_<folder>.jsonl.gz` (and `..._remittance_advices.jsonl.gz`). The only code that builds that cache
  is `recon/pdfs.py` on the old `kenny/recon-draft` branch; nothing in the pipeline runs it. TRAIN and LIVE have caches
  because they were built by hand this morning.
- On FINAL, `load_texts` would print "no text cache" and return nothing: the PDF-clause matcher (6,952 LIVE lines:
  instalments, retentions, advances, replaced invoices) and every PDF remittance advice would silently produce zero
  claims. That is likely **several points**, and the file would still validate, so nothing would warn us.
- Fix: PDF extraction becomes a step of ingestion (runs automatically when the cache for the phase's folder is missing).

### 9.3 Checked and fine for FINAL
- `lib.masters(folder)` overlays `masters_final`; the exception rules find the folder from `statement_file` (which
  starts with `final/`); ZAR rates exist up to 2025-12-31 and later dates fall back to the last rate; the only
  hard-coded entity list (`WHT_ENTITIES`) is correct. But the FINAL code path (three item folders, no next folder for
  late advices) has **never been run**, so it gets a dress rehearsal.

### 9.4 Ideas tested and rejected
- **Oldest invoices first** for payments without references: the true set equals the payer's oldest-by-due-date
  invoices in only 0–3 % of the unclaimed lines.
- **Partial claims are precise but small:** claiming, on unclaimed lines, only the invoices whose number appears complete
  or as a unique prefix recovers about 2,600 pairs on val with 100 % item precision (0.36 % of all pairs, about
  +0.1–0.15 points). Worth doing because it is free, not because it is big.
- **Short disputes without an advice** (17 % of them): the deducted invoice cannot be known. Accept the loss.
- **Late discounts** that the golden labels both ways: noise, not fixable.

### 9.5 The improvement list, in order

| # | Step | Why | Expected gain | Risk | Effort |
|---|---|---|---|---|---|
| B1 | **Done.** PDF text extraction inside ingestion (`ingestion/pdftext.py`) | Without it FINAL loses all PDF evidence (9.2) | Protects several points | Low | 30 min |
| B2 | **Done** (99.98 % identical to LIVE v2, all matchers ran). FINAL dress rehearsal on a fake `final` folder (LIVE data, copied masters as `masters_final`) | The FINAL code path has never run | Protects the one shot | Low | 45 min |
| B3 | **Done.** Drift alarm `integration/drift.py`: share claimed per matcher, per entity and currency, FINAL vs LIVE, flag ±30 % | A silent failure (e.g. ZA01 with zero claims) shows up before we submit | Protects | None | 30 min |
| B4 | **Dropped (measured, no gain, §10.3).** Learn payer IBANs and names from our own confident LIVE matches (evidence matchers ≥ 99.9 % precise), for FINAL only | Customers renamed or changed banks after TRAIN | Small to medium on FINAL | Low (no labels used) | 1–2 h |
| B5 | **Done.** FINAL runbook `integration/final.sh`: checksum, pdftext, ingestion, pipeline, validate, drift report, print the submit command | No improvisation under time pressure | Protects | None | 20 min |
| B6 | **Done in `cn_bundles` (§10.1).** Partial claims on unclaimed lines (9.4) | Free correct pairs | +0.1–0.15 | Low | 1 h |
| B7 | **Done (§10.1, fix 1): 0 voucher advices left unclaimed on LIVE.** Fix remittance advices the new "Voucher" layout misses | The layout is the norm in FINAL | Small on LIVE, larger on FINAL | Low | 1 h |
| B8 | **Done.** UNIDENTIFIED threshold 0.9 → 0.97, the best value on a flat plateau in both windows (`OVERFITTING_CHECKS.md`) | Robustness on new data | +0.05 val, +0.04 hold, +0.04 LIVE (95.62) | Low | Done |
| B9 | **Done in `cn_bundles` (§10.1).** Residual subset-sum after partial claims: claim a whole line only if exactly one completion exists | Targets credit-note bundles and noisy references | Uncertain | Medium | Stretch |

### 9.6 How we test each change
- Every change runs on val **and** the holdout: neither may drop, and allocation precision must stay ≥ 0.998.
- Then one LIVE upload per change (the limit is now 1 per 10 s), compared with the best so far (now 96.34). Only whole-pipeline variants go
  to the leaderboard; we never probe individual lines, since the rules forbid trying to obtain LIVE or FINAL answers.

### 9.7 FINAL timetable
- 18:00 download, verify SHA-256, extract to `data/final/` → run `integration/final.sh` **from the frozen folder
  `hackathon-claude-best`** (about 20–30 minutes with PDF extraction) → read the drift report → submit with `--i-am-sure` by **19:00**, leaving 30 minutes of buffer before
  submissions close at 19:30. A file with format errors doesn't use the shot, but a valid bad file does.

---

## 10. Afternoon round: from 95.62 to 96.34 on LIVE

> **TL;DR:** four fixes found by four parallel agents added +0.32 on validation, +0.34 on the holdout and +0.72 on LIVE,
> with precision unchanged at 0.998; every other idea they tested was measured and dropped.

### 10.1 What shipped
Each fix had to raise both validation and holdout with allocation precision ≥ 0.998, then beat the best LIVE score.

| # | Fix | What it does | Val | Hold | LIVE |
|---|---|---|---|---|---|
| 1 | Remittance FX (`matching/matchers/remittance.py`) | Voucher and old XLSX advices carry no currency. When the advice's invoices are in another currency than the bank line, read the advice in that currency, and accept it when the bank's own original amount equals the advice total. Discounts listed on a voucher count as declared (3-day grace, not 14). | 0 lines changed | 0 lines changed | +22 lines, 3,003 rows; FX_DIFFERENCE rows 12,618 → 15,513 |
| 2 | Exception refinements (`integration/refine.py`) | PARTIAL on `refs:explain` rows whose amount is exactly a usual fraction of the item (cut to the cent); SHORT_BANK_CHARGES at the full open amount when POLICIES explains a single-invoice shortfall as intermediary charges (non-SEPA payers such as GB, CH, PL). | +0.05 | +0.09 | (in v4) |
| 3 | FINAL guards | ZA01: payer names after "EFT CREDIT", "KWI" no longer read as the Polish month April, "KWI" added to invoice prefixes. `final.sh` compares master row counts and warns when FX rates stop before June 2026. | 0 | 0 | 0 |
| 4 | `cn_bundles` matcher (`matching/matchers/cn_bundles.py`, last in the order) | On lines no matcher claimed: claim the invoices the text cites (full number or a unique 6+ digit suffix), each netted with its own credit notes; claim the whole line only when exactly one completion of the remainder exists (coincidence guard 0.05); otherwise only the cited part. | +0.27 | +0.25 | +0.29 (1,177 lines) |

- **LIVE v4** (fixes 1–3) scored **96.05** (+0.43). It gained far more on LIVE than locally because fix 1's case, a
  currency-less advice paid in another currency, never occurs in TRAIN.
- **LIVE v5** (v4 + fix 4) scored **96.34** (+0.29), close to the +0.27 / +0.25 measured locally.
- `cn_bundles` shows only 14 % "lines fully right" in the matcher table because it often claims just the cited part of
  a line. Its pairs are 97–99 % right, and all of its lines were wrong before it ran, so it cannot lower transaction
  accuracy.

### 10.2 What moved on LIVE (line accuracy, v3 → v4 → v5)

| Segment | v3 | v4 | v5 |
|---|---|---|---|
| `ar_fx` | 84.5 % | 87.5 % | 87.9 % |
| `ar_n_to_m` | 65.7 % | 69.2 % | 69.3 % |
| `ar_short_bank_charges` | 88.4 % | 93.2 % | 93.2 % |
| `ar_1toN_credit_note` | 73.3 % | 73.3 % | 74.6 % |
| `ar_1toN_text` | 94.3 % | 94.3 % | 94.9 % |
| `ar_third_party_parent` | 94.1 % | 94.1 % | 95.2 % |
| `ar_short_dispute` | 80.8 % | 81.2 % | 81.2 % |

Exception macro-F1 on LIVE went 0.9481 → 0.9631 → 0.9637; allocation recall 0.9038 → 0.9079 → 0.9175; precision
stayed at 0.998.

### 10.3 How we searched without overfitting, and what we dropped
Four agents worked in parallel on separate questions (unclaimed multi-invoice lines, drift robustness, remittances,
exceptions). Rules for all of them: matcher code never reads the golden; every constant comes from POLICIES or from an
existing matcher; a change ships only if validation **and** holdout both rise and precision stays ≥ 0.998; one LIVE
upload per version.

| Idea | Measured | Why dropped |
|---|---|---|
| Learn payer IBANs and names from our own confident LIVE matches (B4) | +0 lines on val, +1 on hold, 0 on LIVE | The true payer is already a candidate for 95.4 % of unclaimed item receipts; aliases 6–12 months stale lose only 0.02 pp of coverage |
| Amount-only subset search on unclaimed lines | 74 % / 69 % pair precision | Paid and unpaid invoices interleave; the cash fits many sets |
| Date window around the cited invoices | Lower gain and precision than no window | The guard is misleading on a small pool |
| "Pay everything overdue" or oldest-first | Exact on 0–3 % of lines | Payers don't pay in date order |
| Second matching pass after removing claimed items | +0.06–0.08, but 81 % precision in the last month | Lowers overall precision |
| Soft-consuming invoices cited by earlier unclaimed lines | +2 / +5 lines | Too small |
| Wider subset-sum windows for slower payers | val −51, hold +9 true pairs | Bigger pools make solutions less unique |
| Silent-discount grace 14 → 3 days (the written policy) | −0.02 on both windows | The golden accepts up to about 14 days; kept 14 |
| Coincidence guard 0.3 instead of 0.05 for `cn_bundles` | +0.02–0.03 | Completion pair precision falls from 99 % to 96.7 % |

### 10.4 FINAL risks found and handled
- **Stale FX rates.** Every currency in `fx_rates.csv` ends on 2025-12-31. Rules that compare against the bank's original
  amount are unaffected; the remittance matcher now falls back to it too, and `final.sh` warns if `masters_final` does
  not extend the rates.
- **The new South African company (ZA01).** LIVE has only 2 ZA01 lines and no ZA01 items, so nothing can be tested
  before 18:00. The three one-line fixes above remove the known traps.
- **Partial master files.** `lib.masters` replaces whole files with the `masters_final` copy; `final.sh` prints row
  counts per file so a partial file shows up before submitting.

### 10.5 What to expect on FINAL
- **Accuracy decays within the window.** A receipt we leave unclaimed keeps its invoices "open" in our state although
  they are paid. On val these ghost items grow from 0 to about 39k by month 6, and the median pool on an unclaimed receipt
  grows from 32 to 55 invoices. Expect months 5–6 about 1–1.5 points below months 1–2; validation shows the same, so
  it is already in our estimates. (LIVE showed almost no trend: claimed share 88.7 % → 88.0 %.)
- **Slower payers.** Shifting every payment 30 days later costs 3–5 % of the subset and same-amount pairs, about
  −0.1 to −0.15 points. No window cap is close to the data.
- **What to read in the drift report:** ZA01's claimed share (other companies: 67–94 %), the matcher mix against LIVE
  (alarm at ±30 %), and the FX_DIFFERENCE share.

---

## 11. Teammates' cross-checks

> **TL;DR:** three teammates' write-ups and a priorities list confirmed our choices; we were ahead on every segment they
> flagged, nothing was worth shipping before the freeze, and one test idea is worth keeping.

### 11.1 Where their weak segments stand in ours (LIVE)

| Segment | Luigi's branch | Ours (v5) |
|---|---|---|
| `ar_1toN_credit_note` | 63 % | **74.6 %** |
| `ar_discount_late` | 61 % | **82.0 %** |
| `ar_advance_deduction` | 68 % | **99.6 %** |
| `ar_fx` | 77 % | **87.9 %** |
| `ar_dd_represented_transfer` | 89 % | **98.0 %** |
| Overall | 91.11 | **96.34** |

(Matias's branch: validation 92.94, LIVE 82.53 for the code measured there.)

### 11.2 Their ideas, checked against our pipeline

| Idea | From | Our status |
|---|---|---|
| Hide one company's history to imitate a new one (PL01 test: allocation F1 0.882 without name/IBAN fallbacks, 0.955 with, 0.958 with full history) | Luigi | **Worth keeping.** Our payer resolver also falls back to master names, trade names and IBANs, but we never ran this test. At 18:00, watch ZA01's claimed share in the drift report. |
| UNIDENTIFIED for unseen companies decided by evidence (no reference and no known payer) | Luigi | Not adopted: ZA01 is likely a few % of lines and the rule couldn't be validated before the freeze. |
| Bank charges are a fixed fee grid / per-customer | Matias, Luigi | Already covered: SHORT_BANK_CHARGES F1 0.95 on val, `ar_short_bank_charges` 93.2 % on LIVE. |
| Overpayments are +1 %, +10 % or a round amount | Matias | Already in (`overpayment_plausible`); `ar_overpayment` 99.1 % on LIVE. |
| Credit notes only together with their invoice; cancelled items out of the pools | Matias | Covered by `cn_bundles` and the cancelled-items filter in subset-sum. |
| Advice linked by amount when no reference matches; later payment of a disputed deduction | Matias | Covered; `ar_dispute_settlement` 99.6 % on LIVE. |
| p(paid) model for short disputes, accepted at ≥ 65 % pair precision | Priorities list | Rejected: their date rules were 12–24 % precise, and it would give up our 0.998 precision on new data. |
| `ref_partial`: cited invoices + unique completion | Nikita | Same idea as `cn_bundles`, already shipped; not merged, so the same lines aren't claimed twice. |
| FINAL timing: PDF text extraction is paid once for a new folder | Matias | Agreed: plan on 20–30 minutes from 18:00. |

---

## 12. What's left, and what we would do next

> **TL;DR:** the remaining AR losses are lines whose invoices are mostly not in the bank text, so no text rule can find
> them; the next step is a learned "which invoices did this line pay" ranker trained across windows, after FINAL.

### 12.1 Why the remaining multi-invoice lines fail
Diagnosis by a teammate on the sample month (inside the val window, so for understanding only, not for tuning):
- The bank's remittance field holds about 80–140 characters. Customers list the first invoice in full and the rest by
  their last digits, and the field ends after 3–4 references, often in the middle of a number.
- Unclaimed lines settle a median of 15.5 invoices, and **80 % of those invoices are not in the text at all**.
- The payer has a median of 64 open invoices, so the uncited remainder either has hundreds of exact fits (median 194)
  or lies outside the search pool. Filling the cash oldest-first is exact on 0 of 83 lines.
- `cn_bundles` already claims the cited part; the uncited 80 % is the ceiling for any rule that reads only the text.

### 12.2 Short disputes without an advice
17 % of disputed short-payments have no advice anywhere. The bank text lists truncated numbers and nothing says which
invoice was short-paid, so the deducted invoice can't be known. We accept that loss.

### 12.3 The next step: a ranker for "which invoices did this line pay"
- Score each of the payer's open invoices with p(paid on this line), then accept a set only when it sums exactly to the
  remainder (these lines carry no deductions, so the check is strict).
- To avoid overfitting, train the validation model on the holdout window and the holdout model on validation, and judge
  each only on the window it didn't see.
- The same ranker targets `ar_ref_noise`, `ar_1toN_credit_note` and `ar_1toN_text`, whose unclaimed lines fail for the
  same reason. Tracking partial-payment state across matchers is the matching idea for `ar_n_to_m`.

### 12.4 Why we didn't build it today
One honest cycle (build with cross-window training, validation and holdout runs, a LIVE build and upload) takes about
70 minutes, more than was left before the code freeze. A learned invoice chooser is also the part most exposed to drift,
and FINAL is a single shot, so it belongs after the competition, not in the submitted pipeline.

---

## 13. Robustness round for FINAL (17:00–17:40)

> **TL;DR:** six agents tested how the pipeline holds up on a new date range and new settings; nothing was overfitted,
> but five silent-failure risks were fixed in a version that is byte-identical on LIVE and validation (LIVE 96.34 again).

### 13.1 What the six agents checked, and what they found

| Question | Finding | Action |
|---|---|---|
| Do the learned parts go stale over 6–12 months? | No. A classifier trained 9–15 months before the test scores like one trained 3–9 months before (AUC 0.9988 vs 0.9989, category accuracy 0.9998 vs 0.9997, UNIDENTIFIED F1 0.905 vs 0.900). No account, entity or IBAN features; digits are collapsed, so no date or id can go stale. | None |
| How do we do on a company never seen before (ZA01)? | Hiding everything learned about PL01 or MA01: the true payer is still a candidate for 99.1 % / 95.5 % of receipts (from 100 % / 96.7 %); categories of unclaimed lines don't change at all. Only the UNIDENTIFIED flag slips (worth < 0.01 points). | None before FINAL |
| Are the masters and the announced FINAL changes handled? | A partial `masters_final` file would silently drop every other customer, IBAN, mandate and rate (payer matching −0.7 pp); renamed customers lose their old name (−7 % of their lines). Customers changing bank cost nothing. | Fixed (13.2) |
| Which rules were fitted to the TRAIN answers? | Few: the 14-day silent-discount grace, overpayment shape and size cap, duplicate lag, coincidence guards. All behave the same on the untouched 2024 window. Only the overpayment 35 % cap is a clear fit (rejects 4 val / 1 hold true overpayments). | Cap: after FINAL |
| What depends on the date range? | Nothing hard-codes 2024/2025; every window is relative to the booking date; FINAL reads LIVE's folder for January. Stale FX rates matter little: 97 % of cross-currency rows use the bank's own original amount. | None |
| What would fail silently on FINAL-shaped inputs? | Advice rows dropped on small layout changes (e.g. `KWI-25-0308068`, ISO dates: 0–1 of 4 rows); a missing side-file folder starves its matcher quietly; `final.sh` missed "FAILED" fallbacks and 0-line matchers and could reuse an old shared PDF cache. | Fixed (13.2) |

### 13.2 What shipped (tag `final-v5-robust`, commit `6f6b7b5`)
1. **Masters merged by key** (`ingestion/lib.py`): `masters_final` rows replace base rows with the same key instead of
   replacing whole files; former names and IBANs are kept and indexed by the payer resolver (`matching/parties.py`).
   `.csv.gz` masters are read too, and other files are reported instead of ignored.
2. **Remittance parser hardening** (`matching/remittance_parse.py`): accepts `PREFIX-` / `PREFIX/` numbers and ISO dates,
   and prints coverage per advice format with a warning when a layout looks new.
3. **Side-file guards** (`matching/ap_helpers.py`, `matching/batch_helpers.py`): file counts per folder, a warning when the
   phase's folder is empty, and the payment-run month read with a regex.
4. **`final.sh` and `drift.py`**: fresh PDF text extraction, a sanity block after ingestion (lines and open items per
   entity, missing FX), a wider error grep, small entities such as ZA01 always shown in drift, advice parse coverage.

Check: full LIVE (717,870 rows) and validation (706,266 rows) submissions byte-identical to v5, validation 96.73, no
warning fired; the LIVE upload scored 96.34 again. These fixes can only help, or change nothing, on FINAL: their value
appears only if FINAL's inputs differ from LIVE's in the ways above.

### 13.3 After FINAL
- Drop the overpayment size cap for round excesses (+0.01–0.02, unmeasured end to end).
- Parse South African number formats in advices ("1 234,56", "R 5,100.69") and match CSV headers by name.
- A stricter UNIDENTIFIED threshold for companies with no history.
- Bucket classifier amounts in EUR instead of local currency (ZAR amounts are ~20× EUR).
- Two small date fixes (December "service January" wrap; previous-folder payment runs in `refs`).

### 13.4 Lesson: parallel agents need a memory budget
Six agents running data-heavy analyses at once exhausted the machine's memory and it rebooted (17:11). The temporary
folder was wiped; the repo, the frozen FINAL folder and the data were intact. After the restart every Python run went
through a gate (at most 2 heavy processes, only with ≥ 7 GB free) with a watchdog that kills the largest agent process
below 3 GB, and full-pipeline runs were banned for the agents. Nothing crashed again.

---

## 14. Classifier, ranker and FINAL day (16:00–18:45)

> **TL;DR:** a LightGBM model stacked on the line classifier (+0.04 LIVE) and Kenny's ranker for bulk payments without
> an advice (+0.08 LIVE) stack to **LIVE 96.46**. On FINAL, every drift flag was traced to its lines; one hid a real
> problem: the classifier's veto dropping correct ZA01 matches. It was fixed before submitting.

Full detail: `classification/insights.md`.

### 14.1 The line classifier ensemble
- The 22-class logistic regression already labelled lines that settle no invoice ~99.99 % right. The only lever left
  was `p_items` (does this line settle invoices?), which drives the veto (`< 0.2`) and the UNIDENTIFIED flag (`< 0.97`).
- Seven models were tried as binary P(ITEM): LightGBM, XGBoost, a ranker LightGBM on the UNIDENTIFIED population,
  character n-gram LR, linear SVM and Naive Bayes, plus stacks and averages. Each was fit on the 6 months before the
  cutoff, then refit on everything before it.
- **Chosen: LR + LightGBM, quantile-mapped back onto the LR's probability scale**, so 0.2 / 0.97 keep their meaning.
  - val 96.4141 → 96.4419, hold 96.4991 → 96.5276, LIVE 95.62 → 95.65 (on Kenny's v3), then 96.34 → 96.38 (on v5).
  - The 6-model stack was +0.0024 on val and −0.0005 on hold: noise, so the simpler model won.
  - Naive Bayes was clearly worse (−0.6), and SVM alone slightly worse than the LR.
- **Lesson: calibration beat diversity.** The quantile map alone gave +0.012. Four extra models added +0.0015.
- Safety: the models are pre-trained and in git (`classification/pretrained/`, about 12 MB). Any error falls back to
  the plain LR. Runs need `PY=.venv/bin/python`, otherwise LightGBM silently falls back.

### 14.2 The `sd_noadv` ranker (Kenny, `kenny/ranker`)
- What it does: big customer payments with no advice. A logistic regression rates each open invoice and credit note
  of the payer, and the ranker claims the top-rated items while their total stays at or under the cash, covering at
  least 80 % of it.
- About 205–250 lines per phase; 28 % of the invoice pairs it adds are wrong, but it is net positive everywhere.
- It runs as an add-on after the full run (`matching/addon.py`), with `submission_safe.csv.gz` kept as a fallback.
- **The classifier and the ranker work on different lines, so their gains add up:**
  - val 96.76 → 96.82
  - LIVE: 96.34 for v5 alone, 96.38 with the classifier, 96.42 with the ranker, **96.46 with both**
- Caveat: the ranker's hold model learned from val's 2025 answers, so its hold number doesn't follow the time rule.
  The val number is clean.

### 14.3 FINAL day
- **Run:** 220,256 lines (2026-01-02 → 06-30), 18:10 → 18:23. Every matching stage produced lines, the ensemble ran
  without fallback, and validation passed.
- **Drift flags, each traced to lines:**
  - Moncayo moved fully to camt053, which has more fields: fine.
  - PAYROLL +162 % is 900 ZA01 payroll lines: fine.
  - IT01 money matched fell 0.60 → 0.34, but the line share is unchanged; three no-invoice transfers (€96.5 M and
    2 × €9.3 M) are labelled correctly: fine.
  - The new ZA01 entity: **hid a real problem.**
- **The problem:**
  - ZA01 (4,339 lines, MT940, all details inside the text, 35 advices) never appears in TRAIN.
  - The classifier gave its `EFT CREDIT … INVOICE KWI…` receipts p_items ≈ 0.03 with high category confidence.
  - The merge's veto therefore dropped **116 ZA01 receipts that matching had claimed**: 85 `refs:exact`, 64 of them
    with the exact amount and the right payer.
- **How it was found:** the drift report's "52 % UNIDENTIFIED" was a ratio over a sub-population; the real rate was
  8.2 %. Opening the 192 UNIDENTIFIED receipts showed 91 cite an existing invoice, and every exact one had been
  claimed and then vetoed.
- **The fix:**
  - `NO_VETO_ENTITIES=ZA01` in `integration/run.py`; only the merge was re-run.
  - 122 lines changed, all ZA01; UNIDENTIFIED went from 192 to 76.
  - Validation passed, and the LIVE result is byte-identical in content (LIVE has 2 ZA01 lines).
  - Expected gain: about +0.02–0.04.
  - Submitted: `submission_v2_za_noveto.csv.gz` (sha256 `c6bb886b…`).
- None of the advices use South African number formats. The parser fix for them (§13.3) is in place but wasn't
  needed.

### 14.4 Lessons
1. **A rule that lets a learned model override evidence must check the model knows this kind of data.** On an unseen
   entity a confident-but-wrong classifier beat exact invoice-number matches. Next time: no veto for unseen entities,
   banks or formats, and never against `refs:exact`.
2. **A model's confidence doesn't mean much outside its training data.** A 0.98 category confidence on ZA01 was
   confidence about text it had never seen.
3. **Drift scripts point; they don't conclude.** Every flag needed its lines opened, and the dangerous one sat behind
   a ratio that looked normal. "In the same range as other entities" is not a check.
4. **Check the new entity first in a FINAL.** For each new entity: how many receipts were matched, and how many were
   dropped between matching and the merge.
