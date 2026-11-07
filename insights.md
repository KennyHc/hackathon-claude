# Insights: Close the Books bank reconciliation (Talky × Anthropic, 7 Nov 2026)

Everything we learned while building the pipeline on branch `kenny/full-pipeline`. Each insight says what we found, the
number behind it, and why it matters or what we did about it. Numbers come from our own runs (logs in
`integration/out/*.log`), the agents' reports, the code and the LIVE leaderboard response. Where a number is uncertain,
the text says so.

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

## Summary: the 7 most important insights

(This file merges two independent write-ups of the same findings, one by a dedicated agent and one by the main
session; every number was checked against the logs, the code or the LIVE response.)

1. **Rules beat everything else here, and precision is the whole game.** Deterministic rule-based matchers decide 88 %
   of bank lines; one small logistic regression labels the rest (lines that settle no invoice). The pipeline submits invoice pairs at **99.9 % precision** and scores **96.36** on val,
   **96.46** on the untouched holdout, and the first LIVE upload (an older version) scored **93.73**. The organisers'
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
   golden, and the holdout window that no rule was tuned on scores 96.46 vs 96.36 on val. Both windows sit inside
   TRAIN, though, so neither tests the announced drift (new bank format, new remittance layout, a new ZAR subsidiary in
   FINAL). Only LIVE and FINAL measure that (4.6).
7. **What's left is genuinely ambiguous.** The remaining losses are mostly lines where several sets of invoices add up to
   the same amount, or disputed short-payments with no remittance advice anywhere. We chose to leave those unclaimed
   rather than guess, because a wrong claim costs more than a miss.

---

## 1. The data and where the points are

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

### 4.5 LIVE vs local
- Teammate's earlier measurement (starter + classifier): val 13.08, LIVE 13.38. Local val tracked LIVE within about 0.3.
- Our first LIVE upload (file `submission_v1.csv.gz`, 704,576 rows, built at 12:05; uploaded at 12:38, three minutes
  before the v2 file finished writing at 12:41) scored **93.73** (allocation F1
  0.944, precision 0.998, recall 0.896; transaction accuracy 0.964; exception macro-F1 0.880). That file came from code
  between our 90.81 and about 94.3 val measurements, so it's roughly in line; we can't pin its exact val equivalent.
- On LIVE v1 the weakest segments were `ar_short_bank_charges` 35 %, `ar_fx` 49 %, `ar_n_to_m` 55 %, `ar_1toN_credit_note`
  72 %, `ar_discount_late` 75 % and `ar_overpayment` 80 %. The second-wave matchers in v2 targeted exactly these (on val:
  `ar_fx` 87 %, `ar_n_to_m` 68 %, overpayment and bank-charge F1 above 0.93). LIVE v2 (708,637 rows, validated; built
  from the code that scores 96.40 / 96.36 on val) has not been scored yet at the time of writing. The start-state fix
  only changes how val and hold are built, not LIVE, so v2 doesn't need rebuilding.
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

Exception macro-F1 went from **0.716** in the first full run to **0.962**: DUPLICATE_PAYMENT 0.110 → 0.966,
SHORT_BANK_CHARGES 0.368 → 0.937, OVERPAYMENT 0.575 → 0.992, UNIDENTIFIED 0.614 → 0.956.

---

## 7. Where the remaining points go (val, honest run)

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

---

## 8. Engineering lessons

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
