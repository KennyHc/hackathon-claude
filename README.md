<p align="center">
  <img src="assets/claude.svg" alt="Claude" height="64">
</p>

<h1 align="center">Talky × Anthropic hackathon</h1>

<p align="center"><b>Bank reconciliation · Close The Books · Madrid, 7 Nov 2026</b></p>

<p align="center">🏆 <b>1st of 16 teams</b> (about 80 participants). Our pipeline matches ~200k bank lines to invoices<br>
and scored <b>96.71 / 100 on the unseen FINAL set</b>. The organisers' starter scores 11.22.<br>
We held first place for <b>6 h 04 min</b> (64 % of the competition): 12:39–18:00 on LIVE, then 18:47 to the close on FINAL.<br>
We were only ever first or last on the board, never in between.</p>

| | Starter baseline | **Our pipeline** |
|---|---|---|
| FINAL leaderboard (Jan–Jun 2026, unseen) | not submitted | **96.71** |
| LIVE leaderboard (Jul–Dec 2025) | 11.02 | **96.46** |
| val (Jan–Jun 2025, local answer key) | 11.22 | **96.83** |
| Organisers' reference points | naive ≈19, well-worked rules ≈61 | |

More detail: [insights.md](insights.md) (the full story, with numbers) ·
[classification/insights.md](classification/insights.md) (model selection) · [PIPELINE.md](PIPELINE.md) ·
[OVERFITTING_CHECKS.md](OVERFITTING_CHECKS.md) · [BEST.md](BEST.md).

---

## 1. The problem

A fictional industrial group has **17 operating companies, about 50 bank accounts and 11 currencies**. A 12th
currency (ZAR) arrives with a new South African company in FINAL. Every half year the banks send about **200,000 bank
lines**. Each line is little more than an amount, a date and a scrap of text, such as
`EFT CREDIT POOL SUPPLY AZUL LTD MARCH PAYMENT`.

For **every bank line** we must say:
1. **Which open items it settles and for how much.** Items are customer and supplier invoices and credit notes.
   One line can settle up to **1,540** items.
2. **Or, if it settles nothing, what it is:** one of 21 categories such as bank fees, payroll, tax, intercompany or
   cash pool.
3. **Any exception explaining a mismatch**, from 10 codes:

   | Code | Meaning |
   |---|---|
   | `PARTIAL` | only part of the invoice was paid |
   | `OVERPAYMENT` | more than the invoice was paid |
   | `SHORT_DISCOUNT` | short by a discount |
   | `SHORT_WITHHOLDING` | short by withheld tax |
   | `SHORT_BANK_CHARGES` | short by bank charges |
   | `SHORT_DISPUTE` | short by a disputed amount |
   | `DUPLICATE_PAYMENT` | the invoice was paid twice |
   | `RETURNED` | the payment came back |
   | `FX_DIFFERENCE` | the difference comes from currency conversion |
   | `UNIDENTIFIED` | nobody can tell what the line pays |

**Why it's hard:**
* The bank text rarely names the invoices. References are truncated, mistyped or missing.
* Many invoices often add up to the same amount.
* Customers deduct discounts, withholding tax and disputed amounts, and pay in other currencies.
* Some evidence sits only in side files: payment runs, direct-debit batches, card and cheque files, remittance
  advices in CSV, XLSX, email or PDF, and about 130k invoice PDFs.

**Score** = 100 × (0.5 · allocation F1 + 0.3 · transaction accuracy + 0.2 · exception macro-F1):
* **Allocation F1** counts (line, item, amount) pairs. An amount counts if it's within max(0.01, 0.5 %).
* **Transaction accuracy** counts a line as right only if every row on it is right.
* **Exception macro-F1** averages the 10 codes equally.

**The trap:** a wrong match costs more than a missing one, so guessing is punished.

Data windows:

| Window | Period | Lines | Answer key |
|---|---|---|---|
| TRAIN | Jul 2021 – Jun 2025 | 1.49 M | yes |
| LIVE | Jul – Dec 2025 | 203,083 | no (leaderboard, 20 uploads a day) |
| FINAL | Jan – Jun 2026 | 220,256 | no (one shot) |

---

## 2. What we built: evidence first, models second

```
data/<phase>/  ──►  ingestion/  ──►  matching/ (cascade of matchers, strongest evidence first)  ──┐
                       │                                                                         ├──►  integration/run.py  ──►  submission.csv.gz
                       └──────────►  classification/ (line classifier: p_items + category)  ─────┘      merge · exception rules · veto · validate
```

### 2.1 Ingestion (`ingestion/`)
* Normalises bank lines, open items and master data.
* Extracts the text of every PDF (`pdftotext`).
* Parses remittance advices: CSV, XLSX, email TXT and PDF, including South African number formats such as
  `1 234,56` and `R 5,100.69`.
* Merges `masters_final` by key, keeping former names and IBANs.

### 2.2 Matching cascade (`matching/`)
Matchers run in a fixed order, most certain first. **A line claimed by a stronger matcher is never reused by a weaker
one.** Precision is on val, measured as lines fully right:

| # | Matcher | Evidence | Lines claimed | Precision |
|---|---|---|---|---|
| 1 | `payment_runs` | Our supplier payment-run files | 60,361 | 100.0 % |
| 2 | `dd_batches` | Direct-debit batches, returns, re-presentations | 5,984 | 99.6 % |
| 3 | `cards_cheques` | Card settlements and cheque deposits | 3,304 | 100.0 % |
| 4 | `remittance` | Customer remittance advices | 7,862 | 100.0 % |
| 5 | `pdf_cases` | Clauses printed in invoice PDFs (instalments, retentions, advances, replaced invoices) | 6,336 | 99.9 % |
| 6 | `refs` | Invoice numbers in the bank text (exact, truncated, typo'd) | 43,223 | 99.5 % |
| 7 | `subset_sum` | Payer's open invoices that add up to the amount, accepted only when a coincidence-risk estimate says the match is unlikely to be chance | 39,035 | 99.5 % |
| 8 | `same_amount` | Several same-amount invoices: service month in the text, else oldest first | 7,292 | 98.0 % |
| 9 | `short_payments` | One invoice short by bank charges or withholding, or overpaid | 345 | 98.2 % |
| 10 | `ap_refunds` | Supplier refunding its credit note | 2,513 | 100.0 % |
| 11 | `duplicates` | Second payment of an already-paid invoice | 622 | 99.8 % |
| 12 | `cn_bundles` | Cited invoices plus their own credit notes on lines nobody claimed | ~1,100 | pair precision ≈99 %³ |
| + | `sd_noadv` (add-on) | **Learned ranker**: bulk customer payments without an advice. A logistic regression rates each open item of the payer, and the ranker claims the top-rated items that fit the cash | ~205 | pair precision 0.72 |

³ `cn_bundles` often claims only the part of a payment the text cites, so its "lines fully right" is low (14 %), but
the pairs it adds are almost all right.

A payer resolver (`matching/parties.py`) finds who paid from IBANs, names, trade names, former names and aliases
learned from TRAIN.

### 2.3 Line classifier (`classification/`)
For every line it outputs **`p_items`** (the probability the line settles items) and one of **21 categories**.
* **Base:** a 22-class logistic regression on sparse token features.
* **Ensemble:** plus a **LightGBM binary P(ITEM)** model with counterparty-history features, stacked with the LR and
  **quantile-mapped back onto the LR's probability scale**, so the tuned thresholds keep their meaning.
* **Models compared:** LightGBM, XGBoost, character n-gram LR, linear SVM, Naive Bayes, and stacks and averages of
  them. The 6-model stack was within noise of LR + LightGBM, so the simpler model won.
  Details: [classification/insights.md](classification/insights.md).
* **Results:** category accuracy ~99.99 % on lines that settle nothing; settles-items AUC 0.996.

### 2.4 Integration (`integration/`)
* **Merge:** the matchers' rows, plus one category row for every line no matcher claimed.
* **Exception rules engine** (`classification/exceptions.py`): applies [docs/POLICIES.md](docs/POLICIES.md) literally,
  down to the rounding (deductions half-up, partials truncated, overpayments exactly +1 % / +10 % or a round
  amount). Given the true matches on val it scores **99.76 macro-F1**.
* **Veto:** a matcher's claim is dropped when the classifier is very sure the line settles nothing (`p_items < 0.2`
  with confidence ≥ 0.9). It is skipped for entities the classifier never saw (`NO_VETO_ENTITIES=ZA01`).
* **UNIDENTIFIED:** flagged only when `p_items < 0.97`.
* **Validation:** format checks, every line present.
* **Local scorer:** `integration/score.py` gives the official score plus per-segment and per-exception results.

---

## 3. Comparison with the baseline

The baseline is the organisers' starter (`docs/starter/starter.py`): an exact amount plus one of our document numbers
in the bank text. Every unmatched line gets a default label. Both are scored with the official scorer on **val**, the
only window where the answer key is on our machine.

### 3.1 Score components (val)

| Component | Weight | Starter | **Ours** |
|---|---|---|---|
| Allocation F1 | 50 % | 0.083 (precision 0.992, recall 0.043) | **0.964** (precision 0.997, recall 0.932) |
| Transaction accuracy | 30 % | 0.230 | **0.977** |
| Exception macro-F1 | 20 % | 0.009 | **0.967** |
| **Score** | | **11.22** | **96.83** |

**What this shows:**
* **Precision was never the baseline's problem; recall was.** Its matches are 99.2 % correct, but it finds only
  4.3 % of the pairs.
* We kept precision (99.7 %) and raised recall to 93.2 % by reading the side files and the evidence in order of
  strength.

### 3.2 Exceptions (val, F1)

| Exception | Golden rows | Starter | **Ours** |
|---|---|---|---|
| PARTIAL | 12,511 | 0.000 | **0.966** |
| OVERPAYMENT | 400 | 0.000 | **0.998** |
| SHORT_DISCOUNT | 15,624 | 0.000 | **0.982** |
| SHORT_BANK_CHARGES | 171 | 0.000 | **0.950** |
| SHORT_WITHHOLDING | 4,330 | 0.000 | **0.983** |
| SHORT_DISPUTE | 3,036 | 0.000 | **0.918** |
| DUPLICATE_PAYMENT | 707 | 0.000 | **0.966** |
| UNIDENTIFIED | 4,585 | 0.085² | **0.981** |
| RETURNED | 4,092 | 0.000 | **0.994** |
| FX_DIFFERENCE | 15,793 | 0.000 | **0.929** |

² The starter marks all 103k lines it can't match as UNIDENTIFIED: recall 1.0, precision 0.045.

### 3.3 Segments (val, share of lines fully right)
The answer key tags every line with one of 71 case types. Out of those 71:
* **The starter gets 0 % on 57 of them,** covering 133k of the 200k lines.
* **We are at 99 % or better on 45 of them.**

The 25 largest:

| Segment | What it is | Lines | Starter | **Ours** |
|---|---|---|---|---|
| `ar_ref_noise` | Customer payment, reference truncated, typo'd or missing | 42,462 | 0.0 % | **96.3 %** |
| `ap_payment_multi` | Our payment covering several supplier invoices | 24,058 | 0.0 % | **100.0 %** |
| `ar_1to1` | One payment, one invoice, clean reference | 19,143 | 98.6 % | **100.0 %** |
| `ap_payment_single` | Our payment of one supplier invoice | 17,368 | 55.7 % | **100.0 %** |
| `nonitem_bank_fees` | Bank fees | 6,625 | 100.0 % | **100.0 %** |
| `ar_short_discount` | Paid minus an early-payment discount | 6,317 | 0.0 % | **96.3 %** |
| `ar_same_amount_fifo` | Several same-amount invoices, oldest first | 6,151 | 0.0 % | **97.1 %** |
| `ar_1toN_text` | Many invoices named in the bank text | 6,124 | 0.0 % | **95.0 %** |
| `ap_direct_debit` | Supplier collects by direct debit | 5,444 | 24.2 % | **100.0 %** |
| `ap_paid_by_holding` | The holding pays for a subsidiary | 4,798 | 4.6 % | **100.0 %** |
| `ar_instalment` | Invoice paid in instalments (PDF clause) | 4,275 | 0.0 % | **99.7 %** |
| `ar_same_amount_month` | Same-amount invoices, service month in the text | 4,242 | 0.0 % | **97.6 %** |
| `ar_partial` | Partial payment | 3,679 | 0.0 % | **98.4 %** |
| `ar_dd_returned` | Returned direct debit | 3,512 | 0.0 % | **99.5 %** |
| `unidentified` | Nobody can tell what it pays | 3,221 | 100.0 %² | 98.2 % |
| `ar_1toN_credit_note` | Invoices paid net of credit notes | 3,184 | 0.0 % | **76.7 %** |
| `nonitem_bank_fees_intl` | International bank fees | 3,093 | 100.0 % | **100.0 %** |
| `ap_credit_note_netted` | Supplier payment net of credit notes | 2,591 | 0.0 % | **100.0 %** |
| `ap_partial` | Partial supplier payment | 2,536 | 0.0 % | **100.0 %** |
| `ap_supplier_refund` | Supplier refunds its credit note | 2,515 | 30.1 % | **99.9 %** |
| `ar_cheque_deposit` | Cheque deposit | 1,950 | 0.0 % | **100.0 %** |
| `ar_n_to_m` | Several payments for several invoices | 1,857 | 0.0 % | **72.6 %** |
| `nonitem_cash_pool` | Cash-pool sweeps | 1,554 | 0.0 % | **100.0 %** |
| `ar_dd_batch` | Direct-debit batch (up to hundreds of invoices) | 1,530 | 0.0 % | **99.9 %** |
| `ar_short_withholding` | Paid minus withholding tax | 1,401 | 0.0 % | **98.1 %** |

**Where we are still weakest** (val accuracy below 90 %):

| Segment | Accuracy |
|---|---|
| `ar_n_to_m` | 72.6 % |
| `ar_1toN_credit_note` | 76.7 % |
| `ar_short_dispute` | 82.8 % |
| `ar_discount_late` | 84.0 % |
| `ar_third_party_confirming` | 85.4 % |
| `ar_fx` | 87.7 % |

Most of these are lines where several sets of invoices add up to the same amount, or disputes with no advice. We
leave those unclaimed rather than guess.

### 3.4 FINAL matched val
On FINAL the official breakdown was:
* allocation F1 96.2 % (precision 99.6 %, recall 93.0 %)
* transaction accuracy 97.8 %
* exception macro-F1 96.3 %

That's almost exactly val. The weak segments also landed where val predicted:

| Segment | FINAL | val |
|---|---|---|
| `ar_ref_noise` | 96.5 % | 96.3 % |
| `ar_1toN_credit_note` | 75.9 % | 76.7 % |
| `ar_short_dispute` | 81.8 % | 82.8 % |
| `ar_n_to_m` | 72.5 % | 72.6 % |

---

## 4. Score progression

| Step | val | LIVE |
|---|---|---|
| Starter | 11.22 | 11.02 |
| First full pipeline (8 parallel matcher agents) | 90.04 | 93.73 (first upload) |
| Second-wave matchers, exception rules, honest start state | 96.36 | 95.58 |
| UNIDENTIFIED threshold 0.97 | 96.41 | 95.62 |
| Remittance FX, exception refinements, ZA01 prep | 96.46 | 96.05 |
| `cn_bundles` (credit-note bundles) | 96.73 | 96.34 |
| + LR + LightGBM classifier ensemble | 96.76 | 96.38 |
| + `sd_noadv` ranker | 96.82 | **96.46** |
| + ZA01 veto fix, overpayment cap removed | 96.83 | (no ZA01 on LIVE) |
| **FINAL** | | **96.71** |

---

## 5. Leakage and generalisation

* **Time-split learning.** val (Jan–Jun 2025) learns only from data before 2025-01-01. Jan–Jun 2024 is an untouched
  holdout nobody tuned on: 96.84 there against 96.73 on val with the same code. LIVE and FINAL retrain on all of
  TRAIN.
* **No matcher reads the answer key.** The learned parts (classifier, ranker, aliases) are fit only on data before
  each cutoff.
* **FINAL scored above LIVE** (96.71 vs 96.46) on new data with a new South African company, and precision was the
  same on every window.
* **The FINAL-day lesson.** The drift checks flagged the new ZA01 company. Opening the lines behind the flag showed the
  classifier's veto had dropped **116 correct ZA01 matches**, 85 of them exact invoice-number matches, because it had
  never seen that company and was confidently wrong. We kept the veto off for ZA01 before submitting.
  **Rule: a model never overrides exact evidence on data it has never seen.**

---

## 6. How to run

```bash
git clone https://github.com/KennyHc/hackathon-claude.git && cd hackathon-claude
uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt   # or python3.12 -m venv .venv
brew install libomp poppler            # LightGBM/XGBoost runtime + pdftotext (macOS)
classification/restore_models.sh       # put the pre-trained classifier models in place
# extract the dataset into data/: data/masters data/train data/live data/final … (never committed)
```

| Task | Command |
|---|---|
| Whole FINAL run (ingestion → matching → classifier → merge → validate → drift → ranker add-on); prints the submit command | `PY=.venv/bin/python integration/final.sh` |
| Any phase | `integration/pipeline.sh live` |
| Build the local test phase once | `.venv/bin/python integration/make_val.py` |
| Score on val | `.venv/bin/python integration/score.py --phase val --file integration/out/val/submission.csv.gz` |
| Drift: outputs | `.venv/bin/python integration/drift.py --phase final` |
| Drift: inputs | `.venv/bin/python integration/drift_inputs.py --phase final --ref live` |

`PY` must point to the venv. Without it the classifier silently falls back to the plain LR.

The submit step needs the team token in an environment variable. It's never written to a file:
`read -rs HK_TOKEN && export HK_TOKEN && .venv/bin/python integration/submit.py FINAL <file> --i-am-sure`.

## 7. Repository map

| Path | What |
|---|---|
| `ingestion/` | Parsing and normalisation of bank lines, items, masters, PDFs and advices |
| `matching/` | Matcher cascade (`matchers/`), payer resolver, remittance parser, `addon.py` |
| `classification/` | Line classifier (LR + LightGBM ensemble, `pretrained/` models), exception rules engine |
| `integration/` | Merge, validation, scoring, drift checks, `final.sh`, submit |
| `docs/` | The organisers' problem statement, policies, data dictionary, format, starter |
| `atlas/`, `dashboard/`, `guide/` | Visual data briefing, results dashboard, illustrated guide |
| `insights.md` | Everything we learned, with numbers |

## 8. How we worked

* **Four humans, each running several Claude Code agents.** Each folder had one owner, so agents never edited the
  same files.
* **The interfaces between pieces were written down** in [docs/CONTRACTS.md](docs/CONTRACTS.md).
* **A pinned GitHub issue was the team board:** agents posted `DONE`, `NEED @owner` and `INFO` (`scripts/board.sh`).
* **The best version was always tagged and frozen,** so an experiment could never cost the lead.
* **FINAL was one command** with drift checks built in. Only one person submitted, and the token lived only in an
  environment variable.
* **A memory gate for parallel agents** was added after six data-heavy agents crashed the laptop.

## 9. What we'd do next

Precision is 99.6 %, so the gap is **recall (93 %)**:
* credit-note netting for `ar_1toN_credit_note`
* n-to-m payments
* prefix matching of truncated invoice numbers
* unique exact subsets over the payer's whole open pool
* the remainders of partly paid invoices

On the model side:
* no veto for entities, banks or formats the classifier never saw, and never against an exact reference match
* per-entity UNIDENTIFIED thresholds
