# Close the Books — Game Plan

> Talky × Anthropic Claude Code Hackathon · Saturday 7 Nov 2026 · times are Madrid time
>
> Written by Kenny (KennyHc). This is the team's strategy: the case types, what each is worth, the order of the
> matchers and the time boxes. **How the code is laid out:** the single `recon/` package in §5 was split into four
> owned folders. Where each piece of §5 lives now:
>
> | §5 module | Now in | Piece |
> |---|---|---|
> | `load.py`, `text.py`, `parties.py`, `pdfs.py`, `remittance.py` (step 0 prep, PDF clauses, advices) | `ingestion/` | A |
> | `matchers/` batches, remittance, refs, subset_sum, same_amount (steps 2–6) | `matching/` | B |
> | `matchers/nonitem.py` + `exceptions.py` (steps 1, 7, 8) | `classification/` | C |
> | `ledger.py`, `pipeline.py`, `eval.py`, `submit.py` | `integration/` | D |
>
> The interfaces between pieces are in `docs/CONTRACTS.md`. Paths in this plan: the starter/scorer are in
> `docs/starter/` (not `data/starter/`). Kenny's first drafts of `recon/load.py` and `recon/pdfs.py` were saved
> before the repo was restructured: the ingestion owner can ask for them on the board and reuse them in `ingestion/`.

---

## 1. The task in one paragraph

A company receives ~200,000 bank movements every six months. For each one we must answer:
**"What does this money correspond to?"**

- Money **in** from a customer → which of our invoices does it pay (and how much of each)?
- Money **out** to a supplier → which of their invoices are we paying?
- Neither (a bank fee, salaries, taxes, a transfer between our own accounts…) → **which category** is it?
- And if the cash doesn't exactly match the paperwork → **why** (discount, partial payment, bank charge,
  currency difference, returned direct debit, duplicate…)? This is the **exception**.

We write all answers into one CSV and upload it. A computer compares it to the hidden correct answers.

---

## 2. The clock ⏰

| Time | What happens | What we do |
|---|---|---|
| **now → 18:00** | LIVE phase (Jul–Dec 2025 data) open | Build, test locally on TRAIN, submit to LIVE often (max 20/day, 1 every 3 min) |
| **~17:30** | — | **Code freeze.** Only bug fixes after this. |
| **18:00** | FINAL data (Jan–Jun 2026) published | Download, run the pipeline, check, **submit ONCE** |
| **19:45** | **Project deadline** | Video (≤ 5 min), GitHub repo link, written description |
| **20:30** | Reveal | Final ranking = the single FINAL score |

⚠️ **Two deliverables, not one:** the FINAL CSV **and** the project (video + repo + description). The jury also
judges the *approach*, not just the score.

---

## 3. How we're scored (and what that means for strategy)

```
Score = 100 × ( 0.5 × Allocation F1  +  0.3 × Transaction accuracy  +  0.2 × Exception macro-F1 )
```

| Part | Weight | Plain English | 💡 Strategy consequence |
|---|---|---|---|
| **Allocation F1** | 50 % | Did we pair each bank line with the right invoices, at the right amount (±0.5 %)? Counted **per invoice pair**. | Lines that pay *hundreds* of invoices (direct-debit batches, big customers with remittance advices) carry most of the pairs → **batch files and remittance advices are worth a lot.** |
| **Transaction accuracy** | 30 % | Is the **whole** bank line 100 % right (every invoice, every amount, the exception)? Counted **per bank line**. | The many "small" lines matter here → **noisy-reference single customers and supplier payments.** |
| **Exception macro-F1** | 20 % | How well do we detect **each** exception type? Each of the 10 types counts **equally**. | A rare type (OVERPAYMENT, 400 cases) counts as much as a common one (FX_DIFFERENCE, 15k) → **do a decent job on every type, even rare ones.** |

**Reference scores:** starter ≈ 11 · naive (exact amount + reference) ≈ **19** · well-worked rules ≈ **61** · perfect = 100.

**Precision matters:** a wrong pair hurts more than an empty one. When we are unsure, it's often better to say
`UNIDENTIFIED` than to guess.

---

## 4. What the data looks like

Everything lives in `data/` (git-ignored). Each phase folder (`train/`, `live/`, later `final/`) has:

| File | What it is | Analogy |
|---|---|---|
| `bank_transactions.csv.gz` | The bank movements (amount, date, free text, payer name, IBAN…) | The bank statement |
| `ar_invoices` / `ap_invoices` / `credit_notes` | Invoices we issued / received, credit notes | The filing cabinet |
| `open_items_at_start` (LIVE/FINAL) | Invoices still unpaid when the phase begins | The "to-collect" list |
| `direct_debit_batches/` | Lists of customers we charged by direct debit (1 bank line = up to 400 invoices) | Our debit order list |
| `card_settlements/` · `cheque_deposits/` | What's inside a card-payout or cheque-deposit line | Deposit slips |
| `payment_runs/` | Which supplier invoices each of our outgoing payments covers | Our payment orders |
| `remittance_advices/` | Big customers' notes: "this payment covers invoices X, Y, Z, minus 50 € for damaged goods" | Letters from customers |
| `pdfs/` | Invoice PDFs with **clauses that exist nowhere else** (instalments, retention, "replaces invoice X", "May services") | The small print |
| `masters/` | Customers, suppliers, bank accounts, payment terms, exchange rates | The address book |
| `train/golden_allocations.csv.gz` | **The correct answers** for 4 years (TRAIN only) + a `segment` label naming the case type | The answer key 🔑 |

### Where the work is (recent TRAIN half-year, ~200k bank lines)

| Case type (`segment`) | % of lines | Example of what it looks like | How we'll solve it |
|---|---|---|---|
| `ar_ref_noise` | **21 %** | `"PAYMENT"` from *VBW Wassertechnik*, 11,756.23 € = 4 open invoices of that customer | Identify customer → find combination of their open invoices that sums to the amount |
| `ap_payment_multi` | **12 %** | Our transfer, text lists `INV0005403 INV0005461…` + end-to-end id `PR01…` | Look up `payment_runs/` by end-to-end id |
| `ar_1to1` / `ap_payment_single` | 18 % | One reference, one invoice, exact amount | Reference + amount match (easy) |
| Non-item lines (fees, cash pool, payroll…) | ~7 % | `"ZBA SWEEP TO HEADER"`, `"COMISION MANTENIMIENTO"` | Text/bank-code classifier learned from TRAIN |
| `ar_short_discount` | 3 % | Paid 2 % less, within discount days | Payment terms + date rule → `SHORT_DISCOUNT` |
| `ar_same_amount_fifo` / `_month` | 5 % | Monthly fee, several identical open invoices | Oldest first, or the month named in the text ("SERVICE MAY 2025") |
| `ar_1toN_text` / `ar_1toN_credit_note` | 5 % | `"RE-NR RE250205509 RE250205898 …"` | Extract all references (fuzzy) + credit notes netted |
| Direct debits, cards, cheques | ~4 % (but **~120k pairs!**) | One line = a whole batch | Batch files |
| `ar_short_dispute` / `ar_1toN_remittance` | 1 % (but **~160k pairs!**) | Big customer, "REMITTANCE 9053060237", 98 invoices | Parse remittance advices (csv/xlsx/txt/pdf) |
| Instalments, retentions, advances, replaced invoices | ~3 % | `"RATE 2/3"` → `PARTIAL` | PDF clauses |
| Returns, duplicates, overpayments, FX, withholding | ~4 % | `RETURNED`, `FX_DIFFERENCE`… | Exception rules (POLICIES.md §3) |

---

## 5. How the solution works (architecture)

Think of it as an **assembly line**. Each bank line goes through a sequence of "matchers", from most certain to
least certain. The first matcher that is confident claims the line. Whatever is left at the end is `UNIDENTIFIED`.

```
                ┌──────────────────────────────────────────────────────────┐
 bank lines ──▶ │ 0. Prepare: load files, normalise text, build indexes    │
                │    + Ledger: tracks how much of each invoice is still open│
                └──────────────────────────────────────────────────────────┘
                                          │
     1. Non-item classifier  ──▶ BANK_FEES, CASH_POOL, PAYROLL, TAX…  (no invoices)
     2. Batch files          ──▶ direct debits, cards, cheques, payment runs, returns
     3. Remittance advices   ──▶ big customers' multi-invoice payments + disputes
     4. Explicit references  ──▶ invoice numbers found in the text (fuzzy)
     5. Who-paid + amount    ──▶ identify customer/supplier → subset-sum of their open items
     6. Same-amount rules    ──▶ month named in text, else oldest first (FIFO)
     7. Leftovers            ──▶ UNIDENTIFIED / best category
                                          │
                ┌──────────────────────────────────────────────────────────┐
                │ 8. Exception engine: compare cash vs invoices →           │
                │    discount? partial? FX? withholding? dispute? …        │
                └──────────────────────────────────────────────────────────┘
                                          │
                               submission.csv.gz ✅
```

**Key ideas that make it work:**

1. **The ledger.** We process bank lines in date order and keep a running "still open" balance per invoice.
   That stops us paying an invoice twice and lets us spot `DUPLICATE_PAYMENT` and `PARTIAL` correctly.
2. **Learn from the answer key.** TRAIN tells us, for 4 years, which bank text/IBAN belongs to which customer.
   We build a lookup **"payer name or IBAN in the bank → customer"** from it. Customers are the same in LIVE/FINAL.
3. **Fuzzy references.** Invoice numbers appear without prefix, with typos, swapped digits or cut off.
   We index every invoice number in several forms (full, digits only, last N digits) and allow 1-digit errors.
4. **Subset-sum.** When there's no reference, we search for the combination of the customer's open invoices
   (oldest first, minus credit notes) that adds up to the amount — and, for exceptions, to the amount
   *plus* a discount/withholding/bank charge.
5. **PDFs pre-read once.** Run `pdftotext` on all ~150k PDFs at the start and extract the special clauses into a
   small table (`invoice → instalments / retention % / advance / replaces X / service month / new IBAN`).

### Code layout

```
recon/
  load.py        read a phase (train/live/final) + masters into dataframes
  ledger.py      open-amount tracking per invoice
  text.py        normalise names & references, fuzzy number matching
  parties.py     "who paid?" → customer / supplier (IBAN, name, learned aliases, parent/group, confirming)
  pdfs.py        pdftotext + clause extraction (cached to data/cache/)
  remittance.py  parse csv / xlsx / txt / pdf advices into (payer, ref, invoice, amount, deduction, reason)
  matchers/      one file per matcher (nonitem, batches, remittance, refs, subset_sum, same_amount)
  exceptions.py  decide the exception code per row (POLICIES.md §3)
  pipeline.py    run all steps for a phase → submission
eval.py          run on a TRAIN window, score with the official scorer, print per-segment accuracy + diff vs last run
submit.py        upload to the leaderboard (token from $HK_TOKEN env var, never in code)
```

### How we test without wasting submissions

The official scorer is in `data/starter/bankrec_score.py`. We **simulate LIVE using TRAIN**:
take Jan–Jun 2025, build "open items at 1 Jan 2025" from the answer key, run the pipeline, score it.
Same shape as LIVE, but with answers, so we can see **per-segment accuracy** and fix the worst/biggest segments first.
LIVE submissions are then used to confirm the TRAIN gains carry over.

---

## 6. Step-by-step plan with time boxes

Each step ends with a measurable result. **Expected scores are rough guesses**, to check we're on track.

### Phase 0 — Foundation · 09:45 → 10:30
- [ ] `load.py`, `ledger.py`, open-items-at-start builder for a TRAIN window
- [ ] `eval.py`: local score + per-segment table on TRAIN Jan–Jun 2025
- [ ] `submit.py` (you register the team and `export HK_TOKEN` yourself)
- [ ] Start the PDF text extraction running in the background (it takes a while)
- **Done when:** the starter-level pipeline scores locally, and one LIVE submission succeeds end-to-end.

### Phase 1 — The big, easy wins · 10:30 → 12:30
- [ ] Non-item classifier (bank code + keywords, learned from TRAIN) → fees, cash pool, payroll, taxes…
- [ ] Batch files: direct debits, card settlements, cheque deposits, payment runs (by end-to-end id / date+amount)
- [ ] Explicit references with fuzzy matching (1:1 and 1:N, with credit notes)
- [ ] "Who paid" lookup (IBAN + names + learned aliases from TRAIN)
- [ ] Subset-sum over the payer's open invoices
- **Target:** ~35–45 locally · LIVE submission #2

### Phase 2 — Remittances & exceptions · 12:30 → 14:30
- [ ] Parse remittance advices (all 4 formats, incl. the Sept-2025 layout change) → multi-invoice payments + `SHORT_DISPUTE`
- [ ] Exception engine: `SHORT_DISCOUNT` (terms + date + 3 days grace), `PARTIAL`, `OVERPAYMENT`,
      `SHORT_WITHHOLDING` (1.25/1.5/10 %), `SHORT_BANK_CHARGES` (≤ 60 € and ≤ 2 %), `FX_DIFFERENCE` (ECB rates ± spread)
- [ ] `RETURNED` (R-transactions, chargebacks, rejected transfers) and `DUPLICATE_PAYMENT`
- **Target:** ~50–55 locally · LIVE submission #3

### Phase 3 — The small print · 14:30 → 16:30
- [ ] PDF clauses: instalments, retentions/release, advance deduction, "anula y sustituye", service month, rappel, new IBAN
- [ ] Same-amount rules: month in text, else FIFO
- [ ] Third-party payers: parent company, confirming bank ("ORD. …"), collection agency ("EXPTE …"), payment factory ES00
- [ ] Tune: when to say `UNIDENTIFIED` instead of guessing (look-alike amounts)
- **Target:** 60+ locally · several LIVE submissions

### Phase 4 — Make it FINAL-proof · 16:30 → 17:45
Known changes in FINAL (Jan–Jun 2026):
- [ ] Banco del Moncayo now in camt.053 format (different text layout) — check LIVE Oct–Dec already works
- [ ] New subsidiary **ZA01** (South Africa, currency **ZAR**, invoice numbers "KWI…", renamed 1 Apr 2026) → read `final/masters_final/`
- [ ] IT01 new collections account from 1 Feb 2026 (old one still valid)
- [ ] Customers renamed / new banks / slower payments → don't over-rely on learned aliases; keep fallbacks
- [ ] Pipeline runs with **one command** on a fresh folder; runtime < 15 min; validator passes on LIVE
- 🧊 **17:45 code freeze**

### Phase 5 — FINAL submission · 18:00 → ~18:45
1. Download `final` data, check SHA-256, extract
2. `python -m recon.pipeline --phase final`
3. Sanity checks: every bank line present, no format errors (run the official validator locally), score-like stats look like LIVE
4. **Submit once** (a file rejected for format errors doesn't count, so we can fix and retry)

### Phase 6 — Project deliverable · 18:45 → 19:45
- [ ] `README.md`: the approach (this assembly line), results table (local + LIVE scores per phase), how to run
- [ ] Make the GitHub repo public (or share access)
- [ ] Description: key ideas · stack & models (Python, pandas, pdftotext, Claude Code) · what we'd do next · team members
- [ ] Video ≤ 5 min, 720p: problem → approach → demo → score progression → lessons learned
- [ ] Submit on the **Proyecto** page before **19:45**

---

## 7. Rules of the road

- **Never** put the team token in code or commit it. `read -rs HK_TOKEN && export HK_TOKEN` in your own terminal.
- Leaderboard: 1 submission / 3 min, 20 / day per phase; asking for an upload URL already uses a slot.
- Only TRAIN has answers. Don't try to obtain LIVE/FINAL answers any other way.
- Any tool or model is allowed (including Claude).
- Commit after each phase with its local and LIVE score, so the history tells the story for the video.

## 8. Risks & fallbacks

| Risk | Fallback |
|---|---|
| A matcher is slow on 200k lines | Index by (entity, customer, amount); cap subset-sum search size |
| PDF extraction too slow | Only extract PDFs of invoices that are open/relevant; run in background early |
| Running out of time | Phases are ordered by points per hour — stop anywhere and still have a valid, improving submission |
| FINAL breaks on new data (ZA01, formats) | Phase 4 dry-run on LIVE; unmatched lines still get a valid `UNIDENTIFIED` row, never a crash |

## 9. Glossary

| Term | Meaning |
|---|---|
| **AR / AP** | Accounts Receivable (customers owe us) / Accounts Payable (we owe suppliers) |
| **Open item** | An invoice or credit note not yet fully paid (`ARI-`, `ARC-`, `API-`, `APC-`) |
| **Credit note** | A "negative invoice" (refund/discount) that reduces what the customer owes |
| **Remittance advice** | A customer's note listing which invoices a payment covers |
| **Direct debit (SEPA)** | We pull money from customers' accounts; one bank line = whole batch |
| **R-transaction** | A direct debit bounced back by the customer's bank → `RETURNED` |
| **Confirming / factoring** | A bank pays the supplier on the customer's behalf |
| **Cash pool** | Automatic sweeps between group accounts to a central "header" account |
| **Norma 43 / MT940 / camt.053** | Spanish / SWIFT / ISO 20022 bank statement formats |
| **FIFO** | First In, First Out → pay the oldest invoice first |
| **Subset-sum** | Finding which items in a list add up to a target amount |
