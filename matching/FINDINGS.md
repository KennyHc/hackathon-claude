# Findings: first iteration of the matching pipeline (branch `matias/full-pipeline`)

Everything below was measured on our data today (2026-11-07) unless marked *(agent-reported)*: those numbers come from a
sub-agent's own val run of its matcher in isolation and were not re-measured matcher by matcher, but the combined pipeline
numbers in section 1 were measured end to end with the team's `integration/run.py` + `integration/score.py`.

## 1. Results so far (val = TRAIN 2025-01..06, LIVE-shaped, held out)

| Step | val score | alloc F1 (P / R) | txn acc | exc macro-F1 |
|---|---|---|---|---|
| Starter as shipped (integration baseline) | 11.22 | | | |
| Starter + classifier (team pipeline v0) | 13.08 | 0.08 | | |
| + batches + remittance | 41.91 | 0.574 (0.9996 / 0.402) | 0.167 | 0.411 |
| + AP payment runs | 64.66 | 0.810 (0.9998 / 0.680) | 0.479 | 0.491 |
| + AR references | 82.89 | 0.910 (0.9986 / 0.836) | 0.886 | 0.539 |
| + shortpay (FX, withholding/discount, over/under-payments) + duplicates (`8de49b7`) | 90.66 | 0.926 (0.9986 / 0.863) | 0.919 | 0.840 |
| **+ pdf_cases (instalments, retentions, advances, replaced invoices) (current)** | **91.37** | **0.928 (0.9989 / 0.866)** | **0.932** | **0.852** |

Points of the current run: alloc 46.38 · txn 27.95 · exc 17.03. pdf_cases: ar_instalment 0.64 -> 0.93 line acc, ar_retention
0.005 -> off the loss list, PARTIAL recall 0.66 -> 0.81. Still losing most: SHORT_BANK_CHARGES (1.0 pt, recall 0.33),
OVERPAYMENT (0.73, recall 0.47), ar_ref_noise (0.68), UNIDENTIFIED (0.27), PARTIAL (0.25), ar_1toN_credit_note (0.22),
ar_n_to_m (0.19). (Commit `cb19bc9` says "val 90.66" in its message; that number is `8de49b7`'s, the pdf_cases code it added
measured 91.37.)
**LIVE (official leaderboard): 82.53** for the same code (commit 686146c): alloc F1 90.6% (P 99.9%, R 82.8%), txn accuracy
88.6%, exception macro-F1 53.4%, amount-weighted accuracy 90.1%. Val predicted 82.89: **val tracks LIVE within 0.4 points**,
so the rules are not fitted to val. Weakest LIVE segments: ar_fx, ar_duplicate_payment, ar_overpayment, ar_short_bank_charges
(all 0%), ar_retention 0.3%, ar_short_withholding 2.5%, ar_advance_deduction 23%, ar_replaced_invoice 28%, ar_n_to_m 32%,
ar_discount_late 33%, ar_short_discount 44%, ar_1toN_credit_note 52%, ar_instalment 63%; ar_ref_noise 87.8% on ~42k lines.

Matchers, in priority order (a line claimed by an earlier matcher is never touched by a later one):

| Matcher | val lines | pair precision | main segments |
|---|---|---|---|
| `batches` | 9,142 | 99.94% | ar_dd_batch, ar_card_settlement, ar_cheque_deposit (all ~100% recall), ar_dd_returned 98.5%, ar_dd_represented_transfer 82% recall |
| `ap` | 62,464 | 100.00% | ap_payment_multi/single, credit_note_netted, partial, paid_by_holding, direct_debit, supplier_refund ~99-100% recall; ap_confirming 82% *(agent-reported per segment)* |
| `remittance` | 6,353 | 99.98% | ar_1toN_remittance 97% recall, ar_short_dispute 83% recall |
| `refs` | 82,710 (after gate) | 98.65% before gate *(agent-reported)* | ar_1to1 99.5% recall, ar_ref_noise line acc 89%, same_amount fifo/month ~96-98% |

## 2. Data facts worth knowing

- **train-lite is not lighter for matching.** Its CSVs (bank, invoices, credit notes, golden, batch files, payment runs) are
  byte-identical to `train`; only raw statements (574 vs 2,300), PDFs (33k vs 132k) and remittances (13k vs 51k) are cut.
  It is nested one level down: `data/train-lite/train/`. Use `sample` (1 month) and `val` (200k lines) for light work.
- Every file in the downloads has a macOS `._name` twin (163-byte junk). Skip names starting with `._` when globbing.
- TRAIN: 1,489,931 bank lines (2021-07..2025-06), 5.4M golden rows, 3.96M AR / 1.53M AP invoices. Golden row count is
  dominated by batch-like segments: ar_dd_batch 714k, ar_short_dispute 689k, ar_ref_noise 677k, ap_payment_multi 598k.
- Non-item lines are ~2.5% of rows but have clean text templates: digits masked, 246 templates learned before 2024-07
  predicted the category of 27,246 test lines (>= 2024-07) with **100% precision, 95% recall** (classification's area).
- Look-alike traps are real: the generic text `SEPA-UEBERWEISUNG EREF+NOTPROVIDED SVWZ+PAYMENT` appears on 148 sample
  lines of which only 4 are UNIDENTIFIED. Never match on text or amount alone.

## 3. Deterministic rules that work (and the traps behind them)

**Direct-debit batches.** Bank credit = `batch_total` on the same account, collection date <= booking date <= +7 days:
unique for all 12,505 TRAIN batches. ~90% of receipts carry the exact invoice number in `remittance_info`; the other
71,308 read `RECIBO nnnnnn` = the **trailing digits** of the invoice (`FV21-0205710` -> `RECIBO 0205710`). The same digits
repeat across years, so a suffix match alone is ambiguous: disambiguate with amount + customer-name prefix + issue_date <=
collection date. That resolved 100% of the 71,308 at 100% precision.

**Card settlements.** Bank amount = sum of `net`; one row per order at **gross**; `order_number` = invoice number.
10,051/10,051 TRAIN settlements exact.

**Payment runs (AP).** The run reference is on **every** payment-run bank line (358,079/358,079 in TRAIN):
- camt: full 19-char `end_to_end_id` `PR..........-nnnn` -> exact transfer;
- N43/MT940: `ref1`/`ref2` truncated to 16 chars (`PR020022061001-0`) -> run id only, pick the transfer inside the run by amount;
- confirming: `CF............` in text/ref2.
Linking by run id removed the date-window misses (an amount+date window gave only 72.6% exact AP lines over all of TRAIN).
Traps: the same run id is debited from two accounts (key on debtor account too); supplier invoice numbers are **not unique
across suppliers** (~50k TRAIN run rows matched several invoices); beneficiary IBAN -> `suppliers.csv` covers only ~51% of
rows, so the supplier is chosen per transfer by votes of rows with a unique exact-amount match *(agent-reported)*;
`open_items_at_start` lacks party name / payment method, filling them from `masters/suppliers.csv` took AP direct debits from
84% to 100% *(agent-reported)*.

**Remittance advices.** Bank text carries only the customer's payment reference (`AVISO 8826032134`, `REM/2023/0385`);
link by that reference + payer name + total, then each advice line -> item by normalised document number. The advice
often arrives after the cash.

**AR references** *(agent-reported precision by method on val)*: `ref_exact` 99.95% (34.5k lines), `party_subset` 99.04%
(35.6k), `party_oldest` (FIFO) 99.34%, `party_subset_month` 98.6%, `partial` 98.1%, `ref_short_discount` 100%. Weak tiers
(below the 0.65 confidence gate, dropped by `matching/run.py`): `party_subset_r2` 73.8%, `ref_short_withholding` 80.8%
(10% withholding coincides with partial payments), `party_oldest_r2` 90.9%, `ref_short_charges` 90%.
Subset-sum risk is gated structurally: expected number of coincidental subsets = #subsets of size <= k / cash in cents.

## 4. Where the remaining ~17 points are (current val run)

Exceptions (each of the 10 codes is worth 2 points of macro-F1):

| exception | gold | pred | precision | recall | points lost |
|---|---|---|---|---|---|
| OVERPAYMENT | 400 | 1 | 0 | 0 | 2.00 |
| SHORT_BANK_CHARGES | 171 | 0 | 0 | 0 | 2.00 |
| DUPLICATE_PAYMENT | 707 | 98 | 0.55 | 0.08 | 1.73 |
| FX_DIFFERENCE | 15,793 | 2,232 | 1.00 | 0.14 | 1.50 |
| SHORT_WITHHOLDING | 4,330 | 2,401 | 1.00 | 0.55 | 0.57 |
| PARTIAL | 12,511 | 8,732 | 0.91 | 0.64 | 0.50 |
| SHORT_DISCOUNT | 15,624 | 11,567 | 1.00 | 0.74 | 0.30 |
| UNIDENTIFIED | 4,585 | 3,850 | 0.94 | 0.79 | 0.28 |
| SHORT_DISPUTE | 3,036 | 2,291 | 1.00 | 0.75 | 0.28 |
| RETURNED | 4,092 | 4,026 | 0.99 | 0.97 | 0.04 |

Most misses are lines no matcher claims yet (the engine `classification/exceptions.py` is ~99.8 macro-F1 when given the
right items), so they are matching gaps:
- **FX**: `refs` skips lines whose currency differs from the item's (ar_fx 0% line accuracy). Use `orig_amount`/`orig_currency`.
- **OVERPAYMENT / SHORT_BANK_CHARGES / withholding**: one quoted invoice, cash slightly above or below; needs a payer +
  reference match that tolerates the gap.
- **DUPLICATE_PAYMENT**: second payment of an already-settled item; needs the ledger across the phase start (the first
  payment is often in the previous phase).

Line accuracy, segments still losing the most: ar_ref_noise 0.73 pts (acc 0.885), ar_short_discount 0.55 (0.42),
ar_instalment 0.23 (0.64), ar_1toN_credit_note 0.22 (0.54), ar_short_withholding 0.20 (0.04), ar_n_to_m 0.19 (0.33),
ar_partial 0.16, ar_retention 0.14 (0.005), ar_fx 0.11 (0), ar_duplicate_payment 0.10 (0), ar_overpayment 0.06 (0).
Instalments, retentions, advances and replaced invoices need PDF clauses (not built yet in this branch).

Smaller known issues: ~300 rows reference an item not open in the phase (integration warns); 18% of ap_confirming pairs
miss because their run file is in the previous phase (`ap.match(..., use_prior=True)` exists, off by default for memory).

## 5. Method and guard-rails (how we tried not to overfit)

- Judge = `val` only, looked at at milestones; development on `sample` and on TRAIN slices before 2025.
- No leakage: anything learned from labels (refs' IBAN/name aliases: 2,639 + 74) uses TRAIN before 2025-01-01 for val;
  for live/final the cut is 2025-07-01 (all of TRAIN). Most rules are structural (run ids, batch totals, mandate ids) and
  use no labels at all.
- No hard-coded ids/names/dates; thresholds from POLICIES.md. Precision gate: methods under ~97% get confidence < 0.65 and
  are dropped. The 0.65 gate itself was chosen from per-method val precision (mild tuning on the judge, not swept).
- Drift: anything unparseable or ambiguous becomes "no claim", never a crash; untested on ZA01/ZAR and Moncayo camt texts.

## 6. Practical notes (Windows box, 16 GB RAM)

- `.venv` from `scripts/setup.sh` came up empty here; rebuilt with `uv venv --clear --python 3.12 .venv` +
  `uv pip install --python .venv/Scripts/python.exe -r requirements.txt`. On Windows the interpreter is
  `.venv/Scripts/python.exe`, not `.venv/bin/python`. Set `PYTHONUTF8=1`.
- `integration/make_val.py` crashes at the very end writing `data/val/README.txt` (cp1252 can't encode `→`); the data is
  complete. `integration/run.py` fails to print its score on Windows (path error); run `integration/score.py` directly.
- Memory: 5 parallel agents each loading TRAIN crashed the PC. Two agents at a time on `val`/`sample`, one python process
  each (<1.5 GB), was stable. Full val pipeline: ~6 min; live: ~6 min.

## 7. How to reproduce

```bash
.venv/Scripts/python.exe classification/run.py --phase val          # p_items (needed by refs)
.venv/Scripts/python.exe matching/run.py --phase val                # -> matching/out/val/allocations.parquet + report
.venv/Scripts/python.exe integration/run.py --phase val             # merge + exceptions + validate
.venv/Scripts/python.exe integration/score.py integration/out/val/submission.csv.gz --phase val
```
Same with `--phase live` for the submission file. Code map: `matching/lib/data.py` (loaders + parquet cache),
`matching/lib/evalutil.py` (per-segment precision/recall), `matching/run.py` (orchestrator, `ORDER`, `GATES`),
`matching/matchers/{batches,ap,remittance,refs}.py`, `matching/AGENT_BRIEF.md` (rules given to the sub-agents).

## 8. Suggested next steps, by points

1. FX lines in refs (orig_amount) and the one-invoice short/over payments: ~5.5 exception points + allocation.
2. DUPLICATE_PAYMENT with a ledger that starts from the previous phase: ~1.7.
3. PDF clauses (instalments, retentions, advances, replaced invoices) + multi-invoice credit-note netting.
4. `use_prior` for confirming/returns on live/final once memory allows.
