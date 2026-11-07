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
| + pdf_cases (instalments, retentions, advances, replaced invoices) | 91.37 | 0.928 (0.9989 / 0.866) | 0.932 | 0.852 |
| **+ shortpay fee grid + overpayment +1 %/+10 % + near-invoice tie-break (current)** | **92.94** | **0.928 (0.9989 / 0.866)** | **0.933** | **0.929** |

Points of the current run: alloc 46.38 · txn 27.98 · exc 18.58. pdf_cases: ar_instalment 0.64 -> 0.93 line acc, ar_retention
0.005 -> off the loss list, PARTIAL recall 0.66 -> 0.81. Fee grid / overpayment shapes (see s2): SHORT_BANK_CHARGES recall
0.33 -> 0.99 (precision 1.00), OVERPAYMENT recall 0.47 -> 0.85 (precision 0.99); new methods chg_1_fee(_pick),
ref_over_pct, over_1_pct(_pick) all 100 % pair-precise on val.
Still losing most: ar_ref_noise (0.68 txn pts), UNIDENTIFIED (0.27 exc pts, recall 0.79), PARTIAL (0.25, recall 0.81),
ar_1toN_credit_note (0.22), SHORT_DISPUTE (0.20, recall 0.82), FX_DIFFERENCE (0.19, recall 0.83), ar_n_to_m (0.19),
OVERPAYMENT (0.17), ar_partial (0.16). (Commit `cb19bc9` says "val 90.66" in its message; that number is `8de49b7`'s, the pdf_cases code it added
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
- **Bank charges are a fixed fee grid.** Every SHORT_BANK_CHARGES deduction in TRAIN is one of 6 EUR fees
  {10, 15, 20, 25, 35, 45} converted at a fixed rate per currency (USD x1.185, CAD x1.48, AUD x1.58): 24 (currency, amount)
  values. The grid learned before 2025 covers 100 % of 2025. An unquoted shortfall that is on the grid is a charge
  (val 113/113); off the grid it almost never is (2.6 %). `shortpay_core.fee_grid(cut)` learns it from labels before the cut.
- **Overpayments have three shapes.** All 2,838 single-invoice OVERPAYMENT lines of TRAIN are exactly +1 %, exactly +10 %
  (to the cent) or a whole-unit extra (5 / 10 / 100 …). +1 % / +10 % is near-certain evidence even without a quoted
  reference (val 153/153); a non-round, non-percentage excess is mostly a coincidence (over_1_odd 4.9 %).

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

### 5b. Iteration 2 (from val 92.94): parallel agents, one segment each

Where the points were at 92.94: allocation precision 99.9 % but recall 86.6 % (98.6k of 735k golden pairs missed), and
~99 % of the missed pairs sit on lines that NO matcher claims. At this precision one recovered pair is worth ~0.00004 pts.
Missed pairs by segment: ar_ref_noise 31.6k (~1.2 pts), ar_1toN_credit_note 22.2k (~0.9), ar_short_dispute 18.1k (~0.7, all
on unclaimed lines), ar_1toN_text 9.6k (~0.4), ar_n_to_m 4.8k (~0.2), ar_fx 2.7k (~0.1 + FX_DIFFERENCE recall 0.83).

Set-up (so agents can't interfere with each other or with the judge):
- One **git worktree per agent** (`../aa-wt/{refs,netting,dispute,fx}`, branches `wt/<name>`, from `f5d819a`), `data/` a
  junction to the shared read-only data, own copies of the caches and of the 92.94 allocations as baseline. Each agent owns
  disjoint files: refs -> `refs.py`, `refs_text.py`, `party.py`; netting -> a new matcher `netting.py` (+ its `ORDER` slot);
  dispute -> `remittance.py`, `remit_*.py`; fx -> `shortpay*.py`. Agents commit to their branch only; merged and re-measured
  together afterwards (gains don't simply add up: matchers compete for the same lines).
- **Memory**: `matching/tools/slot.py` caps heavy python runs at 2 machine-wide (16 GB box, 5 concurrent runs crashed it).
- **Same guard-rails for every agent**: matcher code never reads the golden of the phase it predicts (val golden only in
  `matching/tests/` for measurement); labels used for learning only from TRAIN before 2025-01-01 for val (all TRAIN for
  live/final); no hard-coded ids/names/amounts/dates; every new rule justified by POLICIES or a TRAIN-before-2025 statistic;
  every emitted method >= 97 % pair-precise on val, weaker tiers below the 0.65 gate; no fine threshold sweeps on val;
  per-segment comparison against the baseline at each milestone; never crash on live/final.
- Each agent was to write its own report (`matching/tests/<area>/FINDINGS_<name>.md`).

**Outcome: stopped after ~40 min on a change of objectives, before any full-pipeline val run.** Nothing of iteration 2 is
in `matias/full-pipeline`; val stays **92.94**. Each agent's work is committed on its own local branch (not pushed) in
`../aa-wt/<name>`; the last commit on each is labelled `WIP ... UNMEASURED`. Numbers below are the agents' own stand-alone
measurements (one matcher run on top of the 92.94 allocations), not official scores: gains will be smaller once matchers
compete for the same lines. No agent wrote its FINDINGS_<name>.md; this is the record.

| agent / branch | what was found / built | measured (stand-alone, val) | state |
|---|---|---|---|
| refs `wt/refs` (`d56c68d`) | Credit-note structure in refs' pools: (1) cancellation credit notes ("Anulacion de factura") and the invoices they cancel never reach a bank line (POLICIES s5.7; TRAIN 2024: 0 of 3,126) -> removed from every pool; (2) a credit note enters an UNREFERENCED subset only if quoted or if the invoice it rectifies is in the pool (TRAIN 2024: return credit notes on ar_1toN_credit_note lines come with their own invoice 95.7 %); (3) solutions with a linked credit note must contain its invoice. Fewer candidates -> lower coincidence risk -> more unique solutions pass the gate. Uses `credit_notes.related_document/reason` (inputs, not labels). | **+1,214 lines, +3,798 correct pairs**, emitted precision 99.51 % -> 99.56 % | committed, best candidate to merge first |
| dispute `wt/dispute` (`e408e86`) | Remittance advices: (1) amount-only 1:1 links of an advice to a bank line when no reference matches, accepted only if every advice line resolves to an open item; (2) the later payment of a disputed deduction (POLICIES s3: "the disputed part stays open, it may be paid later") linked back through the advice line to its unique item. | t1_amt **870/870** pairs right, settle **620/620**; on a TRAIN 2024-Q2 slice 99.8 % / 100 % | committed, second candidate |
| netting `wt/netting` (`93e552c`, WIP) | New matcher `netting.py` (after refs, before shortpay): a quoted document is mandatory as anchor; quoted invoices + <= 2 unquoted + credit notes netted == cash; a credit note only with its invoice; n-to-m splits at fractions learned from TRAIN before the cut (0.4 / 0.5 / 0.6) -> PARTIAL. | first run: net_fill1 90.6 % (30 lines), net_fill2 96.6 % (7), net_ref_frac 1/3 -> all kept below the gate; the emitted tiers were not yet measured (agent was profiling runtime) | WIP, not ready |
| fx `wt/fx` (`fcd6e47`, WIP) | Diagnosis of ar_fx: of the unclaimed FX lines, 225 end in `no_solution` — multi-invoice payments (median ~10 golden items, payer pool ~56) quoting **truncated / prefix-less** numbers ("INV 115608 116333 ...", "INV# 10124…"); the payer is usually found (no_party 8 %). 145 lines get `fx_party_subset`. In progress: prefix-token cover set, withholding on a truncated quote list, service-month fix, all on the FX path. | none yet | WIP, not ready |

Lessons from the run: (a) four agents + 2 heavy slots on 16 GB spent much of the time queued (`[slot] all 2 slots busy`
in every log): with this box, 2 agents is the useful maximum; (b) agents iterate stand-alone (one matcher on a fixed
prior) quickly, but the official number needs the full pipeline (~9 min, everything re-runs when `matching/lib` changes),
so measure milestones early; (c) worktrees kept the agents fully isolated (no clashes, no half-edited files measured).
To resume: merge `wt/refs` (d56c68d only) and `wt/dispute` (e408e86 only) one at a time into `matias/full-pipeline`, full
val run after each, keep only what raises the official score.

## 6. Practical notes (Windows box, 16 GB RAM)

- `.venv` from `scripts/setup.sh` came up empty here; rebuilt with `uv venv --clear --python 3.12 .venv` +
  `uv pip install --python .venv/Scripts/python.exe -r requirements.txt`. On Windows the interpreter is
  `.venv/Scripts/python.exe`, not `.venv/bin/python`. Set `PYTHONUTF8=1`.
- `integration/make_val.py` crashes at the very end writing `data/val/README.txt` (cp1252 can't encode `→`); the data is
  complete. `integration/run.py` fails to print its score on Windows (path error); run `integration/score.py` directly.
- Memory: 5 parallel agents each loading TRAIN crashed the PC. Two agents at a time on `val`/`sample`, one python process
  each (<1.5 GB), was stable.
- **Timings (7 matchers, one process alone):** val matching ~9 min (refs 326 s, ap 88 s, shortpay 48 s, batches 37 s,
  duplicates 22 s, pdf_cases 14 s, remittance 7 s); integration ~25 s. Two runs at once roughly double both.
- **First run on a new phase pays PDF text extraction once:** pdf_cases took 721 s on the first LIVE run (14 s on val),
  then `matching/out/cache/<phase>/pdf_text.parquet` is reused. **FINAL will pay this too** (~12 min) on top of ~9 min of
  matching + ~70 s of classification: budget ~25 min from 18:00.
- **Matcher cache (`run.py`, `871b5fd`):** each matcher's output is stored in `matching/out/<phase>/cache/` under a chained
  fingerprint (its source, every earlier matcher, `matching/lib` + `integration/lib`, data file stats, classifier labels,
  gates). Unchanged matchers load in 0 s; changing one re-runs it and everything after it. Any edit in `matching/lib` re-runs
  everything (it can't know which matcher uses which helper). `--no-cache` forces a full run. Verified on sample: cached
  output identical to a fresh run (values and dtypes), 86 s -> 19 s.
- **Run-to-run variation:** the same code gave 91.37 and 91.35 on val (55 SHORT_DISCOUNT rows differ): some matcher depends
  on an ordering that is not fully fixed. Small; worth pinning down before trusting differences under ~0.05.
- **One agent per folder at a time.** A Claude desktop session left running kept launching val runs in parallel (same
  output file, three heavy processes) and one of its unfinished changes got committed by the other agent. Check
  `tasklist | grep python` before a long run.

## 7. How to reproduce

```bash
.venv/Scripts/python.exe classification/run.py --phase val          # p_items (needed by refs)
.venv/Scripts/python.exe matching/run.py --phase val                # -> matching/out/val/allocations.parquet + report
.venv/Scripts/python.exe integration/run.py --phase val             # merge + exceptions + validate
.venv/Scripts/python.exe integration/score.py integration/out/val/submission.csv.gz --phase val
```
Same with `--phase live` for the submission file. Code map: `matching/lib/data.py` (loaders + parquet cache),
`matching/lib/evalutil.py` (per-segment precision/recall), `matching/run.py` (orchestrator, `ORDER`, `GATES`),
`matching/matchers/{batches,ap,remittance,pdf_cases,refs,shortpay,duplicates}.py`, `matching/AGENT_BRIEF.md` (rules given
to the sub-agents). Diagnostics (val golden, never fed to matchers): `matching/tests/shortpay/weak_precision.py` (precision of
every shortpay method incl. the ones below the gate, and what the wrong lines really are),
`matching/tests/shortpay/diag_over_chg.py` (why a golden OVERPAYMENT / SHORT_BANK_CHARGES line was not claimed).

## 8. Suggested next steps, by points (updated at val 92.94)

Done since the first version of this list: FX lines and one-invoice short/over payments (shortpay), DUPLICATE_PAYMENT
(duplicates), PDF clauses (pdf_cases), bank-fee grid and overpayment shapes.
1. Merge and measure iteration 2's two committed branches (s5b): `wt/refs` credit-note structure (+3.8k pairs stand-alone),
   `wt/dispute` amount-only advice links + dispute settlements (+1.5k pairs stand-alone).
2. ar_ref_noise / ar_1toN_text: 41k missed pairs (~1.6 pts), mostly unclaimed multi-invoice lines.
3. ar_1toN_credit_note / ar_n_to_m: 27k missed pairs (~1.1 pts): finish `wt/netting`.
4. ar_short_dispute: 18k missed pairs (~0.7 pts), all on unclaimed lines.
5. FX multi-invoice payments with truncated numbers (225 lines): finish `wt/fx`.
6. Run-to-run variation (s6) and `use_prior` for confirming/returns on live/final.
7. LIVE: the last LIVE file (`matching/out/live/allocations.parquet`, 15:04) is from the 91.37 code; a 92.94 LIVE build was
   started and stopped. Rebuild before handing a file over (~9-10 min alone).
