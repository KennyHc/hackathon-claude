# Plan — branch `Luigi` (whole workflow, one owner)

Each teammate runs the whole workflow on their own branch; at the end we compare **val scores**
(`integration/score.py --phase val`, tracks LIVE within ~0.3 pts) and build the final pipeline from the best parts.
This branch: @ZhengJinFa168, orchestrated by Claude (Opus) with 4 Opus **coordinators**, each owning one area, writing
its own plan, getting it reviewed by **Gemini 4 Argon (high)**, and implementing through **Gemini 3.8 Flash (high)**
workers in throwaway worktrees.

Starting point (on main, already here): `make_val.py` (val phase), `validate/score/submit/run.py`, `integration/lib`
(fx + ledger), `classification/` (line classifier: p_items AUC 0.996, categories 100 % on no-item lines; exception
engine: 99.76 macro-F1 given perfect allocations). **Val 13.08 · LIVE 13.38.** Allocation F1 is 0.08: matching is
where the points are.

## Points at stake on val (txn-accuracy pts + allocation-F1 pts, if a segment were perfect)

| Coordinator | Segments (pts) | ≈ pts |
|---|---|---|
| **K2 batch** | ap_payment_multi 9.4 · ar_short_dispute 7.3 · ar_dd_batch 6.6 · ar_1toN_remittance 3.9 · ap_payment_single 3.8 · ap_fx 3.3 · ar_card_settlement 1.7 · ap_direct_debit 1.2 · ar_dd_returned 0.8 · ar_cheque_deposit 0.7 · ap_confirming 0.7 · ap_factored 0.5 · returns/chargebacks/represented 0.3 | **~40** |
| **K3 transfers** | ar_ref_noise 12.7 · ar_1to1 4.2 · ar_1toN_text 3.4 · ar_1toN_credit_note 2.7 · ap_paid_by_holding 1.6 · ar_short_discount 1.4 · ar_same_amount_fifo 1.3 · ap_credit_note_netted 1.3 · ap_partial 1.2 · ar_fx 1.1 · ar_same_amount_month 0.9 · ar_instalment 0.9 · ar_partial 0.8 · ar_n_to_m 0.8 · supplier_refund 0.6 · withholding 0.5 · discount_late 0.3 · long tail (retention, third-party, duplicates, overpayment, replaced, new IBAN, advance) ~1.5 | **~37** |
| **K4 C+D** | non-item lines ~3 (done) · UNIDENTIFIED 0.7 · exception macro-F1 **20** (rules done; needs right amounts + pass-through codes) · arbitration between K2/K3 | **~20 + glue** |
| **K1 ingestion** | enables K2 (remittances, PDFs) and K3 (aliases, refs) | — |

## Architecture on this branch

```
data/<phase>/ ─▶ K1 ingestion/run.py ─▶ ingestion/out/<phase>/{remittance_lines,doc_clauses,party_aliases,refs}.parquet
                                    │
   K2 matching/batch/run.py ◀───────┤  → matching/out/<phase>/alloc_batch.parquet   (claims lines first: most certain)
   K3 matching/text/run.py  ◀───────┘  → matching/out/<phase>/alloc_text.parquet    (skips K2's lines; ledger after K2)
                                    │
   classification/run.py (done)  → classification/out/<phase>/line_labels.parquet
   K4 integration/run.py: alloc_batch > alloc_text (conflict rule by confidence) + C categories + assign_exceptions()
                        → integration/out/<phase>/submission.csv.gz → validate → score (val) / submit (live, final)
```

Allocation parquet columns (all matchers): `bank_txn_id, item_id, amount (item currency, POLICIES §2 convention),
exception (guess; RETURNED / DUPLICATE_PAYMENT / SHORT_DISPUTE are passed through), confidence (0–1, per line),
method`. Shared code: `integration/lib/fx.py`, `integration/lib/ledger.py` (open balances in booking order),
`classification/exceptions.py`. Every entry point: `.venv/bin/python <piece>/run.py --phase {sample,val,live,final}`.

**No leakage:** for `val`, learn only from train booked < 2025-01-01 (aliases, priors, models); for live/final, all of
train. `data/val/golden_allocations.csv.gz` is for scoring only.

## Coordinator scope

- **K1 `ingestion/`**: (1) remittance advices csv/xlsx/txt/pdf → `remittance_lines` (payer, payment_ref, doc refs,
  amount_paid, deduction + reason); (2) PDF clauses (instalments, retention, advance deduct, replaces, service month,
  new IBAN, factoring) → `doc_clauses`; (3) `party_aliases`: counterparty name/IBAN/UltmtDbtr → party_id from masters +
  train golden (time-cut for val); (4) `refs`: normalised document-number index (full, digits-only, last-N, 1-typo).
  Deliver **v0 of (3)+(4) and csv/xlsx remittances within ~45 min**, then PDFs.
- **K2 `matching/batch/`**: direct-debit batches (+ R-returns → RETURNED, re-presented), card settlements (gross,
  chargebacks), cheque deposits (+ returned cheques), payment runs by end-to-end id (multi, single, FX, confirming,
  factored), remittance-driven matching (+ SHORT_DISPUTE deductions). Writes `alloc_batch.parquet`.
- **K3 `matching/text/`**: customer/supplier transfers: explicit refs (fuzzy) → who-paid (aliases) + subset-sum over
  that party's open items incl. credit notes and the explained-shortfall variants (discount/withholding/bank charges/FX)
  → same-amount rules (month in text, else FIFO) → instalments/partials/retention via `doc_clauses` → duplicates /
  overpayments via the ledger. Precision first: leave a line unmatched rather than guess. Writes `alloc_text.parquet`.
- **K4 `classification/` + `integration/`**: merge of the two allocation files (conflicts, confidence thresholds),
  retune `--p-items` / `--unid-p` on real outputs, exception engine on real (not golden) allocations, UNIDENTIFIED
  lookalikes, end-to-end timing, FINAL runbook, `--repair` drill, project write-up (approach + scores).

## How each coordinator works

1. Read `CLAUDE.md`, `docs/{POLICIES,DATA_DICTIONARY,FORMAT,CONTRACTS,PLAN_LUIGI}.md`, its folder.
2. Write `<folder>/PLAN.md` (tasks, files, measurable target on val per segment, order by points/hour).
3. Get it reviewed by Argon (`agy … --model gemini-4-argon-high`, read-only), fold the critique in.
4. Implement through Flash workers (`agy … --model gemini-3.8-flash-high`), each in its own worktree with an allow-list
   (`.venv/bin/python`, `ls/cat/head/grep/wc`; reads of `data/` + `.venv`), one small task each (≤ ~20 min), then a
   diff gate (only that worker's paths) before copying into the folder. The coordinator may also write code itself.
5. Measure on val after every merge (`integration/run.py --phase val` + per-segment accuracy), keep `<folder>/FINDINGS.md`
   (what worked, what didn't, numbers) — that file is what we compare with teammates at the end.
6. Never: write outside its folder, write `data/`, commit/push, use the git remote, touch `HK_TOKEN`, submit.
   The orchestrator commits each folder after a diff check.

## Timeline (Madrid)

| When | What |
|---|---|
| 13:00–13:20 | plan reviewed (Argon) · coordinators launched |
| ~13:45 | K1 v0 aliases + refs + csv/xlsx remittances |
| ~14:45 | K2 + K3 v1 on val → integrate → **LIVE #2** |
| 15:00–16:30 | iterate by points/hour; LIVE every ~45 min when val improves ≥ 1 pt |
| 16:30 | **dry run**: clean checkout → `live` end to end, timed, validate |
| 17:30 | code freeze (bug fixes only) |
| 18:00 | FINAL data → run → validate → `submit.py --phase FINAL --final-i-am-sure` (human) by ~18:45 |
| 19:45 | project: repo, description, video |

## Risks

- K3 subset-sum blow-up → cap candidates per party (open, ±window), time budget per line.
- K2/K3 claim the same line → K4 rule: batch wins unless its confidence < K3's.
- Wrong amount convention kills exception points → K4 checks exception F1 on real allocations every merge.
- FINAL run time → full `live` run must finish < 20 min; measured at the 16:30 dry run.
- `.github/CODEOWNERS` on this branch gives one owner everything: never merge it back to main.
