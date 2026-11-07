# Plan — branch `Luigi` (whole workflow, one owner)

Each teammate runs the whole workflow on their own branch; at the end we compare **val scores**
(`integration/score.py --phase val`, tracks LIVE within ~0.3 pts) and build the final pipeline from the best parts.
This branch: @ZhengJinFa168, orchestrated by Claude (Opus) with 4 Opus **coordinators**. Each owns one area, writes its
own plan, gets it reviewed by **Gemini 4 Argon (high)**, and implements itself and/or through **Gemini 3.8 Flash
(high)** workers. Reviewed by Argon (13:00): the split below is Argon's (parsers live with the matcher that uses them,
one cascade owner), to avoid an ingestion bottleneck.

Starting point: `make_val.py` (val phase), `validate/score/submit/run.py`, `integration/lib` (fx, ledger, masters),
`classification/` (line classifier: p_items AUC 0.996, categories 100 % on no-item lines; exception engine: 99.76
macro-F1 given perfect allocations). **Val 13.08 · LIVE 13.38.** Allocation F1 is 0.08: matching is where the points are.

## Coordinators and points at stake on val (txn-acc + alloc-F1 pts if the segment were perfect)

| | Folder | Segments (pts) | ≈ pts |
|---|---|---|---|
| **K1 batch** | `matching/batch/` | ap_payment_multi 9.4 · ar_short_dispute 7.3 · ar_dd_batch 6.6 · ar_1toN_remittance 3.9 · ap_payment_single 3.8 (if in payment runs) · ap_fx 3.3 · ar_card_settlement 1.7 · ap_direct_debit 1.2 · ar_dd_returned 0.8 · ar_cheque_deposit 0.7 · ap_confirming 0.7 · ap_factored 0.5 · represented / chargebacks / returned cheques / returned transfers 0.3 | **~40** |
| **K2 refs** | `matching/refs/` | ar_1to1 4.2 · ar_1toN_text 3.4 · ar_1toN_credit_note 2.7 · ap_credit_note_netted 1.3 · ar_fx 1.1 · ar_instalment 0.9 · ar_n_to_m 0.8 · retention / release / replaced / advance-deduction (PDF clauses) 0.5 · ap_supplier_refund 0.6 | **~16** |
| **K3 party** | `matching/party/` | ar_ref_noise 12.7 · ap_paid_by_holding 1.6 · ar_short_discount 1.4 · ar_same_amount_fifo 1.3 · ap_partial 1.2 · ar_same_amount_month 0.9 · ar_partial 0.8 · withholding 0.5 · discount_late 0.3 · third-party / parent / agency / confirming 0.4 · new IBAN 0.1 · duplicates / overpayment 0.2 · unidentified_lookalike (don't match!) 0.2 | **~22** |
| **K4 cascade** | `matching/run.py`, `matching/common.py`, `classification/`, `integration/` | one allocations file, ledger consistency, exception macro-F1 (**2 pts per code**, incl. rare OVERPAYMENT / SHORT_BANK_CHARGES / DUPLICATE_PAYMENT) on real allocations, UNIDENTIFIED, thresholds, masters_final, FINAL readiness | **~20 + glue** |

## Architecture

```
data/<phase>/ ──▶ matching/run.py (K4): one chronological cascade over the phase's items
                   1. matching/batch/matcher.py   (K1)  batch files, payment runs, remittances
                   2. matching/refs/matcher.py    (K2)  explicit references, PDF clauses
                   3. matching/party/matcher.py   (K3)  who-paid + subset-sum, same-amount, shortfalls
                 → matching/out/<phase>/allocations.parquet  (one file; each line claimed by at most one stage;
                   an item never cleared beyond its open amount)
classification/run.py → classification/out/<phase>/line_labels.parquet  (p_items + categories; done)
integration/run.py   → merge + classification/exceptions.py → submission.csv.gz → validate → score / submit
```

### The matcher interface (all three stages implement exactly this)

```python
# matching/<stage>/matcher.py      stage ∈ batch | refs | party
def match(phase: str, lines: pd.DataFrame, items: pd.DataFrame, claimed: pd.DataFrame, **ctx) -> pd.DataFrame:
    """lines:   bank lines of the phase NOT claimed by an earlier stage (all bank_transactions columns, str).
       items:   integration.lib.ledger.load_items(phase): ITEM_COLS + EXTRA_COLS (payment_terms, po_number,
                related_document, party_name, payment_method, cn_reason); open_amount = balance at phase start.
       claimed: allocations of earlier stages (bank_txn_id,item_id,amount,exception,confidence,method,booking_date).
                Subtract them in booking-date order before using an item's balance on a given date.
       ctx:     line_labels (classification parquet or None), masters_dir (integration.lib.masters.masters_dir(phase)).
       returns: DataFrame bank_txn_id, item_id, amount, exception, confidence, method — rows only for lines this
                stage claims. amount in ITEM currency, POLICIES §2: full open amount when the shortfall is explained
                (discount, withholding, bank charges, FX, overpayment); cash applied for PARTIAL / SHORT_DISPUTE;
                credit notes positive. exception: your guess (RETURNED / DUPLICATE_PAYMENT / SHORT_DISPUTE are
                passed through as given; others are recomputed by classification/exceptions.py)."""
```

Each stage also has `matching/<stage>/run.py --phase P` for development: loads lines + items, reads earlier stages'
`matching/out/P/alloc_<stage>.parquet` as `claimed` when present, calls `match`, writes
`matching/out/P/alloc_<stage>.parquet`, then scores it alone with
`.venv/bin/python integration/run.py --phase val --b matching/out/val/alloc_<stage>.parquet --out …` (prints
per-segment accuracy and per-exception F1). Stages must not import each other.

**No leakage:** for `val`, learn only from train booked < 2025-01-01 (aliases, priors, models); for live/final, all of
train. `data/val/golden_allocations.csv.gz` is for scoring only. **FINAL drift:** use
`integration.lib.masters` (`final/masters_final/`: ZA01, ZAR, KWI… numbering, renames, new IBANs, 2026 FX rates).

## How each coordinator works

1. Read `CLAUDE.md`, `docs/{POLICIES,DATA_DICTIONARY,FORMAT,CONTRACTS,PLAN_LUIGI}.md`, `docs/EDA.md`, its folder.
2. Write `<folder>/PLAN.md` (tasks by points/hour, measurable val target per segment) — **≤ 15 min**.
3. Argon review: `tools/agy/argon.sh <prompt-file> <out>` (read-only); fold the critique in.
4. Build: write core code yourself; hand well-bounded chunks (a parser, a feature, an eval script) to Flash:
   `tools/agy/flash.sh <kN-name> <brief.md> <allowed-path>…` (snapshot copy, allow-listed commands), then
   `tools/agy/gate.sh <name>` and `tools/agy/accept.sh <name>` (copies only allowed files). Review before accepting.
5. Measure on val after every change; keep `<folder>/FINDINGS.md` (what worked, what didn't, numbers) — compared with
   teammates at the end.
6. Never: write outside its folder, write `data/`, git commit/push, touch `HK_TOKEN`, submit. The orchestrator
   commits each folder after a diff check.

## Timeline (Madrid)

| When | What |
|---|---|
| 13:05 | coordinators launched |
| 13:20 | each `PLAN.md` reviewed by Argon |
| ~14:30 | K1/K2/K3 v1 on val · K4 cascade v1 → integrate → **LIVE #2** |
| 14:30–16:30 | iterate by points/hour; LIVE every ~45 min when val improves ≥ 1 pt |
| 16:30 | **dry run**: clean checkout → `live` end to end, timed (< 20 min), validate |
| 17:30 | code freeze (bug fixes only) |
| 18:00 | FINAL data → run → validate → `submit.py --phase FINAL --final-i-am-sure` (human) by ~18:45 |
| 19:45 | project: repo, description, video |

## Risks (from Argon's review)

- Double allocation / broken balances → one cascade, stages see `claimed`; K4 caps items at open amount.
- Wrong amount convention → loses both allocation F1 and exception F1; K4 checks exception F1 on real output each merge.
- FINAL drift (masters_final, ZA01/ZAR, 2026 FX) → `integration.lib.masters`, tested with a mock folder before 16:30.
- Subset-sum blow-up → bound candidates per party and time per line; precision first (an empty answer beats a wrong one).
- `.github/CODEOWNERS` on this branch gives one owner everything: never merge it back to main.
