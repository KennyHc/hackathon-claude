# K1 batch — plan (stage 1 of the cascade, `claimed` is empty)

Goal: claim, with high precision, every bank line whose items are listed in a **batch/detail file** (payment runs,
DD batches, card settlements, cheque deposits) or in a **remittance advice**, plus the returns of those operations.
Output `matching/out/<phase>/alloc_batch.parquet` (bank_txn_id,item_id,amount,exception,confidence,method).

## Val segments (lines / pairs in val golden) and how each is linked

| # | Segment | val lines / pairs | ≈ pts | Link bank line → detail | Detail → item |
|---|---|---|---|---|---|
| 1 | ap_payment_multi + single + fx + factored | 24.1k/85k + 17.4k/17k + 1.2k/45k + 1.4k/3.5k | 17.0 | camt `end_to_end_id` = PR…-NNNN exact; N43/MT940: ref1/ref2 = e2e truncated to 16 → run prefix + debtor account + amount (orig_amount / `/OCMT/` for FX) + date | `document_number` + `|document_amount|` in AP items of the account's entity (ES00 also pays ES01/ES02/PT01) |
| 2 | ap_confirming | 788/8.1k | 0.7 | N43 ref2 = run_id `CF…`; one debit per maturity date → subset of the run's transfers summing to the line (group by item due_date) | same |
| 3 | ar_dd_batch | 1.5k/94k | 6.6 | batch_id in text (`EINZUG 0225…`, `REMESA SEPA CORE …`, `REMISE …`) + account + amount = batch_total | DD `remittance_info` = our document_number (AR, entity); amount = open |
| 4 | ar_dd_returned + represented | 3.5k/3.5k + ~1k | 1.5 | mandate in text (`MREF+`, `MANDATO`, `RUM`) + amount → latest earlier DD receipt with that mandate/amount | the item of that receipt; RETURNED, amount = receipt amount |
| 5 | ar_short_dispute + ar_1toN_remittance (+ remit-covered others) | 1.1k/105k + 1.2k/55k | 11.2 | advice `payment_ref` token in bank text/ref + amount = advice total (± few days, same entity) | advice document → AR/ARC item by document_number (normalised; no-prefix / customer voucher → fallback by amount within party); SHORT_DISPUTE amount = paid, deductions with reason |
| 6 | ar_card_settlement + chargeback | 1.25k/21.6k + 36 | 1.7 | `MID <merchant> <date>` / `REMISE CB <mid> DU ddmmyy` / N43 `ON <mid> ddmm` + amount = Σ net | order_number → AR item (document_number / PO); amount = gross |
| 7 | ar_cheque_deposit + returned cheque | 1.95k/6.2k + 66 | 0.8 | deposit_id `RCH…` in text | cheque amount + stub_reference → item of drawer; returned: cheque number |
| 8 | ap_direct_debit | 5.4k/5.4k | 1.2 | supplier DD: name + amount + text `FRA <num>` | AP item by document number / amount within supplier (may be left to K2/K3 if refs are weak) |
| 9 | ap_returned_transfer | 171/478 | 0.2 | `RETURN AC01 … PR…-NNNN` / name + amount | items of the returned transfer, RETURNED |

Order of work (points/hour): 1 → 3 → 2 → 4 → 6 → 7 → 5 (remittance parser delegated to Flash `k1-remit` at 13:03,
I build the matcher on top of it when it lands) → 9 → 8.

Val targets (per-segment line accuracy from `integration/run.py`): payment runs ≥ 90 %, DD batch ≥ 90 %,
returns ≥ 80 %, card/cheque ≥ 85 %, remittance segments ≥ 60 % (short_dispute needs exact deduction handling),
**precision of claimed pairs ≥ 97 %** overall. Runtime on val/live < 3 min (PDF parsing cached).

## Rules
- Detail files of a phase + those of `train` from the 2–3 months before the phase start (runs/batches/advices issued
  before the cash). They are inputs, not labels → no leakage. Golden only for scoring.
- Item resolution needs a unique (entity, document_number, amount) hit; ambiguous → drop the row; if a line loses any
  of its documents, drop the whole line (precision first) unless the remaining docs exactly sum to the line.
- Balances: track a local ledger in booking-date order (items paid twice → DUPLICATE_PAYMENT later; returns re-open).
- Exceptions passed: RETURNED, SHORT_DISPUTE, FX_DIFFERENCE (when line currency ≠ item currency), others left to
  classification/exceptions.py.

## Deliverables
`matching/batch/{matcher.py, run.py, pr.py, dd.py, cards.py, cheques.py, remit.py, remit_parse.py}`, FINDINGS.md.
