# K2 refs — plan (stage 2 of the cascade)

Stage 2: lines not claimed by K1 (batch). I claim a line only when an explicit document reference in the bank text
resolves to open items whose amounts (± credit notes, ± explained shortfall) explain the cash. Precision first: an
empty answer (left to K3 / classification) beats a wrong one.

## Segments and val targets (val = 2025-01..06, ~200k lines)

| segment | val lines | pts if perfect | v1 target (line acc) | how |
|---|---|---|---|---|
| ar_1to1 | 19.1k | 4.2 | 95 % | exact/normalised doc number + amount == open (or explained shortfall) |
| ar_1toN_text | 6.1k | 3.4 | 85 % | multi-ref; suffix refs ("FT25-01-13658 14309" → prefix completion); sum == cash |
| ar_1toN_credit_note | 3.2k | 2.7 | 70 % | + open credit notes of the cited invoices via `related_document`, also of same party; CN rows positive |
| ap_credit_note_netted | 2.6k | 1.3 | 70 % | same on AP (debit lines), APC "AB-<inv>" / related_document |
| ar_fx | 0.7k | 1.1 | 50 % | orig_amount/orig_currency or fx.convert, 3 % band; many are remittance (K1) |
| ar_instalment | 4.3k | 0.9 | 70 % | "PLAZO/RATA/INSTALMENT k/n" or PDF n vencimientos: cash ≈ total/n → amount = cash (PARTIAL unless last) |
| ar_n_to_m | 1.9k | 0.8 | 40 % | all-full-but-one partial; choose the item whose remainder is a round share |
| retention / release / replaced / advance | 2.5k | 0.5 | 50 % | PDF clauses (Flash: pdf_clauses.py): cash = total − retention / − advance; release = open remainder; replaced → follow "anula y sustituye" |
| ap_supplier_refund | 2.5k | 0.6 | 80 % | receipt citing APC number or its related invoice ("REEMBOLSO FRA X", "CREDIT NOTE RINV…") |

## Design (matching/refs/matcher.py)

1. **Doc index** over items: key = upper alnum of document_number (and po_number), plus digit-only key, plus
   "strip letter prefix" key; values = item rows; restricted per line to the line's entity (AR: receipts → ARI/ARC/APC;
   AP: debits → API/APC/ARC-paid-out). Payment factory ES00 → ES01/ES02/PT01 for AP.
2. **Ref extraction** from text, ref1, ref2, remittance_info, end_to_end_id: tokens ≥ 4 chars with ≥ 3 digits;
   multi-ref completion: a short numeric token after a full ref inherits the full ref's prefix (same length);
   truncated tail tokens → prefix lookup within the party.
3. **Fuzzy** (only within the party established by exact hits, or within entity when unique): 1-substitution and
   adjacent-transposition variants on the digit part, last-N suffix (N≥4) match within party.
4. **Ledger**: process lines chronologically; open balance = open_amount at phase start − claimed (K1, booking order)
   − my own earlier allocations.
5. **Amount resolution** (cash converted to item currency when FX): try in order and accept the first that matches
   within max(0.01, 0.5 %):
   full sum (invoices − CNs) · + linked open CNs (related_document) · discount within terms (payment_terms.csv,
   booking ≤ issue + days + 3) · withholding 1.25/1.5/10 % · bank charges (≤ 60 EUR, ≤ 2 %) · FX (≤ 3 %) ·
   overpayment (single item, cash > open) · instalment (cash ≈ total/n) · retention (cash = total·(1−r)) ·
   advance (cash = total − advance) · partial-one (n_to_m) · single item cash < open → PARTIAL (only when text says
   "A CUENTA"/instalment, else leave for K3).
   If the text is clearly truncated and cash > sum of cited items → emit cited items at full open (pairs, low conf).
6. Exceptions are recomputed by classification/exceptions.py; I emit my best guess.

## Tasks by points/hour

1. (core, me, 40 min) index + extraction + ledger + full/CN/shortfall resolution → ar_1to1, 1toN_text, credit notes, AP CN.
2. (Flash k2-pdf, background) `pdf_clauses.py` → `matching/out/<phase>/pdf_clauses.parquet`.
3. (me, 20 min) instalment / partial-one / FX / supplier refund.
4. (me, 15 min) PDF-driven: retention, release, advance, replaced invoice.
5. Measure on val after each step (integration/run.py --b alloc_refs), FINDINGS.md.

## Risks
- Stealing lines from K3 with wrong items (ar_ref_noise has refs that are noise): require amount explanation.
- Runtime on live: vectorised index build, per-line work bounded (≤ 40 candidate items, subset search only for ≤ 12).
