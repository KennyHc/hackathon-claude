# K1 batch — findings (val, ~14:05)

Alone: **64.49** (allocF1 0.797, P 0.9994 R 0.662 · txn 0.462 · exc 0.540). Runtime val 36 s, live 37 s.
Segments (line exact incl. exception): ap_payment_multi 99.7 % · ap_payment_single 99.9 · ar_dd_batch 94.5 ·
ar_short_dispute 79.9 · ar_1toN_remittance 86.6 · ap_fx 94.7 · ar_card_settlement 100 · ap_direct_debit 97.5 ·
ar_dd_returned 95.9 · ar_cheque_deposit 99.4 · ap_confirming 76.9 · ap_factored 95.3 · ap_returned_transfer 86.0 ·
ar_cheque_returned 97 · ar_card_chargeback 64. Also ap_paid_by_holding 100, ap_credit_note_netted 99.7, ap_partial 99.6.

What made the difference: e2e / run ids are not unique across entities → key by (account, e2e) and (entity, run_id)
(pair precision 0.98 → 1.000); supplier invoice numbers collide → resolve by beneficiary IBAN, name, amount, currency;
N43/MT940 cut e2e to 16 chars → prefix + account + amount (FX amount from orig_amount or /OCMT/); aggregated runs mix
currencies → convert, accept on transfer count + total within 1 %; confirming = one debit per maturity date with
"N PROVEEDORES"; DD returns by mandate + amount, latest collection ≥ 2 days before (median lag 7), returned items looked
up in all AR files; remittance deductions: stated reason → SHORT_DISPUTE (paid), RET/1.25/1.5/10 % → withholding (full),
in-terms discount → full, else PARTIAL (paid).
Next: ar_short_dispute 207 missed (advices after phase end / text-listed → refs) · ap_confirming 182 (bounded subset) ·
ar_dd_batch 66 · ar_dd_returned ~130 (prefer re-presentation link) · ap_duplicate_payment 66 (not handled).
Exceptions engine: adds 73 wrong DUPLICATE_PAYMENT and turns 43 empty + 48 FX_DIFFERENCE rows into PARTIAL on batch rows.
