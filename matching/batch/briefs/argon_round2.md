Review this short plan for round 2 of coordinator K1 (matching/batch/, stage 1 of the cascade; context docs/PLAN_LUIGI.md,
docs/POLICIES.md, matching/batch/*.py). Read-only. Answer in <= 300 words: risks, wrong ordering, anything that would
cost precision on LIVE/FINAL.

1. remit_io.py: fuzzy header detection for csv/xlsx (synonym table EN/ES/FR/DE/IT/PT for document, date, gross,
   deduction, reason, paid, order; any column order; header row found by >= 3 synonym hits). remit.py: when some advice
   rows don't resolve, keep the resolved rows only if sum(paid of resolved rows) (+ credit notes) equals the bank amount
   within 0.01, else, if the unresolved rows are <= 10% of the advice and the resolved paid sum is consistent... no: drop
   the advice. Test on perturbed copies of val advices (renamed/reordered columns, garbage row) in a scratch folder.
2. pr.py: fx.convert with masters_dir(phase); dd.py mandate regex generic ([A-Z]{2,5}-[A-Z]{2}-\d{4,8} plus any
   mandate id known in customers.csv / DD files).
3. New AP matchers (precision-first, claim only unique hits):
   a. ap_supplier_refund (val acc 0.39, 2.5k lines): credit line from a supplier ("DEVOLUCION SALDO A FAVOR", "ABONO
      AB-…", "CREDIT NOTE …") → the unique open APC credit note of that supplier (entity group incl. ES00 payment
      factory) with open amount == line amount; supplier identified by name in text/counterparty or by the APC number
      in text.
   b. ap_confirming leftovers (182 lines): maturity grouping with weekend/holiday shift (max due date in
      (previous booking day, booking date]) and a bounded subset search on transfer count N.
   c. card chargebacks: Spanish "RETROCESION CONTRACARGO", 6-month settlement look-back.
   d. ap_duplicate_payment: a payment-run document whose supplier invoice is already closed / already paid in-phase →
      the closed item, exception DUPLICATE_PAYMENT, amount = paid.
All new logic behind flags defaulting off until the full pipeline stays >= 90.00 on val.
