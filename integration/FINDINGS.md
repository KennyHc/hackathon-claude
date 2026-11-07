# K4 cascade + integration — findings (val, 14:02)

Full pipeline (batch → refs → party + PDF clauses, C, exceptions): **88.47** = alloc 0.9177 (P 0.993 R 0.853) ·
txn 0.9258 · exc 0.7402. Path: 49.30 (refs+party) → 49.94 (exception fixes) → 75.17 (+batch) → 88.45 (+PDF clauses,
newer stages) → 88.47 (K2 requests).

Per exception F1: SHORT_BANK_CHARGES 0.22 (122/171 golden rows on unclaimed lines) · DUPLICATE 0.52 · OVERPAYMENT 0.53 ·
PARTIAL 0.75 · UNIDENTIFIED 0.86 · SHORT_DISPUTE 0.88 · SHORT_DISCOUNT 0.89 · FX 0.90 · WITHHOLDING 0.93 · RETURNED 0.94.
Precision ≥ 0.91 everywhere; remaining loss is recall → needs stages to claim those lines.

Exception engine fixes (upper bound 99.76 → 99.88): bank charges only cross-border AR; instalment k/N text rule
(1,865/1,865 and 735/735); normalize_amounts (§2) on single-invoice lines; overpay guard (cash > 1.1 × amount: 603/701
predicted OVERPAYMENT were wrong items); matcher "" kept for instalment_last / settlement.
Cascade: one line per stage, items capped at open balance (golden through cascade changes 1/735,474 rows).
FINAL: masters merged with masters_final by key; mock FINAL (ZA01/ZAR) VALID in 233 s; unseen-entity unclaimed receipts
→ UNIDENTIFIED. Timing: val 326 s, live ~8 min (pipeline.sh). Risk: stage runtimes balloon under machine load.
