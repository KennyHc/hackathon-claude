# K2 refs — findings (val, 13:55)

| version | val alone | lines claimed | line-exact | change |
|---|---|---|---|---|
| v0 | – | 75.3k | 0.916 | first build |
| v0.1 | 32.98 | 72.9k | 0.995 | strict rules over all candidate parties before weak ones; AR-only partials; "Anulación" CNs block their invoice |
| v0.3 | 38.06 | 83.8k | 0.998 | suffix completion, party-suffix, entity last-6 index, N43 38-char joins, truncated-list fill (4.3k @ 99.6 %), 1-typo fuzzy (97 %), PDF clauses (instalment/retention/advance/replaced 100 %) |

With K1 first: batch + refs = 60.21. Per segment (cascade): ar_1to1 98.6 % · 1toN_text 84.0 · instalment 72.6 ·
1toN_credit_note 34.7 · n_to_m 41.4 · retention_release 92.5 · retention 45.7 · replaced 71.9 · advance 42.2 ·
supplier_refund 39.1 · ar_fx 22.2 — precision 96.5–100 % everywhere.
Dropped: reference-rate FX within 3 % without orig_amount (0/18 correct).
Next: keep exception "" for method instalment_last / settlement (≈0.5–1 pt), credit-note lines with truncated refs
(≈1 pt), short bank charges rule, tighten n_to_m / same-party-exact-amount (82 %).
