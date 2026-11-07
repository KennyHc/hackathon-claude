# K3 party — findings (val, 13:55)

| run | val | allocF1 (P/R) | txnAcc | excF1 |
|---|---|---|---|---|
| party alone (--no-claimed) | 52.65 | 0.446 (0.957/0.290) | 0.745 | 0.401 |
| batch + refs | 75.24 | 0.862 (0.999/0.757) | 0.663 | 0.614 |
| batch + refs + party | **89.10** | 0.919 (0.993/0.855) | 0.927 | 0.767 |

Segments (cascade): ar_ref_noise 0.938 · short_discount 0.949 · same_amount_fifo 0.962 · month 0.971 · partial 0.761 ·
withholding 0.957 · discount_late 0.629 · third-party confirming 0.858 / agency 0.728 / parent 0.954 · new_iban 0.971 ·
duplicate 0.786 · overpayment 0.510 · unidentified_lookalike wrongly claimed 0.6 %.
Biggest win: keep several candidate payers per alias key and let the amount decide (ar_ref_noise 0.705 → 0.858 alone).
Method precision (line-exact): single 99.8 % · subset 97.9 · month 100 · FIFO-equal 91 · discount 98.8 · withholding 97.4 ·
window 92 · cited partial 98 · cited duplicate 100 · uncited duplicate 93 · cited overpayment 71 · CN subset 87.5 · late discount 87.
Disabled (imprecise): pay all group members (0 %), uncited single-open-item PARTIAL (55 %).
Next: rank ambiguous ar_ref_noise subsets (~2.6k lines), late/mixed discounts, uncited OVERPAYMENT, partial chains + typos,
harvest aliases from claimed lines at runtime (new IBANs, Moncayo camt switch).
Runtime: val 40–90 s, live 58 s warm / 115 s cold.
