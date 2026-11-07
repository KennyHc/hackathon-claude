# K3 party — plan (stage 3 of the cascade)

Owner: K3 coordinator (branch `Luigi`). Folder `matching/party/`, output `matching/out/<phase>/alloc_party.parquet`.
Interface: `matcher.match(phase, lines, items, claimed, **ctx)` per docs/PLAN_LUIGI.md.

## What the data says (val, golden)

| segment | lines | items/line | exc | notes |
|---|---|---|---|---|
| ar_ref_noise | 42,462 | 2.2 (74 % = 1) | none | single party, no CN, same currency; 54 % camt (counterparty_name), rest name in text |
| ar_short_discount | 6,317 | 1.1 | SHORT_DISCOUNT | cash = Σ total·(1−pct) for items in window (issue + discount_days + 3) |
| ar_same_amount_fifo | 6,151 | 1 | – | several equal open invoices → oldest by issue date |
| ap_paid_by_holding | 4,798 | 2.6 | – | n43, ES00 account pays ES01/ES02/PT01 supplier items; name in text ("TRANSFERENCIA A …") |
| ar_same_amount_month | 4,242 | 1 | – | month named in text → invoice issued in month+1 |
| ar_partial | 3,679 | 1 | PARTIAL (half) / – (remainder) | remainder = exact match on tracked balance |
| ap_partial | 2,536 | 4.9 | PARTIAL on one | text lists our supplier refs "FRAS …" |
| ar_short_withholding | 1,401 | 3.2 | SHORT_WITHHOLDING | MX/BR/MA; cash = Σ total·(1−r), r∈{1.25,1.5,10}% |
| unidentified_lookalike | 1,364 | – | UNIDENTIFIED | amount matches an unrelated item → require party identification |
| ar_discount_late | 788 | 3.7 | PARTIAL | discount taken after window → amount = cash on the short item |
| third party (confirming/agency/parent) | 1,278 | 1–9 | – | ORD. <customer>, EXPTE, group parents |
| new IBAN / overpayment / duplicate | 278 / 400 / 641 | 1 | – / OVERPAYMENT / DUPLICATE_PAYMENT | |

## Method

1. **Who paid** (party resolver, `party.py`): normalise text (upper, no accents, collapse spaces, drop N43 ` | `),
   strip bank boilerplate prefixes (DIRECT CREDIT, ACH CREDIT, /ORDP/, PIX RECEBIDO, TT IN, FP, BONIFICO DA,
   VIR SEPA RECU /DE, TRANSFERENCIA(S) DE/A, TRANSF. A SU FAVOR, …), then keys = prefixes of the remaining string at
   fixed lengths (8…40 chars) + full counterparty_name + counterparty IBAN.
   Alias table learned from TRAIN golden lines whose items have a single party (val: booked < 2025-01-01; live/final:
   all train): key → party counts, kept when purity ≥ 0.9. Plus master keys (customers/suppliers name, trade_name,
   iban; name prefixes at every truncation length). Longest key wins. Party set extended with group members
   (`group_id`/`parent_customer_id`) only as a fallback when the payer's own items don't fit.
2. **Which items** (`subset.py`): candidates = open items of the party (receipts: ARI + ARC as negative;
   payments: API + APC), same currency, issue_date ≤ booking_date + 3, open balance > 0.01 after `claimed` and our own
   earlier allocations (chronological ledger). Tries in order, first hit wins:
   a. single item exact (several equal → month named in text, else FIFO oldest issue date);
   b. all open items / oldest-k prefix (FIFO);
   c. bounded subset-sum (≤ 24 candidates nearest in date, DFS with pruning, 50 ms cap; ambiguous → prefer
      oldest set; if two different minimal solutions → skip);
   d. explained shortfall variants of a–c: early-payment discount within terms (SHORT_DISCOUNT, full open amount),
      withholding 1.25/1.5/10 % for MX/BR/MA customers (SHORT_WITHHOLDING), late discount (PARTIAL, cash),
      bank charges ≤ min(60 EUR, 2 %) for non-SEPA (SHORT_BANK_CHARGES);
   e. single invoice partial: cash < open, one obvious item (only item or text names it) → PARTIAL, amount = cash;
      overpayment: cash > open of single obvious item → OVERPAYMENT.
   Precision first: no party → no claim (unidentified_lookalike stays UNIDENTIFIED via classification).
3. **Duplicates**: a receipt exactly equal to an item of the party that was fully settled in the last 60 days and no
   open fit → DUPLICATE_PAYMENT (amount = cash).

## Tasks by points/hour

| # | task | val pts at stake | target (seg txn-acc) | who | time |
|---|---|---|---|---|---|
| 1 | party resolver + alias table (core) | 12.7 + all | ar_ref_noise party-hit ≥ 95 % | me | 35 min |
| 2 | ledger + exact subset-sum + FIFO/month | 12.7 + 2.2 | ar_ref_noise ≥ 85 %, fifo ≥ 90 %, month ≥ 80 % | me | 30 min |
| 3 | discount / withholding / late discount | 2.2 | short_discount ≥ 80 %, withholding ≥ 70 % | me / Flash eval | 20 min |
| 4 | AP side: holding + partial (FRAS refs) | 2.8 | holding ≥ 70 %, ap_partial ≥ 40 % | me | 20 min |
| 5 | ar_partial / overpayment / duplicate | 1.0 | ≥ 50 % | later | 20 min |
| 6 | per-segment eval script (Flash) | – | – | Flash | bg |

v1 by 14:30 = tasks 1–3 (+4 if time). Runtime target: val < 5 min, live < 10 min.
