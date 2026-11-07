# matching (piece B)

**Mission:** for each bank line, find the open items it settles and the amount of each (item currency, positive), and
the exception on each item row (PARTIAL, SHORT_DISCOUNT, SHORT_BANK_CHARGES, FX_DIFFERENCE, …). Allocation F1 is 50 % of
the score, so this is the biggest piece.

**Reads:** `ingestion/out/<phase>/*` (until v1 lands: build from `data/<phase>/` directly, same column names).
**Writes:** `matching/out/<phase>/allocations.parquet` — `docs/CONTRACTS.md` §B.
**Entry point:** `python matching/run.py --phase sample`

Rules for amounts, tolerances and how to pick items: `docs/POLICIES.md` §2, §3, §5. Learn from
`data/train/golden_allocations.csv.gz` (it has `segment` and `difficulty`: use them to see which cases you're missing).
