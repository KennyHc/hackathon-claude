# ingestion (piece A)

**Mission:** turn the raw inputs into clean tables everyone else builds on. Ship a rough v1 FAST (others are waiting on
it), then improve.

**Reads:** `data/<phase>/` (all files) + `data/masters/`.
**Writes:** `ingestion/out/<phase>/` — see `docs/CONTRACTS.md` §A for exact schemas:
`bank_lines.parquet`, `open_items.parquet`, `batch_lines.parquet`, `remittance_lines.parquet`, `doc_clauses.parquet`,
`party_aliases.parquet`.
**Entry point:** `python ingestion/run.py --phase sample`

Priorities: 1) `bank_lines` + `open_items` (unblock B and C) → 2) `batch_lines` (DD / card / cheque / payment runs) →
3) `remittance_lines` (csv/xlsx/txt first, then pdf) → 4) `doc_clauses` from PDFs → 5) `party_aliases`.
Note: `bank_transactions.csv.gz` is already normalised; the value you add is the derived columns (names, refs, EUR).
