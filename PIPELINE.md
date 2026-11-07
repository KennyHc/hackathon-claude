# Full pipeline (branch `kenny/full-pipeline`)

Worktree: `/Users/kenny/Documents/GitHub/hackathon-claude-kenny` (a separate checkout; `data/` is a symlink to the main
checkout's data, read-only). Background: `docs/PLAN.md`, `docs/POLICIES.md` (the labels follow it exactly),
`docs/DATA_DICTIONARY.md`, `docs/FORMAT.md`. Issues #1–#28 on GitHub describe each piece.

## Run it

```bash
cd /Users/kenny/Documents/GitHub/hackathon-claude-kenny
python3 ingestion/run.py --phase sample          # already done for sample/val/live; rerun only if ingestion changes
python3 matching/run.py --phase sample --only <your_matcher>   # develop one matcher
python3 integration/run.py --phase sample --all  # matching (all, in ORDER) + classification + merge
python3 integration/score.py --phase sample      # official score + segments by points lost + per-method precision
python3 integration/score.py --phase sample --dump ar_ref_noise   # wrong lines of a segment -> CSV
```

Phases: `sample` = TRAIN 2025-06 (≈38k lines, fast) · `val` = TRAIN 2025-01..06 (≈200k lines, LIVE-shaped, use to
confirm) · `live` · `final`. Labelled phases have `ingestion/out/<phase>/golden.parquet` (with `segment`).
Side files (payment_runs/, direct_debit_batches/, card_settlements/, cheque_deposits/, remittance_advices/, pdfs/) are
read from `ctx.data_path(...)` (sample and val read `data/train/`). PDF text is cached in
`data/cache/pdftext_<train|live>.jsonl.gz` and `data/cache/pdftext_<train|live>_remittance_advices.jsonl.gz`
(one JSON per line: `{"file": "<path relative to pdfs/ or remittance_advices/>", "text": "..."}`).

## Inputs you get (`matching.common.Ctx`)
- `ctx.bank`: bank lines (all columns of bank_transactions + `direction, amount_abs, amount_eur, party_name_raw,
  party_name_norm, doc_tokens` (uppercase alnum tokens with ≥3 digits) `, doc_refs` (digits only, no leading zeros))
- `ctx.items`: items open at the phase start or issued during it: `item_id, item_type (ARI/ARC/API/APC), side (AR/AP),
  entity, party_id, party_name, party_name_norm, document_number, document_number_norm, document_alnum, issue_date,
  due_date, currency, total_amount, open_amount, payment_terms, payment_method, po_number, po_norm, related_document, reason`
- indexes: `ctx.items_by_id`, `ctx.open_items`, `ctx.by_docnorm`, `ctx.by_alnum`, `ctx.by_party` (open items oldest first)
- `ctx.masters` (customers, suppliers, bank_accounts, payment_terms, fx_rates, categories, entities)

## Rules for parallel work
- **Only create/edit your own files** (listed in your brief). Shared files (`matching/common.py`, `matching/run.py`,
  `ingestion/*`, `integration/*`, `classification/run.py`) belong to the lead; if you need a change there, say so in your
  final report — meanwhile put helpers in your own module.
- Develop on `sample`; run `val` once you're happy (memory: several agents share 24 GB; avoid running `val` in a loop).
- Precision first: a wrong line costs more than an unclaimed one. Aim ≥ 95 % precision per method.
- Don't commit, don't push, don't touch the main checkout (`/Users/kenny/Documents/GitHub/hackathon-claude`).
- Final report: what you built (files), per-segment accuracy before/after and your method's precision on `sample` and
  `val`, known gaps, and anything the lead must change in shared files.
