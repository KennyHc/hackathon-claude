# K4 plan — cascade + integration (branch Luigi, 13:05)

Points at stake: exception macro-F1 (20 pts, 2 per code), the glue that keeps K1-K3 points (no double allocation,
no over-allocation), UNIDENTIFIED (receipt with no open item), thresholds, FINAL readiness. Val baseline 13.08.

## Tasks (in order, with time boxes and measurable targets)

| # | By | Task | Done when |
|---|---|---|---|
| 1 | 13:45 | `matching/common.py`: `load_lines(phase)`, `open_at(items, claimed, date)` (balance per item as of a date given earlier stages), `stage_frame()` schema helper, `SUB_COLS`. `matching/run.py --phase P [--stages batch,refs,party] [--out]`: load lines + `ledger.load_items` + line_labels; import `matching/<stage>/matcher.py` via importlib (missing / exception → logged, stage skipped); feed `lines_left`, `items`, `claimed` (+booking_date); enforce (a) a line is claimed by one stage only (later stage rows for an already-claimed line dropped), (b) per stage, sort by booking date and walk a `Ledger`-like balance: an allocation beyond the open balance is trimmed to the balance (or dropped if ≤ 0.01) — except RETURNED (re-opens) and DUPLICATE_PAYMENT (no balance effect); log counts per stage; write `allocations.parquet` (CONTRACTS §B) + `alloc_<stage>.parquet` + `summary.csv` | dummy stage in a temp dir: claims, conflicts, over-allocation and a crashing stage all handled; val run < 2 min with no stages |
| 2 | 14:15 | Exceptions on real allocations: run `integration/run.py --phase val` with whatever stage outputs exist; per-code F1 via score.py; diagnose code by code (confusion vs golden on the lines both cover). Known risks: OVERPAYMENT fires on any single-invoice line with cash > amount (also FX-converted/ partial-trimmed); DUPLICATE_PAYMENT only for items settled earlier in the run (items closed before val start are absent → matching must say it); SHORT_BANK_CHARGES needs country from masters (FINAL: masters_final). Also: a golden "fully paid" item whose real allocation was trimmed by cascade → PARTIAL false positive. | per-code F1 table in FINDINGS.md, each fix measured |
| 3 | 14:30+ | Tune `integration/run.py --p-items` / `--unid-p` on val with the real cascade; UNIDENTIFIED for receipts that no stage claims and C says "no item" | best thresholds recorded |
| 4 | 16:00 | FINAL readiness: `masters.masters_path` per file (masters_final may be partial: fall back per file, not per dir); exceptions.py reads payment_terms/customers/suppliers via `masters_path`; mock final phase (`integration/make_mock_final.py` → `data_mock/final` symlinks? no: data/ is read-only, so a mock data_dir under `integration/out/mock_data/final/` built from live + a fake masters_final) and run the pipeline with `--data-dir`; `integration/pipeline.sh P` (classification → matching → integration → validate) timed on live < 20 min; `integration/RUNBOOK.md` | live end-to-end timed; mock final runs clean |

## Design decisions
- Cascade owns ledger consistency, not the stages: stages may still be careless; the cascade trims.
- Trim rule: if `amount > open + tol` where tol = max(0.01, 0.5%·open) → amount := open (the line still settles it);
  if open ≤ 0.01 and exception not DUPLICATE_PAYMENT/RETURNED → drop the row (log); a line left with 0 rows is
  unclaimed (later stages may take it). Stages run one after another, so stage k sees what 1..k-1 cleared.
- Exceptions from classification/exceptions.py on the merged allocations (one Ledger pass over the whole file).
- Never crash: every stage is optional; integration/run.py already tolerates missing inputs.

## Risks
- Stage interface drift (column names, dtypes) → `common.normalize_stage_output()` coerces and validates.
- Runtime on live (≈ 200k lines): stages' cost dominates; the cascade itself is O(rows).
- FINAL masters_final partial / different columns → per-file fallback + mock test.
