# RUNBOOK — FINAL at 18:00 (branch Luigi)

One command runs everything for a phase: `integration/pipeline.sh P` =
classification/run.py → matching/run.py (cascade batch → refs → party, + PDF clauses) → integration/run.py
(merge, amount fix, exceptions, overpay guard, validate; scores when P has a golden). Log in
`integration/out/P/pipeline_<time>.log`. Exit 0 = valid submission at `integration/out/P/submission.csv.gz`.

Timings (val/live, this machine): see the last section; budget < 20 min.

## Before 18:00 (dry run, ~16:30)
```bash
git status                                   # clean tree, everything committed by the orchestrator
integration/pipeline.sh live                 # end to end, timed; must print VALID ✓
.venv/bin/python integration/lib/test_lib.py # masters_final merge + fx + ledger tests
# mock FINAL (ZA01 / ZAR / masters_final deltas), never touches data/:
.venv/bin/python integration/make_mock_final.py --out integration/out/mock_data
BANKREC_DATA=$PWD/integration/out/mock_data integration/pipeline.sh final   # runs on the mock; must be VALID
```
After the mock run delete `integration/out/final/`, `matching/out/final/`, `classification/out/final/` and
`integration/out/masters_merged/final/` so nothing from the mock is reused.

## 18:00 — FINAL
1. Download the FINAL data into `data/final/` (the human). Check:
   ```bash
   ls data/final data/final/masters_final
   gzcat data/final/bank_transactions.csv.gz | wc -l          # ≈ 220k + 1
   gzcat data/final/bank_transactions.csv.gz | cut -d, -f7 | sort | sed -n '2p;$p'   # 2026-01 → 2026-06
   ```
   If the masters come as `data/final/masters_final/` nothing to do; if they come elsewhere (e.g. `data/masters_final/`
   or inside a zip), put/unpack them to `data/final/masters_final/` — `integration/lib/masters.py` merges every file
   with `data/masters/` by key (masters_final wins), whether it has full files or only changes.
2. Run: `integration/pipeline.sh final` (fresh: no stale `matching/out/final`, `integration/out/masters_merged/final`).
3. Read the log: every stage `ok`, rows per stage similar to live (batch ≈ 30 %, refs ≈ 25 %, party ≈ 27 % of lines);
   `VALID ✓`. If a stage crashed it is skipped (logged): the file is still valid — fix only if there is time
   (≥ 30 min before 19:30), otherwise submit what we have.
4. `.venv/bin/python integration/validate.py integration/out/final/submission.csv.gz --phase final` → exit 0.
5. Human, own terminal: `read -rs HK_TOKEN && export HK_TOKEN` then
   `.venv/bin/python integration/submit.py integration/out/final/submission.csv.gz --phase FINAL --final-i-am-sure`.
   One shot; deadline **19:30**. Target: submitted by 18:45.

## If something breaks
| Symptom | Do |
|---|---|
| classification fails | pipeline continues (integration falls back to defaults); rerun `classification/run.py --phase final` alone to see the error |
| one stage errors / times out (1800 s; batch/pr took 559 s on live under heavy machine load) | it is skipped; to retry one stage: `.venv/bin/python matching/run.py --phase final --stages batch,refs,party` after the fix; `--timeout 0` disables the limit |
| a stage is slow but its own `matching/<stage>/run.py` already wrote `matching/out/final/alloc_<stage>.parquet` | `MATCH_ARGS=--fallback-files integration/pipeline.sh final` |
| INVALID | `.venv/bin/python integration/validate.py SUB --phase final --repair integration/out/final/submission_fixed.csv.gz`, check the log of fixes, submit the fixed file |
| masters_final has different column names | `masters.py` keeps base columns + new ones; check `integration/out/masters_merged/final/*.csv` |
| no time at all | `integration/run.py --phase final` with whatever `matching/out/final/allocations.parquet` exists (or none) still writes a valid file |

## Knobs (integration/run.py; defaults chosen on val)
`--overpay-max 0.1` (drop single-invoice lines with cash > 1.1 × amount) · `--b-exc-fallback 0` (exceptions from
classification/exceptions.py only, matchers' RETURNED / DUPLICATE_PAYMENT / SHORT_DISPUTE passed through) ·
`--unid-p 0.9` · `--p-items 0.5` · `--no-amount-fix`. matching/run.py: `--overclaim drop|dup`, `--conflict rows|line`.
