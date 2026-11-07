# integration (piece D)

**Mission:** own the end-to-end pipeline and the **one FINAL submission**. Owner: @ZhengJinFa168 (also owns
`docs/CONTRACTS.md`).

| Script | What it does |
|---|---|
| `make_val.py` | builds `data/val/` (LIVE-shaped, train 2025-01→06, with golden). Run once. |
| `run.py --phase P` | merges `matching/out/P/allocations.parquet` + `classification/out/P/line_labels.parquet` (+ `classification/exceptions.py`) → `integration/out/P/submission.csv.gz`, validates, scores if P has labels. Missing inputs are skipped, so it always produces a valid file. `--p-items` = arbitration threshold; `--b/--c` override input paths. |
| `validate.py SUB --phase P [--repair OUT]` | official scorer format checks (label-less golden) + all lines present + size limits + unknown item ids (warning). `--repair` writes a fixed copy and logs every fix. Exit 0 = safe to upload. |
| `score.py SUB --phase val --label X` | official score + components, **per-exception P/R/F1** (as the scorer counts), segments losing the most points; appends to `integration/out/scores.csv` and shows the change vs the last run. `val` is the team's official local score. |
| `submit.py SUB --phase LIVE` | **owner only, run by the human in their own terminal** (`read -rs HK_TOKEN && export HK_TOKEN`). Validates first, local quota guard (1/3 min, 20/day; asking for an upload URL burns a slot), logs every attempt to `integration/out/submissions.csv` (never the token). FINAL needs `--final-i-am-sure` and refuses a second accepted FINAL. `--dry-run` = no network. |
| `scoring.py` | shared helpers (official Golden, per-exception counts). |
| `lib/` | shared `fx.py` (ECB rates as-of, conversions) and `ledger.py` (open balance of each item through time); importable by any piece: `sys.path.insert(0, ROOT); from integration.lib import fx, ledger`. |

```bash
.venv/bin/python integration/run.py --phase val                      # local score
.venv/bin/python integration/run.py --phase live                     # then: integration/submit.py … --phase LIVE
```

Baselines (starter as shipped): **sample 11.28 · val 11.22** (allocF1 0.083, txnAcc 0.230, excF1 0.009).
Starter on val: `docs/starter/starter.py` only knows sample/train/live/final, so run it with a `--data` dir whose `live`
links to `data/val` (and `train` to `data/train`).
