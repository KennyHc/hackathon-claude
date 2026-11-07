# Best version: what to submit for FINAL

**Git tag `final-v5-robust`** (same output as `best-live-96.34`) on branch `kenny/full-pipeline` · frozen copy in
`/Users/kenny/Documents/GitHub/hackathon-claude-best` (a separate checkout of the tag; nothing else edits it).

| Where | Score |
|---|---|
| LIVE leaderboard (v5, uploaded 16:38, sub `s_bcbcad868f0264faf72c`) | **96.34**: allocation F1 0.9562 (precision 0.9983, recall 0.9175), transaction accuracy 0.9753, exception macro-F1 0.9637 |
| earlier bests | v4 96.05 (tag `best-live-96.05`), v3 95.62 (tag `best-live-95.62`) |
| Validation (TRAIN Jan–Jun 2025, tuned on) | 96.73 |
| Untouched holdout (TRAIN Jan–Jun 2024) | 96.84 |

Code = v5 (merge `b7457de`) + FINAL robustness guards that leave val, hold and LIVE byte-identical: masters_final merged
by key with former names/IBANs kept, remittance parser hardening + coverage warnings, side-file guards, `final.sh` /
`drift.py` checks, `.csv.gz` masters. v5 = v4 + the `cn_bundles` matcher (cited invoices and their credit notes on unclaimed lines,
unique completion or the cited part only). v4 = v3 + remittance FX for currency-less advices, voucher discount grace, exception refinements
(`integration/refine.py`), ZA01 one-liners, masters/FX-rates check in `final.sh`.
The submitted LIVE file and its leaderboard result are kept in the frozen folder under `BEST_RESULTS/` (not in git:
outputs and files over 5 MB stay out of the repo).

## FINAL, from the frozen folder

```bash
cd /Users/kenny/Documents/GitHub/hackathon-claude-best
integration/final.sh                      # or: integration/final.sh FINAL.zip <sha256>
```

It wipes old `final` artefacts, extracts PDF text, runs the pipeline, validates, prints the drift report vs LIVE and
then prints the submit command (it never submits). Data: unzip FINAL into
`/Users/kenny/Documents/GitHub/hackathon-claude/data/final/` (the frozen folder's `data` points there).

## Rule for anything newer
A change replaces this version only if it beats validation **and** the holdout **and** a LIVE upload (96.34), before
the code freeze at 17:45 (FINAL data lands at 18:00). Then add a new tag `best-live-<score>` (never move a pushed tag), check it out in the frozen folder and update this file.
