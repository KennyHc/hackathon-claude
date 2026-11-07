# Best version: what to submit for FINAL

**Git tag `best-live-96.05`** on branch `kenny/full-pipeline` · frozen copy in
`/Users/kenny/Documents/GitHub/hackathon-claude-best` (a separate checkout of the tag; nothing else edits it).

| Where | Score |
|---|---|
| LIVE leaderboard (v4, uploaded 15:57, sub `s_1b8ad6ac2abbfa6e0333`) | **96.05**: allocation F1 0.9511 (precision 0.9985, recall 0.9079), transaction accuracy 0.9746, exception macro-F1 0.9631 |
| previous best (v3, tag `best-live-95.62`) | 95.62 |
| Validation (TRAIN Jan–Jun 2025, tuned on) | 96.46 |
| Untouched holdout (TRAIN Jan–Jun 2024) | 96.59 |

Code = commit `78151ca`: v3 + remittance FX for currency-less advices, voucher discount grace, exception refinements
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
A change replaces this version only if it beats validation **and** the holdout **and** a LIVE upload (96.05), before
the code freeze at 17:45 (FINAL data lands at 18:00). Then add a new tag `best-live-<score>` (never move a pushed tag), check it out in the frozen folder and update this file.
