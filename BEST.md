# Best version: what to submit for FINAL

**Git tag `best-live-95.62`** on branch `kenny/full-pipeline` · frozen copy in
`/Users/kenny/Documents/GitHub/hackathon-claude-best` (a separate checkout of the tag; nothing else edits it).

| Where | Score |
|---|---|
| LIVE leaderboard (v3, uploaded 14:18, sub `s_270b3120f3e3cdf21522`) | **95.62**: allocation F1 0.9487 (precision 0.9983, recall 0.9038), transaction accuracy 0.9741, exception macro-F1 0.9481 |
| Validation (TRAIN Jan–Jun 2025, tuned on) | 96.41 |
| Untouched holdout (TRAIN Jan–Jun 2024) | 96.50 |

Code = commit `f39a6a9` (UNIDENTIFIED threshold 0.97); later commits up to the tag only change documents and visuals.
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
A change replaces this version only if it beats validation **and** the holdout **and** a LIVE upload (95.62), before
the code freeze at 17:00. Then move the tag: `git tag -f best-live-<score> <commit>` and update this file.
