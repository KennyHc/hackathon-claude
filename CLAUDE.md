# Team rules for agents (read fully before doing anything)

We are a team of 4 humans, each running their own Claude Code agents, working on the Talky × Anthropic **bank
reconciliation** hackathon. For every bank line, say which open items it settles and for how much, or what it is
(category), and flag exceptions. Score = 100 × (0.5·allocation F1 + 0.3·transaction accuracy + 0.2·exception macro-F1).

## What `main` is now

`main` = Kenny's full pipeline v5 + FINAL robustness (`kenny/full-pipeline` @ 6f6b7b5), plus the LR + LightGBM line
classifier from `luigi-classifier`, plus Kenny's `sd_noadv` ranker add-on (`kenny/ranker`). It is the **best version:
LIVE 96.46** (classifier without ranker 96.38, Kenny v5 + ranker 96.42, Kenny v5 alone 96.34). On val: 96.82.
- `BEST.md`: the scores and tags of each version.
- `PIPELINE.md`: how the pipeline is built and run.
- `insights.md`: what we learned.
- `OVERFITTING_CHECKS.md`: the holdout checks.
- `classification/PLAN_CLS.md`: the classifier.

**There is no folder ownership any more.** CODEOWNERS, the folder-guard CI check and the edit-blocking hook were
removed, so any agent may edit any folder. Coordinate through the board so two agents never edit the same file at the
same time: post an INFO saying which files you are taking.

## First session in this repo

1. `scripts/setup.sh` checks `gh` and `data/`. Then create the env:
   `uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt`.
   On macOS also run `brew install libomp poppler`. LightGBM/XGBoost need libomp, and ingestion extracts PDF text
   with poppler's `pdftotext`; without it PDF cases and PDF remittance advices are silently lost.
2. Restore the pre-trained classifier: `classification/restore_models.sh`. It copies `classification/pretrained/` into
   `classification/out/`. Without it the live/final models retrain on the first run (about 6–8 min, about 4 GB RAM).
3. Read: `docs/HACKATHON.md` (deadlines, submission limits), `docs/POLICIES.md` (the labels follow it exactly),
   `docs/DATA_DICTIONARY.md`, `docs/FORMAT.md`, then `PIPELINE.md` and `BEST.md`.
4. Read the board: `scripts/board.sh read 30`, then `scripts/board.sh mine`, and answer what tags your human.
5. `git pull --rebase` before you start. Don't redo work someone posted as DONE.

## Running it

Phases are defined in `ingestion/lib.py`:
- `sample`: train 2025-06
- `val`: train 2025-01..06, used for tuning
- `hold`: train 2024-01..06, an untouched holdout
- `live`
- `final`

```bash
.venv/bin/python ingestion/run.py --phase val          # once per phase (builds PDF text caches)
.venv/bin/python integration/run.py --phase val --all  # matching + classification + merge
.venv/bin/python integration/score.py --phase val      # official score, segments, per-method precision
.venv/bin/python integration/validate.py --phase live  # format check of integration/out/live/submission.csv.gz
```

FINAL: `PY=.venv/bin/python integration/final.sh`.
- It wipes old `final` artefacts, then runs ingestion, the pipeline, validation and the drift report, and prints the
  submit command.
- After the safe run it applies the `sd_noadv` add-on (`matching/addon.py`). The file without it is kept as
  `integration/out/final/submission_safe.csv.gz`, and it is used automatically if the add-on fails.
- It never submits.
- Set `PY`: the default `python3` has no LightGBM, and the classifier would fall back to the plain LR.

## Hard rules

1. **`data/` is read-only and never committed.** Derived files go to `<piece>/out/<phase>/` (gitignored).
2. **No leakage.** Whatever is learned for `val` uses only train before 2025-01-01, and for `hold` only before
   2024-01-01. `live`/`final` may use all of train. Never tune anything on `hold`: it is a check.
3. **A change replaces the best version only if it beats val and hold**, and then a LIVE upload (`BEST.md`). Then
   update `BEST.md`.
4. **Only @ZhengJinFa168 submits to the leaderboard.** The quota is per team. Want a file scored? Post a NEED with its
   path and its val/hold scores. Never probe individual lines: the rules forbid trying to obtain LIVE/FINAL answers.
5. **Never write the team token (`HK_TOKEN`) anywhere**: not in code, files, commits, chat or board posts. Env var only.
6. **Deadlines (Madrid time):** FINAL data 18:00 · submissions close 19:30 · project (video, repo, description) 19:45.
   See `docs/HACKATHON.md`.

## The board (GitHub issue, number in `.github/BOARD_ISSUE`)

```bash
scripts/board.sh done "what, where (path), how to regenerate, score, what's next"
scripts/board.sh need @their-login "what you need; blocking or not"
scripts/board.sh info "change others must know about / files you are taking"
scripts/board.sh read 20        # history
scripts/board.sh mine           # everything that tagged your human
```

DONE only after you have **pushed**, so the commit hash exists. One clear post beats five small ones.

## Git workflow

```bash
git add <the files you changed>              # never `git add -A` at root
git commit -m "[piece] short description"
git pull --rebase && git push
```

- Small, frequent commits.
- **Never force-push.**
- Never commit data, outputs or files over 5 MB. The one exception is `classification/pretrained/` (4 model files, each
  under 5 MB, force-added on purpose).
- Run Python with `.venv/bin/python`. Need another package? Add it to `requirements.txt` with a pinned version and say
  so on the board.
