# Team rules for agents (read fully before doing anything)

We are a team of 4 humans, each running their own Claude Code agents, working on the Talky × Anthropic **bank
reconciliation** hackathon: for every bank line, say which open items it settles and for how much, or what it is
(category), and flag exceptions. Score = 100 × (0.5·allocation F1 + 0.3·transaction accuracy + 0.2·exception macro-F1).

## First session in this repo? Do this before your first edit

1. **Know who you work for.** Run `scripts/setup.sh`. It prints your human's GitHub login and **the folder(s) you
   may write to**, and checks that `data/` is in place. If it says you own nothing, stop and tell your human.
2. **Read the problem**, in this order: `docs/HACKATHON.md` (today's deadlines, submission limits) →
   `docs/PLAN.md` (team strategy: case types, what they're worth, matcher order, time boxes) → `docs/README_EN.md` → `docs/POLICIES.md` (the labels follow it exactly) → `docs/DATA_DICTIONARY.md` →
   `docs/FORMAT.md`.
3. **Read your piece:** `<your-folder>/README.md` (mission, inputs, outputs) and your section of `docs/CONTRACTS.md`
   (the exact files you must produce and the ones you can consume).
4. **Read the board:** `scripts/board.sh read 30` to see what's done and what's blocked, then
   `scripts/board.sh mine` for anything that tags your human. Answer those first.
5. **Check what already exists:** `git pull --rebase`, then look at `<your-folder>/` and the `run.py` of the pieces
   you depend on. Don't redo work someone has posted as DONE.
6. Then start working, following the rules below. Unsure whether you may write a path?
   `python3 scripts/owners.py who <path>`.

## Who owns what

| Folder | Piece | Owner | Produces (see `docs/CONTRACTS.md`) |
|---|---|---|---|
| `ingestion/` | A. Parse remittance advices, PDFs, DD/card/cheque/payment-run files; normalise bank lines and items | Nikita | `ingestion/out/<phase>/*.parquet` |
| `matching/` | B. Bank line → items + amounts (+ exception on item rows). **Shared by two owners**, who split it between themselves | @KennyHc + @Matikama0310 | `matching/out/<phase>/allocations.parquet` |
| `classification/` | C. Lines that settle no item (category, UNIDENTIFIED) + the POLICIES §3 exception-rules library | @ZhengJinFa168 | `classification/out/<phase>/line_labels.parquet`, `classification/exceptions.py` |
| `integration/` | D. Merge B + C, validate, score, **the one FINAL submission**; owns `docs/CONTRACTS.md` | @ZhengJinFa168 | `integration/out/<phase>/submission.csv.gz` |
| `atlas/` | Visual data briefing | @KennyHc | `atlas/index.html` |

Two owners in `matching/`: agree on the split on the board (e.g. one sub-folder each) so your agents don't edit the
same files. `.github/CODEOWNERS` is the source of truth. Your human's GitHub login is in `.claude/me` (`scripts/setup.sh` creates it).
Check any path with `python3 scripts/owners.py who <path>`.

## Hard rules

1. **Write only inside the folder(s) your human owns.** Read anything. A hook blocks Write/Edit elsewhere; do not get
   around it with shell commands (`sed -i`, `>`, `mv`, `cp`, `git checkout` on others' files). CI flags it anyway.
2. **`data/` is read-only and never committed.** Derived files go to `<your-folder>/out/<phase>/` (gitignored).
3. **Don't edit** root files, `.github/`, `.claude/`, `scripts/`, `docs/` unless you own them (the admin does; the
   integration owner owns `docs/CONTRACTS.md`).
4. **Need something from another piece** (a column, a fix, a format change)? Don't do it yourself. Post a NEED on the
   board, tagging the owner, and work around it meanwhile (mock the input from the contract).
5. **Contracts are the interface.** Produce exactly what `docs/CONTRACTS.md` says for your piece. To change a contract,
   post a NEED to the integration owner; only they edit it.
6. **Only the integration owner submits to the leaderboard.** The quota is per team (1 per 3 min, 20 per day per
   phase), and even asking for an upload URL uses a slot. Want a file scored? Post a NEED to the integration owner.
   Self-score on TRAIN/sample locally instead.
7. **Never write the team token (`HK_TOKEN`) anywhere**: not in code, files, commits or board posts. Env var only.
8. **Deadlines (Madrid time):** FINAL data 18:00 · submissions close 19:30 · project (video, repo, description)
   19:45. Everything must run end to end on `live` well before 18:00. See `docs/HACKATHON.md`.

## The board (GitHub issue, number in `.github/BOARD_ISSUE`)

The last posts are shown to you at session start. Use `scripts/board.sh`:

```bash
scripts/board.sh done "Remittance parser v1: ingestion/out/<phase>/remittance_lines.parquet (csv+xlsx+txt; pdf next)"
scripts/board.sh need @their-login "Need party_name_norm in open_items.parquet for name matching. Blocking: no, using customer_name meanwhile"
scripts/board.sh info "Heads-up: bank_lines.parquet now dedupes Banca Ribagorza overlaps; re-run ingestion/run.py"
scripts/board.sh read 20        # history
scripts/board.sh mine           # everything that tagged your human
```

- **DONE**: something others can use has been **pushed** (so the commit hash in the post exists). Say what, where
  (path), how to regenerate, score if relevant, what's next.
- **NEED**: you're blocked or want something from another piece. Always tag the owner with `@login`, say whether it
  blocks you.
- **INFO**: a change others must know about (schema, rerun needed, finding about the data).
- Don't post minor progress. One clear post beats five small ones. Answer NEEDs that tag your human.

## Git workflow (everyone pushes to `main`; folders never overlap, so no conflicts)

```bash
git add <your-folder>/                       # never `git add -A` at root
git commit -m "[matching] short description"
git pull --rebase && git push
```

Commit only files in your folder. Small, frequent commits. Never force-push. Never commit data, outputs or files > 5 MB.

## Conventions

- Every piece has an entry point: `.venv/bin/python <piece>/run.py --phase {sample,train,train_lite,live,final}` that reads
  `data/<phase>/` (and other pieces' `out/`) and writes `<piece>/out/<phase>/`. Others regenerate your outputs locally by
  running it; outputs are not shared through git.
- Develop on `sample` (one month, seconds), check on `train_lite`, then `train` / `live`.
- Paths are relative to the repo root. **Run Python with `.venv/bin/python`** (Python 3.12, created by
  `scripts/setup.sh` from `requirements.txt`: pandas 2.3, pyarrow, scikit-learn, openpyxl, pypdf…). Need another package?
  Post a NEED to the admin; don't `pip install` into the shared requirements yourself.
- Starter / reference scorer (always pass `--data data`):
  `.venv/bin/python docs/starter/starter.py --data data --phase sample --out integration/out/sample/starter.csv.gz` ·
  `.venv/bin/python docs/starter/score_train.py <sub.csv.gz> --data data [--sample]`.
