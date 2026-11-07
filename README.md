# Close the tabs: bank reconciliation hackathon

Team repo for the Talky × Anthropic bank-reconciliation hackathon (problem: [docs/README_EN.md](docs/README_EN.md) · deadlines, submission and delivery: [docs/HACKATHON.md](docs/HACKATHON.md) · team strategy: [docs/PLAN.md](docs/PLAN.md)).

## Setup (once per person)

```bash
git clone <repo-url> close-the-tabs && cd close-the-tabs
# put the extracted dataset folders in data/  (gitignored, never committed)
#   data/masters  data/sample  data/train  data/train_lite  data/live  (data/final when released)
scripts/setup.sh
claude              # start Claude Code from the repo root
```

`setup.sh` checks your GitHub login, shows which folder is yours, checks `data/` and creates the Python 3.12 environment in
`.venv/` from `requirements.txt` (needs [uv](https://docs.astral.sh/uv/) or `python3.12`).

## How we work

- **One folder per piece, one owner per folder** ([.github/CODEOWNERS](.github/CODEOWNERS)):
  `ingestion/` · `matching/` · `classification/` · `integration/`. You and your agents write only in yours.
- **Interfaces between pieces** are written down in [docs/CONTRACTS.md](docs/CONTRACTS.md) (owned by integration).
- **Team board** = the pinned GitHub issue. Agents (and humans) post `DONE` / `NEED @someone` / `INFO` there with
  `scripts/board.sh`; tagging `@login` sends that person a GitHub notification. Turn on notifications for the issue
  (or the GitHub mobile app) so you see them.
- Everyone pushes to `main` (`git pull --rebase && git push`). Agent rules and first-session checklist: [CLAUDE.md](CLAUDE.md).

## Enforcement (3 layers)

1. `CLAUDE.md`: rules and first-session checklist every agent reads at startup.
2. `.claude/settings.json` hook: blocks agent Write/Edit outside your folder and anything in `data/`.
   (It can't see shell commands, which is what layer 3 is for.)
3. `folder-guard` GitHub Action: on each push, flags files outside the pusher's folder, anything in `data/`, and files
   over 5 MB, and posts the problem on the board.
