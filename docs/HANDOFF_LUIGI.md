# Handoff — branch `Luigi` (state at 13:25 Madrid)

Read this first if you are a new Claude session taking over (previous one ran out of usage).

## Where things are
- Plan: `docs/PLAN_LUIGI.md` (Argon-reviewed). Matcher interface, folders, timeline, rules are there.
- Baseline: val **13.08**, LIVE #1 **13.38** (starter matches + classifier). Submission log: `integration/out/submissions.csv`.
- Four Opus coordinators were launched at 13:05 (they stop when the old session ends). Their work in progress is
  committed as "WIP snapshot 13:25"; each folder has `PLAN.md`, `ARGON_REVIEW.md`, and (when written) `FINDINGS.md`:
  - **K1 batch** `matching/batch/` — DD batches, cards, cheques, payment runs, remittances (~40 pts)
  - **K2 refs** `matching/refs/` — fuzzy invoice refs, credit notes, PDF clauses (~16 pts)
  - **K3 party** `matching/party/` — who-paid aliases + bounded subset-sum, same-amount, shortfalls (~22 pts)
  - **K4 cascade** `matching/run.py`, `matching/common.py`, `classification/`, `integration/` (pipeline.sh, exceptions on real output, thresholds, masters_final, RUNBOOK)
- Shared libs: `integration/lib/{fx,ledger,masters}.py` (masters_final support for FINAL). Tooling: `tools/agy/`
  (`argon.sh` review, `flash.sh` + `gate.sh` + `accept.sh` for Gemini Flash workers).

## To resume
1. `git checkout Luigi && git pull`; run `scripts/setup.sh` if `.venv` is missing.
2. Read each folder's `PLAN.md` / `FINDINGS.md`; check each stage runs:
   `.venv/bin/python matching/<stage>/run.py --phase val` then
   `.venv/bin/python integration/run.py --phase val --b matching/out/val/alloc_<stage>.parquet --out /tmp/s.csv.gz`.
3. Relaunch coordinators (Opus subagents) with the prompts' substance: own one folder, follow PLAN_LUIGI.md, Argon
   review, Flash for bounded chunks, report val numbers. Only the orchestrator commits (`git add <folder>`), never
   `.github/CODEOWNERS` to main.
4. Full pipeline: `matching/run.py --phase P` (cascade) → `integration/run.py --phase P` (or `integration/pipeline.sh`).
5. Uploads: the human runs `integration/submit.py` in their own terminal (`read -rs HK_TOKEN && export HK_TOKEN`).
   Never put the token in chat or files. LIVE #1 used 1 of 20 slots today.

## Deadlines (Madrid)
16:30 dry run on live (timed) · 17:30 code freeze · 18:00 FINAL data → run → validate → human submits FINAL once
(`--final-i-am-sure`) · 19:45 project (repo, description, video).

## Open items for the human
- The HK token was pasted in chat once: ask the organisers to reset it before FINAL.
- Nikita's GitHub login (main's `ingestion/` is still locked); `docs/EDA.md` + `scripts/eda.py` not committed.
