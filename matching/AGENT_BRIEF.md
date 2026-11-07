# Shared brief for matching sub-agents (read fully before coding)

Goal: maximise the hackathon score = 100 x (0.5 allocation F1 + 0.3 per-line accuracy + 0.2 exception macro-F1) on **unseen**
data (LIVE Jul-Dec 2025, FINAL Jan-Jun 2026). We win on generalisation, not on TRAIN fit. A wrong pair costs more than a
missing one: **claim a bank line only when the evidence is unambiguous.**

## Repo and tooling
- Repo root `C:\Users\dell\Documents\GitHub\aa`. Run everything from the root: `PYTHONUTF8=1 .venv/Scripts/python.exe ...`
  (shared env: pandas 2.3.3, numpy, pyarrow, scikit-learn, scipy, openpyxl, pypdf; no other installs).
- Start scripts with `import sys; sys.path.insert(0, '.')`, then `from matching.lib import data, evalutil`.
- Read first: `docs/POLICIES.md` (labels follow it exactly: s2 amounts, s3 exceptions, s5 how to pick items),
  `docs/DATA_DICTIONARY.md`, `docs/CONTRACTS.md` section B, `matching/lib/data.py`, `matching/lib/evalutil.py`,
  `atlas/` (scenario catalogue: how each segment looks).
- Phases: `sample` (June 2025, tiny: use for debugging only), `val` (**held-out judge**, 2025-01..06, LIVE-shaped: same files as
  live incl. `open_items_at_start`), `train` (2021-07..2025-06, heavy), `live` (no golden).

## Interface every matcher implements
File `matching/matchers/<name>.py`:
```python
def match(phase: str, claimed: set[str] | None = None) -> pd.DataFrame:
    """Return one row per (bank line, item): bank_txn_id, item_id, amount, exception, confidence, method.
    Never return lines in `claimed`. amount = cleared amount in ITEM currency, positive (POLICIES s2).
    exception = your best guess or None. confidence in [0,1], same value on every row of a line. method = rule name."""
```
Also write your result to `matching/out/<phase>/<name>.parquet` when run as `python -m matching.matchers.<name> --phase val`
and print `evalutil.report(...)` for it. `matching/run.py` (owned by the orchestrator) calls `match()` in priority order.

## Files you may write (and nothing else)
Your own `matching/matchers/<name>.py`, `matching/lib/<name>_*.py`, `matching/tests/<name>/*`. Do **not** edit
`matching/lib/data.py`, `matching/lib/evalutil.py`, `matching/run.py`, other agents' files, or anything outside `matching/`
(a hook blocks it and it breaks the team rules). Need a change elsewhere? Say so in your final report. Do not run git
commit/push, do not post to the board, never write or print `HK_TOKEN`.

## Anti-overfitting rules (these decide whether we win)
1. **Judge on `val`, develop elsewhere.** Iterate on `sample` and on slices of `train` from 2021-2024. Look at `val` numbers
   at milestones only (a handful of times), never tune a constant until val improves. Never read `val` golden except through
   `evalutil.report`.
2. **No leakage:** anything *learned* from labels (aliases, templates, weights) may use `data.golden("train")` rows booked
   **before 2025-01-01** only. For `live`/`final` the orchestrator will re-learn on all of train; so learning code must take a
   `cut` argument and be re-runnable.
3. **No hard-coded ids, names, dates, accounts or entity quirks.** Numeric thresholds come from POLICIES.md or are fitted on
   train<2025 with the fit code kept in the repo. Prefer structural rules (a run id in a reference field, a mandate id, a batch
   total) over fitted ones.
4. **Expect drift** (POLICIES s6): new bank formats (Moncayo camt.053 from 2025-10), new entity ZA01 (ZAR, "KWI..." numbers),
   renamed customers, new IBANs, a remittance layout change in Sept 2025, slower payments. A rule must degrade to "no claim",
   never to a crash or a forced guess. Wrap risky parsing in try/except per record and count failures.
5. **Precision gate:** every rule reports precision on val *and* on a train slice from a different year. If a rule is < 97%
   precise it must lower `confidence` (and say so) so the orchestrator can drop it below the gate.
6. Report **both** val and a second out-of-period check when cheap. A rule that only works on one month is a bug.

## Machine limits (shared by 5 agents at once: 15.7 GB RAM, 20 threads)
- Keep your python process under ~2 GB: `pd.read_parquet(path, columns=[...])`, categoricals, drop unused columns, delete
  big frames. Never load all of `data/train` golden or invoices at once unless you slice by date first. One heavy process at a
  time per agent, no multiprocessing pools, no background jobs left running when you finish.
- `data.*("val")` is cached in `matching/out/cache/val/*.parquet` (already built). Do not delete caches.

## Final report (keep it short, factual)
1. What each rule does (one line each) and which golden segments it targets.
2. Val metrics from `evalutil.report`: precision, recall, per-segment table; plus the second-period check.
3. What does not work yet and why (with a counted example), and drift risks.
4. Files written, public functions, anything needed from other pieces.
Do not claim a number you did not measure.

## Facts already verified by the lead (use as hints, re-check on val)
- Direct-debit batch credit: `batch_total`+account+date<=7d -> the batch (unique in train). Receipts: `remittance_info` is the
  exact document number for ~90%; the rest read `RECIBO nnnnnn` = last digits of the invoice number (same digits repeat across
  years, so disambiguate with amount, customer name and issue_date <= collection date; this resolved 100% of 71k receipts).
- Card settlement: bank amount = sum of `net`; one row per order at **gross**; order_number = invoice document number. 100% exact.
- Payment runs (AP): every payment-run bank line carries the run reference in `end_to_end_id` (camt, 19 chars `PR..-nnnn`),
  or in `ref1`/`ref2` truncated to 16 chars (N43/MT940: run id + `-0`); confirming lines carry `CF...` in the text/ref2. Use the
  run id plus the amount of the transfer inside the run. Document numbers are NOT unique across suppliers: resolve the item with
  supplier (beneficiary IBAN -> suppliers.csv covers only ~51% of rows) + amount; partials have amount != invoice total.
- Non-item templates (fees, cash pool...) are classification's job; do not claim those lines.
- Do not match on amount alone: look-alike amounts exist on purpose (`unidentified_lookalike`).
