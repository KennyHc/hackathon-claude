# Rules for this job (read first, follow exactly)

You are a worker on a bank-reconciliation hackathon repo (see CLAUDE.md, docs/POLICIES.md, docs/DATA_DICTIONARY.md,
docs/FORMAT.md, docs/CONTRACTS.md, docs/PLAN_LUIGI.md). Your working directory is a copy of the repo.
HARD LIMITS
- Create/edit files ONLY under the paths listed in "YOU OWN" below. data/ is READ-ONLY. Do not delete files.
- Shell: the ONLY allowed commands, typed exactly like this, from the working-directory root, with no `cd`, pipes, `&&`
  or redirects:  .venv/bin/python <script.py> [args]  |  .venv/bin/python -c "<code>"  |  ls / cat / head / grep / wc <path>
  Anything else is denied and ENDS your run. No git, no pip.
- Python 3.12 with pandas 2.3, numpy, scipy, scikit-learn, pyarrow, openpyxl, pypdf. Read CSVs with
  pd.read_csv(path, dtype=str, keep_default_na=False).
- Shared code you may import (don't edit): integration/lib/fx.py, integration/lib/ledger.py,
  classification/exceptions.py, integration/scoring.py.
- No leakage: for phase val learn only from data/train rows booked < 2025-01-01; for live/final from all of train.
  data/val/golden_allocations.csv.gz is for evaluation scripts only.
- Finish by writing a short REPORT (what you built, how to run, measured numbers, known gaps) to the report path in your
  brief, then stop.
