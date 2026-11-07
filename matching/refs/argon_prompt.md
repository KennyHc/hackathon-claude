You are reviewing the plan of coordinator K2 (refs) in a bank-reconciliation hackathon (repo in cwd). Read
docs/POLICIES.md, docs/PLAN_LUIGI.md, docs/FORMAT.md and matching/refs/PLAN.md. Score = 100*(0.5 alloc F1 + 0.3 txn
accuracy + 0.2 exception macro-F1). ~80 minutes of build time remain for a v1.
Critique matching/refs/PLAN.md in <= 400 words: (1) the 3 biggest risks to precision (false claims that steal lines
from the later party/subset-sum stage) and how to gate them; (2) ordering of the amount-resolution rules and
ambiguities (e.g. discount vs withholding vs bank charges vs partial); (3) what to cut for v1 to hit the time box;
(4) any POLICIES rule the plan gets wrong (amount convention, credit notes, instalments, retention, replaced invoices,
supplier refunds). Be concrete. Do not write code.
