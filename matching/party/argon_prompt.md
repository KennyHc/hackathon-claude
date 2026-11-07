Review the plan in matching/party/PLAN.md for stage 3 ("party") of a bank-reconciliation matcher.
Context: docs/PLAN_LUIGI.md (interface, cascade), docs/POLICIES.md (labels follow it exactly, esp. §2, §3, §5),
docs/DATA_DICTIONARY.md, docs/EDA.md, integration/lib/ledger.py. Score = 100*(0.5 allocation F1 + 0.3 txn accuracy + 0.2 exception macro-F1).
We have ~80 minutes to a working v1, must run on ~200k lines in < 10 min.
Give a short, critical review (max 40 lines): (1) the biggest risks to precision/recall in the party resolver
(alias learning from train golden, truncated names, renamed customers, parents) and the subset-sum (ambiguity, blow-up),
(2) what to cut or reorder for points/hour, (3) concrete pitfalls in amount/exception conventions for the segments listed,
(4) anything about leakage for val (train < 2025-01-01 only). Be specific, no generic advice.
