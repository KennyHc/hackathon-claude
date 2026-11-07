Review the plan in matching/batch/PLAN.md (coordinator K1 "batch" of a bank-reconciliation hackathon; context:
CLAUDE.md, docs/PLAN_LUIGI.md (matcher interface), docs/POLICIES.md, docs/DATA_DICTIONARY.md, docs/EDA.md sections 5-6,
docs/FORMAT.md). Data samples: data/val/{payment_runs,direct_debit_batches,card_settlements,cheque_deposits,
remittance_advices}/ and data/val/bank_transactions.csv.gz. About 3 hours of build time remain; FINAL data arrives 18:00.
Be concrete and brief (<= 400 words): (1) the 5 biggest risks to precision/recall in the linking keys proposed, with the
concrete fix; (2) anything mis-ordered by points/hour; (3) traps in amount conventions (FX, credit notes netted,
SHORT_DISPUTE, RETURNED, confirming per maturity date, aggregated runs, card gross vs net); (4) live/final drift that
would break this (camt migration of Banco del Moncayo 2025-10-01, remittance layout change Sept 2025, ZA01/KWI).
