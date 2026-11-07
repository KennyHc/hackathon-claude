# Classifier zoo + ensemble: plan

Branch `luigi-classifier` (worktree `Close_the_tabs-cls`), cut from tag `best-live-95.62` (val 96.41, hold 96.50,
LIVE 95.62). Only `classification/` changes; matching and integration stay as they are.

## 1. Where the classifier can still earn points

The 22-class LR (`categories.py`) already categorises golden no-item lines 99.98-99.99 % right. Integration uses three
outputs:

| Use | Rule | Score part |
|---|---|---|
| veto a matcher claim | p_items < 0.2 and confidence >= 0.9 | allocation F1 (50 %): a false drop loses good pairs |
| category row for unclaimed lines | argmax no-item class | txn accuracy (30 %): already ~100 % on true no-item lines |
| UNIDENTIFIED flag | category = UNIDENTIFIED and p_items < 0.97 | 1/10 of exception macro-F1 (20 %) |

So the lever is **p_items**, especially its ranking among *unclaimed receipts whose best no-item class is
UNIDENTIFIED*: true UNIDENTIFIED lines vs item lines the matchers missed (a miss is wrong either way; flagging it
UNIDENTIFIED adds a false positive). Upper bound: UNIDENTIFIED F1 0.956 -> 1.0 is ~0.09 points; the veto drops ~19
true item lines on val (tiny). Expected realistic gain: +0.01 to +0.04 points. Anything smaller than that is noise
in how members are trained, so the selection rule (beat val AND hold on the full pipeline) matters.

Design consequence: zoo members are **binary P(ITEM)** models. The category distribution among no-item classes stays
the LR's conditional one: P(cat) = (1 - p_items) * P_lr(cat | not ITEM). With p_items = P_lr(ITEM) this reproduces
the LR exactly, so the reference is a special case. (A 22-class GBM would mostly re-learn what the LR already gets
99.99 % right, at 22x the trees; dropped unless time allows.)

## 2. Model zoo (`classification/zoo.py`)

Shared: the exact `categories._features` sparse binary features, computed once for all 1.49M TRAIN lines into a cache
(global vocab df >= 2; each fit re-filters columns by df on its own training rows, so no future vocab leaks into
which columns exist... see 4). Same rows and weights as the LR: ITEM subsampled to 300k (rng 0) and up-weighted to its
true frequency, all no-item lines kept.

| id | model | features |
|---|---|---|
| lr | current 22-class LR, C=4 (reference) | sparse binary, df >= 3 |
| lgb | LightGBM binary, 400 trees, lr 0.05, 63 leaves, colsample 0.5, bagging 0.8 | sparse binary df >= 20 + dense |
| xgb | XGBoost binary, hist, 400 trees, depth 8, eta 0.08, colsample 0.5 | sparse binary df >= 20 + dense |
| chr | binary LR on char_wb 2-4-gram TF-IDF (text | remittance | counterparty, digits -> 0, 200k feats) + sparse binary | |

Dense extras: log10 |amount|, cents==0, multiple of 100, day of month, weekday, sign; and for the counterparty
account and the normalised counterparty name: has-key, log(1 + prior lines), smoothed prior ITEM / UNIDENTIFIED /
other-category rates. **History is strictly prior:** a line sees labelled lines dated before
min(its own date, history end), where history end = the fit's cutoff. Training rows therefore see only their own past
(time-ordered, no self-leak; this is the out-of-fold equivalent for an expanding window); prediction rows see history
up to the cutoff.

## 3. Ensemble

* Stacker = binary LR (C=1) on the members' logits [lr, lgb, xgb, chr], plus single-member Platt calibration and a
  plain probability average as simpler alternatives.
* Fit recipe for a phase with cutoff C: members trained on TRAIN < C - 6 months -> predict TRAIN [C - 6 m, C) ->
  fit stacker there -> members refit on < C -> stacker applied to the phase lines.
  * val: members < 2024-07-01, stacker on 2024-07..12, refit < 2025-01-01.
  * hold: same a year earlier (< 2023-07-01, 2023-07..12, refit < 2024-01-01).
  * live/final: stacker on TRAIN 2025-01..06 with the val-cutoff members, members refit on all of TRAIN.
* Nothing is fit on the val or hold windows. Isotonic calibration on the same slice is a candidate only if the
  Platt-style stacker is visibly off near 0.2 / 0.97 (checked on the slice, not on val).
* Output composed back into the 22-class matrix, then the unchanged post-processing (keyword rules, reversal pairs).

## 4. Leakage controls

* Every fitted object (LR, members, stacker, history tables) is keyed by its cutoff and only touches TRAIN lines dated
  before it; the history tables are built from lines < cutoff only.
* The global column list of the feature cache is built on all of TRAIN, but it only defines *which strings exist*;
  each fit keeps columns by their df on its own training rows (same rule as the LR's MIN_DF), so a column that only
  appears after the cutoff has no training signal and is dropped.
* Hold is never used to choose anything; it is a check only. Thresholds stay as they are (P_MIN 0.2, CONF 0.9,
  UNID_P 0.97); any threshold change is only proposed, with val and hold evidence.

## 5. Evaluation (`classification/eval_cls.py`)

Matching output is fixed, so each labels candidate is re-merged with the unchanged `integration.run.merge` and scored
with the official scorer in-process (~30 s), plus: p_items log loss / AUC vs "has item rows", category accuracy on
golden no-item lines, UNIDENTIFIED P/R/F1 as the scorer counts them, veto lines and false drops.
**Selection:** a candidate replaces the LR only if its full score beats the baseline on val AND hold; ties go to the
simpler model.

## 6. Runtime and FINAL safety

* Budget: train-feature cache ~5 min once; each member fit 1-5 min (n_jobs 4); 5 cutoffs per member
  (2023-07, 2024-01, 2024-07, 2025-01, all). Labelling a phase: features ~40 s + predict < 30 s.
* `categories.label` keeps working alone; the ensemble path loads pre-trained models only and falls back to the
  plain LR on any load/predict error (logged), and `run.py` still falls back to the baseline after that.
* live/final models (cutoff 9999-12-31) are pre-trained before 16:45 so FINAL only predicts. Unseen tokens /
  entities (ZA01, ZAR) are simply absent from the vocab; history features are 0 for unseen counterparties.
* Shared Mac: every heavy run under `tools/locked.sh val`, n_jobs <= 4, no parallel training.

## 7. Timeline
15:00 baseline queued (lock) + zoo code + smoke test (done on a 5 % sample) · 16:00 interim: baseline + first members
on val · 16:45 choice + live/final pre-trained · 17:30 freeze.

## 8. Changes after the Argon review (`classification/ARGON_REVIEW_CLS.md`)
* **compose() precision bug fixed:** LR probabilities are now computed in float64 and the no-item conditional is
  renormalised from the no-item columns themselves (no more 1/(1 - P(ITEM)) blow-up when P(ITEM) rounds to 1).
* **Operating point vs fixed thresholds:** UNID_P 0.97 was tuned on the LR's (over-confident) scale. Every stacker
  candidate also gets a `_qm` twin: a monotone quantile map, fit on the stacker slice, that sends the stacker's
  distribution onto the LR's. It keeps the LR's operating points at 0.2 / 0.97 and changes only the ranking.
* **Decision population:** new candidates `pop_lr+<m>`: a stacker fit only on slice lines that are receipts whose
  best no-item LR class is UNIDENTIFIED, applied only to that population (LR p_items elsewhere, so the veto is
  untouched). New member `lgbr`: LightGBM trained only on ITEM / UNIDENTIFIED receipts.
* **History covariate shift:** the prior-count feature is capped at 30 (early TRAIN rows have little history).
* **ZAR / currency:** the dense log-amount uses amount in EUR (ingestion's fx_eur), raw amount if no rate.
* **Compute:** xgb dropped from the first pass (redundant with lgb); val first, hold only for what survives val,
  live/final only for the winner. FINAL inference loads pre-saved history tables (`hist_v1_<cutoff>.pkl`) and
  pre-trained members/stacker; the fallback is the plain LR (`categories.label`), not run.py's dummy baseline.

## 9. Results (full pipeline re-merge, fixed matching output, official scorer; 4-decimal score)

| candidate | val | hold | UNID F1 val / hold | veto lines (false) val / hold |
|---|---|---|---|---|
| lr (baseline, = tag best-live-95.62) | 96.4141 | 96.4991 | 0.971 / 0.972 | 23 (18) / 19 (10) |
| **lr+lgb, quantile-mapped (chosen)** | **96.4419** | **96.5276** | 0.980 / 0.982 | 10 (5) / 11 (2) |
| lr+lgb+xgb+chr, quantile-mapped | 96.4434 | 96.5276 | 0.980 / 0.982 | 10 (5) / 10 (0) |
| pop_lr+lgbr, quantile-mapped | 96.4425 | 96.5275 | 0.981 / 0.982 | 10 (5) / 11 (2) |
| avg(lr, lgb, xgb, chr) | 96.4395 | 96.5220 | | |
| lgb raw | 96.4362 | 96.5174 | | |
| lr+lgb (no quantile map) | 96.4298 | 96.5119 | | |

Chosen by val among the tied top group (within 0.0015 of the best), simplest wins: `CONFIG = {"members": ["lgb"],
"kind": "full_qm"}` in `classification/ensemble.py`. Hold confirmed (+0.0285), it was not used to choose.
Thresholds: veto off (P_MIN 0) is worse with the new model (val 96.4376, hold 96.4931); UNID_P 0.95 / 0.99 do not
beat 0.97 on both windows. No integration change proposed.

Added at the human's request (same features, sigmoid-calibrated by 3-fold CV inside the training history):
| candidate | val | hold |
|---|---|---|
| svm raw (LinearSVC C=0.1, calibrated) | 96.4029 | 96.4915 |
| cnb raw (ComplementNB, calibrated) | 95.8016 | 95.8424 |
| lr+svm, quantile-mapped | 96.4298 | 96.5101 |
| lr+cnb, quantile-mapped | 96.4151 | 96.5014 |
| stack lr+lgb+xgb+chr+svm+cnb, quantile-mapped | 96.4443 | 96.5271 |
Neither changes the choice (the 6-member stack is +0.0024 on val, -0.0005 on hold vs the chosen lr+lgb_qm).
