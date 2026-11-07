# Classifier insights: model selection, the ensemble, and how it fits the pipeline

Branch `luigi-final` (also on `main`). Owner: classification (Luigi). Everything below was measured with the
official scorer on the full pipeline: fixed matching output, re-merged by the unchanged `integration/run.py`.

## TL;DR

* The classifier gives every bank line `p_items` (the chance it settles open items) plus one of 21 categories.
  The categories were already ~99.99 % right. **The only lever left was `p_items`.**
* We kept Kenny's 22-class logistic regression (LR) and added **one LightGBM binary model for P(ITEM)**, stacked
  with the LR. The result is **quantile-mapped back onto the LR's probability scale**, so the tuned thresholds
  (0.2 / 0.97) still mean the same thing.
* Gain: **+0.03 points** on val and on the untouched hold window, **+0.03 to +0.04 on LIVE**. That's small, but it
  holds on every window and it adds to every matching improvement.
* LIVE progression: 95.62 → **95.65** with the classifier → Kenny v5 96.34 → **96.38** v5 + classifier →
  **96.46** v5 + classifier + Kenny's `sd_noadv` ranker (current best, what FINAL runs).

## 1. Where the points were

Integration uses the classifier in three places:

| Use | Rule in `integration/run.py` | Score part it touches |
|---|---|---|
| Veto a matcher claim | `p_items < 0.2` and category confidence ≥ 0.9 | allocation F1 (50 %): a wrong veto drops good pairs |
| Category for unclaimed lines | argmax over the no-item classes | transaction accuracy (30 %): already ~100 % |
| UNIDENTIFIED flag | category = UNIDENTIFIED and `p_items < 0.97` | 1/10 of exception macro-F1 (20 %) |

So the gain comes from **ranking `p_items` well** on one hard population: unclaimed incoming payments whose best
no-item class is UNIDENTIFIED. These are either truly unidentified, or item payments the matchers missed.
The ceiling was small: UNIDENTIFIED F1 going from 0.956 to 1.0 is worth about 0.09 points. That set the bar:
**anything below ~0.01 is training noise**, so a candidate had to win on val *and* hold.

**Design consequence:** the zoo members are **binary P(ITEM)** models. The category split stays the LR's:
P(cat) = (1 − p_items) · P_lr(cat | not ITEM). With p_items = P_lr(ITEM) this gives back the LR exactly, so the
baseline is a special case and nothing else in the pipeline changes. A 22-class GBM would have re-learned what the
LR already gets 99.99 % right, at 22× the trees.

## 2. The model zoo

All members share the LR's exact sparse token features (`categories._features`), computed once for the 1.49 M
TRAIN lines. They also use the same rows and weights: ITEM subsampled to 300 k and re-weighted to its true
frequency, all no-item lines kept.

| id | Model | Extra features |
|---|---|---|
| `lr` | Kenny's 22-class LR, C=4 (reference) | none |
| `lgb` | LightGBM binary, 400 trees, lr 0.05, 63 leaves, colsample 0.5, bagging 0.8 | dense (below) |
| `lgbr` | LightGBM trained only on ITEM vs UNIDENTIFIED receipts | dense |
| `xgb` | XGBoost binary, hist, 400 trees, depth 8, eta 0.08 | dense |
| `chr` | Binary LR on character 2–4-gram TF-IDF of text, remittance info and counterparty (digits → 0) | none |
| `svm` | LinearSVC C=0.1, sigmoid-calibrated by 3-fold CV | none |
| `cnb` | ComplementNB, calibrated the same way | none |

**Dense extras (trees only):**
* Amount: log of the amount in EUR (`fx_eur`, so ZAR isn't 20× bigger), round cents, multiple of 100, day of
  month, weekday, sign.
* Counterparty history, for both the counterparty account and the normalised name: whether a key exists,
  log(1 + prior lines) capped at 30, and smoothed prior rates of ITEM, UNIDENTIFIED and other categories.

**History is strictly prior:** a line only sees labelled lines dated before min(its own date, the fit's cutoff),
with no same-day matches. Training rows see only their own past, so this works like out-of-fold for an expanding
window.

## 3. The ensemble recipe

* **Stacker:** a binary LR (C=1) on the members' logits. Simpler alternatives were also run: a plain probability
  average and single-member Platt scaling.
* **Fit recipe for a cutoff C:**
  1. Train the members on TRAIN before C − 6 months.
  2. Predict the 6 months [C − 6 m, C) and fit the stacker there.
  3. Refit the members on everything before C.
  4. Apply the stacker to the phase.
* **Windows:**
  * val: members before 2024-07-01, stacker on 2024-07..12, refit before 2025-01-01.
  * hold: the same, one year earlier.
  * live/final: stacker on 2025-01..06, members refit on all of TRAIN.
* **Quantile mapping (`_qm`):** UNID_P = 0.97 was tuned on the LR's over-confident scale. A calibrated stacker pulls
  the upper tail down, which silently moves the operating point and floods UNIDENTIFIED with false positives. So a
  monotone map, fit on the stacker slice, sends the stacker's distribution onto the LR's. **The thresholds keep
  their meaning; only the ranking changes.** This was the single most important trick: without it,
  `lr+lgb` scores 96.4298 on val, with it 96.4419.
* **Population variant (`pop_`):** a stacker fit and applied only to the UNIDENTIFIED-gate population (LR
  elsewhere, veto untouched). It tied the full version, so the simpler full version won.

## 4. Results (4-decimal official score)

| Candidate | val | hold | UNID F1 val / hold | Vetoes (false) val / hold |
|---|---|---|---|---|
| `lr` (baseline = tag best-live-95.62) | 96.4141 | 96.4991 | 0.971 / 0.972 | 23 (18) / 19 (10) |
| **`lr+lgb`, quantile-mapped (chosen)** | **96.4419** | **96.5276** | 0.980 / 0.982 | 10 (5) / 11 (2) |
| `lr+lgb+xgb+chr`, quantile-mapped | 96.4434 | 96.5276 | 0.980 / 0.982 | 10 (5) / 10 (0) |
| `pop_lr+lgbr`, quantile-mapped | 96.4425 | 96.5275 | 0.981 / 0.982 | 10 (5) / 11 (2) |
| 6-member stack (+svm +cnb), quantile-mapped | 96.4443 | 96.5271 | | |
| avg(lr, lgb, xgb, chr) | 96.4395 | 96.5220 | | |
| `lgb` alone | 96.4362 | 96.5174 | | |
| `lr+lgb`, not quantile-mapped | 96.4298 | 96.5119 | | |
| `lr+svm`, quantile-mapped | 96.4298 | 96.5101 | | |
| `lr+cnb`, quantile-mapped | 96.4151 | 96.5014 | | |
| `svm` alone (calibrated) | 96.4029 | 96.4915 | | |
| `cnb` alone (calibrated) | 95.8016 | 95.8424 | | |

**What it bought:** UNIDENTIFIED F1 went from 0.971 to 0.980, and wrong vetoes dropped from 18 to 5 on val
(10 to 2 on hold).

## 5. Why `lr+lgb` and not the bigger stacks

* **Selection rule, written down before seeing results:** beat the baseline on val AND hold; among candidates
  within 0.0015 of the best on val, the simplest wins. Hold is a check, never used to choose.
* The 6-member stack is +0.0024 on val but −0.0005 on hold against `lr+lgb`: well inside the noise. It costs
  6 models to train, ship and keep alive on FINAL, plus 6 places where something can go wrong.
* `xgb` is nearly redundant with `lgb` (same features, same family). `chr`, `svm` and `cnb` see the same tokens the
  LR already sees, so they add little new signal.
* **Naive Bayes is clearly worse** (−0.6): its independence assumption double-counts correlated tokens. A linear
  **SVM alone is slightly worse than the LR**, because its calibrated probabilities are coarser at the top.
* **Lesson:** at this level, **calibration and the operating point mattered more than model diversity**.
  The quantile map gave more than adding four more models.

## 6. Thresholds checked (no change made)

* Veto off (P_MIN 0) with the new model: val 96.4376, hold 96.4931, both worse. Keep the veto.
* UNID_P 0.95 and 0.99: neither beats 0.97 on both windows. Kept at 0.97.

## 7. Leakage controls

* Every fitted object (LR, members, stacker, history tables) is keyed by its cutoff and only sees TRAIN lines
  before it: val before 2025-01-01, hold before 2024-01-01, live/final all of TRAIN.
* The global vocabulary only decides which strings exist. Each fit keeps columns by their frequency on its own
  training rows, so a token that only appears after the cutoff gets no weight.
* Nothing was ever fit on the val or hold windows. An independent review (Argon) audited the recipe and found
  **no target leakage**. It also found the bugs below.

## 8. Bugs the review caught (fixed before any number above)

* **float32 cancellation in `compose()`:** when P_lr(ITEM) rounded to 1.0, the no-item renormalisation divided by
  1e-12. Category confidences then blew up past 1 (up to 50,000×) and falsely triggered the 0.9 veto. Fixed with
  float64 and renormalising from the no-item columns themselves.
* **History shift:** early TRAIN rows have no history, while test rows have years of it. The prior-count feature is
  now capped at 30.
* **Currency-blind amounts:** raw log-amount made ZAR look 20× bigger to the trees. It now uses the EUR amount.
* **FINAL inference loaded the 1.49 M-row training cache.** The history tables are now pre-saved
  (`hist_v1_<cutoff>.pkl`).

## 9. Production safety

* `classification/ensemble.py` loads **pre-trained models only**. On any error it falls back to the plain LR
  (`categories.label`), and logs it. It never falls back to run.py's dummy baseline, which would tank the score.
* The models are in git under `classification/pretrained/`, force-added because `.gitignore` excludes them; each is
  under 5 MB, about 12 MB in total:
  * the LR (`model_v1_9999-12-31.npz`, 4.3 MB)
  * the LightGBM (3.5 MB)
  * the stacker and quantile map (2.5 MB)
  * the history tables (2.1 MB)

  `classification/restore_models.sh` puts them back in place.
* Unseen tokens and new entities (ZA01 / ZAR on FINAL) simply aren't in the vocabulary, and history features are 0
  for new counterparties. The model then leans on the LR tokens and the amount features.
* `requirements.txt` pins `lightgbm==4.7.0` and `xgboost==3.4.1`. On a Mac they need `brew install libomp`.
* Runs need `PY=.venv/bin/python`. Without it the LightGBM import fails, the LR fallback kicks in, and about 0.03
  points are lost quietly.

## 10. How it combined with the rest of the pipeline

| Version | val | hold | LIVE |
|---|---|---|---|
| Kenny best-live-95.62 (LR only) | 96.41 | 96.50 | 95.62 |
| + classifier | 96.44 | 96.53 | 95.65 |
| Kenny v5 (cn_bundles etc.) | 96.73 | 96.84 | 96.34 |
| v5 + classifier | 96.76 | | 96.38 |
| Kenny v5 + `sd_noadv` ranker | 96.79 | 96.91* | 96.42 |
| **v5 + classifier + ranker (FINAL build)** | **96.82** | | **96.46** |

\* The ranker's hold model was trained on val's 2025 labels, so this hold number doesn't follow our time rule.
The val number is clean.

The classifier and the ranker work on different lines (the ranker claims unclaimed bulk payments; the classifier
decides vetoes and UNIDENTIFIED), so **their gains add up almost exactly**: +0.04 and +0.08 on LIVE give +0.12.

## 11. Things tried around the classifier that we dropped

* **Voucher layout fix (B7):** the claimed loss doesn't exist. LIVE loses no Voucher row, the fix changes nothing on
  LIVE, and the FINAL exposure is well under 0.05 points. Not merged.
* **n-to-m matching:** explored, then handed to a teammate. Its worktree and branch were deleted.

## 12. FINAL day: drift checks and the ZA01 veto problem

**The run** (`PY=.venv/bin/python integration/final.sh`, 18:10 → 18:23, exit 0):
* 220,256 bank lines from 2026-01-02 to 2026-06-30, and 1.16 M open items.
* Every matching stage produced lines.
* The classifier ran as the LR + LightGBM ensemble, with no fallback.
* The ranker added 249 lines.
* Validation: 220,256 of 220,256 lines.

**Drift checks** (`drift.py` and `drift_inputs.py` against LIVE, plus manual checks on the raw files). Each flag was
traced to specific lines:

| Flag | Cause | Verdict |
|---|---|---|
| Moncayo n43 gone | The bank moved fully to camt053 (half-way there on LIVE), which carries more fields | fine |
| PAYROLL +162 % | 900 ZA01 `NOMINA` payroll lines; other entities are unchanged | fine |
| IT01 money matched 0.60 → 0.34 | The share of lines matched is unchanged (0.924). Three transfers without invoices weigh in: €96.5 M intercompany and 2 × €9.3 M internal, labelled correctly | fine |
| ZA01 everywhere | New entity: 4,339 lines, MT940, every detail inside the text field, only 35 advices | **hid a real problem (below)** |

Exchange rates cover every currency up to 2026-06-30, ZAR included. All 22,041 PDFs gave text. None of the advices
use South African number formats.

**The problem: the veto silently dropped good ZA01 matches.**
* The merge drops a matcher's claim when the classifier says p_items < 0.2 with category confidence ≥ 0.9.
* ZA01 never appears in TRAIN, so its `EFT CREDIT … INVOICE KWI…` texts got p_items ≈ 0.03.
* The veto removed **116 ZA01 receipts that matching had claimed**: 85 by `refs:exact`, 64 of them with the exact
  invoice amount and the right payer. They came out as UNIDENTIFIED.
* Outside ZA01 the veto fired on 29 lines, the usual number.

**How we found it:**
* The drift report said "ZA01: 52 % of receipts flagged UNIDENTIFIED". That number turned out to be measured among
  the receipts whose best category is UNIDENTIFIED; over all 2,344 ZA01 receipts the rate was 8.2 %.
* Looking directly at those 192 UNIDENTIFIED receipts showed that 91 cite an invoice number that exists.
* All 64 exact-amount cases had been claimed by matching and dropped by the merge.

**The fix:** `NO_VETO_ENTITIES=ZA01` in `integration/run.py` keeps matcher claims for entities the classifier
never saw.
* Re-ran only the merge, on the same matching and classifier outputs.
* 122 lines changed, all ZA01; ZA01 UNIDENTIFIED went from 192 to 76.
* Validation passed.
* It can't be tested on val or LIVE, which have no ZA01 receipts. Expected gain: about +0.02 to +0.04.

Files in `integration/out/final/`:
* `submission_v2_za_noveto.csv.gz`, the recommended one
* `submission_v1_ranker.csv.gz`, without the fix

**Lessons:**
* **A rule that lets a learned model override evidence must check that the model has seen this kind of data.** The
  veto was tuned on entities with years of history; on an unseen entity, a confident-but-wrong classifier beats an
  exact invoice-number match. The general fix is to skip the veto for entities, banks or formats absent from
  training, or when the matcher's evidence is exact (`refs:exact`).
* **A model's confidence doesn't mean much outside its training data.** A 0.98 category confidence on ZA01 was
  confidence about text it had never seen.
* **Drift scripts point; they don't conclude.** Every flag here was explained only by opening the lines behind it,
  and the dangerous one was hidden behind a ratio that looked normal. "In the same range as other entities" is not
  a check.
* **The new entity is the riskiest place in a FINAL.** Next time, check the new entities before submitting: how
  many receipts each piece of the pipeline matched, and how many were dropped between matching and the merge.

## 13. What we'd do with more time

* **Learn the thresholds jointly with the ranking** (one quantile-mapped operating point per entity) instead of
  global 0.2 / 0.97. Watch for overfitting: there are few UNIDENTIFIED lines per entity.
* **Counterparty history across entities** (the same payer in ES01 and PT01), which would help new entities like
  ZA01 that start with no history.
* **Veto only with in-distribution evidence:** skip it for unseen entities, banks or formats, and never against `refs:exact`.
* **Feed the classifier's `p_items` into the matchers** as a prior, for example to skip subset-sum on lines below
  0.1. Today it is only used after matching.
