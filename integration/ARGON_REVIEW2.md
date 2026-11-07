* **1. Leakage Deferred (#5) & Partial `masters_final` Bug (Highest Impact)**
  * *Wrong assumption*: Treating cross-phase item/master leakage as "only if cheap" in [`PLAN.md`](file:///Users/luigizheng/Desktop/Close_the_tabs/integration/PLAN.md#L41) risks a mirage 90.00 `val` score that collapses on LIVE/FINAL.
  * *FINAL risk*: [`_load_masters`](file:///Users/luigizheng/Desktop/Close_the_tabs/classification/exceptions.py#L38-L84) takes one `masters_dir` without per-file fallback. If `masters_final` is partial, missing files (e.g., `payment_terms.csv`) silently load empty, zeroing `SHORT_DISCOUNT`.

* **2. Learned `SHORT_BANK_CHARGES` Table (#2) — Policy Violation, Leakage & FINAL Drift**
  * *Wrong assumption*: [`POLICIES.md` §3](file:///Users/luigizheng/Desktop/Close_the_tabs/docs/POLICIES.md#L46) defines `SHORT_BANK_CHARGES` deterministically (non-SEPA, ≤60 EUR, ≤2%), not by customer propensity. Current errors come from [`EURO_SEPA_COUNTRIES`](file:///Users/luigizheng/Desktop/Close_the_tabs/classification/exceptions.py#L32) having only 11 countries and late discounts falling through to SBC in [`_explained_shortfall`](file:///Users/luigizheng/Desktop/Close_the_tabs/classification/exceptions.py#L461-L493) and [`assign_exceptions`](file:///Users/luigizheng/Desktop/Close_the_tabs/classification/exceptions.py#L423-L440).
  * *Leakage & FINAL drift*: Train customer/country tables overfit known entities and fail in FINAL ([`POLICIES.md` §6](file:///Users/luigizheng/Desktop/Close_the_tabs/docs/POLICIES.md#L117-L119)), where new subsidiary `ZA01` and new/renamed customers have zero history—causing [`normalize_amounts`](file:///Users/luigizheng/Desktop/Close_the_tabs/classification/exceptions.py#L496-L558) to flip valid SBCs into `PARTIAL` with wrong amounts.

* **3. New-Entity `UNIDENTIFIED` Rule (#3) — FINAL Drift & Leaky Validation**
  * *Wrong assumption*: Flipping unclaimed receipts with invoice refs or known payers to `CUSTOMER_ADVANCE` in [`run.py`](file:///Users/luigizheng/Desktop/Close_the_tabs/integration/run.py#L150-L159) conflates B-matcher misses with proformas ([`POLICIES.md` §4](file:///Users/luigizheng/Desktop/Close_the_tabs/docs/POLICIES.md#L69)), hurting `UNIDENTIFIED` recall.
  * *FINAL drift & leakage*: `ZA01` uses unseen `"KWI…"` refs and renamed parties. Leave-one-entity-out on `val` leaks because C (`p_items`) and matchers were trained on all [`train_entities`](file:///Users/luigizheng/Desktop/Close_the_tabs/integration/run.py#L65-L67).

* **4. Ordering & Guard Bugs (#1, #4)**
  * [`drop_wild_overpay`](file:///Users/luigizheng/Desktop/Close_the_tabs/integration/run.py#L73-L84) (`--overpay-max 0.1`) drops true `OVERPAYMENT` >10% before [`assign_exceptions`](file:///Users/luigizheng/Desktop/Close_the_tabs/classification/exceptions.py#L328-L331).
  * In [`exceptions_from_c`](file:///Users/luigizheng/Desktop/Close_the_tabs/integration/run.py#L51-L60), [`normalize_amounts`](file:///Users/luigizheng/Desktop/Close_the_tabs/classification/exceptions.py#L496-L558) mutates `alloc["amount"]` *before* [`KEEP_EMPTY_METHODS`](file:///Users/luigizheng/Desktop/Close_the_tabs/integration/run.py#L131-L135) / `--trust-methods`, corrupting [`Ledger`](file:///Users/luigizheng/Desktop/Close_the_tabs/classification/exceptions.py#L151) balances.

* **5. Cut List (2h to 16:30 dry run)**
  * **Cut**: #2 (`--sbc-model`), #3 (`unid_evidence` stub + LOEO), and broad #1 (`--trust-methods`) tuning.
  * **Do first**: #5 (leakage audit + per-file `masters_path`) and #4 (full SEPA list, block late-discount-to-SBC fallthrough, fix overpay/amount ordering).
