"""Bulk customer payments WITHOUT a remittance advice (golden segment ar_short_dispute, the 17 % no other matcher
claims): a big customer pays many open invoices (minus credit notes, minus disputed amounts) in one transfer.

Learned ranker (logistic regression, payer-agnostic features) gives p(paid on this line) to every open item of the
resolved payer; we claim a SUBSET at full open amount with exception "" (never invent SHORT_DISPUTE amounts):

  candidate line : unclaimed incoming line, no FX; among the resolved payers keep those whose pool (unclaimed open AR
                   items, same currency, issued in the 365 days up to booking) has >= 8 items, cash >= 5 x median open
                   amount, and signed open total due in [booking-90d, booking+30d] >= cash; exactly one must remain
                   (several -> the unique IBAN-resolved one, else skip)
  claim rule     : items with p >= P_MIN in descending p (tie: due date, item_id), added while the signed running total
                   stays <= cash; claim the line only if that total >= COVER x cash. Items taken by an earlier claim
                   of this stage are skipped (lines processed in booking order).

Model: matching/models/sd_noadv_<window>.json (coefficients + feature list). val uses the model trained on hold
labels, hold the one trained on val labels, live/final the one trained on both. No model -> no rows.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from matching.common import ALLOC_COLS, ROOT, Ctx

NAME = "sd_noadv"
P_MIN = 0.60
COVER = 0.80
MIN_POOL = 8
CASH_X_MEDIAN = 5.0
LOOKBACK = 365
MODEL_FOR = {"val": "hold", "hold": "val"}       # cross-window; anything else -> "both"
DD_BINS = [-30, -15, 0, 15, 30, 60, 90]            # days booking - due (credit notes: booking - issue)
FEATURES = ([f"dd_bin{k}" for k in range(len(DD_BINS) + 1)] +
            ["issue_age_m", "is_cn", "amt_ratio", "rank_due", "cum_ratio", "cum_le1", "cited_full",
             "cited_suffix", "cited_prefix", "log_pool"])
CREDIT = {"ARC"}


def _no_fx(t) -> bool:
    oc = getattr(t, "orig_currency", None)
    return oc is None or (isinstance(oc, float) and np.isnan(oc)) or str(oc) in ("", "None", "nan") or str(oc) == str(t.currency)


def _tokens(t) -> set[str]:
    dr = getattr(t, "doc_refs", None)
    if dr is None:
        return set()
    try:
        return {str(x) for x in dr if x is not None and str(x)}
    except TypeError:
        return set()


def candidates(ctx: Ctx) -> pd.DataFrame:
    """One row per (candidate line, pool item) with FEATURES + cash/open amounts. Payer-agnostic."""
    from matching.parties import PartyResolver
    pr = PartyResolver.get(ctx)
    used = set(ctx.allocations.item_id) if getattr(ctx, "allocations", None) is not None and len(ctx.allocations) else set()
    op = ctx.open_items
    op = op[op.item_type.isin(["ARI", "ARC"]) & ~op.item_id.isin(used)]
    op = op[["item_id", "item_type", "party_id", "currency", "issue_date", "due_date", "open_amount",
             "document_number_norm"]]
    by_party = {k: v for k, v in op.groupby("party_id")}
    b = ctx.bank[(ctx.bank.amount > 0) & ~ctx.bank.bank_txn_id.isin(ctx.claimed)]
    out = []
    for t in b.sort_values(["booking_date", "bank_txn_id"]).itertuples(index=False):
        if not _no_fx(t):
            continue
        try:
            cands = pr.resolve(t)
        except Exception:
            continue
        if not cands:
            continue
        bd = pd.Timestamp(t.booking_date)
        cash = float(t.amount)
        ok = []
        for cid, (strength, src) in sorted(cands.items()):
            pool = by_party.get(cid)
            if pool is None:
                continue
            iss = pd.to_datetime(pool.issue_date, errors="coerce")
            pool = pool[(pool.currency == t.currency) & (iss <= bd) & (iss >= bd - pd.Timedelta(days=LOOKBACK))]
            if len(pool) < MIN_POOL or cash < CASH_X_MEDIAN * float(pool.open_amount.median()):
                continue
            due = pd.to_datetime(pool.due_date, errors="coerce").fillna(pd.to_datetime(pool.issue_date, errors="coerce"))
            rec = pool[(due >= bd - pd.Timedelta(days=90)) & (due <= bd + pd.Timedelta(days=30))]
            if float(np.where(rec.item_type.isin(CREDIT), -rec.open_amount, rec.open_amount).sum()) < cash:
                continue
            ok.append((cid, src, pool))
        if len(ok) > 1:          # several feasible payers: a unique IBAN hit decides, else skip
            ok = [o for o in ok if str(o[1]).startswith("iban")]
        if len(ok) != 1:
            continue
        cid, _, pool = ok[0]
        f = pool.copy()
        iss = pd.to_datetime(f.issue_date, errors="coerce")
        due = pd.to_datetime(f.due_date, errors="coerce").fillna(iss)
        f["is_cn"] = f.item_type.isin(CREDIT).astype(float)
        dd = (bd - due).dt.days.astype(float)
        binix = np.searchsorted(DD_BINS, dd.values, side="right")
        for k in range(len(DD_BINS) + 1):
            f[f"dd_bin{k}"] = (binix == k).astype(float)
        f["issue_age_m"] = ((bd - iss).dt.days / 30.0).clip(0, 12)
        f["amt_ratio"] = (f.open_amount / cash).clip(0, 2)
        f["_due"] = due
        f = f.sort_values(["_due", "item_id"])
        f["rank_due"] = np.arange(len(f)) / len(f)
        signed = np.where(f.is_cn > 0, -f.open_amount, f.open_amount)
        f["cum_ratio"] = (np.cumsum(signed) / cash).clip(-1, 3)
        f["cum_le1"] = (f.cum_ratio <= 1.0).astype(float)
        toks = _tokens(t)
        long_toks = [x for x in toks if len(x) >= 5]
        docs = f.document_number_norm.fillna("").astype(str).tolist()
        full = [float(d != "" and d in toks) for d in docs]
        ends = {x: sum(1 for e in docs if e.endswith(x)) for x in long_toks}
        suf, pre = [], []
        for d in docs:
            s = p = 0.0
            if d:
                for x in long_toks:
                    if len(x) < len(d) and ends[x] == 1 and d.endswith(x):
                        s = 1.0
                    if len(x) >= 6 and len(x) < len(d) and d.startswith(x):
                        p = 1.0
            suf.append(s)
            pre.append(p)
        f["cited_full"], f["cited_suffix"], f["cited_prefix"] = full, suf, pre
        f["log_pool"] = np.log(len(f))
        f["bank_txn_id"] = t.bank_txn_id
        f["booking_date"] = t.booking_date
        f["cash"] = cash
        f["payer"] = cid
        out.append(f.drop(columns=["_due"]))
    if not out:
        return pd.DataFrame(columns=["bank_txn_id", "item_id"] + FEATURES)
    return pd.concat(out, ignore_index=True)


def load_model(phase: str) -> dict | None:
    p = os.path.join(ROOT, "matching", "models", f"sd_noadv_{MODEL_FOR.get(phase, 'both')}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def predict(m: dict, c: pd.DataFrame) -> np.ndarray:
    X = c[m["features"]].to_numpy(dtype=float)
    z = X @ np.asarray(m["coef"], dtype=float) + float(m["intercept"])
    return 1.0 / (1.0 + np.exp(-z))


def claim(c: pd.DataFrame) -> list[tuple]:
    """c has p, open_amount, is_cn, cash, booking_date, due ordering columns. Returns ALLOC rows."""
    rows, taken = [], set()
    c = c.copy()
    c["_due"] = pd.to_datetime(c.due_date, errors="coerce").fillna(pd.to_datetime(c.issue_date, errors="coerce"))
    for txn, g in c.sort_values(["booking_date", "bank_txn_id"]).groupby(["booking_date", "bank_txn_id"], sort=True):
        txn = txn[1]
        cash = float(g.cash.iloc[0])
        g = g[(g.p >= P_MIN) & ~g.item_id.isin(taken)].sort_values(["p", "_due", "item_id"], ascending=[False, True, True])
        tot, pick = 0.0, []
        for r in g.itertuples(index=False):
            s = -r.open_amount if r.is_cn > 0 else r.open_amount
            if tot + s <= cash + 0.005:
                tot += s
                pick.append(r)
        if not pick or tot < COVER * cash:
            continue
        conf = round(float(np.mean([r.p for r in pick])), 3)
        for r in pick:
            taken.add(r.item_id)
            rows.append((txn, r.item_id, round(float(r.open_amount), 2), "", conf, NAME))
    return rows


def run(ctx: Ctx) -> pd.DataFrame:
    m = load_model(ctx.phase)
    if m is None:
        return pd.DataFrame(columns=ALLOC_COLS)
    c = candidates(ctx)
    if c.empty:
        return pd.DataFrame(columns=ALLOC_COLS)
    c["p"] = predict(m, c)
    return pd.DataFrame(claim(c), columns=ALLOC_COLS)
