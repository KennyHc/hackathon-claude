"""Deterministic exception / amount engine (POLICIES s2 + s3). Pure function of (bank line, item set, cash), no matcher needed.

    from matching.matchers.exception_rules import assign
    out = assign(alloc, "val")           # alloc: bank_txn_id, item_id, amount [+ other cols]  (item rows only)

What it does, per bank line, in booking-date order (a within-run ledger tracks the open amount of every item):
  RETURNED           invoice rows on a line whose cash runs against the side's normal direction (AR cash out / AP cash in),
                     or line id in `reversal_ids`. amount = last applied amount of the item (ledger).
  DUPLICATE_PAYMENT  invoice already fully settled earlier in the run (ledger open ~ 0). amount = cash paid again.
  FX_DIFFERENCE      line currency != item currency, fully cleared within the 3 % tolerance (replaces SD / SW / SBC / empty).
  SHORT_WITHHOLDING  AR customer in MX/BR/MA, deficit = 1.25 / 1.5 / 10 % of the invoice totals (all invoices of the line).
  SHORT_DISCOUNT     AR invoices with discount terms, booking_date <= issue + discount_days + grace, subset-sum on the deficit.
  PARTIAL            same deficit but outside the window (late discount), or any unexplained deficit placed on the item whose
                     deficit fraction is a known partial fraction (fitted on train < cut), else last invoice.
  SHORT_BANK_CHARGES AR transfer, single invoice, deficit <= 60 EUR-eq and <= 2 %, nothing else explains it.
  OVERPAYMENT        AR single invoice, cash > open amount. amount = open amount.
  SHORT_DISPUTE      only when the caller supplies `dispute_ids` (bank_txn_id set) [+ optional `dispute_by_fraction`].
  amount             full open amount for fully cleared rows (s2); cash applied for PARTIAL / SHORT_DISPUTE / DUPLICATE.

Context kwargs (all optional): bank, items (frames; default data.bank/data.items(phase)), reversal_ids, dispute_ids,
params (dict, see exc_params), dispute_by_fraction (bool), return_ledger (bool).
Every rule degrades to "no exception" on missing data; per-line failures are counted in assign.stats.
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, ".") if "." not in sys.path else None
from matching.lib import data, exc_params

P_FUZZY_MIN = 5; P_FUZZY_WH = 0.35
WITHHOLD_COUNTRIES = ("MX", "BR", "MA")
CN_TYPES = ("ARC", "APC")


# ------------------------------------------------------------------ helpers
def _fx_tables():
    fx = data.fx_rates().sort_values("date")
    t = {}
    for c, d in fx.groupby("currency"):
        t[c] = (d.date.values.astype("datetime64[D]").astype(np.int64), d.units_per_eur.values.astype(float))
    t["EUR"] = (np.array([0]), np.array([1.0]))
    return t


def _rate(tab, cur, day):
    """units of `cur` per EUR on `day` (int days since epoch), last available rate rule; None if unknown."""
    if cur not in tab:
        return None
    d, r = tab[cur]
    i = np.searchsorted(d, day, side="right") - 1
    if i < 0:
        i = 0  # before first fixing: use first
    return float(r[i])


def _subsets(vals, late, target, tol, maxn=22):
    """All subsets of `vals` (cents, ints) summing to target +-tol. Returns best (mask list) by (late count, error, idx) or None."""
    k = len(vals)
    if k == 0 or k > maxn:
        return None
    s = np.zeros(1, dtype=np.int64)
    l = np.zeros(1, dtype=np.int16)
    for v, lt in zip(vals, late):
        s = np.concatenate([s, s + v])
        l = np.concatenate([l, l + lt])
    err = np.abs(s - target)
    ok = np.flatnonzero((err <= tol) & (s > 0))
    if ok.size == 0:
        return None
    key = np.lexsort((ok, err[ok], l[ok]))
    best = int(ok[key[0]])
    return [j for j in range(k) if (best >> j) & 1]


# ------------------------------------------------------------------ engine
class _Ctx:
    pass


def _prepare(alloc, phase, ctx):
    bank = ctx.get("bank")
    if bank is None:
        bank = data.bank(phase)
    items = ctx.get("items")
    if items is None:
        items = data.items(phase)
    bcols = ["bank_txn_id", "booking_date", "amount", "currency", "orig_amount", "orig_currency"]
    bank = bank[bcols].rename(columns={"amount": "cash", "currency": "bcur"}).drop_duplicates("bank_txn_id")
    icols = ["item_id", "item_type", "side", "party_id", "issue_date", "currency", "total_amount", "open_amount", "payment_terms", "payment_method"]
    it = items[icols].rename(columns={"currency": "icur"}).drop_duplicates("item_id")
    a = alloc.reset_index(drop=True).copy()
    a["_ord"] = np.arange(len(a))
    x = a[["_ord", "bank_txn_id", "item_id", "amount"]].rename(columns={"amount": "in_amt"})
    x = x.merge(bank, on="bank_txn_id", how="left").merge(it, on="item_id", how="left")
    x = x.sort_values(["booking_date", "bank_txn_id", "_ord"], kind="mergesort").reset_index(drop=True)
    return a, x


def assign(alloc: pd.DataFrame, phase: str, **ctx) -> pd.DataFrame:
    P = ctx.get("params") or exc_params.load()
    reversal_ids = set(ctx.get("reversal_ids") or ())
    dispute_ids = set(ctx.get("dispute_ids") or ())
    dispute_by_fraction = bool(ctx.get("dispute_by_fraction", True))
    a, x = _prepare(alloc, phase, ctx)
    pt = data.payment_terms().set_index("code")
    disc = {c: (float(r.early_payment_discount_pct) / 100.0, int(r.discount_days)) for c, r in pt.iterrows() if pd.notna(r.early_payment_discount_pct) and pd.notna(r.discount_days)}
    fxt = _fx_tables()
    try:
        cu = data.customers().set_index("customer_id").country.to_dict()
    except Exception:
        cu = {}
    grace = int(P["grace_days"])
    fee_cap = float(P["bank_fee_cap_eur"]); fee_rel = float(P["bank_fee_cap_rel"]); fx_tol = float(P["fx_tolerance"]) + float(P["fx_slack"])
    wh_rates = P["withholding_rates"]
    pfr = [(float(f), w) for f, w in P["partial_fractions"]]  # (deficit fraction, weight)
    dfr = set(round(float(f), 4) for f in P["dispute_fractions"]); dmin = int(P["dispute_min_items"])

    n = len(x)
    out_amt = x.in_amt.values.astype(float).copy()
    out_exc = np.array([None] * n, dtype=object)
    stats = defaultdict(int)

    # columns -> numpy
    tid = x.bank_txn_id.values; iid = x.item_id.values
    cash = x.cash.values.astype(float); bcur = x.bcur.values; oam = x.orig_amount.values.astype(float); ocur = x.orig_currency.values
    bday = x.booking_date.values.astype("datetime64[D]").astype(np.int64) if x.booking_date.notna().all() else None
    bd_na = x.booking_date.isna().values
    if bday is None:
        bday = np.where(bd_na, 0, x.booking_date.fillna(pd.Timestamp("1970-01-01")).values.astype("datetime64[D]").astype(np.int64))
    inamt = x.in_amt.values.astype(float)
    itype = x.item_type.values; side = x.side.values; party = x.party_id.values
    iday = x.issue_date.fillna(pd.Timestamp("1970-01-01")).values.astype("datetime64[D]").astype(np.int64)
    icur = x.icur.values; tot = x.total_amount.values.astype(float); op0 = x.open_amount.values.astype(float)
    terms = x.payment_terms.values; meth = x.payment_method.values

    open_amt = {}   # item -> open amount (ledger)
    last_app = {}   # item -> last applied amount
    known = ~(pd.isna(x.item_type).values | pd.isna(x.cash).values)

    bounds = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1], True])
    for bi in range(len(bounds) - 1):
        lo, hi = bounds[bi], bounds[bi + 1]
        idx = np.arange(lo, hi)
        idx = idx[known[idx]]
        if idx.size == 0:
            continue
        try:
            _line(inamt, idx, tid, iid, cash, bcur, oam, ocur, bday, itype, side, party, iday, icur, tot, op0, terms, meth, out_amt, out_exc,
                  open_amt, last_app, reversal_ids, dispute_ids, dispute_by_fraction, disc, fxt, cu, grace, fee_cap, fee_rel, fx_tol,
                  wh_rates, pfr, dfr, dmin, stats)
        except Exception:  # never crash: no claim
            stats["line_errors"] += 1

    res = a.copy()
    o = pd.DataFrame({"_ord": x._ord.values, "_amt": out_amt, "_exc": out_exc}).sort_values("_ord")
    res["amount"] = np.where(known[np.argsort(x._ord.values)], o._amt.values, res.amount.values)
    exc = o._exc.values
    if "exception" in res:
        res["exception"] = np.where(known[np.argsort(x._ord.values)], exc, res["exception"].values)
    else:
        res["exception"] = exc
    res = res.drop(columns="_ord")
    assign.stats = dict(stats)
    return res


assign.stats = {}
DEBUG = {}   # bank_txn_id -> number of distinct candidate invoices matching a known partial fraction (diagnostics only)


def _line(inamt, idx, tid, iid, cash, bcur, oam, ocur, bday, itype, side, party, iday, icur, tot, op0, terms, meth, out_amt, out_exc,
          open_amt, last_app, reversal_ids, dispute_ids, dispute_by_fraction, disc, fxt, cu, grace, fee_cap, fee_rel, fx_tol,
          wh_rates, pfr, dfr, dmin, stats):
    i0 = idx[0]
    sd = 1.0 if side[i0] == "AR" else -1.0
    c = cash[i0] * sd                       # >0: normal direction
    is_cn = np.array([itype[i] in CN_TYPES for i in idx])
    O = np.array([open_amt.get(iid[i], op0[i] if not np.isnan(op0[i]) else tot[i]) for i in idx], dtype=float)

    def setrow(k, amt, exc):
        i = idx[k]
        out_amt[i] = round(float(amt), 2)
        out_exc[i] = exc
        if exc == "RETURNED":
            open_amt[iid[i]] = O[k] + out_amt[i]
        elif exc != "DUPLICATE_PAYMENT":
            open_amt[iid[i]] = max(O[k] - out_amt[i], 0.0)
        if exc != "RETURNED" and exc != "DUPLICATE_PAYMENT":
            last_app[iid[i]] = out_amt[i]

    # ---- RETURNED
    if (~is_cn).any() and (c < -0.005 or tid[i0] in reversal_ids):
        for k, i in enumerate(idx):
            amt = last_app.get(iid[i])
            if amt is None:
                amt = O[k] if tot[i] <= 0 or abs(O[k] - tot[i]) > 0.01 else tot[i]
            setrow(k, amt, "RETURNED")
        stats["returned"] += 1
        return
    # ---- credit notes only
    if is_cn.all():
        for k in range(len(idx)):
            setrow(k, O[k] if O[k] > 0.005 else tot[idx[k]], None)
        return
    ccash = abs(c)
    # ---- DUPLICATE: settled invoices
    settled = (~is_cn) & (O <= 0.005)
    active = np.flatnonzero(~settled)
    ccur = None
    if settled.any():
        single = (~is_cn).sum() == 1 and not is_cn.any() and len(idx) == 1
        k = int(np.flatnonzero(settled)[0])
        i = idx[k]
        if single:
            amt = _cash_in_item_ccy(ccash, bcur[i0], oam[i0], ocur[i0], icur[i], bday[i0], fxt)
            amt = amt if amt is not None else tot[i]
            ref = last_app.get(iid[i], tot[i])
            same_ccy = icur[i] == bcur[i0] or (isinstance(ocur[i0], str) and ocur[i0] == icur[i])
            # exact repeat of an already settled invoice -> DUPLICATE; a smaller payment is the remainder of a dispute/partial that the
            # ledger could not see -> plain clearing of what was paid
            if abs(amt - ref) <= max(0.01, 0.005 * ref) or abs(amt - tot[i]) <= max(0.01, 0.005 * tot[i]) or (not same_ccy and amt >= 0.95 * ref):
                out_amt[i] = round(float(amt), 2); out_exc[i] = "DUPLICATE_PAYMENT"; stats["duplicate"] += 1
            else:
                out_amt[i] = round(float(min(amt, tot[i])), 2); stats["settled_remainder"] += 1
                out_exc[i] = None if same_ccy else "FX_DIFFERENCE"   # remainder paid in a foreign currency is still an FX line
            return
        # settled invoice inside a multi-item line: the ledger is probably wrong (an earlier deduction we could not place) -> reopen
        for k in np.flatnonzero(settled):
            O[k] = tot[idx[k]]
        active = np.arange(len(idx))
    act = active
    inv = [k for k in act if not is_cn[k]]
    cns = [k for k in act if is_cn[k]]
    D = float(sum(O[k] for k in inv) - sum(O[k] for k in cns))
    cur_set = set(icur[idx[k]] for k in act)
    sdct = side[i0] == "AR"

    def finish_clean(label=None, label_rows=None):
        for k in act:
            setrow(k, O[k], label if (label and not is_cn[k]) else None)

    # ---- FX path
    if len(cur_set) == 1 and list(cur_set)[0] != bcur[i0] and isinstance(bcur[i0], str):
        ic = list(cur_set)[0]
        Cp = _cash_in_item_ccy(ccash, bcur[i0], oam[i0], ocur[i0], ic, bday[i0], fxt)
        if Cp is None:
            for k in act:
                setrow(k, O[k], None)
            return
        exact = (isinstance(ocur[i0], str) and ocur[i0] == ic and not np.isnan(oam[i0]))
        diff = D - Cp
        lim = fx_tol * max(D, 1.0)
        if abs(diff) <= max(lim, 0.02):
            stats["fx"] += 1
            for k in act:
                setrow(k, O[k], None if is_cn[k] else "FX_DIFFERENCE")
            return
        if diff < 0:  # more than tolerance in excess: still an FX line; flag cleared invoices
            for k in act:
                setrow(k, O[k], None if is_cn[k] else "FX_DIFFERENCE")
            return
        # deficit beyond FX tolerance -> partial / dispute on one invoice, others FX
        tolc = 0.02 if exact else fx_tol * D
        sel = _partial_pick(inv, O, iday, iid, idx, diff, tolc, pfr, dfr, dispute_by_fraction, tid[i0] in dispute_ids, dmin)
        for k in act:
            if sel is not None and k == sel[0]:
                setrow(k, max(O[k] - diff, 0.0) if not exact else sel[2], sel[1])
            else:
                setrow(k, O[k], None if is_cn[k] else "FX_DIFFERENCE")
        return

    # ---- same currency (or unknown): deficit analysis
    diff = D - ccash
    n_inv = len(inv)
    ctol = 0.015
    if all(meth[idx[k]] == "card" for k in inv):
        finish_clean(); return
    if abs(diff) <= ctol:
        finish_clean(); return
    if diff < 0:
        if sdct and n_inv == 1 and not cns:
            k = inv[0]
            setrow(k, O[k], "OVERPAYMENT"); stats["overpayment"] += 1
        else:
            finish_clean()
        return
    # deficit
    tid_i = tid[i0]
    hyps = []      # systematic deficits: (name, {k: (label, deficit)})
    tolh = lambda m: 0.01 + 0.009 * np.sqrt(max(m, 1))   # rounding noise of m per-item cent roundings (3 sigma)
    el = []; d_c = []; late = []
    if sdct:
        ctry = cu.get(party[idx[inv[0]]]) if inv else None
        if ctry in WITHHOLD_COUNTRIES:
            for r in wh_rates.get(ctry, wh_rates.get("*", [])):
                hyps.append(("wh", {k: ("SHORT_WITHHOLDING", r * O[k]) for k in inv}))
        for k in inv:
            t = terms[idx[k]]
            if isinstance(t, str) and t in disc and abs(O[k] - tot[idx[k]]) <= 0.01:
                el.append(k)
        if el:
            d_c = [int(round(round(disc[terms[idx[k]]][0] * tot[idx[k]], 2) * 100)) for k in el]
            late = [0 if bday[i0] <= iday[idx[k]] + disc[terms[idx[k]]][1] + grace else 1 for k in el]
            # customers take the discount on every eligible invoice of a line (fitted on train, see matching/tests/exceptions/)
            seen = set()
            for sel_h in ([j for j in range(len(el))], [j for j in range(len(el)) if not late[j]], [j for j in range(len(el)) if late[j]]):
                if sel_h and tuple(sel_h) not in seen:
                    seen.add(tuple(sel_h))
                    hyps.append(("disc", {el[j]: ("PARTIAL" if late[j] else "SHORT_DISCOUNT", d_c[j] / 100.0) for j in sel_h}))

    def apply(asg, extra=None):
        for k in act:
            if extra is not None and k == extra[0]:
                setrow(k, extra[2], extra[1])
            elif k in asg:
                lab, dfc = asg[k]
                setrow(k, O[k] if lab not in ("PARTIAL", "SHORT_DISPUTE") else O[k] - dfc, lab)
            else:
                setrow(k, O[k], None)

    # 0) caller-supplied cash-applied amounts (e.g. from a remittance advice / payment run): trust them when they explain the deficit
    hk = [k for k in inv if not np.isnan(inamt[idx[k]]) and O[k] - inamt[idx[k]] > max(0.01, 0.005 * O[k])]
    if hk and abs(sum(O[k] - inamt[idx[k]] for k in hk) - diff) <= max(0.02, 0.002 * D):
        asg = {}
        for k in hk:
            dfc = O[k] - inamt[idx[k]]
            t = terms[idx[k]]
            lab = None
            if sdct and isinstance(t, str) and t in disc and abs(dfc - round(disc[t][0] * tot[idx[k]], 2)) <= 0.011:
                lab = "PARTIAL" if bday[i0] > iday[idx[k]] + disc[t][1] + grace else "SHORT_DISCOUNT"
            elif sdct and cu.get(party[idx[k]]) in WITHHOLD_COUNTRIES and any(abs(dfc - r * O[k]) <= 0.011 for r in wh_rates.get(cu.get(party[idx[k]]), [])):
                lab = "SHORT_WITHHOLDING"
            else:
                lab = "SHORT_DISPUTE" if tid_i in dispute_ids else "PARTIAL"
                if lab == "PARTIAL" and dispute_by_fraction and len(inv) >= dmin and round(dfc / O[k], 2) in dfr:
                    lab = "SHORT_DISPUTE"
            asg[k] = (lab, dfc)
        apply(asg); stats["hinted"] += 1
        return
    # 1) exact systematic hypotheses
    for name, asg in hyps:
        if abs(diff - sum(v[1] for v in asg.values())) <= tolh(len(asg)):
            apply(asg); stats["withholding" if name == "wh" else "discount"] += 1
            return
    # 2) discount subset (customer took the discount only on some invoices)
    if el:
        sol = None
        win = [j for j in range(len(el)) if not late[j]]
        lat = [j for j in range(len(el)) if late[j]]
        if win and lat:   # every in-window invoice took the discount, a subset of the late ones did too (late -> PARTIAL)
            r1 = int(round(diff * 100)) - sum(d_c[j] for j in win)
            if r1 > 0:
                sub = _subsets([d_c[j] for j in lat], [1] * len(lat), r1, max(2, int(np.ceil(0.5 * len(el))) + 1))
                if sub is not None:
                    sol = win + [lat[j] for j in sub]
        if sol is None:
            sol = _subsets(d_c, late, int(round(diff * 100)), max(2, int(np.ceil(0.5 * len(el))) + 1))
        if sol is not None:
            apply({el[j]: ("PARTIAL" if late[j] else "SHORT_DISCOUNT", d_c[j] / 100.0) for j in sol})
            stats["discount"] += 1
            return
    # 2b) fuzzy systematic hypotheses on big lines: withholding / discount on every invoice plus a few disputed invoices that cannot be
    #     located without the remittance advice (their deductions make the line's deficit deviate from the pure rate)
    if sdct and n_inv >= int(P_FUZZY_MIN):
        sinv = sum(O[k] for k in inv)
        for name, asg in hyps:
            if name == "wh":
                r_ = list(asg.values())[0][1] / O[list(asg)[0]] if asg else 0
                if r_ > 0 and abs(diff / sinv - r_) <= P_FUZZY_WH * r_:
                    apply(asg); stats["withholding_fuzzy"] += 1
                    return
        for name, asg in hyps:
            if name == "disc" and all(v[0] == "SHORT_DISCOUNT" for v in asg.values()) and len(asg) >= int(P_FUZZY_MIN):
                if diff >= sum(v[1] for v in asg.values()) - tolh(len(asg)):
                    apply(asg); stats["discount_fuzzy"] += 1
                    return
    # 3) bank charges (single AR invoice)
    if sdct and n_inv == 1 and not cns and all(meth[idx[k]] != "dd" for k in inv):
        k = inv[0]
        r = _rate(fxt, bcur[i0], bday[i0]) if isinstance(bcur[i0], str) else None
        eur = diff / r if r else None
        if eur is not None and eur <= fee_cap and diff <= fee_rel * D:
            setrow(k, O[k], "SHORT_BANK_CHARGES"); stats["bank_charges"] += 1
            return
    # 4) systematic deficit + one partial/dispute on another invoice (needs a known fraction: exact, unique-ish)
    for name, asg in hyps:
        res = diff - sum(v[1] for v in asg.values())
        if res > 0.02:
            sel = _partial_pick(inv, O, iday, iid, idx, res, tolh(len(asg)), pfr, dfr, dispute_by_fraction, tid_i in dispute_ids, dmin,
                                exclude=set(asg), strict=True)
            if sel is not None:
                apply(asg, sel); stats["composite"] += 1
                return
    # 5) partial / dispute alone
    sel = _partial_pick(inv, O, iday, iid, idx, diff, 0.015, pfr, dfr, dispute_by_fraction, tid_i in dispute_ids, dmin)
    apply({}, sel)
    stats["partial" if sel else "unresolved"] += 1
    DEBUG[tid_i] = getattr(_partial_pick, 'last_n', -1)
    if sel:
        stats["partial_cands_%s" % min(_partial_pick.last_n, 3)] += 1


def _cash_in_item_ccy(ccash, bc, oamt, oc, ic, day, fxt):
    """Cash (positive, line currency) expressed in the item currency. Uses the instructed amount when it is in the item currency."""
    if isinstance(oc, str) and oc == ic and not np.isnan(oamt):
        return abs(float(oamt))
    if bc == ic:
        return ccash
    rb = _rate(fxt, bc, day); ri = _rate(fxt, ic, day)
    if rb is None or ri is None:
        return None
    return ccash / rb * ri


def _partial_pick(inv, O, iday, iid, idx, diff, tolc, pfr, dfr, dispute_by_fraction, in_dispute, dmin=10**9, exclude=(), strict=False):
    """Choose the invoice that carries the deficit. Returns (k, exception, applied_amount) or None."""
    if not inv:
        return None
    label = "SHORT_DISPUTE" if in_dispute else "PARTIAL"
    cands = []
    for k in inv:
        if k in exclude or O[k] <= diff + 0.005:
            continue
        for f, w in pfr:
            if abs(diff - f * O[k]) <= max(tolc, 0.011):
                cands.append((-w, abs(diff - f * O[k]), -iday[idx[k]], k, f))
    if cands:
        cands.sort()
        _, _, _, k, f = cands[0]
        _partial_pick.last_n = len(set(c[3] for c in cands))
        lab = label
        if dispute_by_fraction and not in_dispute and round(f, 4) in dfr and len(inv) >= dmin:
            lab = "SHORT_DISPUTE"
        return k, lab, O[k] - diff
    _partial_pick.last_n = 0
    if strict:
        return None
    if len(inv) == 1:
        k = inv[0]
        if O[k] > diff:
            return k, label, O[k] - diff
        return None
    return None  # multi-invoice line, no known partial fraction: no claim (a guess was ~15 % precise on train 2024)
