"""AP matcher: supplier-side bank lines (payment runs, confirming, holding, returned transfers, duplicates, direct debits, refunds).

    python -m matching.matchers.ap --phase val
Interface: match(phase, claimed=None) -> DataFrame[bank_txn_id,item_id,amount,exception,confidence,method]
"""
from __future__ import annotations
import argparse, itertools, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from matching.lib import data, ap_data, ap_units as U

NAME = "ap"
TOLC = 0.011


def _cents(x):
    return int(round(float(x) * 100))


def _all_sums(amts):
    """array of subset sums; index = bitmask over amts."""
    sums = np.zeros(1, dtype=np.int64)
    for a in amts:
        sums = np.concatenate([sums, sums + a])
    return sums


def _subset_solutions(amts, target, limit=5):
    """index subsets of amts (cents) summing to target (meet in the middle, numpy). n <= 40."""
    n = len(amts)
    if n == 0 or n > 40:
        return []
    h = n // 2
    L, R = amts[:h], amts[h:]
    sl, sr = _all_sums(L), _all_sums(R)
    order = np.argsort(sr, kind="stable")
    srs = sr[order]
    need = target - sl
    lo = np.searchsorted(srs, need, side="left")
    hi = np.searchsorted(srs, need, side="right")
    out = []
    for i in np.nonzero(hi > lo)[0]:
        for j in order[lo[i]:hi[i]]:
            idx = [k for k in range(h) if (int(i) >> k) & 1] + [h + k for k in range(n - h) if (int(j) >> k) & 1]
            if idx:
                out.append(tuple(idx))
                if len(out) >= limit:
                    return out
    return out


def _roll(d):
    """next business day (Sat/Sun roll to Monday)."""
    d = pd.Timestamp(d)
    wd = d.weekday()
    return d + pd.Timedelta(days=(2 if wd == 5 else 1 if wd == 6 else 0))


def _line_amount_match(line, unit_cur, unit_amt):
    """bank amount equals the unit amount (account currency, or instructed amount in unit currency)."""
    if line.currency == unit_cur and abs(line.amount_abs - unit_amt) <= 0.011:
        return True
    oc = line.orig_currency
    if isinstance(oc, str) and oc == unit_cur and pd.notna(line.orig_amount) and abs(abs(line.orig_amount) - unit_amt) <= 0.011:
        return True
    return False


def core(bank: pd.DataFrame, runs: pd.DataFrame, items: pd.DataFrame, claimed=None) -> pd.DataFrame:
    claimed = claimed or set()
    rows = U.prep_runs(runs)
    cand = U.resolve_rows(rows, items)
    alias = U.build_alias(rows, cand)
    picks, unit_party = U.decide_rows(rows, cand, alias)
    rows["unit"] = rows.run_id + "|" + rows.end_to_end_id.astype(str)
    pk = picks.set_index("row_id")
    rows = rows.join(pk[["item_id", "status", "total_amount", "open_amount"]], on="row_id")
    units = rows.groupby("unit").agg(run_id=("run_id", "first"), run_type=("run_type", "first"), exec_date=("execution_date", "first"),
                                     cur=("currency", "first"), tamt=("transfer_amount", "first"), bnorm=("bnorm", "first"),
                                     acct=("debtor_account_id", "first"), n=("row_id", "size"))
    prior_ids = set(items.item_id[items.prior]) if "prior" in items else set()
    rows_by_unit = {u: d for u, d in rows.groupby("unit")}
    ucols = units.reset_index()
    uinfo = {x.unit: x for x in ucols.itertuples()}
    runs_units = {r: [x.unit for x in g.itertuples()] for r, g in ucols.groupby("run_id")}
    run_type = dict(zip(ucols.run_id, ucols.run_type))

    b = bank[(bank.direction == "out") & ~bank.bank_txn_id.isin(claimed)].copy()
    tok = [U.extract_run((r.end_to_end_id, r.ref1, r.ref2, r.remittance_info, r.text)) for r in b.itertuples()]
    b["run"] = [(t[0] + "@" + a) if t[0] else None for t, a in zip(tok, b.account_id)]
    b["sfx"] = [t[1] for t in tok]
    b = b[b.run.notna() & b.run.isin(runs_units.keys())].sort_values(["booking_date", "bank_txn_id"])

    line_units = {}      # bank_txn_id -> (list of unit ids, method, conf)
    used_units = {}      # unit -> first bank line
    dup_units = set()    # bank_txn_id flagged as duplicate payment
    conf_lines = []
    for r in b.itertuples():
        ru = runs_units[r.run]
        rt = run_type[r.run]
        if rt == "aggregated":
            line_units[r.bank_txn_id] = (list(ru), "run_aggregated", 0.95)
            continue
        if rt == "confirming":
            conf_lines.append(r)
            continue
        cands = []
        meth = ""
        if r.sfx and len(r.sfx) == 4:
            u = f"{r.run}|{r.run.split('@')[0]}-{r.sfx}"
            if u in uinfo and _line_amount_match(r, uinfo[u].cur, uinfo[u].tamt):
                cands = [u]
                meth = "run_e2e"
        if not cands:
            cands = [u for u in ru if _line_amount_match(r, uinfo[u].cur, uinfo[u].tamt)]
            meth = "run_amount"
            if len(cands) > 1:
                t = U.norm_core(r.text) + " " + U.norm_core(getattr(r, "counterparty_name", None))
                nm = [u for u in cands if uinfo[u].bnorm and uinfo[u].bnorm in t]
                if nm:
                    cands = nm
                    meth = "run_amount_name"
            if len(cands) > 1:
                free = [u for u in cands if u not in used_units]
                if free and len({uinfo[u].n for u in cands}) == 1:
                    cands = free[:1]
                    meth = "run_amount_order"
        if len(cands) != 1:
            continue
        u = cands[0]
        conf = 0.97 if meth in ("run_e2e", "run_amount") else 0.85
        line_units[r.bank_txn_id] = ([u], meth, conf)
        if u in used_units:
            dup_units.add(r.bank_txn_id)
        else:
            used_units[u] = r.bank_txn_id

    # returned transfers: incoming credit that quotes the run unit (PRnnnnnnnnnnnn-nnnn) and has the same amount
    forced = {}
    bi = bank[(bank.direction == "in") & ~bank.bank_txn_id.isin(claimed)]
    for r in bi.itertuples():
        run, sfx = U.extract_run((r.end_to_end_id, r.remittance_info, r.text, r.ref1, r.ref2))
        if not run or not sfx or len(sfx) != 4:
            continue
        u = f"{run}@{r.account_id}|{run}-{sfx}"
        if u in uinfo and _line_amount_match(r, uinfo[u].cur, uinfo[u].tamt):
            line_units[r.bank_txn_id] = ([u], "returned_transfer", 0.95)
            forced[r.bank_txn_id] = "RETURNED"

    # confirming: one debit per maturity date = whole units whose latest document due date (+0/30/60/90 d, weekend roll) is that date
    due = items.drop_duplicates("item_id").set_index("item_id").due_date
    rows["due"] = rows.item_id.map(due)
    udue = rows.groupby("unit").due.max()
    by_run = {}
    for r in conf_lines:
        by_run.setdefault(r.run, []).append(r)
    for run, lines in by_run.items():
        uids = runs_units[run]
        taken = set()
        res_ = {}
        for r in lines:   # pass 1: a single unit with exactly this amount
            ex = [u for u in uids if _line_amount_match(r, uinfo[u].cur, uinfo[u].tamt)]
            res_[r.bank_txn_id] = ex
        def _solve(r, offsets):
            sol = []
            for u in uids:
                if u in taken:
                    continue
                d = udue.get(u)
                if pd.isna(d):
                    continue
                if any(_roll(d + pd.Timedelta(days=e)) == r.booking_date for e in offsets):
                    sol.append(u)
            if not sol:
                return None
            tgt = _cents(r.amount_abs)
            amts = [_cents(uinfo[u].tamt) for u in sol]
            if sum(amts) == tgt:
                return sol
            if len(sol) > 40:
                return None
            ss = _subset_solutions(amts, tgt, limit=5)
            if len(ss) == 1:
                return [sol[i] for i in ss[0]]
            return None
        for offsets in ((0,), (0, 30), (0, 30, 60), (0, 30, 60, 90)):
            for r in lines:
                if r.bank_txn_id in line_units:
                    continue
                sol = _solve(r, offsets)
                if sol:
                    line_units[r.bank_txn_id] = (sol, "confirming_subset", 0.93)
                    taken |= set(sol)
        for r in lines:
            if r.bank_txn_id in line_units:
                continue
            ex = [u for u in res_[r.bank_txn_id] if u not in taken]
            if len(ex) == 1:
                line_units[r.bank_txn_id] = (ex, "confirming_single", 0.9)
                taken.add(ex[0])

    # allocation rows
    out = []
    bl = bank.set_index("bank_txn_id")
    for t, (ulist, meth, conf) in line_units.items():
        line = bl.loc[t]
        for u in ulist:
            d = rows_by_unit[u]
            d = d[d.item_id.notna()]
            for rr in d.itertuples():
                out.append((t, rr.item_id, rr.damt, rr.status, rr.kind, rr.currency, rr.total_amount, meth, conf, t in dup_units, line.currency, None, rr.open_amount, rr.item_id in prior_ids))
    a = pd.DataFrame(out, columns=_empty().columns)
    a["forced"] = a.bank_txn_id.map(forced)
    a = pd.concat([a, _direct_debits(bank, items, claimed, set(a.bank_txn_id) | claimed, set(a.item_id)), _supplier_refunds(bank, items, claimed, set(a.bank_txn_id))], ignore_index=True)
    return _exceptions(a, bank)


def _exceptions(a, bank):
    """Ledger in booking order: DUPLICATE_PAYMENT if the item was already settled (earlier line of the phase, the same run
    transfer debited twice, or settled before the phase when use_prior). RETURNED on returned transfers. Nothing else."""
    if a.empty:
        return a
    a = a.copy()
    a["bdate"] = a.bank_txn_id.map(bank.set_index("bank_txn_id").booking_date)
    a = a.sort_values(["bdate", "bank_txn_id"], kind="stable")
    paid = {}
    exc = []
    if "forced" not in a:
        a["forced"] = None
    for r in a.itertuples():
        if r.forced == "RETURNED":   # the returned transfer re-opens the item: a later re-payment is not a duplicate
            exc.append("RETURNED")
            paid[r.item_id] = paid.get(r.item_id, 0.0) - r.amount
            continue
        e = None
        if r.kind == "API":
            init = 0.0 if r.prior else (r.open if pd.notna(r.open) else r.total)
            rem = init - paid.get(r.item_id, 0.0)
            if rem <= TOLC or r.dup:
                e = "DUPLICATE_PAYMENT"
            # PARTIAL / FX_DIFFERENCE / discounts are left to classification/exceptions.py (team rule)
            paid[r.item_id] = paid.get(r.item_id, 0.0) + r.amount
        exc.append(e)
    a["exception"] = exc
    return a.drop(columns=["bdate", "forced"])


def _name_hit(pname_norm, text_norm):
    """first 12 chars of the (suffix-less) supplier name appear in the bank text (N43 truncates names to 16 chars)."""
    k = pname_norm[:12].strip()
    return len(k) >= 6 and k in text_norm


def _line_text(r):
    return U.norm_name(" ".join(str(x) for x in (getattr(r, "counterparty_name", None), r.text, r.ref2, r.remittance_info) if isinstance(x, str)))


def _empty():
    return pd.DataFrame(columns=["bank_txn_id", "item_id", "amount", "status", "kind", "icur", "total", "method", "confidence", "dup", "bcur", "exception", "open", "prior"])


def _direct_debits(bank, items, claimed, done_lines, used_items):
    """Supplier pulls the money: out line, no run reference, amount = invoice total, booked on the due date (+weekend roll),
    supplier name in the text. Items with payment_method dd (or unknown, e.g. open_items_at_start)."""
    cand = items[(items.item_type == "API") & items.payment_method.fillna("dd").eq("dd") & items.due_date.notna() & ~items.item_id.isin(used_items) & ~items.get("prior", False)].copy()
    if cand.empty:
        return _empty()
    cand["cents"] = (cand.open_amount * 100).round().astype(int)
    cand["tcents"] = (cand.total_amount * 100).round().astype(int)
    cand["pn"] = cand.party_name.map(U.norm_core)
    cand["roll"] = cand.due_date.map(_roll)
    byc = {}
    for c in cand.itertuples():
        byc.setdefault((c.currency, c.cents), []).append(c)
        if c.tcents != c.cents:
            byc.setdefault((c.currency, c.tcents), []).append(c)
    b = bank[(bank.direction == "out") & ~bank.bank_txn_id.isin(done_lines)]
    out, taken = [], set()
    for r in b.sort_values(["booking_date", "bank_txn_id"]).itertuples():
        cs = byc.get((r.currency, int(round(r.amount_abs * 100))), [])
        if not cs:
            continue
        txt = _line_text(r)
        hit = [c for c in cs if c.item_id not in taken and 0 <= (r.booking_date - c.roll).days <= 3 and c.issue_date <= r.booking_date and _name_hit(c.pn, txt)]
        if len({c.item_id for c in hit}) != 1:
            continue
        c = hit[0]
        taken.add(c.item_id)
        out.append((r.bank_txn_id, c.item_id, r.amount_abs, "exact", "API", c.currency, c.total_amount, "direct_debit", 0.95, False, r.currency, None, c.open_amount, bool(getattr(c, "prior", False))))
    return pd.DataFrame(out, columns=_empty().columns) if out else _empty()


def _supplier_refunds(bank, items, claimed, done_lines):
    """Incoming credit equal to an APC credit-note total, with the supplier name in the text."""
    cand = items[(items.item_type == "APC")].copy()
    if cand.empty:
        return _empty()
    cand["cents"] = (cand.total_amount * 100).round().astype(int)
    cand["pn"] = cand.party_name.map(U.norm_core)
    byc = {}
    for c in cand.itertuples():
        byc.setdefault((c.currency, c.cents), []).append(c)
    b = bank[(bank.direction == "in") & ~bank.bank_txn_id.isin(done_lines | claimed)]
    out, taken = [], set()
    for r in b.sort_values(["booking_date", "bank_txn_id"]).itertuples():
        cs = byc.get((r.currency, int(round(r.amount_abs * 100))), [])
        if not cs:
            continue
        txt = _line_text(r)
        hit = [c for c in cs if c.item_id not in taken and c.issue_date <= r.booking_date and _name_hit(c.pn, txt)]
        if len({c.item_id for c in hit}) != 1:
            continue
        c = hit[0]
        taken.add(c.item_id)
        out.append((r.bank_txn_id, c.item_id, c.total_amount, "exact", "APC", c.currency, c.total_amount, "supplier_refund", 0.95, False, r.currency, None, c.total_amount, False))
    return pd.DataFrame(out, columns=_empty().columns) if out else _empty()


def match(phase: str, claimed=None, start=None, end=None, use_prior: bool = False) -> pd.DataFrame:
    bank = ap_data.bank_slice(phase, start, end)
    runs = ap_data.runs_slice(phase, start, end, use_prior=use_prior)
    items = ap_data.ap_items(phase, start, end, use_prior=use_prior)
    a = core(bank, runs, items, claimed)
    del runs, items
    a["amount"] = a.amount.astype(float).round(2)
    a["confidence"] = a.confidence.astype(float)
    a = a.drop_duplicates(["bank_txn_id", "item_id"])
    return a[["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"]].reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="val")
    a = ap.parse_args()
    from matching.lib import evalutil
    res = match(a.phase)
    os.makedirs(f"matching/out/{a.phase}", exist_ok=True)
    res.to_parquet(f"matching/out/{a.phase}/{NAME}.parquet", index=False)
    evalutil.report(res, a.phase, name=NAME, top=30)
