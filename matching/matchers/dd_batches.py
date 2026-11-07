"""Direct-debit batches (SEPA / BACS / ACH / BECS) and their returns.

* Batch credit: the bank line on the collection account whose amount == batch_total (batch id in the text when the bank
  shows it, collection date ± a few days) clears every receipt of the batch; each receipt -> its invoice via
  remittance_info (full number, or 'RECIBO <trailing digits>'), the mandate's customer and the amount.
* Return: a debit carrying the mandate id (RUECKLASTSCHRIFT / DEVOLUCION RECIBO / IMPAYE PRLV / SDD RETURN ...) ->
  the latest earlier receipt with that mandate and amount -> its invoice, exception RETURNED, amount = receipt amount.
* Re-presented receipts (re_presentation=Y) are ordinary rows of a later batch (no exception).
"""
from __future__ import annotations

import re

import pandas as pd

from ingestion import text
from matching.batch_helpers import Resolver, get_resolver, side_files
from matching.common import Ctx, alloc_frame

NAME = "dd_batches"

MANDATE = re.compile(r"\bVTM-[A-Z]{2}-\d{7}\b")
NOT_RETURN = re.compile(r"COMISI|GEBUEHR|GEBÜHR|ENTGELT|FRAIS|COMMISSION|\bFEE\b|CHARGE", re.I)
RCODE = re.compile(r"\b([A-Z]{2}\d\d)\b")
MIN_RESOLVED = 0.9


def _batches(ctx: Ctx) -> pd.DataFrame:
    d = side_files(ctx, "direct_debit_batches")
    if d.empty:
        return d
    d["amount"] = d.amount.astype(float).round(2)
    d["batch_total"] = d.batch_total.astype(float).round(2)
    return d


def _link_batches(ctx: Ctx, d: pd.DataFrame) -> dict[str, str]:
    """batch_id -> bank_txn_id"""
    bank = ctx.unclaimed()
    cr = bank[bank.amount > 0]
    key = {}
    for t in cr[["bank_txn_id", "account_id", "amount", "booking_date"]].itertuples(index=False):
        key.setdefault((t.account_id, round(t.amount, 2)), []).append((t.booking_date, t.bank_txn_id))
    text_of = dict(zip(cr.bank_txn_id, cr.text.fillna("") + " " + cr.ref1.fillna("") + " " + cr.ref2.fillna("")))
    heads = d.drop_duplicates("batch_id")[["batch_id", "account_id", "batch_total", "collection_date"]]
    out: dict[str, str] = {}
    used: set[str] = set()
    for h in heads.itertuples(index=False):
        cands = key.get((h.account_id, h.batch_total), [])
        if not cands:
            continue
        cd = pd.Timestamp(h.collection_date)
        near = [(abs((pd.Timestamp(bd) - cd).days), tid) for bd, tid in cands
                if -2 <= (pd.Timestamp(bd) - cd).days <= 6 and tid not in used]
        if not near:
            continue
        with_id = [x for x in near if h.batch_id in text_of[x[1]]]
        pick = min(with_id or near)
        if not with_id and len(near) > 1 and sorted(near)[0][0] == sorted(near)[1][0]:
            continue
        out[h.batch_id] = pick[1]
        used.add(pick[1])
    return out


def _words(s: str) -> set[str]:
    return {w for w in re.split(r"[^A-Z0-9]+", text.strip_accents((s or "").upper())) if len(w) >= 3}


def _paid_after_return(ctx: Ctx, d: pd.DataFrame, res: Resolver, rows: list, link: dict[str, str]):
    """Credit transfer that pays a direct-debit receipt that came back (returned in this phase, or collected before the
    phase and still open at its start). Normal allocation, no exception.

    Per item, the events of this phase (collection by a batch line 'C', return 'R') decide whether it is open for a
    transfer at a given date: payable when the last event before the date is a return."""
    start = ctx.bank.booking_date.min()
    day = dict(zip(ctx.bank.bank_txn_id, ctx.bank.booking_date))
    taken = {r[0] for r in rows}
    events: dict[str, list[tuple[str, str]]] = {}
    for tid, iid, _, exc, *_ in rows:
        events.setdefault(iid, []).append((day[tid], "R" if exc == "RETURNED" else "C"))
    # open at the phase start although collected before it (and that collection is not booked in this phase)
    it = ctx.open_items[(ctx.open_items.side == "AR") & (ctx.open_items.payment_method == "dd")]
    open_ids = set(it.item_id)
    before = d[(d.collection_date < start) & ~d.batch_id.isin(link)]
    for r in before[["account_id", "remittance_info", "amount", "mandate_id"]].drop_duplicates().itertuples(index=False):
        ent = r.account_id[:4]
        iid, _ = res.resolve(ent, r.remittance_info, r.amount, res.parties_for(ent, mandate=r.mandate_id))
        if iid in open_ids:
            events.setdefault(iid, []).append(("0000", "R"))
    for ev in events.values():
        ev.sort()
    cand = {i: ev for i, ev in events.items() if any(k == "R" for _, k in ev)}

    def payable(i: str, on: str) -> bool:
        last = [k for dt, k in cand[i] if dt < on]
        return bool(last) and last[-1] == "R"

    pool = res.pool_by_id
    by_key: dict[tuple, list[str]] = {}
    for iid in cand:
        if iid in pool.index:
            p = pool.loc[iid]
            by_key.setdefault((p.entity, round(float(p.total_amount), 2)), []).append(iid)
    bank = ctx.unclaimed()
    cr = bank[(bank.amount > 0) & ~bank.bank_txn_id.isin(taken)]
    out = []
    used: set[str] = set()
    for t in cr.itertuples(index=False):
        ids = by_key.get((t.entity, round(t.amount, 2)))
        if not ids:
            continue
        ids = [i for i in ids if i not in used and payable(i, t.booking_date)]
        if not ids:
            continue
        blob = " ".join(str(x or "") for x in (t.text, t.counterparty_name, t.remittance_info, t.ref2))
        words = _words(blob)
        digits = re.sub(r"\D", " ", blob).split()

        def score(i):
            p = pool.loc[i]
            nm = _words(p.party_name) - {"LTD", "LLC", "INC", "PTY", "CORP", "GROUP", "PLUS"}
            name_hit = len(nm & words) >= min(2, len(nm)) if nm else False
            dn = p.document_number_norm
            ref_hit = any(len(x) >= 5 and (dn.endswith(x.lstrip("0")) or dn[-7:].startswith(x.lstrip("0")[:6]))
                          for x in digits)
            return 2 * ref_hit + name_hit

        sc = sorted(((score(i), i) for i in ids), reverse=True)
        best = sc[0]
        if best[0] == 0 or (len(sc) > 1 and sc[1][0] == best[0]):
            continue
        # veto: the text cites another document of ours with this amount
        cited = set()
        for tok in list(t.doc_tokens) + re.findall(r"RECIBO\D{0,3}(\d{5,})", blob):
            cited |= {c for c in res.ref_candidates(t.entity, tok) if abs(res.tot.get(c, -1) - t.amount) < 0.006}
        if cited and best[1] not in cited:
            continue
        used.add(best[1])
        out.append((t.bank_txn_id, best[1], t.amount, "", 0.85, NAME + "_transfer"))
    return out


def _later_signal(ctx, res, t, gb: pd.DataFrame, amt: float, rep, coll_on, credits) -> pd.DataFrame:
    """Among several earlier collections of the same mandate and amount, keep the ones whose receipt shows up again
    after the return: re-presented in a later batch, collected by a later batch line, or cited by a later transfer."""
    keep = []
    later_credits = [toks for dt, toks in credits.get((t.entity, amt), ()) if dt > t.booking_date]
    for i, r in enumerate(gb.itertuples(index=False)):
        sig = any(c > t.booking_date for c in rep.get((r.mandate_id, r.remittance_info), ()))
        iid, _ = res.resolve(t.entity, r.remittance_info, amt, res.parties_for(t.entity, mandate=r.mandate_id))
        if iid and not sig:
            sig = any(dt > t.booking_date for dt in coll_on.get(iid, ()))
        if iid and not sig:
            sig = any(iid in res.ref_candidates(t.entity, tok) for toks in later_credits for tok in toks)
        if sig:
            keep.append(i)
    return gb.iloc[keep] if keep else gb


def run(ctx: Ctx):
    d = _batches(ctx)
    if d.empty:
        return alloc_frame([])
    res = get_resolver(ctx)
    rows: list[tuple] = []

    # ---- 1. batch credits -------------------------------------------------------------------------------------------
    link = _link_batches(ctx, d)
    ent_of_line = dict(zip(ctx.bank.bank_txn_id, ctx.bank.entity))
    for bid, grp in d[d.batch_id.isin(link)].groupby("batch_id", sort=False):
        tid = link[bid]
        ent = ent_of_line.get(tid) or grp.account_id.iloc[0][:4]
        used: set[str] = set()
        got = []
        for r in grp.itertuples(index=False):
            parties = res.parties_for(ent, mandate=r.mandate_id, name=r.debtor_name)
            iid, how = res.resolve(ent, r.remittance_info, r.amount, parties, used)
            if iid:
                used.add(iid)
                got.append((tid, iid, r.amount, "", 0.97 if how == "ref" else 0.9, NAME))
        if got and len(got) >= MIN_RESOLVED * len(grp):
            rows += got

    # ---- 2. returns -------------------------------------------------------------------------------------------------
    bank = ctx.unclaimed()
    deb = bank[(bank.amount < 0) & bank.text.fillna("").str.contains(MANDATE)]
    deb = deb[~deb.text.str.contains(NOT_RETURN)]
    by_mandate = dict(tuple(d.sort_values("collection_date", kind="stable").groupby("mandate_id")))
    # receipts re-presented later: (mandate, remittance_info) -> collection dates of the re-presentations
    rep = d[d.re_presentation == "Y"].groupby(["mandate_id", "remittance_info"]).collection_date.agg(list).to_dict()
    claimed_ret: set[str] = set()
    day = dict(zip(ctx.bank.bank_txn_id, ctx.bank.booking_date))
    coll_on: dict[str, list[str]] = {}   # item -> booking dates of the batch lines of this phase that collected it
    for tid, iid, *_ in rows:
        coll_on.setdefault(iid, []).append(day[tid])
    cr = bank[bank.amount > 0]
    credits: dict[tuple, list] = {}
    for c in cr[["entity", "amount", "booking_date", "doc_tokens"]].itertuples(index=False):
        credits.setdefault((c.entity, round(c.amount, 2)), []).append((c.booking_date, c.doc_tokens))
    for t in deb.itertuples(index=False):
        m = MANDATE.search(t.text).group(0)
        g = by_mandate.get(m)
        if g is None:
            continue
        amt = round(-t.amount, 2)
        g = g[(abs(g.amount - amt) < 0.006) & (g.collection_date < t.booking_date) & ~g.end_to_end_id.isin(claimed_ret)]
        if g.empty:
            continue
        blob = " ".join(str(x or "") for x in (t.text, t.ref1, t.ref2, t.end_to_end_id, t.remittance_info))
        ids = set(re.findall(r"\d{11}", blob))
        gb = g[g.batch_id.isin(ids)]
        how = "batch_id"
        if gb.empty:
            code = RCODE.search(t.text)
            lo, hi = (14, 80) if code and code.group(1) == "MD06" else (1, 20)
            bd = pd.Timestamp(t.booking_date)
            lag = (bd - pd.to_datetime(g.collection_date)).dt.days
            gb = g[(lag >= lo) & (lag <= hi)]
            how = "window"
            if len(gb) > 1:   # several collections of the same amount: the returned one is collected / paid again
                gb = _later_signal(ctx, res, t, gb, amt, rep, coll_on, credits) if len(gb) > 1 else gb
            if gb.empty:
                continue
        r = gb.iloc[-1]   # latest earlier collection of that amount
        ent = t.entity
        parties = res.parties_for(ent, mandate=m, name=r.debtor_name)
        iid, _ = res.resolve(ent, r.remittance_info, amt, parties)
        if not iid:
            continue
        claimed_ret.add(r.end_to_end_id)
        conf = 0.97 if how == "batch_id" or len(gb) == 1 else 0.8
        rows.append((t.bank_txn_id, iid, amt, "RETURNED", conf, NAME))

    # ---- 3. a returned receipt paid afterwards by transfer -----------------------------------------------------------
    rows += _paid_after_return(ctx, d, res, rows, link)
    return alloc_frame(rows)
