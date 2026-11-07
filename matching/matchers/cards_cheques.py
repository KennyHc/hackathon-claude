"""Card acquirer settlements, cheque / lockbox deposits, card chargebacks and returned cheques.

* Card settlement: bank credit on the settlement account, amount == Σ net of the settlement (merchant id / date in the
  text), clears every order at GROSS (the acquirer fee is contractual: no exception).
* Cheque deposit: bank credit == deposit_total (deposit id in the text) clears each cheque's invoice
  (stub reference — often truncated or missing — + drawer + amount).
* Chargeback ('CHARGEBACK ****1234 MID …'): debit == gross of an earlier card operation of that card / merchant ->
  RETURNED on its order's invoice.  Returned cheque ('RETURNED ITEM CHK n' / 'CHEQUE IMPAYE n') -> RETURNED on the
  invoice(s) of that cheque.
"""
from __future__ import annotations

import re

import pandas as pd

from matching.batch_helpers import Resolver, get_resolver, side_files
from matching.common import Ctx, alloc_frame

NAME = "cards_cheques"
MIN_RESOLVED = 0.9


def _link(ctx: Ctx, heads: pd.DataFrame, id_col: str, total_col: str, date_col: str, lo: int, hi: int,
          extra_ids=None) -> dict[str, str]:
    """side-file group id -> bank_txn_id (credit on the same account, amount == total, date within [lo, hi] days)."""
    bank = ctx.unclaimed()
    cr = bank[bank.amount > 0]
    key: dict[tuple, list] = {}
    for t in cr[["bank_txn_id", "account_id", "amount", "booking_date"]].itertuples(index=False):
        key.setdefault((t.account_id, round(t.amount, 2)), []).append((t.booking_date, t.bank_txn_id))
    blob = dict(zip(cr.bank_txn_id, cr.text.fillna("") + " " + cr.ref1.fillna("") + " " + cr.ref2.fillna("")))
    out: dict[str, str] = {}
    used: set[str] = set()
    for h in heads.itertuples(index=False):
        gid, acc, tot, dt = getattr(h, id_col), h.account_id, getattr(h, total_col), getattr(h, date_col)
        cands = key.get((acc, tot), [])
        if not cands:
            continue
        d0 = pd.Timestamp(dt)
        near = [(abs((pd.Timestamp(bd) - d0).days), tid) for bd, tid in cands
                if lo <= (pd.Timestamp(bd) - d0).days <= hi and tid not in used]
        if not near:
            continue
        ids = [gid] + list((extra_ids or {}).get(gid, ()))
        with_id = [x for x in near if any(i and i in blob[x[1]] for i in ids)]
        near.sort()
        if not with_id and len(near) > 1 and near[0][0] == near[1][0]:
            continue
        pick = min(with_id or near)
        out[gid] = pick[1]
        used.add(pick[1])
    return out


def _cards(ctx: Ctx, res: Resolver, rows: list):
    cs = side_files(ctx, "card_settlements")
    if cs.empty:
        return cs
    for c in ("gross", "fee", "net"):
        cs[c] = cs[c].astype(float).round(2)
    heads = cs.groupby("settlement_id", sort=False).agg(
        account_id=("account_id", "first"), settlement_date=("settlement_date", "first"),
        merchant_id=("merchant_id", "first"), total=("net", "sum")).reset_index()
    heads["total"] = heads.total.round(2)
    link = _link(ctx, heads, "settlement_id", "total", "settlement_date", -1, 6,
                 extra_ids=dict(zip(heads.settlement_id, zip(heads.merchant_id))))
    ent_of = dict(zip(ctx.bank.bank_txn_id, ctx.bank.entity))
    for sid, grp in cs[cs.settlement_id.isin(link)].groupby("settlement_id", sort=False):
        tid = link[sid]
        ent = ent_of.get(tid) or grp.account_id.iloc[0][:4]
        used: set[str] = set()
        got = []
        for r in grp.itertuples(index=False):
            iid, how = res.resolve(ent, r.order_number, r.gross, None, used)
            if iid:
                used.add(iid)
                got.append((tid, iid, r.gross, "", 0.97, NAME + "_card"))
        if got and len(got) >= MIN_RESOLVED * len(grp):
            rows += got
    return cs


def _cheques(ctx: Ctx, res: Resolver, rows: list):
    ch = side_files(ctx, "cheque_deposits")
    if ch.empty:
        return ch
    ch["amount"] = ch.amount.astype(float).round(2)
    ch["deposit_total"] = ch.deposit_total.astype(float).round(2)
    heads = ch.drop_duplicates("deposit_id")[["deposit_id", "account_id", "deposit_total", "deposit_date"]]
    link = _link(ctx, heads, "deposit_id", "deposit_total", "deposit_date", -1, 6)
    ent_of = dict(zip(ctx.bank.bank_txn_id, ctx.bank.entity))
    for did, grp in ch[ch.deposit_id.isin(link)].groupby("deposit_id", sort=False):
        tid = link[did]
        ent = ent_of.get(tid) or grp.account_id.iloc[0][:4]
        used: set[str] = set()
        got = []
        for r in grp.itertuples(index=False):
            parties = res.parties_for(ent, name=r.drawer)
            iid, how = res.resolve(ent, r.stub_reference, r.amount, parties, used)
            if iid:
                used.add(iid)
                got.append((tid, iid, r.amount, "", 0.95, NAME + "_cheque"))
        if got and len(got) >= MIN_RESOLVED * len(grp):
            rows += got
    return ch


CARD4 = re.compile(r"\*{2,}(\d{4})")
CARD4_ANY = r"\*{2,}\d{4}"
MID = re.compile(r"(?:MID|COMERC)\s*(\d{6,})")
CHK_TXT = r"(?:RETURNED ITEM CHK|CHEQUE IMPAYE|CHQ IMPAYE|CHEQUE DEVUELTO|RETURNED CHECK)\D{0,5}\d{5,}"
CHK = re.compile(r"(?:RETURNED ITEM CHK|CHEQUE IMPAYE|CHQ IMPAYE|CHEQUE DEVUELTO|RETURNED CHECK)\D{0,5}(\d{5,})")


def _returns(ctx: Ctx, res: Resolver, cs: pd.DataFrame, ch: pd.DataFrame, rows: list):
    bank = ctx.unclaimed()
    deb = bank[bank.amount < 0]
    txt = deb.text.fillna("")
    taken: set[tuple] = set()
    # --- card chargebacks
    if not cs.empty:
        cb = deb[txt.str.contains(r"CHARGEBACK|CONTRACARGO|RETROCESSION|IMPAYE CB", regex=True) & txt.str.contains(CARD4_ANY, regex=True)]
        for t in cb.itertuples(index=False):
            card = "****" + CARD4.search(t.text).group(1)
            mid = MID.search(t.text)
            amt = round(-t.amount, 2)
            c = cs[(cs.card == card) & (cs.account_id.str[:4] == t.entity) & (cs.settlement_date < t.booking_date)]
            if mid:
                c = c[c.merchant_id == mid.group(1)]
            c = c[(abs(c.gross - amt) < 0.006) | (abs(c.net - amt) < 0.006)]
            c = c[[(r.settlement_id, r.order_number) not in taken for r in c.itertuples(index=False)]]
            if len(c) != 1:
                continue
            r = c.iloc[0]
            iid, _ = res.resolve(t.entity, r.order_number, r.gross)
            if iid:
                taken.add((r.settlement_id, r.order_number))
                rows.append((t.bank_txn_id, iid, float(r.gross), "RETURNED", 0.95, NAME + "_chargeback"))
    # --- returned cheques
    if not ch.empty:
        rc = deb[txt.str.contains(CHK_TXT, regex=True)]
        for t in rc.itertuples(index=False):
            num = CHK.search(t.text).group(1)
            amt = round(-t.amount, 2)
            c = ch[(ch.cheque_number.str.lstrip("0") == num.lstrip("0")) & (ch.account_id.str[:4] == t.entity)
                   & (ch.deposit_date <= t.booking_date)]
            c = c[abs(c.amount - amt) < 0.006]
            if c.empty:
                continue
            r = c.sort_values("deposit_date").iloc[-1]
            iid, _ = res.resolve(t.entity, r.stub_reference, r.amount, res.parties_for(t.entity, name=r.drawer))
            if iid:
                rows.append((t.bank_txn_id, iid, amt, "RETURNED", 0.95, NAME + "_chq_returned"))


def run(ctx: Ctx):
    res = get_resolver(ctx)
    rows: list[tuple] = []
    cs = _cards(ctx, res, rows)
    ch = _cheques(ctx, res, rows)
    claimed = {r[0] for r in rows}
    ctx_claimed = ctx.claimed
    ctx.claimed = ctx_claimed | claimed
    try:
        _returns(ctx, res, cs, ch, rows)
    finally:
        ctx.claimed = ctx_claimed
    return alloc_frame(rows)
