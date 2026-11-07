"""Customer pays an invoice that is already settled -> DUPLICATE_PAYMENT (POLICIES §3).

Runs LAST: a credit still unclaimed after every other matcher, that repeats an earlier payment of this phase — same
entity, same amount, same payer, 1–14 days later — and that earlier line settled exactly one AR invoice in full
(`ctx.allocations`). The later line gets the item with amount = what was paid again and DUPLICATE_PAYMENT.

Guards (precision first): the payer of the repeat must be the invoice's customer (counterparty name / text words),
a document number cited in the text must be that invoice, and the customer must have no other open invoice of the same
amount still unallocated (that would be a normal payment of a fixed monthly fee, not a duplicate).
"""
from __future__ import annotations

import re

import pandas as pd

from ingestion import lib, text
from matching.common import Ctx, alloc_frame

NAME = "duplicates"
MAX_LAG = 14       # customer pays again within two weeks
MAX_LAG_AP = 70    # we pay a supplier invoice again in a later payment cycle
ACC_CODE = re.compile(r"\b(\d{4}-\d{2}-\d{7})-\d{2}\b")
STOP = {"LTD", "LLC", "INC", "PTY", "CORP", "GROUP", "PLUS", "PRO", "CO", "SL", "SA", "GMBH", "SAS", "SARL", "SRL",
        "SPA", "LDA", "BV", "THE", "AND", "DE", "DEL", "LA", "Y", "SP", "ZOO", "SCOOP", "SLU", "EURL"}


def _words(s: str) -> set[str]:
    return {w for w in re.split(r"[^A-Z0-9]+", text.strip_accents((s or "").upper())) if len(w) >= 3 and w not in STOP}


def _allocs(ctx: Ctx) -> pd.DataFrame:
    a = getattr(ctx, "allocations", None)
    if a is None or len(a) == 0:
        return pd.DataFrame(columns=["bank_txn_id", "item_id", "amount", "exception"])
    return a


def run(ctx: Ctx):
    a = _allocs(ctx)
    if a.empty:
        return alloc_frame([])
    bank = ctx.bank
    bk = bank.set_index("bank_txn_id")
    items = ctx.items_by_id
    # earlier lines that settled exactly one invoice (AR receipt / AP payment), cash == allocated amount
    a = a[a.item_id.notna() & (a.item_id != "")]
    n = a.groupby("bank_txn_id").item_id.transform("size")
    one = a[(n == 1) & a.item_id.isin(items.index)].copy()
    one = one[one.bank_txn_id.isin(bk.index)]
    one["entity"] = one.bank_txn_id.map(bk.entity)
    one["date"] = one.bank_txn_id.map(bk.booking_date)
    one["cash"] = one.bank_txn_id.map(bk.amount).round(2)
    one["side"] = one.item_id.map(items.side)
    one = one[(abs(one.cash.abs() - one.amount) < 0.006)
              & (((one.side == "AR") & (one.cash > 0)) | ((one.side == "AP") & (one.cash < 0)))]
    one = one[one.item_id.map(items.item_type).isin(["ARI", "API"])]
    prior: dict[tuple, list] = {}
    for r in one[["bank_txn_id", "item_id", "entity", "date", "cash"]].itertuples(index=False):
        prior.setdefault((r.entity, r.cash), []).append((r.date, r.bank_txn_id, r.item_id))

    # allocated amount per item (to know which open invoices are still unallocated)
    signed = a.amount.where(a.exception.fillna("") != "RETURNED", -a.amount)
    used = signed.groupby(a.item_id).sum()
    op = ctx.open_items
    free = op[op.open_amount - op.item_id.map(used).fillna(0) > 0.005]
    free_key = set(zip(free.party_id, free.open_amount.round(2)))

    claimed = ctx.claimed | set(a.bank_txn_id)
    cand = bank[~bank.bank_txn_id.isin(claimed)]
    rows = []
    dup_of: set[str] = set()
    for t in cand.itertuples(index=False):
        amt = round(t.amount, 2)
        hits = []
        for d0, tid0, iid in prior.get((t.entity, amt), ()):
            lag = (pd.Timestamp(t.booking_date) - pd.Timestamp(d0)).days
            if 0 <= lag <= (MAX_LAG if amt > 0 else MAX_LAG_AP) and tid0 != t.bank_txn_id:
                hits.append((lag, tid0, iid))
        if not hits:
            continue
        blob = " ".join(str(x or "") for x in (t.text, t.counterparty_name, t.remittance_info))
        words = _words(blob)
        ok = []
        for lag, tid0, iid in hits:
            it = items.loc[iid]
            e0 = bk.loc[tid0]
            same_party = bool(t.party_name_norm) and t.party_name_norm == e0.party_name_norm
            nm = _words(it.party_name)
            name_hit = bool(nm) and len(nm & words) >= min(2, len(nm))
            # same payer text as the first payment (first 3 words after the bank's prefix)
            w0 = _words(" ".join(str(x or "") for x in (e0.text, e0.counterparty_name)))
            txt_hit = len(words & w0 & nm) >= 1 if nm else False
            # 'PAGO TRANSFERENCIAS | 4799-56-5859790-82 …': same ordering-account code as the first payment
            code = ACC_CODE.search(t.text or "")
            if code and code.group(1) in (e0.text or ""):
                txt_hit = True
            if not (same_party or name_hit or txt_hit):
                continue
            ok.append((lag, tid0, iid))
        ok = list({x[2]: x for x in ok}.values())
        if len(ok) != 1:
            continue
        lag, tid0, iid = ok[0]
        it = items.loc[iid]
        # cited documents must include this invoice
        cited = set()
        for tok in t.doc_tokens:
            cited |= set(ctx.by_alnum.get(tok, ()))
        for ref in t.doc_refs:
            cited |= set(ctx.by_docnorm.get(ref, ()))
        cited = {c for c in cited if c in items.index and items.at[c, "side"] == it.side and items.at[c, "entity"] == t.entity}
        if cited and iid not in cited:
            continue
        # a still-open invoice of the same customer and amount -> a normal payment, not a duplicate
        if (it.party_id, abs(amt)) in free_key:
            continue
        # the earlier line(s) must have cleared the item in full (not an instalment / partial payment)
        if used.get(iid, 0.0) < float(it.open_amount) - 0.005:
            continue
        if iid in dup_of:
            continue
        dup_of.add(iid)
        rows.append((t.bank_txn_id, iid, abs(amt), "DUPLICATE_PAYMENT", 0.85 if not cited else 0.92,
                     NAME if amt > 0 else NAME + "_ap"))
    rows += _settled_before_phase(ctx, claimed | {r[0] for r in rows})
    return alloc_frame(rows)


def _settled_before_phase(ctx: Ctx, claimed: set[str]) -> list[tuple]:
    """Early in the phase: a repeat of a payment made just before the phase. Its invoice is not in ctx.items any more
    (settled). Needs the invoice cited in the text (number or digits) + same entity + exact amount + payer name."""
    bank = ctx.bank
    start = bank.booking_date.min()
    until = (pd.Timestamp(start) + pd.Timedelta(days=MAX_LAG + 2)).strftime("%Y-%m-%d")
    cand = bank[(bank.amount > 0) & (bank.booking_date <= until) & ~bank.bank_txn_id.isin(claimed)]
    if cand.empty:
        return []
    pool = settled_items(ctx)
    by_alnum = pool.groupby(["entity", "document_alnum"]).item_id.agg(list).to_dict()
    by_dn = pool.groupby(["entity", "document_number_norm"]).item_id.agg(list).to_dict()
    P = pool.set_index("item_id")
    rows = []
    seen: set[str] = set()
    for t in cand.itertuples(index=False):
        amt = round(t.amount, 2)
        cited = set()
        for tok in t.doc_tokens:
            cited |= set(by_alnum.get((t.entity, tok), ()))
        for ref in t.doc_refs:
            if len(ref) >= 6:
                cited |= set(by_dn.get((t.entity, ref), ()))
        ok = [i for i in cited if abs(P.at[i, "total_amount"] - amt) < 0.006 and i not in seen]
        if len(ok) != 1:
            continue
        iid = ok[0]
        # any open item of ours also cited -> it is a normal payment of that one
        if any(iid2 in ctx.items_by_id.index for tok in t.doc_tokens for iid2 in ctx.by_alnum.get(tok, ())):
            continue
        words = _words(" ".join(str(x or "") for x in (t.text, t.counterparty_name, t.remittance_info)))
        nm = _words(P.at[iid, "party_name"])
        if not nm or len(nm & words) < min(2, len(nm)):
            continue
        seen.add(iid)
        rows.append((t.bank_txn_id, iid, amt, "DUPLICATE_PAYMENT", 0.85, NAME + "_pre"))
    return rows


def settled_items(ctx: Ctx) -> pd.DataFrame:
    """AR invoices issued in the year before the phase that are no longer open at its start."""
    folders = lib.PHASES[ctx.phase][3]
    start = ctx.bank.booking_date.min()
    since = (pd.Timestamp(start) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    it = pd.concat([lib.items_of(f) for f in folders], ignore_index=True)
    it = it[(it.item_type == "ARI") & (it.issue_date >= since) & (it.issue_date < start)]
    it = it[~it.item_id.isin(set(ctx.items.item_id))].drop_duplicates("item_id", keep="last").copy()
    it["document_alnum"] = text.alnum_series(it.document_number.fillna(""))
    it["document_number_norm"] = text.doc_number_norm_series(it.document_number.fillna(""))
    it["total_amount"] = it.total_amount.astype(float).round(2)
    return it
