"""Remittance advices → bank receipts. Advice ↔ line: same entity, amount = advice total, booking date close to the
advice payment date, payment_ref in the bank text when several candidates. Advice documents → AR items."""
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from matching.batch.remit_io import parse_remittances

DISC_RE = re.compile(r"DISC|DTO|DESC|SKONTO|ESCOMPTE|SCONTO|PRONTO|REMISE|RABAT|EARLY", re.I)
WHT_RE = re.compile(r"^RET|WITHH|RETEN|IRPF|ISR|IRRF|TDS", re.I)


def _norm(s):
    return re.sub(r"[^0-9A-Z]", "", str(s).upper())


def _digits(s):
    return re.sub(r"\D", "", str(s))


def resolve_docs(H, D, items, phase=""):
    from integration.lib.masters import read_master
    ar = items[items.item_type.isin(["ARI", "ARC"])]
    by_doc, by_norm, by_party = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in ar[["item_id", "item_type", "entity", "document_number", "party_id", "party_name", "open_amount",
                 "total_amount", "po_number"]].itertuples(index=False):
        by_doc[(r.entity, r.document_number)].append(r)
        by_norm[(r.entity, _norm(r.document_number))].append(r)
        by_party[r.party_id].append(r)
    cust = read_master("customers.csv", phase)
    name2c = defaultdict(set)
    for c, n, t, e in zip(cust.customer_id, cust.name, cust.trade_name, cust.entity):
        name2c[(e, _norm(n))].add(c)
        if t:
            name2c[(e, _norm(t))].add(c)
    for e, n, p in zip(ar.entity, ar.party_name, ar.party_id):
        name2c[(e, _norm(n))].add(p)
    # group members: parent pays for subsidiaries
    grp = defaultdict(set)
    for c, g in zip(cust.customer_id, cust.group_id):
        if g:
            grp[g].add(c)
    c2g = dict(zip(cust.customer_id, cust.group_id))
    ent = dict(zip(H.file, H.entity))
    payer = dict(zip(H.file, H.payer_name))
    ids = []
    cand_cache = {}
    for f, doc, g in zip(D.file, D.document, D.gross):
        e = ent[f]
        want = "ARC" if g < 0 else "ARI"
        c = [r for r in by_doc.get((e, doc), []) if r.item_type == want]
        if not c:
            c = [r for r in by_norm.get((e, _norm(doc)), []) if r.item_type == want]
        if len(c) > 1:
            ps = name2c.get((e, _norm(payer[f])), set())
            c2 = [r for r in c if r.party_id in ps]
            c = c2 or c
        if len(c) > 1:
            c2 = [r for r in c if abs(r.total_amount - abs(g)) < 0.01 or abs(r.open_amount - abs(g)) < 0.01]
            c = c2 or c
        if len(c) == 1:
            ids.append(c[0].item_id)
            continue
        # fallback: payer's (group's) items whose number ends with the advice digits / PO = advice doc
        key = (f, want)
        cand = cand_cache.get(key)
        if cand is None:
            ps = set(name2c.get((e, _norm(payer[f])), set()))
            for p in list(ps):
                if c2g.get(p) and len(grp[c2g[p]]) <= 30:
                    ps |= grp[c2g[p]]
            cand = [r for p in ps for r in by_party.get(p, []) if r.item_type == want and r.entity == e]
            cand = (cand, {}, {})
            for r in cand[0]:
                dg = _digits(r.document_number)
                for k in range(5, len(dg) + 1):
                    cand[1].setdefault(dg[-k:], []).append(r)
                if r.po_number:
                    cand[2].setdefault(r.po_number, []).append(r)
            cand_cache[key] = cand
        d = _digits(doc)
        c = list(cand[2].get(doc, []))
        if not c and len(d) >= 5:
            c = list(cand[1].get(d, []))
        if len(c) > 1:
            c = [r for r in c if abs(r.total_amount - abs(g)) < 0.01 or abs(r.open_amount - abs(g)) < 0.01]
        ids.append(c[0].item_id if len(c) == 1 else "")
    D = D.copy()
    D["item_id"] = ids
    return D


def match(lines, items, ctx=None):
    phase = (ctx or {}).get("phase")
    H, D = parse_remittances(phase)
    if H.empty:
        return []
    D = resolve_docs(H, D, items, phase)
    item_open = dict(zip(items.item_id, items.open_amount))
    item_ccy = dict(zip(items.item_id, items.currency))
    rec = lines[lines.amount.astype(float) > 0].copy()
    rec["amt"] = rec.amount.astype(float).round(2)
    rec["oamt"] = pd.to_numeric(rec.orig_amount, errors="coerce").round(2)
    rec["blob"] = (rec.ref1 + " " + rec.ref2 + " " + rec.text + " " + rec.remittance_info + " " +
                   rec.end_to_end_id).map(_norm)
    by_amt = defaultdict(list)
    for r in rec[["bank_txn_id", "entity", "booking_date", "amt", "oamt", "blob", "currency"]].itertuples(
            index=False):
        by_amt[(r.entity, r.amt)].append(r)
        if r.oamt == r.oamt:
            by_amt[(r.entity, r.oamt)].append(r)
    docs_by_file = {f: d for f, d in D.groupby("file", sort=False)}
    links = []
    for h in H.itertuples(index=False):
        if h.total != h.total or h.total <= 0 or h.file not in docs_by_file:
            continue
        pd_ = h.payment_date or h.received_date
        lo = (pd.Timestamp(pd_) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        hi = (pd.Timestamp(pd_) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        cands = [r for r in by_amt.get((h.entity, round(h.total, 2)), []) if lo <= r.booking_date <= hi]
        ref = _norm(h.payment_ref)
        withref = [r for r in cands if ref and (ref in r.blob or (len(ref) >= 8 and ref[:10] in r.blob))]
        if len(withref) == 1:
            links.append((h.file, withref[0].bank_txn_id, withref[0].currency, 0.95, "remit_ref"))
        elif len(cands) == 1 and not withref:
            links.append((h.file, cands[0].bank_txn_id, cands[0].currency, 0.8, "remit_amt"))
    L = pd.DataFrame(links, columns=["file", "bank_txn_id", "ccy", "conf", "method"])
    # one advice per line and one line per advice
    L = L[~L.bank_txn_id.duplicated(keep=False)]
    from integration.lib.masters import read_master
    pt = read_master("payment_terms.csv", phase)
    terms = {c: (float(p or 0), int(float(d or 0))) for c, p, d in
             zip(pt.code, pt.early_payment_discount_pct, pt.discount_days)}
    item_terms = dict(zip(items.item_id, items.payment_terms))
    item_issue = dict(zip(items.item_id, items.issue_date))
    bdate = dict(zip(lines.bank_txn_id, lines.booking_date))

    def discount_ok(it, g, paid, bd):
        pct, days = terms.get(item_terms.get(it, ""), (0.0, 0))
        if pct <= 0 or not item_issue.get(it):
            return False
        lim = (pd.Timestamp(item_issue[it]) + pd.Timedelta(days=days + 3)).strftime("%Y-%m-%d")
        return bd <= lim and abs(round(g - g * pct / 100.0, 2) - paid) <= 0.02

    rows = []
    for f, bt, ccy, conf, meth in L.itertuples(index=False):
        d = docs_by_file[f]
        if (d.item_id == "").any() or d.item_id.duplicated().any():
            continue
        bd = bdate.get(bt, "")
        for it, g, ded, reason, paid in zip(d.item_id, d.gross, d.deduction, d.reason, d.paid):
            g = abs(g)
            if it.startswith("ARC"):
                rows.append((bt, it, g, "", conf, meth))
                continue
            exc, amt = "", g
            short = g - paid if paid == paid else 0.0
            if short > 0.005:
                rate = short / g if g else 0
                if (reason and WHT_RE.search(reason)) or (not (reason and DISC_RE.search(reason))
                                                         and any(abs(rate - w) < 0.0006 for w in (0.0125, 0.015, 0.10))
                                                         and not (reason and not WHT_RE.search(reason))):
                    amt = g                                    # withholding: fully cleared
                elif reason and not DISC_RE.search(reason):
                    exc, amt = "SHORT_DISPUTE", paid           # stated dispute reason
                elif discount_ok(it, g, paid, bd):
                    amt = g                                    # discount within terms → SHORT_DISCOUNT downstream
                elif reason or ded > 0.005:
                    exc, amt = "PARTIAL", paid                 # late discount / unexplained deduction
                else:
                    exc, amt = "PARTIAL", paid                 # paid less than the document, no reason
            if item_ccy.get(it, ccy) != ccy and exc == "":
                exc = "FX_DIFFERENCE"
            rows.append((bt, it, amt, exc, conf, meth))
    return rows
