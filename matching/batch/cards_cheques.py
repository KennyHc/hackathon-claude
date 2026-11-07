"""Card settlements (acquirer → Σnet credit; items cleared at gross), chargebacks (RETURNED), cheque / lockbox
deposits (deposit_id in text) and returned cheques (cheque number in text)."""
import re
from collections import defaultdict

import pandas as pd

from matching.batch.common import load_detail

DEP_RE = re.compile(r"(RCH\d{11}|[A-Z]{2,4}\d{10,14})")
MID_RE = re.compile(r"(?<!\d)(\d{9})(?!\d)")
CARD_RE = re.compile(r"\*{2,4}(\d{4})")
CHK_RE = re.compile(r"(?:CHK|CHQ|CHEQUE|CHÈQUE|SCHECK|ASSEGNO)\D{0,12}(\d{6,8})")


def _digits(s):
    return re.sub(r"\D", "", str(s))


def _ar_index(items):
    ar = items[items.item_type == "ARI"]
    by_doc, by_party, by_ent_amt = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in ar[["item_id", "entity", "document_number", "party_id", "party_name", "open_amount",
                 "total_amount"]].itertuples(index=False):
        by_doc[(r.entity, r.document_number)].append(r)
        by_party[r.party_id].append(r)
    return by_doc, by_party


def match_cards(lines, items, phase):
    cs = load_detail(phase, "card_settlements")
    if cs.empty:
        return []
    by_doc, _ = _ar_index(items)
    cs["ent"] = cs.account_id.str[:4]
    cs["item_id"] = [(lambda c: c[0].item_id if len(c) == 1 else "")(by_doc.get((e, o), []))
                     for e, o in zip(cs.ent, cs.order_number)]
    cs["g"], cs["n"] = cs.gross.astype(float), cs.net.astype(float)
    S = {}
    by_acc_net = defaultdict(list)
    for sid, d in cs.groupby(["account_id", "settlement_id"], sort=False):
        S[sid] = d
        by_acc_net[(sid[0], round(d.n.sum(), 2))].append(sid)
    by_card = defaultdict(list)
    for r in cs[["merchant_id", "card", "g", "item_id", "settlement_date", "account_id"]].itertuples(index=False):
        by_card[(r.merchant_id, r.card[-4:], round(r.g, 2))].append(r)
    rows, used = [], set()
    for r in lines.itertuples(index=False):
        amt = float(r.amount)
        blob = " ".join([r.ref1, r.ref2, r.text, r.remittance_info])
        if amt > 0:
            cands = [s for s in by_acc_net.get((r.account_id, round(amt, 2)), []) if s not in used]
            if not cands:
                continue
            mids = set(MID_RE.findall(blob))
            c2 = [s for s in cands if S[s].merchant_id.iloc[0] in mids]
            if mids and c2:
                cands = c2
            elif mids:
                continue
            lo = (pd.Timestamp(r.booking_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            cands = [s for s in cands if lo <= S[s].settlement_date.iloc[0] <= r.booking_date]
            if len(cands) != 1:
                continue
            D = S[cands[0]]
            if (D.item_id == "").any() or D.item_id.duplicated().any():
                continue
            used.add(cands[0])
            for it, a in zip(D.item_id, D.g):
                rows.append((r.bank_txn_id, it, a, "", 0.95, "card_settle"))
        elif amt < 0 and "CHARGEBACK" in blob.upper() or (amt < 0 and "RETROCESSION" in blob.upper()):
            cm, mm = CARD_RE.search(blob), MID_RE.findall(blob)
            if not cm:
                continue
            c = [x for mid in mm for x in by_card.get((mid, cm.group(1), round(-amt, 2)), [])
                 if x.item_id and x.settlement_date <= r.booking_date]
            if len({x.item_id for x in c}) == 1:
                rows.append((r.bank_txn_id, c[0].item_id, -amt, "RETURNED", 0.9, "card_chargeback"))
    return rows


def match_cheques(lines, items, phase):
    from integration.lib.masters import read_master
    cq = load_detail(phase, "cheque_deposits")
    if cq.empty:
        return []
    by_doc, by_party = _ar_index(items)
    cust = read_master("customers.csv", phase)
    name2c = defaultdict(set)
    for c, n, t in zip(cust.customer_id, cust.name, cust.trade_name):
        name2c[n.upper()].add(c)
        if t:
            name2c[t.upper()].add(c)
    ar = items[items.item_type == "ARI"]
    by_ent_name = defaultdict(set)
    for e, n, p in zip(ar.entity, ar.party_name, ar.party_id):
        by_ent_name[(e, n.upper())].add(p)
    ids = []
    for ent, stub, drawer, amt in zip(cq.account_id.str[:4], cq.stub_reference, cq.drawer, cq.amount.astype(float)):
        ok = lambda r: abs(r.open_amount - amt) < 0.01 or abs(r.total_amount - amt) < 0.01
        c = [r for r in by_doc.get((ent, stub), []) if ok(r)]
        if len(c) == 1:
            ids.append(c[0].item_id); continue
        parties = by_ent_name.get((ent, drawer.upper()), set()) | {p for p in name2c.get(drawer.upper(), set())}
        cand = [r for p in parties for r in by_party.get(p, []) if r.entity == ent and ok(r)]
        d = _digits(stub)
        c = [r for r in cand if d and _digits(r.document_number).endswith(d)]
        if len(c) == 1:
            ids.append(c[0].item_id); continue
        c = cand if len(cand) == 1 else []
        ids.append(c[0].item_id if c else "")
    cq = cq.copy()
    cq["item_id"] = ids
    cq["amt"] = cq.amount.astype(float)
    D = {k: d for k, d in cq.groupby("deposit_id", sort=False)}
    by_chk = defaultdict(list)
    for r in cq[["cheque_number", "amt", "item_id", "deposit_date", "account_id"]].itertuples(index=False):
        by_chk[(r.cheque_number.lstrip("0"), round(r.amt, 2))].append(r)
    rows = []
    for r in lines.itertuples(index=False):
        amt = float(r.amount)
        blob = " ".join([r.ref1, r.ref2, r.text, r.remittance_info])
        if amt > 0:
            for dep in DEP_RE.findall(blob):
                d = D.get(dep)
                if d is None or d.account_id.iloc[0] != r.account_id:
                    continue
                if abs(d.amt.sum() - amt) > 0.01 or (d.item_id == "").any() or d.item_id.duplicated().any():
                    break
                for it, a in zip(d.item_id, d.amt):
                    rows.append((r.bank_txn_id, it, a, "", 0.93, "cheque_dep"))
                break
        elif amt < 0:
            m = CHK_RE.search(blob.upper())
            if not m:
                continue
            c = [x for x in by_chk.get((m.group(1).lstrip("0"), round(-amt, 2)), []) if x.item_id
                 and x.deposit_date <= r.booking_date]
            if len({x.item_id for x in c}) == 1:
                rows.append((r.bank_txn_id, c[0].item_id, -amt, "RETURNED", 0.9, "cheque_return"))
    return rows


def match(lines, items, ctx=None):
    phase = (ctx or {}).get("phase")
    return match_cards(lines, items, phase) + match_cheques(lines, items, phase)
