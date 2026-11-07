"""Direct-debit batches (our collections): batch credit → all receipts of the batch; R-transactions (returns) →
the receipt with the mandate + amount → its item, RETURNED (amount = receipt amount)."""
import os
import re
from collections import defaultdict

import pandas as pd

from matching.batch.common import load_detail

BATCH_RE = re.compile(r"(?<!\d)(\d{11})(?!\d)")
DD_REPRES = os.environ.get("DD_REPRES", "1") == "1"
MANDATE_RE = re.compile(r"\b[A-Z]{2,5}-[A-Z]{2}-\d{4,10}\b")   # VTM-ES-0006513; tolerant of other prefixes


def _digits(s):
    return re.sub(r"\D", "", str(s))


def resolve_receipts(dd, items, phase=""):
    """item_id per DD receipt: exact (entity, document_number); else customer (mandate / IBAN) + digits suffix of the
    remittance_info; else customer + unique amount among open AR invoices."""
    from integration.lib.masters import read_master
    ar = items[items.item_type == "ARI"]
    by_doc = defaultdict(list)
    by_party = defaultdict(list)
    for r in ar[["item_id", "entity", "document_number", "party_id", "open_amount", "total_amount"]].itertuples(
            index=False):
        by_doc[(r.entity, r.document_number)].append(r)
        by_party[r.party_id].append(r)
    cust = read_master("customers.csv", phase)
    m2c = {m: c for m, c in zip(cust.sepa_mandate_id, cust.customer_id) if m}
    ib2c = defaultdict(set)
    for c, ib in zip(cust.customer_id, cust.iban):
        if ib:
            ib2c[ib].add(c)
    ids, how = [], []
    for ent, ri, mand, ib, amt in zip(dd.account_id.str[:4], dd.remittance_info, dd.mandate_id, dd.debtor_iban,
                                      dd.amount.astype(float)):
        c = by_doc.get((ent, ri), [])
        if len(c) > 1:
            c = [r for r in c if abs(r.total_amount - amt) < 0.01 or abs(r.open_amount - amt) < 0.01] or c
        if len(c) == 1:
            ids.append(c[0].item_id); how.append("doc"); continue
        cid = m2c.get(mand)
        cids = {cid} if cid else (ib2c.get(ib, set()) if ib else set())
        cand = [r for k in cids for r in by_party.get(k, []) if r.entity == ent]
        d = _digits(ri)
        c = [r for r in cand if d and _digits(r.document_number).endswith(d)]
        if len(c) > 1:
            c = [r for r in c if abs(r.total_amount - amt) < 0.01 or abs(r.open_amount - amt) < 0.01]
        if len(c) == 1:
            ids.append(c[0].item_id); how.append("cust_digits"); continue
        # digits suffix within entity (no customer)
        if not cand and d and len(d) >= 6:
            pass
        c = [r for r in cand if abs(r.total_amount - amt) < 0.01]
        if len(c) == 1:
            ids.append(c[0].item_id); how.append("cust_amount"); continue
        ids.append(""); how.append("")
    dd = dd.copy()
    dd["item_id"], dd["how"] = ids, how
    return dd


def _full_ar_docs(phase):
    """(entity, document_number) → item_id over all AR invoice files (train, live, phase): returned receipts can be of
    items already closed at phase start (not in open items). Input documents, not labels."""
    import os
    from matching.batch.common import DATA
    out = {}
    prev = {"final": ["train", "live"], "live": ["train"], "val": ["train"]}.get(phase, [])
    for ph in dict.fromkeys(prev + [phase]):   # never a later phase (no leakage: val must not read live)
        f = os.path.join(DATA, ph, "ar_invoices.csv.gz")
        if os.path.exists(f):
            df = pd.read_csv(f, dtype=str, keep_default_na=False, usecols=["item_id", "entity", "document_number"])
            out.update(zip(zip(df.entity, df.document_number), df.item_id))
    return out


def match(lines, items, ctx=None):
    phase = (ctx or {}).get("phase")
    dd = load_detail(phase, "direct_debit_batches")
    if dd.empty:
        return []
    dd = resolve_receipts(dd, items, phase)
    dd["amt"] = dd.amount.astype(float)
    batches = {}
    for (acc, bid), d in dd.groupby(["account_id", "batch_id"], sort=False):
        batches[(acc, bid)] = d
    by_bid = defaultdict(list)
    for acc, bid in batches:
        by_bid[bid].append(acc)
    # receipts by mandate for returns
    by_mand = defaultdict(list)
    full_doc = _full_ar_docs(phase)
    for r in dd[["batch_id", "collection_date", "account_id", "mandate_id", "amt", "item_id",
                 "remittance_info"]].itertuples(index=False):
        by_mand[r.mandate_id].append(r)

    def _it(x):
        return x.item_id or full_doc.get((x.account_id[:4], x.remittance_info), "")
    # re-presentations (re_presentation=Y) tell which earlier collection came back
    rep_item = defaultdict(list)
    rep_mand = defaultdict(list)
    if DD_REPRES and "re_presentation" in dd.columns:
        for x in dd[dd.re_presentation == "Y"][["batch_id", "collection_date", "account_id", "mandate_id", "amt",
                                                 "item_id", "remittance_info"]].itertuples(index=False):
            i = _it(x)
            if i:
                rep_item[i].append(x.collection_date)
                rep_mand[x.mandate_id].append((x, i))
    rows = []
    for r in lines.itertuples(index=False):
        amt = float(r.amount)
        blob = " ".join([r.ref1, r.ref2, r.text, r.remittance_info, r.end_to_end_id])
        if amt > 0:
            for bid in BATCH_RE.findall(blob):
                accs = by_bid.get(bid, [])
                acc = r.account_id if r.account_id in accs else (accs[0] if len(accs) == 1 else None)
                if acc is None:
                    continue
                D = batches[(acc, bid)]
                tot = float(D.batch_total.iloc[0])
                if abs(tot - amt) > 0.01 and abs(D.amt.sum() - amt) > 0.01:
                    continue
                unres = (D.item_id == "").mean()
                if unres > 0.2:
                    break
                D = D[(D.item_id != "") & ~D.item_id.duplicated(keep=False)]
                for it, a in zip(D.item_id, D.amt):
                    rows.append((r.bank_txn_id, it, a, "", 0.97 if unres == 0 else 0.7,
                                 "dd_batch" if unres == 0 else "dd_batch_part"))
                break
        elif amt < 0:
            mm = [m for m in MANDATE_RE.findall(blob) if m in by_mand]
            if not mm:   # any token that is a known mandate id of our DD files
                mm = [t for t in re.findall(r"[A-Z0-9][A-Z0-9/\-]{7,34}", blob) if t in by_mand]
            if not mm:
                continue
            lim = (pd.Timestamp(r.booking_date) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            cands = [x for x in by_mand.get(mm[0], []) if x.collection_date <= lim
                     and abs(x.amt + amt) <= 0.01]
            bids = set(BATCH_RE.findall(blob))
            if bids:
                c2 = [x for x in cands if x.batch_id in bids]
                cands = c2 or cands
            if not cands:
                continue
            x = max(cands, key=lambda x: x.collection_date)
            it = _it(x)
            if DD_REPRES:
                hi = (pd.Timestamp(r.booking_date) + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
                rp = [c for c in cands if _it(c) and any(r.booking_date < d <= hi for d in rep_item[_it(c)])]
                if rp:
                    x = max(rp, key=lambda x: x.collection_date)
                    it = _it(x)
                elif not it:   # unresolved original: the later re-presentation of the same document names the item
                    dg = _digits(x.remittance_info)[-6:]
                    nx = {i for c, i in rep_mand.get(mm[0], []) if r.booking_date < c.collection_date <= hi
                          and abs(c.amt + amt) <= 0.01 and len(dg) >= 5 and dg in _digits(c.remittance_info + i)}
                    if len(nx) == 1:
                        it = nx.pop()
            if it:
                rows.append((r.bank_txn_id, it, x.amt, "RETURNED", 0.9, "dd_return"))
    return rows
