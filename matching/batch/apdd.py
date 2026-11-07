"""Supplier direct debits (ap_direct_debit): debit line flagged as a DD charge → the unique open AP invoice of that
entity (or ES00 payment-factory group) with amount = line amount and (document number in text, or supplier name in
text). Single invoice per line."""
import re
import unicodedata
from collections import defaultdict

from matching.batch.common import PAYFACTORY

DD_RE = re.compile(r"LASTSCHRIFT|DIRECT DEBIT|ADEUDO|PRELEVEMENT|PRÉLÈVEMENT|ADDEBITO|SDD|DEBITO DIRETO|POLECENIE|"
                   r"MREF\+|RECIBO|PRLV|RUM |\| N \d{10,}", re.I)


def _nm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^0-9A-Z]", "", s.upper())


def match(lines, items, ctx=None):
    ap = items[(items.item_type == "API") & (items.open_amount > 0)]
    by_amt = defaultdict(list)
    for r in ap[["item_id", "entity", "document_number", "party_name", "open_amount", "currency",
                 "issue_date", "payment_method"]].itertuples(index=False):
        by_amt[(r.entity, round(r.open_amount, 2))].append(r)
    rows, used = [], set()
    deb = lines[lines.amount.astype(float) < 0]
    for r in deb.itertuples(index=False):
        blob = " ".join([r.bank_code, r.text, r.remittance_info, r.counterparty_name])
        if not DD_RE.search(blob) or "RUECKLASTSCHRIFT" in blob.upper():
            continue
        amt = round(-float(r.amount), 2)
        ent = r.account_id[:4]
        nb = _nm(blob)
        cands = [c for e in PAYFACTORY.get(ent, [ent]) for c in by_amt.get((e, amt), [])
                 if c.item_id not in used and c.currency == r.currency and c.issue_date <= r.booking_date]
        hit = [c for c in cands if len(_nm(c.document_number)) >= 4 and _nm(c.document_number) in nb]
        if len(hit) != 1:
            hit = [c for c in cands if _nm(c.party_name)[:12] and _nm(c.party_name)[:12] in nb]
            if len(hit) > 1:
                dd = [c for c in hit if c.payment_method == "dd"]
                hit = dd if len(dd) == 1 else hit
                if len(hit) > 1:   # fixed monthly DD: oldest open first
                    hit = sorted(hit, key=lambda c: c.issue_date)[:1] if len({c.party_name for c in hit}) == 1 else []
        if len(hit) == 1:
            used.add(hit[0].item_id)
            rows.append((r.bank_txn_id, hit[0].item_id, amt, "", 0.85, "ap_dd"))
    return rows
