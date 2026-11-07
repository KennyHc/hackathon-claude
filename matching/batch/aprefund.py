"""Supplier refunds of their credit notes (ap_supplier_refund): a receipt from a supplier → the unique open APC credit
note of that supplier (entity, or ES00 payment-factory group) with open amount = line amount, the supplier named in the
text (or the APC number in the text). Refunds of duplicate payments (category SUPPLIER_REFUND, no item) are excluded by
keyword. APC items already used earlier in this stage (netted in a payment run) are skipped."""
import re
import unicodedata
from collections import defaultdict

from matching.batch.common import PAYFACTORY

DUP_RE = re.compile(r"DOBLE|DUPLICAD|DUPLICATE|DOPPEL|DOUBLE|DOPPI|DUPLIC|OVERPAY", re.I)


def _nm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^0-9A-Z]", "", s.upper())


def match(lines, items, ctx=None):
    used = (ctx or {}).get("used_items", set())
    apc = items[(items.item_type == "APC") & (items.open_amount > 0)]
    by_amt = defaultdict(list)
    for r in apc[["item_id", "entity", "document_number", "party_name", "open_amount", "currency",
                  "issue_date"]].itertuples(index=False):
        by_amt[(r.entity, round(r.open_amount, 2))].append(r)
    rows, taken = [], set()
    cre = lines[lines.amount.astype(float) > 0]
    for r in cre.itertuples(index=False):
        amt = round(float(r.amount), 2)
        ent = r.account_id[:4]
        cands = [c for e in PAYFACTORY.get(ent, [ent]) for c in by_amt.get((e, amt), [])
                 if c.item_id not in used and c.item_id not in taken and c.currency == r.currency
                 and c.issue_date <= r.booking_date]
        if not cands:
            continue
        blob = " ".join([r.text, r.remittance_info, r.counterparty_name, r.ref1, r.ref2])
        if DUP_RE.search(blob):
            continue
        nb = _nm(blob)
        hit = [c for c in cands if len(_nm(c.document_number)) >= 4 and _nm(c.document_number) in nb]
        if len(hit) != 1:
            hit = [c for c in cands if len(_nm(c.party_name)) >= 6 and _nm(c.party_name)[:12] in nb]
        if len(hit) == 1:
            taken.add(hit[0].item_id)
            rows.append((r.bank_txn_id, hit[0].item_id, amt, "", 0.85, "ap_refund"))
    return rows
