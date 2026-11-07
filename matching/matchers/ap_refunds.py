"""Supplier refunds of their credit notes (POLICIES §4, §5.9): an incoming transfer from a supplier that refunds one
of ITS credit notes settles the APC item (amount = open amount of the credit note, no exception).

A line is claimed when
  - it is a receipt (direction "in") of an entity with an open APC item of exactly the cash (same currency), and
  - the APC is pointed to by the line: its number (supplier numbering, with or without prefix: R2025/00963,
    AB-INV0002992 -> INV0002992, RFAC2500192) or the number of the invoice it credits is in the text, or the
    supplier is identified by IBAN / name (and no namesake supplier has an open APC of that amount);
  - exactly one APC qualifies.
Refunds of a payment we made twice (category SUPPLIER_REFUND, "PAGO DOBLE", "DUPLICADO") have no APC: lines with
those words are only claimed on an explicit credit-note reference. CN_REFUND PDF clauses (pdf_clauses) raise
confidence but are not required (most APC items have no PDF).
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from ingestion.text import norm_name
from matching import pdf_clauses
from matching.common import Ctx, alloc_frame
from matching.matchers.pdf_cases import _Names

NAME = "ap_refunds"
EPS = 0.011
DUP = re.compile(r"DOBLE|DUPLICA|DOUBLE|TWICE|DOPPEL", re.I)


def _keys(doc: str) -> set[str]:
    """Reference keys of a supplier document number: alnum, alnum without a credit-note prefix, digits."""
    a = re.sub(r"[^A-Z0-9]", "", (doc or "").upper())
    out = {a} if len(a) >= 4 else set()
    m = re.match(r"^(AB|R|RF|NC|CN|GS|AV)(?=[A-Z0-9]*\d)", a)
    if m and len(a) - len(m.group(1)) >= 4:
        out.add(a[len(m.group(1)):])
    d = re.sub(r"\D", "", a).lstrip("0")
    if len(d) >= 5:
        out.add(d)
    return out


def run(ctx: Ctx):
    items = ctx.items_by_id
    apc = ctx.open_items[ctx.open_items.item_type == "APC"]
    if apc.empty:
        return alloc_frame([])
    try:
        cl = pdf_clauses.load_or_build(ctx)
        refund_pdf = set(cl.item_id[(cl.clause_type == "CN_REFUND") & (cl.item_id != "")])
        cl = cl[cl.item_id != ""]
    except Exception:  # noqa: BLE001 - clauses are a bonus
        refund_pdf, cl = set(), None
    if cl is None:
        cl = pd.DataFrame(columns=["clause_type", "party_id", "iban", "old_iban"])
    names = _Names(ctx, cl)

    by_amount: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    ref_keys: dict[str, set[str]] = {}
    for r in apc.itertuples(index=False):
        by_amount[(r.entity, r.currency, int(round(r.open_amount * 100)))].append(r.item_id)
        ref_keys[r.item_id] = _keys(r.document_number) | _keys(r.related_document)

    def same_amount_apcs(pid: str, cents: int, ent: str, ccy: str) -> bool:
        return any(items.at[i, "party_id"] == pid for d in (-1, 0, 1) for i in by_amount.get((ent, ccy, cents + d), ()))

    rows = []
    for t in ctx.unclaimed().itertuples(index=False):
        if t.direction != "in":
            continue
        cash = round(float(t.amount_abs), 2)
        cents = int(round(cash * 100))
        cands = [i for d in (-1, 0, 1) for i in by_amount.get((t.entity, t.currency, cents + d), ())]
        if not cands:
            continue
        toks = set(t.doc_tokens) | set(t.doc_refs)
        text = f"{t.text} {t.remittance_info or ''}"
        cp, tx = norm_name(t.counterparty_name or ""), f" {norm_name(t.text or '')} "
        ib = (t.counterparty_account or "").replace(" ", "")
        hits = []
        for i in cands:
            pid = items.at[i, "party_id"]
            if ref_keys[i] & toks:
                hits.append((i, "ref"))
                continue
            if DUP.search(text):
                continue
            riv = names.rivals(pid, t.entity, cp, tx, ib)
            if riv is None:
                continue
            if any(same_amount_apcs(q, cents, t.entity, t.currency) for q in riv):
                continue   # a namesake supplier has an open credit note of that amount
            hits.append((i, "name"))
        if len({i for i, _ in hits}) != 1:
            continue
        i, via = hits[0]
        conf = 0.95 if via == "ref" or i in refund_pdf else 0.85
        rows.append((t.bank_txn_id, i, round(float(items.at[i, "open_amount"]), 2), "", conf,
                     f"{NAME}:{via}{'_pdf' if i in refund_pdf else ''}"))
    return alloc_frame(rows)
