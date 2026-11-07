"""Evidence for K4's UNIDENTIFIED decision on receipts that no stage claimed.

    from matching.party.unid import unid_evidence
    ev = unid_evidence(lines, items, phase)          # one row per receipt line of `lines` (amount > 0)

Columns:
  bank_txn_id
  has_ref_like            bool  text / remittance_info / ref1 carry an invoice-like reference: a digit run (≥ 5 digits,
                                or ≥ 4 when equal) that is the tail of an AR document number of the line's entity, or
                                an invoice token (INV/FV/FA/FT/NF/RE/KWI/FACTURA/INVOICE… followed by digits)
  payer_known             bool  a customer of the line's entity is identified: learned alias (train history), master
                                name / trade name / IBAN anywhere in the text, or the phase's invoice party names
  best_candidate_customer str   that customer ("" when none)
  evidence                str   which source gave payer_known: alias | iban | name | ""

Leakage: aliases are the same as the party stage's (val: train < 2025-01-01). Uses masters via read_master(phase)
(masters_final merged), so new entities (ZA01) are covered by their master names / IBANs.
"""
from __future__ import annotations

import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import party as P  # noqa: E402
from integration.lib.masters import read_master  # noqa: E402

_TOKEN = re.compile(r"\b(?:INV|INVOICE|FV|FA|FT|NF|RE|KWI|FAC|FACT|FACTURA|FACTURE|FATURA|RECHNUNG|FRA|FRAS|"
                    r"RECHNUNGSNR|RE-NR|FAKTURA)[#.:\- ]*[A-Z]{0,3}\d{3,}")
_RUN = re.compile(r"\d{4,}")


def _doc_index(items: pd.DataFrame):
    """entity → (set of full doc digit strings, set of their tails of length 5..9)."""
    ar = items[items.item_type.isin(["ARI", "ARC"])]
    full, tails = {}, {}
    for e, docs in ar.groupby("entity").document_number:
        f, t = set(), set()
        for x in docs.values:
            d = re.sub(r"\D", "", x).lstrip("0")
            if len(d) < 4:
                continue
            f.add(d)
            for L in range(5, min(len(d), 9) + 1):
                t.add(d[-L:].lstrip("0"))
        full[e], tails[e] = f, t
    return full, tails


def unid_evidence(lines: pd.DataFrame, items: pd.DataFrame, phase: str, **ctx) -> pd.DataFrame:
    import matcher as M
    L = lines[pd.to_numeric(lines.amount, errors="coerce") > 0]
    for c in ("text", "counterparty_name", "counterparty_account", "remittance_info", "ref1"):
        if c not in L:
            L = L.assign(**{c: ""})
    L = L.fillna("")
    start, cutoff = M._cutoffs(phase)
    P.NO_PREFIX.clear()
    if ctx.get("loeo"):
        P.NO_PREFIX.add(ctx["loeo"])
    R = P.Resolver(P.learn_aliases(phase, start, cutoff, log=ctx.get("log", lambda m: None),
                                   loeo=ctx.get("loeo", "")), {})
    cu = read_master("customers.csv", phase)
    names = [(e, pid, nm) for pid, nm, e in zip(cu.customer_id, cu.name, cu.entity)]
    names += [(e, pid, nm) for pid, nm, e in zip(cu.customer_id, cu.trade_name, cu.entity) if nm]
    for pid, nm, e in zip(cu.customer_id, cu.iban, cu.entity):
        if nm:
            R.add(e + "|I:" + nm.replace(" ", ""), pid)
    if "party_name" in items:
        pn = items[items.item_type.isin(["ARI", "ARC"])][["party_id", "party_name", "entity"]].drop_duplicates()
        names += list(zip(pn.entity, pn.party_id, pn.party_name))
    A = P.Anywhere(names)
    full, tails = _doc_index(items)
    out = []
    for t, e, txt, cpn, cpa, rmt, r1 in zip(L.bank_txn_id, L.entity, L.text, L.counterparty_name,
                                            L.counterparty_account, L.remittance_info, L.ref1):
        blob = P.norm(f"{txt} {rmt} {r1}")
        f, tl = full.get(e, set()), tails.get(e, set())
        ref = bool(_TOKEN.search(blob.replace(" | ", "")))
        if not ref:
            for x in _RUN.findall(blob.replace(" | ", "")):
                x = x.lstrip("0")
                if re.fullmatch(r"(19|20)\d\d", x):  # a year, not a reference
                    continue
                if (len(x) >= 5 and x in f) or (len(x) >= 6 and x in tl):
                    ref = True
                    break
        best, src = "", ""
        for p, *_ in R.resolve(e, txt, cpn, cpa):
            if p.startswith("C"):
                best, src = p, "alias"
                break
        if not best:
            for ib in P.ibans_in(f"{txt} {rmt}"):
                v = [p for p, *_ in R.p.get(e + "|I:" + ib, []) if p.startswith("C")]
                if v:
                    best, src = v[0], "iban"
                    break
        if not best:
            v = [p for p, _ in A.find(e, txt, cpn, rmt) if p.startswith("C")]
            if v:
                best, src = v[0], "name"
        out.append((t, ref, bool(best), best, src))
    return pd.DataFrame(out, columns=["bank_txn_id", "has_ref_like", "payer_known", "best_candidate_customer",
                                      "evidence"])
