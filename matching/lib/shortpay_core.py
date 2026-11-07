"""Shared state for the shortpay / duplicates matchers (owner: shortpay matcher).

    from matching.lib import shortpay_core as sc
    st = sc.ARState(phase, prior)        # AR items + ledger after `prior` allocations, per-party / per-entity reference indexes
    pool = st.pool(party_ids, currency, booking_date)      # open item positions of these payers, oldest first
    sc.risk(n, k, cash_cents)            # expected number of coincidental subset sums (structural, no fitted constant)

Everything here reads INPUTS only (bank lines, items, masters, earlier allocations); no labels.
"""
from __future__ import annotations

import os
from collections import defaultdict
from math import comb

import numpy as np
import pandas as pd

from . import data, refs_text as rt

SEPA = {"ES", "FR", "DE", "IT", "PT", "NL", "BE", "AT", "LU", "IE", "FI"}  # POLICIES s3: charges only outside SEPA (engine's list)
WHT_COUNTRIES = {"MX", "BR", "MA"}  # POLICIES s3 SHORT_WITHHOLDING
WHT_RATES = (0.0125, 0.015, 0.10)   # POLICIES s3


def cents(x) -> int:
    return int(round(float(x) * 100))


def fee_grid(cut: str, min_count: int = 2) -> set:
    """{(currency, cents)}: intermediary-bank fees deducted on SHORT_BANK_CHARGES lines of TRAIN booked before `cut`
    (labels, before the cut only: CLAUDE.md 2b).  The fees are a few fixed EUR amounts converted at a fixed rate per currency
    (10/15/20/25/35/45 EUR; USD x1.185, CAD x1.48, AUD x1.58): the grid learned before 2025 covers 100 % of 2025.
    Single-item lines only (fee = item amount - cash); a value must be seen min_count times."""
    try:
        g = data.golden("train")
        s = g[g.exception == "SHORT_BANK_CHARGES"]
        s = s[s.groupby("bank_txn_id").item_id.transform("size") == 1]
        b = data.bank("train").set_index("bank_txn_id")
        bd = pd.to_datetime(s.bank_txn_id.map(b.booking_date))
        s = s[bd < pd.Timestamp(cut)]
        ccy = s.bank_txn_id.map(b.currency)
        fee = ((s.amount - s.bank_txn_id.map(b.amount)) * 100).round().astype("Int64")
        vc = pd.Series(list(zip(ccy, fee))).value_counts()
        return {(c, int(f)) for (c, f), n in vc.items() if n >= min_count and f is not pd.NA and f > 0}
    except Exception:  # no train labels available: the fee rule simply never fires
        return set()


def risk(n: int, k: int, cash: int) -> float:
    """#subsets of size <= k of n items / #possible cent amounts (~cash): expected spurious exact sums."""
    k = max(k, 1)
    return sum(comb(n, j) for j in range(1, min(k, n) + 1)) / max(cash, 100)


def ledger_delta(prior: pd.DataFrame | None) -> pd.Series:
    """item_id -> net cents consumed by earlier allocations (RETURNED re-opens, DUPLICATE_PAYMENT does not consume)."""
    if prior is None or not len(prior):
        return pd.Series(dtype="int64")
    p = prior[["item_id", "amount", "exception"]].dropna(subset=["item_id"]).copy()
    exc = p.exception.fillna("").astype(str)
    sgn = np.where(exc == "RETURNED", -1, np.where(exc == "DUPLICATE_PAYMENT", 0, 1))
    p["d"] = np.round(p.amount.astype(float).values * 100).astype("int64") * sgn
    return p.groupby("item_id").d.sum()


class ARState:
    """AR invoices / credit notes of the phase with their open balance after `prior`."""

    def __init__(self, phase: str, prior: pd.DataFrame | None = None, keep_closed: bool = False):
        cols = ["item_id", "item_type", "side", "entity", "party_id", "document_number", "issue_date", "currency", "open_amount",
                "total_amount", "po_number", "related_document", "payment_terms"]
        cp = os.path.join(data.CACHE, phase, "items.parquet")
        if not os.path.exists(cp):
            data.items(phase)
        it = pd.read_parquet(cp, columns=cols, filters=[("side", "==", "AR")])
        it = it[it.item_type.isin(["ARI", "ARC"])].copy()
        it["cents"] = np.round(it.open_amount.astype(float) * 100).astype("int64")
        it["full"] = np.round(it.total_amount.astype(float) * 100).astype("int64")
        d = ledger_delta(prior)
        if len(d):
            dd = it.item_id.map(d).fillna(0).astype("int64").values
            it["cents"] = np.maximum(0, it.cents.values - dd)
        if not keep_closed:
            it = it[it.cents > 0]
        it = it.sort_values(["issue_date", "item_id"]).reset_index(drop=True)
        cu = data.customers().set_index("customer_id")
        terms = it.payment_terms.fillna(it.party_id.map(cu.payment_terms))
        pt = data.payment_terms().set_index("code")
        it["disc_pct"] = terms.map(pt.early_payment_discount_pct).fillna(0.0).astype(float).values
        it["disc_days"] = terms.map(pt.discount_days).fillna(0.0).astype(float).values
        self.country = cu.country.to_dict()
        self.items = it
        self.item_id = it.item_id.values
        self.c = it.cents.values.copy()
        self.full = it.full.values
        self.typ = it.item_type.values
        self.docs = it.document_number.astype(str).values
        self.pos = np.array([p if isinstance(p, str) else "" for p in it.po_number.values], dtype=object)
        self.cur = it.currency.values
        self.issue = it.issue_date.values
        self.ent = it.entity.values
        self.party = it.party_id.values
        self.pct = it.disc_pct.values
        self.ddays = it.disc_days.values
        self.by_party: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(self.party):
            self.by_party[p].append(i)
        self.ref_idx: dict[tuple, list[int]] = defaultdict(list)
        for i in range(len(it)):
            k = rt.doc_keys(self.docs[i])
            if len(k["full"]) >= 8:
                self.ref_idx[(self.ent[i], k["full"])].append(i)
            if len(k["digits"]) >= 8 and k["digits"] != k["full"]:
                self.ref_idx[(self.ent[i], k["digits"])].append(i)

    def ref_owners(self, toks, entity, typ=None, open_only=True) -> dict:
        """party -> #quoted full document numbers of that party (entity-wide exact lookup)."""
        out: dict[str, int] = {}
        for t in toks:
            for key in {t, rt.digits(t)}:
                for i in self.ref_idx.get((entity, key), ()):
                    if (not open_only or self.c[i] > 0) and (typ is None or self.typ[i] == typ):
                        out[self.party[i]] = out.get(self.party[i], 0) + 1
        return out

    def pool(self, members, currency, bd, types=("ARI", "ARC")) -> list[int]:
        idx = []
        for m in members:
            for i in self.by_party.get(m, ()):
                if self.c[i] > 0 and self.issue[i] <= bd and self.cur[i] == currency and self.typ[i] in types:
                    idx.append(i)
        idx.sort(key=lambda i: (self.issue[i], i))
        return idx
