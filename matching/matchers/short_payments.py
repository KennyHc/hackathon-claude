"""Proposed matcher (for the lead to drop into matching/matchers/short_payments.py, after same_amount in ORDER):
receipts no other matcher claimed, resolved payer, exactly one open invoice explained by SHORT_BANK_CHARGES /
OVERPAYMENT (plausible excess) / SHORT_WITHHOLDING (classification.exceptions.match_short_payment)."""
from __future__ import annotations

import pandas as pd

from classification.exceptions import match_short_payment
from matching.common import ALLOC_COLS, Ctx
from matching.parties import PartyResolver

NAME = "short_payments"


def run(ctx: Ctx) -> pd.DataFrame:
    pr = PartyResolver.get(ctx)
    used = set(ctx.allocations.item_id) if getattr(ctx, "allocations", None) is not None and len(ctx.allocations) else set()
    op = ctx.open_items[~ctx.open_items.item_id.isin(used) & (ctx.open_items.item_type == "ARI")]
    by_party = {k: v.to_dict("records") for k, v in op.groupby("party_id")}
    b = ctx.bank[(ctx.bank.amount > 0) & ~ctx.bank.bank_txn_id.isin(ctx.claimed)]
    rows, taken = [], set()
    for t in b.sort_values(["booking_date", "bank_txn_id"]).itertuples(index=False):
        cands = pr.resolve(t)
        if not cands:
            continue
        line = t._asdict()
        hits = []
        for cid in cands:
            its = [r for r in by_party.get(cid, ()) if r["item_id"] not in taken]
            if its:
                h = match_short_payment(its, t.amount_abs, line, ctx.masters)
                if h:
                    hits.append(h)
        if len(hits) != 1:
            continue
        iid, amt, exc = hits[0]
        taken.add(iid)
        rows.append((t.bank_txn_id, iid, amt, exc, 0.8, f"{NAME}:{exc.lower()}"))
    return pd.DataFrame(rows, columns=ALLOC_COLS)
