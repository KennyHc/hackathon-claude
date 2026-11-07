"""Baseline: a document number of ours appears in the bank text and the amount equals its open amount (1 item)."""
from __future__ import annotations

from matching.common import Ctx, alloc_frame

NAME = "ref_exact"


def run(ctx: Ctx):
    items = ctx.items_by_id
    rows = []
    for t in ctx.unclaimed().itertuples(index=False):
        amt = abs(t.amount)
        side = "AR" if t.amount > 0 else "AP"
        hits = set()
        for tok in t.doc_tokens:
            for iid in ctx.by_alnum.get(tok, ()):
                it = items.loc[iid]
                if it.side == side and it.entity == t.entity and it.currency == t.currency and abs(it.open_amount - amt) < 0.005:
                    hits.add(iid)
        if len(hits) == 1:
            rows.append((t.bank_txn_id, hits.pop(), amt, "", 0.9, NAME))
    return alloc_frame(rows)
