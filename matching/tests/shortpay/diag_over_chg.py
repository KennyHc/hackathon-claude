"""Why are golden OVERPAYMENT / SHORT_BANK_CHARGES lines not claimed? Runs shortpay's per-line solver on those lines only.

    python matching/tests/shortpay/diag_over_chg.py [--alloc matching/out/val/allocations.parquet]

Outcome per line: no_party | no_solution | ambiguous_payers | weak:<method>:<conf> | ok:<method> (+ whether the item is right).
Diagnostic only (uses the val golden); nothing here feeds the matchers.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, ".")
from matching.lib import data, party as partylib  # noqa: E402
from matching.lib import shortpay_core as sc  # noqa: E402
from matching.matchers import shortpay as sp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", default=os.path.join("matching", "out", "val", "allocations.parquet"))
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()
    phase = "val"
    g = pd.read_csv(os.path.join("data", phase, "golden_allocations.csv.gz"))
    al = pd.read_parquet(a.alloc)
    prior = al[~al.method.fillna("").str.match(
        r"^(fx_)?(ref_short_wht|ref_short_disc|ref_over|ref_chg|wht_|disc_|over_|chg_|cn_paid_out|dup_)")]
    claimed_any = set(al.bank_txn_id)
    st = sc.ARState(phase, prior)
    px = partylib.PartyIndex(phase, cut=sp.CUT.get(phase, sp.DEFAULT_CUT))
    from integration.lib import fx
    bank = data.bank(phase).set_index("bank_txn_id")
    cust = data.customers().set_index("customer_id") if hasattr(data, "customers") else None
    for exc in ("OVERPAYMENT", "SHORT_BANK_CHARGES"):
        gl = g[(g.exception == exc) & ~g.bank_txn_id.isin(claimed_any)]
        out, ex = Counter(), defaultdict(list)
        for r in gl.itertuples(index=False):
            row = bank.loc[r.bank_txn_id].to_dict()
            row["bank_txn_id"] = r.bank_txn_id
            stats = Counter()
            cands = px.candidates(row, row["entity"])
            gold_party = st.owner.get(r.item_id) if hasattr(st, "owner") else None
            res = sp._one_in(st, px, fx, row, stats)
            if res is None:
                k = next(iter(stats), "none")
            else:
                right = r.item_id in {st.item_id[i] for i in res["idx"]}
                k = ("ok:" if res["conf"] >= sp.CONF_MIN else "weak:") + res["method"] + (":right" if right else ":WRONG")
            out[k] += 1
            pos = st.pos_of(r.item_id) if hasattr(st, "pos_of") else None
            ex[k].append((r.bank_txn_id, r.item_id, row["amount"], row["currency"], r.amount,
                          str(row.get("counterparty_name"))[:28], str(row.get("counterparty_account"))[:24],
                          str(row.get("remittance_info") or row.get("text"))[:40], [c.customer_id for c in cands][:3]))
        print(f"\n=== {exc}: {len(gl)} golden lines nobody claimed ===")
        for k, n in out.most_common():
            print(f"  {n:4d}  {k}")
        for k, _ in out.most_common(4):
            print(f"  -- {k}")
            for e in ex[k][: a.show]:
                print("     ", e)


if __name__ == "__main__":
    main()
