"""Diagnostics on val misses (scoring data only)."""
import sys, os
import pandas as pd, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SP = sys.argv[2] if len(sys.argv) > 2 else None
seg = sys.argv[1]
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
g = pd.read_csv(os.path.join(ROOT, "data/val/golden_allocations.csv.gz"), dtype=str, keep_default_na=False)
d = pd.read_parquet(os.path.join(ROOT, "matching/party/out/debug_val.parquet"))
b = pd.read_csv(os.path.join(ROOT, "data/val/bank_transactions.csv.gz"), dtype=str, keep_default_na=False)
it = pd.read_parquet(SP) if SP else None
gs = g[g.segment == seg]
gp = gs.merge(it[["item_id", "party_id"]], on="item_id", how="left").groupby("bank_txn_id").party_id.first() if it is not None else None
x = d[d.bank_txn_id.isin(gs.bank_txn_id)].copy()
x["gparty"] = gp.reindex(x.bank_txn_id).values if gp is not None else ""
x["party_ok"] = x.party_id == x.gparty
print(x.groupby("status").agg(n=("party_ok", "size"), party_ok=("party_ok", "mean")))
miss = set(gs.bank_txn_id) - set(d.bank_txn_id)
print("not in debug (no amount?)", len(miss))
for st in ["no_party", "no_fit"]:
    y = x[x.status == st].sample(min(8, (x.status == st).sum()), random_state=2)
    for t, pid, gpp, key in zip(y.bank_txn_id, y.party_id, y.gparty, y.key):
        l = b[b.bank_txn_id == t].iloc[0]
        print(f"--{st} {t} {l.booking_date} {l.bank} {l.entity} {l.amount} {l.currency} | {l.text} | cp={l.counterparty_name} {l.counterparty_account} | res={pid} key={key} gold={gpp}")
        gi = gs[gs.bank_txn_id == t].merge(it, on="item_id", how="left")
        print("   gold:", [(r.item_id, r.amount, r.exception, r.issue_date, r.open_amount) for r in gi.itertuples()][:6])
        if gpp:
            o = it[(it.party_id == gpp)]
            o = o[(o.issue_date <= l.booking_date)].sort_values("issue_date")
            print("   party items(open_at_start≤date):", len(o), [(r.item_id[-7:], r.issue_date, r.open_amount, r.currency) for r in o.tail(8).itertuples()])
