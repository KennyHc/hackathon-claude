"""Show candidate structure for no_fit multi-item lines (val golden, diagnostics only)."""
import sys, os
import pandas as pd, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
seg = sys.argv[1]; nmin = int(sys.argv[3]) if len(sys.argv) > 3 else 2
g = pd.read_csv(os.path.join(ROOT, "data/val/golden_allocations.csv.gz"), dtype=str, keep_default_na=False)
b = pd.read_csv(os.path.join(ROOT, "data/val/bank_transactions.csv.gz"), dtype=str, keep_default_na=False).set_index("bank_txn_id")
it = pd.read_parquet(sys.argv[2])
d = pd.read_parquet(os.path.join(ROOT, "matching/party/out/debug_val.parquet")).set_index("bank_txn_id")
gi = g[g.item_id != ""].merge(b[["booking_date"]], left_on="bank_txn_id", right_index=True)
first_paid = gi.sort_values("booking_date").drop_duplicates("item_id").set_index("item_id").booking_date
gs = g[g.segment == seg].merge(it[["item_id", "party_id"]], on="item_id", how="left")
L = gs.groupby("bank_txn_id").agg(n=("item_id", "size"), gp=("party_id", "first"))
L = L[(L.n >= nmin) & (d.status.reindex(L.index) == "no_fit")]
for t in L.sample(min(6, len(L)), random_state=3).index:
    l = b.loc[t]; gp = L.loc[t, "gp"]
    gold = set(gs[gs.bank_txn_id == t].item_id)
    o = it[(it.party_id == gp) & (it.issue_date <= l.booking_date) & (it.item_type.isin(["ARI", "ARC", "API", "APC"]))].copy()
    o["paid"] = first_paid.reindex(o.item_id).values
    o = o[(o.paid.isna()) | (o.paid >= l.booking_date)].sort_values("issue_date")
    print(f"== {t} {l.booking_date} {l.amount} {l.currency} | {l.text[:100]} | {len(gold)} gold of {len(o)} open")
    print("   ", " ".join(f"{'*' if i in gold else ' '}{r.issue_date[2:]}/{r.due_date[5:]}:{r.open_amount:.0f}{'' if r.item_type in ('ARI','API') else 'CN'}" for i, r in zip(o.item_id, o.itertuples())))
