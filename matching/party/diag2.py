"""Miss breakdown for a segment: status x golden item count, party correctness (val golden, diagnostics only)."""
import sys, os
import pandas as pd, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
seg = sys.argv[1]
g = pd.read_csv(os.path.join(ROOT, "data/val/golden_allocations.csv.gz"), dtype=str, keep_default_na=False)
it = pd.read_parquet(sys.argv[2])
d = pd.read_parquet(os.path.join(ROOT, "matching/party/out/debug_val.parquet"))
s = pd.read_parquet(os.path.join(ROOT, "matching/out/val/alloc_party.parquet"))
gs = g[g.segment == seg].merge(it[["item_id", "party_id"]], on="item_id", how="left")
L = gs.groupby("bank_txn_id").agg(n=("item_id", "size"), gp=("party_id", "first"))
L = L.join(d.set_index("bank_txn_id")[["party_id", "status"]])
L["pok"] = L.party_id == L.gp
L["nb"] = pd.cut(L.n, [0, 1, 2, 4, 8, 16, 1000])
print(pd.crosstab(L.nb, L.status))
print(L.groupby("status").pok.mean().round(3))
