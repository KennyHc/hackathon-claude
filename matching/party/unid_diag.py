"""Examples of unid_evidence 'no evidence' lines that are not UNIDENTIFIED in the val golden (diagnostics)."""
import os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from unid import unid_evidence
from integration.lib.ledger import load_items
ent, combo = sys.argv[1], sys.argv[2]
L = pd.read_csv(os.path.join(ROOT, "data/val/bank_transactions.csv.gz"), dtype=str, keep_default_na=False)
L = L[(L.entity == ent) & (pd.to_numeric(L.amount) > 0)]
s = pd.read_parquet(combo)
L = L[~L.bank_txn_id.isin(set(s.bank_txn_id))]
g = pd.read_csv(os.path.join(ROOT, "data/val/golden_allocations.csv.gz"), dtype=str, keep_default_na=False)
seg = g.drop_duplicates("bank_txn_id").set_index("bank_txn_id")
ev = unid_evidence(L, load_items("val"), "val", loeo=ent).set_index("bank_txn_id")
L = L.set_index("bank_txn_id").join(ev)
L["gold"] = seg.exception.reindex(L.index); L["segment"] = seg.segment.reindex(L.index)
x = L[~L.has_ref_like & ~L.payer_known]
print(pd.crosstab(x.segment, x.gold == "UNIDENTIFIED").sort_values(False, ascending=False).head(12))
for t, r in x[x.gold != "UNIDENTIFIED"].sample(10, random_state=1).iterrows():
    print(r.segment, r.amount, "|", r.text[:100], "| cp", r.counterparty_name, "| rmt", r.remittance_info[:40])
