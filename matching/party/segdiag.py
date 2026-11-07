"""Non-exact lines of a segment in a combo file: who claimed them (method) / unclaimed; examples (val golden)."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
seg, f = sys.argv[1], sys.argv[2]
nex = int(sys.argv[3]) if len(sys.argv) > 3 else 6
pd.set_option("display.width", 250)
s = pd.read_parquet(f)
g = pd.read_csv(os.path.join(ROOT, "data/val/golden_allocations.csv.gz"), dtype=str, keep_default_na=False)
b = pd.read_csv(os.path.join(ROOT, "data/val/bank_transactions.csv.gz"), dtype=str, keep_default_na=False).set_index("bank_txn_id")
sys.path.insert(0, ROOT)
from integration.lib.ledger import load_items
it = load_items("val").set_index("item_id")
gs = g[g.segment == seg]
bad, why = [], {}
for t, gg in gs.groupby("bank_txn_id"):
    ss = s[s.bank_txn_id == t]
    ga = {(i, round(float(a), 2)) for i, a in zip(gg.item_id, gg.amount)}
    sa = {(i, round(float(a), 2)) for i, a in zip(ss.item_id, ss.amount)}
    if ga != sa:
        bad.append(t)
        why[t] = ss.method.iloc[0] if len(ss) else "UNCLAIMED"
print(seg, len(gs.bank_txn_id.unique()), "lines, bad", len(bad), pd.Series(why).value_counts().head(8).to_dict())
for t in pd.Series(bad).sample(min(nex, len(bad)), random_state=0):
    l = b.loc[t]; gg = gs[gs.bank_txn_id == t]; ss = s[s.bank_txn_id == t]
    print(f"-- {t} {l.entity} {l.booking_date} {l.amount} {l.currency} orig={l.orig_amount}{l.orig_currency} | {l.text[:90]} | rmt {l.remittance_info[:40]} | {why[t]}")
    def fmt(i):
        if i not in it.index: return i
        x = it.loc[i]; return f"{i[:3]}{i[-5:]}:{x.document_number}:{x.issue_date}:{x.currency}{x.total_amount:.2f}/{x.open_amount:.2f}:{x.party_id}"
    print("   gold:", [(fmt(i), a, e) for i, a, e in zip(gg.item_id, gg.amount, gg.exception)][:6])
    print("   ours:", [(fmt(i), round(a, 2), e) for i, a, e in zip(ss.item_id, ss.amount, ss.exception)][:6])
