import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
pd.set_option('display.width',250)
g=pd.read_pickle('matching/out/cache/exc_dev/diag.pkl')
f=g[(g.gexc=='FX_DIFFERENCE')&(g.pexc!='FX_DIFFERENCE')]
print(len(f), f.segment.value_counts().head(6).to_dict()); print(f.assign(same=f.bcur==f.icur).same.mean())
print(f.groupby(['bcur','icur']).size().sort_values().tail(8))
d=f[f.bcur!=f.icur]; print(len(d)); print(d.pexc.value_counts()); print(d.groupby('bank_txn_id').size().describe())
t=d.bank_txn_id.drop_duplicates().head(5)
for x in t:
    L=g[g.bank_txn_id==x]; print(x,len(L),L.cash.iloc[0],L.bcur.iloc[0],L.orig_amount.iloc[0],L.icur.iloc[0],L.total_amount.sum(),L.gexc.value_counts().to_dict(),L.pexc.value_counts().to_dict())
