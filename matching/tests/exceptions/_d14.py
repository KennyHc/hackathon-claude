import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
pd.set_option('display.width',250)
g=pd.read_pickle('matching/out/cache/exc_dev/diag.pkl')
fn=g[(g.gexc=='FX_DIFFERENCE')&(g.pexc!='FX_DIFFERENCE')]; fp=g[(g.gexc!='FX_DIFFERENCE')&(g.pexc=='FX_DIFFERENCE')]
print('FN',len(fn)); print(fn.groupby(['segment','pexc']).size()); print('FP',len(fp)); print(fp.groupby(['segment','gexc']).size())
for t in fn.bank_txn_id.drop_duplicates().head(5):
    L=g[g.bank_txn_id==t]; print(t,len(L),L.cash.iloc[0],L.bcur.iloc[0],L.orig_amount.iloc[0],L.orig_currency.iloc[0],L.icur.iloc[0],round(L.total_amount.sum(),2),L.gexc.value_counts().to_dict(),L.pexc.value_counts().to_dict(),L.segment.iloc[0])
