import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
pd.set_option('display.width',250)
g=pd.read_pickle('matching/out/cache/exc_dev/diag.pkl')
f=g[(g.gexc=='PARTIAL')&(g.pexc=='')&(g.segment=='ap_fx')]
print(len(f), f.bank_txn_id.nunique())
for t in f.bank_txn_id.drop_duplicates().head(6):
    L=g[g.bank_txn_id==t]
    inv=L[~L.item_type.isin(['ARC','APC'])]
    p=L[L.gexc=='PARTIAL']
    print(t,'n',len(L),'cash',L.cash.iloc[0],L.bcur.iloc[0],'orig',L.orig_amount.iloc[0],L.orig_currency.iloc[0],'icur',L.icur.iloc[0],'D',round(L.total_amount.sum(),2),'partial amt/tot',(p.amount.round(2).tolist(),p.total_amount.tolist()),'exc',L.gexc.value_counts().to_dict(),'pred',L.pexc.value_counts().to_dict())
