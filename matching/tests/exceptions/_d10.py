import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
pd.set_option('display.width',250)
g=pd.read_pickle('matching/out/cache/exc_dev/diag.pkl')
g['isinv']=~g.item_type.isin(['ARC','APC'])
mis=g[(g.gexc=='SHORT_WITHHOLDING')&(g.pexc=='')]
rows=[]
for t in mis.bank_txn_id.drop_duplicates():
    d=g[g.bank_txn_id==t]; inv=d[d.isinv]
    D=inv.total_amount.sum()-d[~d.isinv].total_amount.sum(); diff=D-d.cash.iloc[0]
    swi=inv[inv.gexc=='SHORT_WITHHOLDING']; r=(diff-(inv[inv.gexc=='SHORT_DISPUTE'].total_amount-inv[inv.gexc=='SHORT_DISPUTE'].amount).sum())/swi.total_amount.sum()
    rows.append(dict(t=t,n=len(d),ncn=(~d.isinv).sum(),nsw=len(swi),ndisp=(d.gexc=='SHORT_DISPUTE').sum(),nother=(d.gexc.isin(['','PARTIAL'])&d.isinv).sum(),diff=round(diff,2),rate=round(r,4),ctry=d.ctry.iloc[0]))
r=pd.DataFrame(rows); print(len(r)); print(r.head(15).to_string()); print(r.ndisp.value_counts().head(), r.nother.value_counts().head())
