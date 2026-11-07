import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from matching.lib import data
pd.set_option('display.width',250)
exec(open('matching/tests/exceptions/_d3.py').read().split("def lines")[0])
g=pd.read_pickle('matching/out/cache/exc_dev/diag.pkl')
pt=data.payment_terms().set_index('code')
g['pct']=g.payment_terms.map(pt.early_payment_discount_pct)/100; g['dd']=g.payment_terms.map(pt.discount_days)
g['inwin']=(g.booking_date-g.issue_date).dt.days<=g.dd+3
g['elig']=g.pct.notna()&g.inwin&(g.item_type=='ARI')
g['isinv']=~g.item_type.isin(['ARC','APC'])
g['gdef']=np.where(g.isinv,g.total_amount-g.amount,0.0)
mis=g[(g.gexc=='SHORT_DISCOUNT')&(g.pexc=='')]
ids=mis.bank_txn_id.drop_duplicates()
print(len(ids))
rows=[]
for t in ids.head(60):
    d=g[g.bank_txn_id==t]; inv=d[d.isinv]
    D=inv.total_amount.sum()-d[~d.isinv].total_amount.sum()
    rows.append(dict(t=t,n=len(d),ncn=(~d.isinv).sum(),diff=round(D-d.cash.iloc[0],2),gdef=round(inv.gdef.sum(),2),sdgold=round(inv[inv.gexc=='SHORT_DISCOUNT'].gdef.sum(),2),
      sdexp=round((inv.pct.fillna(0)*inv.total_amount*inv.elig).sum(),2),nsd=(inv.gexc=='SHORT_DISCOUNT').sum(),nel=inv.elig.sum(),exc=','.join(sorted(set(d.gexc))),seg=d.segment.iloc[0],
      ndisc_terms=inv.pct.notna().sum(), fxline=d.bcur.iloc[0]!=d.icur.iloc[0]))
print(pd.DataFrame(rows).head(40).to_string())
