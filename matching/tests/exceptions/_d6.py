import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from matching.lib import data
pd.set_option('display.width',250)
g=pd.read_pickle('matching/out/cache/exc_dev/diag.pkl')
pt=data.payment_terms().set_index('code')
g['pct']=g.payment_terms.map(pt.early_payment_discount_pct)/100; g['dd']=g.payment_terms.map(pt.discount_days)
g['inwin']=(g.booking_date-g.issue_date).dt.days<=g.dd+3
g['elig']=g.pct.notna()&g.inwin&(g.item_type=='ARI')
g['sd']=g.gexc=='SHORT_DISCOUNT'
# golden deficit per row
g['gdef']=np.where(g.item_type.isin(['ARC','APC']),0,g.total_amount-g.amount)
big=g[(g.n>22)&(g.pexc=='')&g.gexc.isin(['SHORT_DISCOUNT','SHORT_WITHHOLDING','SHORT_DISPUTE','PARTIAL'])]
ids=big.bank_txn_id.drop_duplicates().head(400)
rows=[]
for t in ids:
    d=g[g.bank_txn_id==t]
    inv=d[~d.item_type.isin(['ARC','APC'])]
    D=inv.total_amount.sum()-d[d.item_type.isin(['ARC','APC'])].total_amount.sum()
    rows.append(dict(t=t,n=len(d),ncn=len(d)-len(inv),diff=D-d.cash.iloc[0],gdef=inv.gdef.sum(),sd=d.sd.sum(),nel=d.elig.sum(),nsd_el=(d.sd&d.elig).sum(),exc=','.join(sorted(set(d.gexc))),
      sd_def=(inv.gdef*inv.sd).sum(), ctry=d.ctry.iloc[0], seg=d.segment.iloc[0], icur=d.icur.iloc[0],bcur=d.bcur.iloc[0], sdd_exp=(inv.pct.fillna(0)*inv.total_amount*inv.elig).sum()))
r=pd.DataFrame(rows); print(r.head(25).to_string()); print(r.seg.value_counts())
