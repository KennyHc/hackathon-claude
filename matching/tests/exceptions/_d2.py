import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from matching.lib import data
pd.set_option('display.width',250)
g=pd.read_pickle('matching/out/cache/exc_dev/last_oracle.pkl')
D='matching/out/cache/exc_dev/'
bank=pd.read_parquet(D+'bank.parquet',columns=['bank_txn_id','booking_date']); it=pd.read_parquet(D+'items.parquet')
pt=data.payment_terms().set_index('code')
g=g.merge(bank,on='bank_txn_id').merge(it[['item_id','item_type','payment_terms','issue_date']],on='item_id')
g['pct']=g.payment_terms.map(pt.early_payment_discount_pct); g['dd']=g.payment_terms.map(pt.discount_days)
g['inwin']=(g.booking_date-g.issue_date).dt.days<=g.dd+3
g['elig']=g.pct.notna()&g.inwin&(g.item_type=='ARI')
g['sd']=g.gexc=='SHORT_DISCOUNT'
L=g.groupby('bank_txn_id').agg(nel=('elig','sum'),nsd=('sd','sum'),n=('sd','size'))
L=L[L.nsd>0]
print('sd items not eligible:', (g.sd&~g.elig).sum(), 'of', g.sd.sum())
L['frac']=L.nsd/L.nel
print(L.frac.describe()); print((L.nsd==L.nel).mean(), (L.nsd<L.nel).sum(), len(L))
print(L[L.nsd<L.nel].head(20))
x=g[g.sd&~g.elig]; print(x[['booking_date','issue_date','payment_terms','dd','item_type','gexc']].head())
