import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
D='matching/out/cache/exc_dev/'
g=pd.read_parquet(D+'golden.parquet'); g=g[g.item_id.notna()]
it=pd.read_parquet(D+'items.parquet'); b=pd.read_parquet(D+'bank.parquet',columns=['bank_txn_id','booking_date','amount','currency','orig_amount','orig_currency','entity','account_id','text']).rename(columns={'amount':'cash','currency':'bcur'})
m=g.merge(it[['item_id','item_type','side','party_id','issue_date','currency','total_amount','payment_terms','related_document']].rename(columns={'currency':'icur'}),on='item_id',how='left').merge(b,on='bank_txn_id',how='left')
m['sign']=np.where(m.item_type.isin(['ARC','APC']),-1,1)
m['n']=m.groupby('bank_txn_id').item_id.transform('size')
m['ratio']=m.amount/m.total_amount
print(m.groupby('exception',dropna=False).agg(n=('n','size'),single=('n',lambda s:(s==1).mean()),ratio=('ratio','median'),cn=('sign',lambda s:(s<0).mean())))
print(pd.crosstab(m.segment,m.exception.fillna('-')).to_string())
m.to_pickle('matching/out/cache/exc_dev/m.pkl')
