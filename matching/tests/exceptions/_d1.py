import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
pd.set_option('display.width',250)
g=pd.read_pickle('matching/out/cache/exc_dev/last_oracle.pkl')
D='matching/out/cache/exc_dev/'
bank=pd.read_parquet(D+'bank.parquet',columns=['bank_txn_id','amount','currency','booking_date']); it=pd.read_parquet(D+'items.parquet')
g=g.merge(bank.rename(columns={'amount':'cash'}),on='bank_txn_id').merge(it[['item_id','total_amount','item_type','payment_terms','issue_date','currency']].rename(columns={'currency':'icur'}),on='item_id')
g['n']=g.groupby('bank_txn_id').item_id.transform('size')
miss=g[(g.gexc=='SHORT_DISCOUNT')&(g.pexc=='')]
L=g[g.bank_txn_id.isin(miss.bank_txn_id)]
s=L.groupby('bank_txn_id').agg(n=('n','first'),nsd=('gexc',lambda x:(x=='SHORT_DISCOUNT').sum()),cn=('item_type',lambda x:x.isin(['ARC']).sum()),cash=('cash','first'),D=('total_amount','sum'))
print(s.n.describe()); print(s.head(10)); print((s.n>22).mean(), (s.cn>0).mean())
