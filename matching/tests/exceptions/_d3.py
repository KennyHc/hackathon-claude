import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from matching.lib import data
pd.set_option('display.width',250)
g=pd.read_pickle('matching/out/cache/exc_dev/last_oracle.pkl')
D='matching/out/cache/exc_dev/'
bank=pd.read_parquet(D+'bank.parquet',columns=['bank_txn_id','amount','currency','booking_date','orig_amount','orig_currency']).rename(columns={'amount':'cash','currency':'bcur'}); it=pd.read_parquet(D+'items.parquet')
g=g.merge(bank,on='bank_txn_id').merge(it[['item_id','total_amount','item_type','payment_terms','issue_date','currency','party_id','side']].rename(columns={'currency':'icur'}),on='item_id')
g['n']=g.groupby('bank_txn_id').item_id.transform('size')
cu=pd.read_csv('data/masters/customers.csv',dtype=str).set_index('customer_id').country
g['ctry']=g.party_id.map(cu)
g.to_pickle('matching/out/cache/exc_dev/diag.pkl')
def lines(mask,k=6,cols=None):
    L=g[g.bank_txn_id.isin(g[mask].bank_txn_id)]
    s=L.groupby('bank_txn_id').agg(n=('n','first'),cash=('cash','first'),bcur=('bcur','first'),icur=('icur','first'),D=('total_amount','sum'),ctry=('ctry','first'),gold=('gexc',lambda x:','.join(sorted(set(x)))),pred=('pexc',lambda x:','.join(sorted(set(x)))),orig=('orig_amount','first'))
    s['diff']=s.D-s.cash
    return s
print('SW missed'); s=lines((g.gexc=='SHORT_WITHHOLDING')&(g.pexc=='')); print(s.n.describe()); print(s.head(8)); print(s.bcur.eq(s.icur).mean())
print('PARTIAL FP'); s=lines((g.gexc=='')&(g.pexc=='PARTIAL')); print(s.n.describe()); print(s.gold.value_counts().head()); print(s.head(8))
