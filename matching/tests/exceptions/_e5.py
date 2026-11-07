import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
pd.set_option('display.width',250)
m=pd.read_pickle('matching/out/cache/exc_dev/m.pkl')
cu=pd.read_csv('data/masters/customers.csv',dtype=str)[['customer_id','country']]
m=m.merge(cu,left_on='party_id',right_on='customer_id',how='left')
w=m[m.exception=='SHORT_WITHHOLDING']
L=m[m.bank_txn_id.isin(w.bank_txn_id)]
g=L.groupby('bank_txn_id')
def f(d):
    inv=d[d.sign>0]; cn=d[d.sign<0]; fl=inv[inv.exception=='SHORT_WITHHOLDING']
    D=inv.amount.sum()-cn.amount.sum(); diff=D-d.cash.iloc[0]
    return pd.Series(dict(ninv=len(inv),ncn=len(cn),nfl=len(fl),other=len(inv)-len(fl),rate=diff/fl.total_amount.sum(),country=d.country.iloc[0],party=d.party_id.iloc[0]))
r=L.groupby('bank_txn_id').apply(f)
print(r.describe()); print(r.groupby('country').rate.describe())
print(r.groupby('party').rate.nunique().value_counts())
print(r.round({'rate':4}).groupby(['country','rate']).size())
print(pd.crosstab(r.ninv,r.other).head(8))
# country of items with withholding exception vs those paying same customer without
print(m[m.country.isin(['MX','BR','MA'])&(m.side=='AR')].groupby(['country',m.exception.fillna('-')]).size())
