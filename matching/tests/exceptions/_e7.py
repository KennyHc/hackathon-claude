import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
pd.set_option('display.width',250)
m=pd.read_pickle('matching/out/cache/exc_dev/m.pkl')
m['r']=(m.amount/m.total_amount).round(2)
for e in ['SHORT_DISPUTE','PARTIAL']:
    d=m[m.exception==e]
    print(e); print(d.groupby('segment').r.apply(lambda s:s.value_counts(normalize=True).head(7).round(3).to_dict()).to_string())
d=m[m.exception=='SHORT_DISPUTE']
print(d.groupby('bank_txn_id').size().describe())
# in dispute lines: other rows
L=m[m.bank_txn_id.isin(d.bank_txn_id)]
print(L.exception.fillna('-').value_counts()); print(L.n.describe())
