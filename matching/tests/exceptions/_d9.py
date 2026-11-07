import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
pd.set_option('display.width',250)
g=pd.read_pickle('matching/out/cache/exc_dev/diag.pkl')
for t in ['BT-2024-0006928','BT-2024-0007901']:
    d=g[g.bank_txn_id==t]
    p=d[d.gexc=='PARTIAL']
    print(p[['item_id','amount','total_amount','payment_terms','issue_date','booking_date','item_type']].assign(r=lambda x:x.amount/x.total_amount))
    print(d.groupby('gexc').size(), d.payment_terms.value_counts().to_dict())
