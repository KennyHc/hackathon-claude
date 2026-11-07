import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
g=pd.read_pickle('matching/out/cache/exc_dev/last_oracle.pkl')
D='matching/out/cache/exc_dev/'
gold=pd.read_parquet(D+'golden.parquet'); gold=gold[gold.item_id.notna()]
bank=pd.read_parquet(D+'bank.parquet',columns=['bank_txn_id','booking_date','amount','text'])
fp=g[(g.gexc=='')&(g.pexc=='DUPLICATE_PAYMENT')]
print(fp.segment.value_counts().head(10))
ids=fp.item_id.head(6).tolist()
for i in ids[:4]:
    h=gold[gold.item_id==i].merge(bank,on='bank_txn_id').sort_values('booking_date')
    print(i); print(h[['bank_txn_id','booking_date','amount_x','amount_y','exception','segment','text']].to_string())
