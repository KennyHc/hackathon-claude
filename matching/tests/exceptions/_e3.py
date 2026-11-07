import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',90)
m=pd.read_pickle('matching/out/cache/exc_dev/m.pkl')
s=m[m.exception=='SHORT_BANK_CHARGES']
print(s[['bank_txn_id','entity','bcur','icur','cash','orig_amount','orig_currency','amount','total_amount','text']].head(12).to_string())
print(s.groupby(['bcur','icur']).size())
print((s.total_amount-s.amount).describe(), ((s.total_amount-s.cash)/s.total_amount).describe())
# lines of bank charges: how does text look in other ar lines with USD/ non-EUR same text
print(s.text.str[:25].value_counts().head(10))
# withholding
w=m[(m.segment=='ar_short_withholding')]
print(w.exception.value_counts(dropna=False))
w1=w[w.n==1]; print(w1.exception.value_counts(dropna=False))
print(w1.head(5)[['cash','amount','total_amount','icur','bcur','party_id','exception']])
