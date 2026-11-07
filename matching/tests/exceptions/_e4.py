import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
pd.set_option('display.width',250)
m=pd.read_pickle('matching/out/cache/exc_dev/m.pkl')
s=m[(m.n==1)&(m.item_type.isin(['ARI']))&(m.bcur==m.icur)&(m.cash>0)]
s=s.assign(short=(s.total_amount-s.cash).round(2), rel=lambda d:(d.total_amount-d.cash)/d.total_amount)
bc=s[s.exception=='SHORT_BANK_CHARGES']
print(bc.short.round(2).value_counts().head(25))
c=s[(s.short>0.005)&(s.short<=60)&(s.rel<=0.02)]
print(c.exception.fillna('-').value_counts()); print(c.groupby(['exception','segment'],dropna=False).size())
print(s[(s.short<-0.005)].exception.fillna('-').value_counts())
