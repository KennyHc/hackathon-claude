import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
m=pd.read_pickle('matching/out/cache/exc_dev/m.pkl')
d=m[m.exception.isin(['PARTIAL','SHORT_DISPUTE'])&(m.booking_date<'2025-01-01')]
d=d.assign(f=(1-d.amount/d.total_amount).round(2))
d=d[d.f.isin([0.03,0.05,0.1,0.2,0.3,0.5,0.7,0.4,0.6,0.67,0.75,0.83])]
b=pd.cut(d.n,[0,1,2,3,5,8,12,16,22,30,50,100,3000])
print(pd.crosstab(b,d.exception,normalize='index').round(3).join(pd.crosstab(b,d.exception).add_prefix('n_')))
