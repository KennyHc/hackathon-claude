import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from matching.lib import data
m=pd.read_pickle('matching/out/cache/exc_dev/m.pkl')
pt=data.payment_terms().set_index('code')
m['pct']=m.payment_terms.map(pt.early_payment_discount_pct); m['dd']=m.payment_terms.map(pt.discount_days)
m['days']=(m.booking_date-m.issue_date).dt.days
s=m[(m.exception=='SHORT_DISCOUNT')]
print(s.pct.value_counts(dropna=False)); print((s.days-s.dd).describe()); print((s.days-s.dd).value_counts().sort_index().head(30))
# late partial
e=m[(m.pct.notna())&(m.side=='AR')]
e['late']=e.days-e.dd
print(pd.crosstab(e.late.clip(-5,40), e.exception.fillna('-')).to_string())
