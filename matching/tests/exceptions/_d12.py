import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from matching.matchers import exception_rules as er
pd.set_option('display.width',250)
g=pd.read_pickle('matching/out/cache/exc_dev/diag.pkl')
g=g[(g.bcur!=g.icur)&(g.n==1)&(g.side=='AP')|(g.bcur!=g.icur)&(g.n==1)&(g.side=='AR')]
fx=er._fx_tables()
g['day']=g.booking_date.values.astype('datetime64[D]').astype(np.int64)
g['exact']=(g.orig_currency==g.icur)&g.orig_amount.notna()
def conv(r):
    if r.exact: return abs(r.orig_amount)
    rb=er._rate(fx,r.bcur,r.day); ri=er._rate(fx,r.icur,r.day)
    return abs(r.cash)/rb*ri if rb and ri else np.nan
g['cp']=g.apply(conv,axis=1)
g['rel']=(g.total_amount-g.cp)/g.total_amount
for ex in [True,False]:
    d=g[g.exact==ex]; print('exact',ex,len(d))
    print(d.groupby(d.gexc.replace('', 'clean')).rel.describe(percentiles=[.01,.05,.5,.95,.99]).round(4).to_string())
d=g[g.exact&(g.rel>0.003)&(g.rel<0.12)]
print(d.groupby(['gexc',d.rel.round(2)]).size().to_string())
print(d.segment.value_counts().head())
# ledger-less caveat: FX rows with rel>0.12
d=g[g.exact&(g.rel>=0.12)&(g.gexc=='FX_DIFFERENCE')]; print(d.segment.value_counts().head())
