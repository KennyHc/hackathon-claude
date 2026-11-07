import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
pd.set_option('display.width',250)
g=pd.read_pickle('matching/out/cache/exc_dev/diag.pkl')
g['nb']=pd.cut(g.n,[0,1,5,22,100,2000])
for e in ['SHORT_DISCOUNT','SHORT_WITHHOLDING','PARTIAL','SHORT_DISPUTE','FX_DIFFERENCE']:
    d=g[g.gexc==e]
    print(e, 'miss by size bucket'); print(d.groupby('nb',observed=True).apply(lambda x:pd.Series({'rows':len(x),'rec':(x.pexc==e).mean().round(3)})).T.to_string())
d=g[(g.gexc=='PARTIAL')&(g.pexc=='')]; print(d.segment.value_counts().head(8)); 
d=g[(g.gexc=='')&(g.pexc=='PARTIAL')]; print(d.segment.value_counts().head(8))
