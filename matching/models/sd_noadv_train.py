"""Rebuild the sd_noadv models (reproducibility; run from anywhere).
  python matching/models/sd_noadv_train.py 
"""
import sys, os, json
import os as _o
W=_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
S=_o.path.join(W,'matching','out','sd_noadv')
sys.path.insert(0,W)
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from matching.matchers import sd_noadv as M
C={ph:pd.read_parquet(f'{S}/cand_{ph}.parquet') for ph in ('val','hold')}
def fit(df):
    lr=LogisticRegression(C=1.0,class_weight=None,random_state=0,max_iter=2000).fit(df[M.FEATURES].to_numpy(float),df.y.to_numpy())
    return {"features":M.FEATURES,"coef":[round(float(x),6) for x in lr.coef_[0]],"intercept":round(float(lr.intercept_[0]),6),
            "n_rows":int(len(df)),"n_pos":int(df.y.sum()),"P_MIN":M.P_MIN,"COVER":M.COVER}
os.makedirs(f'{W}/matching/models',exist_ok=True)
models={'hold':fit(C['hold']),'val':fit(C['val']),'both':fit(pd.concat([C['val'],C['hold']]))}
for k,m in models.items(): json.dump(m,open(f'{W}/matching/models/sd_noadv_{k}.json','w'),indent=1)
print(dict(zip(M.FEATURES,models['both']['coef'])))
tol=lambda g: np.maximum(0.01,0.005*np.abs(g))
for ph in ('val','hold'):
    c=C[ph].copy(); m=models[M.MODEL_FOR[ph]]
    c['p']=M.predict(m,c)
    rows=pd.DataFrame(M.claim(c),columns=['bank_txn_id','item_id','amount','exception','confidence','method'])
    r=rows.merge(c[['bank_txn_id','item_id','g_amount','g_exc','segment','payer','booking_date']],on=['bank_txn_id','item_id'],how='left')
    r['pair']=r.g_amount.notna(); r['tp']=r.pair&((r.amount-r.g_amount).abs()<=tol(r.g_amount)+1e-9)
    seen=set(C['hold' if ph=='val' else 'val'].payer)
    r['unseen']=~r.payer.isin(seen); r['month']=r.booking_date.str[:7]
    print(f'\n=== {ph}: lines {r.bank_txn_id.nunique()} rows {len(r)} pair-hit {r.pair.mean():.3f} pair+amount precision {r.tp.mean():.3f}  TP {r.tp.sum()} FP {(~r.tp).sum()}')
    for k in ['month','segment','unseen']:
        print(r.groupby(k).agg(lines=('bank_txn_id','nunique'),rows=('tp','size'),prec=('tp','mean')).round(3).to_string())
    # line-level fully right?
    g=pd.read_parquet(f'{W}/ingestion/out/{ph}/golden.parquet',columns=['bank_txn_id','item_id'])
    gn=g[g.bank_txn_id.isin(r.bank_txn_id)&(g.item_id!='')].groupby('bank_txn_id').size()
    print('golden pairs on claimed lines',gn.sum(),' TP',r.tp.sum(),' recall-on-lines',round(r.tp.sum()/gn.sum(),3))
    rows.to_parquet(f'{S}/rows_{ph}.parquet',index=False)
