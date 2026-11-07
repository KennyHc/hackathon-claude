"""Rebuild the sd_noadv models (reproducibility; run from anywhere).
  python matching/models/sd_noadv_build.py val|hold
"""
import sys, os, time
import os as _o
W=_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
sys.path.insert(0,W); os.chdir(W)
import pandas as pd
from matching.common import Ctx
from matching.matchers import sd_noadv as M
ph=sys.argv[1]; _o.makedirs(_o.path.join(W,'matching','out','sd_noadv'),exist_ok=True); t=time.time()
ctx=Ctx(ph)
base=pd.read_parquet(f'matching/out/{ph}/allocations.parquet')
ctx.allocations=base; ctx.claimed=set(base.bank_txn_id)
c=M.candidates(ctx)
g=pd.read_parquet(f'ingestion/out/{ph}/golden.parquet',columns=['bank_txn_id','item_id','amount','exception','segment'])
g=g[g.item_id!=''].rename(columns={'amount':'g_amount','exception':'g_exc'})
c=c.merge(g[['bank_txn_id','item_id','g_amount','g_exc']],on=['bank_txn_id','item_id'],how='left')
c['y']=c.g_amount.notna().astype(int)
seg=pd.read_parquet(f'ingestion/out/{ph}/golden.parquet',columns=['bank_txn_id','segment']).drop_duplicates('bank_txn_id')
c=c.merge(seg,on='bank_txn_id',how='left')
c.to_parquet(f'{W}/matching/out/sd_noadv/cand_{ph}.parquet',index=False)
print(ph,'lines',c.bank_txn_id.nunique(),'rows',len(c),'pos',c.y.sum(),f'{time.time()-t:.0f}s')
print(c.drop_duplicates('bank_txn_id').segment.value_counts().head(15))
