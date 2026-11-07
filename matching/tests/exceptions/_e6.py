import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
pd.set_option('display.width',250)
m=pd.read_pickle('matching/out/cache/exc_dev/m.pkl')
x=m[m.segment.isin(['ar_n_to_m','ap_partial','ap_payment_run_aggregated','ar_instalment'])&(m.n>1)]
x=x.assign(rk_issue=x.groupby('bank_txn_id').issue_date.rank(method='first'),rk_amt=x.groupby('bank_txn_id').total_amount.rank(method='first'),isp=x.exception.eq('PARTIAL'))
for seg,d in x.groupby('segment'):
    p=d[d.isp]; lines=d.groupby('bank_txn_id').isp.sum()
    print(seg,'lines',len(lines),'npartial/line',lines.value_counts().head(4).to_dict())
    p=p.assign(first=p.rk_issue==1,last=p.rk_issue==p.n,big=p.rk_amt==p.n,small=p.rk_amt==1)
    print(p[['first','last','big','small']].mean().round(3).to_dict())
    print('ratio',(p.amount/p.total_amount).round(2).value_counts().head(5).to_dict())
