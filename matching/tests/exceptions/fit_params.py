"""Fit exception_rules constants from TRAIN golden booked before `cut` (default 2025-01-01). Needs build_dev.py output (2023-2024 slice).
Fits: deficit fractions of PARTIAL / SHORT_DISPUTE rows, which fractions are dispute-only, withholding rates per country (check only)."""
import sys, json, argparse
sys.path.insert(0, '.')
import pandas as pd, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--cut', default='2025-01-01'); ap.add_argument('--out', default='matching/lib/exc_params.json')
a = ap.parse_args()
m = pd.read_pickle('matching/out/cache/exc_dev/m.pkl')
m = m[m.booking_date < a.cut]
d = m[m.exception.isin(['PARTIAL', 'SHORT_DISPUTE']) & m.total_amount.gt(0)]
d = d.assign(f=(1 - d.amount / d.total_amount).round(2))
d = d[d.f > 0.0]
from matching.lib import data
pt = data.payment_terms().set_index('code').early_payment_discount_pct / 100.0
d = d[~(d.exception.eq('PARTIAL') & ((d.payment_terms.map(pt) - d.f).abs() < 0.0051))]   # late-discount rows belong to the discount rule
# discount-late PARTIALs are explained by the discount rule (deficit = pct), not by fractions: drop deficit 0.02/0.01/0.03 PARTIAL rows
cnt = d.groupby(['f', 'exception']).size().unstack(fill_value=0)
cnt['n'] = cnt.sum(axis=1)
cnt = cnt[cnt.n >= 0.01 * cnt.n.sum()]
print(cnt.sort_values('n', ascending=False))
fr = [[float(f), float(r.n)] for f, r in cnt.iterrows()]
dis = [float(f) for f, r in cnt.iterrows() if r.get('SHORT_DISPUTE', 0) / r.n >= 0.2]
# smallest line size n* such that, for fraction-matched PARTIAL/DISPUTE rows, the dispute share of lines with >= n* items first exceeds 50 %
dd = d[d.f.isin(dis + list(cnt.index))]
sh = dd.groupby('n').exception.apply(lambda s: (s == 'SHORT_DISPUTE').mean())
nmin = next((int(n) for n in sorted(sh.index) if n >= 5 and sh[n:].head(10).mean() > 0.5), 17)
print('dispute_min_items', nmin)
json.dump({'partial_fractions': fr, 'dispute_fractions': dis, 'dispute_min_items': nmin, 'fit_cut': a.cut}, open(a.out, 'w'), indent=1)
print('written', a.out, fr, dis)
