"""Oracle eval: golden item sets (train 2023-2024 dev slice, or val/sample via evalutil) stripped to cash+item set -> assign -> compare.
  python matching/tests/exceptions/oracle.py --lo 2024-01-01 --hi 2024-04-01 [--mode full|cash] [--dispute-by-fraction]"""
import sys, argparse, time
sys.path.insert(0, '.')
import pandas as pd, numpy as np
from matching.lib import data
from matching.matchers.exception_rules import assign
ap = argparse.ArgumentParser()
ap.add_argument('--lo', default='2023-01-01'); ap.add_argument('--hi', default='2025-01-01')
ap.add_argument('--mode', default='full'); ap.add_argument('--no-dispute-fraction', action='store_true')
ap.add_argument('--phase', default='dev'); ap.add_argument('--eval-lo', default=None, help='score only lines booked from this date (earlier lines = ledger warm-up)')
a = ap.parse_args()
EXC = ['SHORT_DISCOUNT', 'SHORT_BANK_CHARGES', 'SHORT_WITHHOLDING', 'SHORT_DISPUTE', 'OVERPAYMENT', 'PARTIAL', 'DUPLICATE_PAYMENT', 'FX_DIFFERENCE', 'RETURNED']


def prf(g, p, label=EXC):
    rows = []
    for e in label:
        tp = ((g == e) & (p == e)).sum(); fp = ((g != e) & (p == e)).sum(); fn = ((g == e) & (p != e)).sum()
        P = tp / (tp + fp) if tp + fp else float('nan'); R = tp / (tp + fn) if tp + fn else float('nan')
        rows.append((e, tp, fp, fn, P, R, 2 * P * R / (P + R) if tp else 0.0))
    return pd.DataFrame(rows, columns=['type', 'tp', 'fp', 'fn', 'P', 'R', 'F1']).round(3)


if a.phase == 'dev':
    D = 'matching/out/cache/exc_dev/'
    g = pd.read_parquet(D + 'golden.parquet'); g = g[g.item_id.notna()]
    bank = pd.read_parquet(D + 'bank.parquet'); items = pd.read_parquet(D + 'items.parquet')
    bank = bank[(bank.booking_date >= a.lo) & (bank.booking_date < a.hi)]
    g = g[g.bank_txn_id.isin(set(bank.bank_txn_id))]
    phase = 'train'
else:
    phase = a.phase
    g = data.golden(phase); g = g[g.item_id.notna()]; bank = data.bank(phase); items = data.items(phase)
g = g.copy()
alloc = g[['bank_txn_id', 'item_id']].copy()
tot = items.set_index('item_id').open_amount
alloc['amount'] = alloc.item_id.map(tot) if a.mode == 'full' else g.amount.values
t = time.time()
out = assign(alloc, phase, bank=bank, items=items, dispute_by_fraction=not a.no_dispute_fraction)
print('assign secs', round(time.time() - t), 'rows', len(out), 'stats', assign.stats)
from matching.matchers import exception_rules as _er
g['ncand'] = g.bank_txn_id.map(_er.DEBUG)
g['pexc'] = out.exception.values; g['pamt'] = out.amount.values
if a.eval_lo:
    bd = bank.set_index('bank_txn_id').booking_date
    keep = g.bank_txn_id.map(bd) >= a.eval_lo
    g = g[keep.values].copy()
g['gexc'] = g.exception.fillna('')
g['pexc'] = g.pexc.fillna('')
print(prf(g.gexc, g.pexc).to_string())
f1 = prf(g.gexc, g.pexc)
print('macro F1 (9 types, UNIDENTIFIED excluded):', round(f1.F1.mean(), 4))
ok = (g.amount - g.pamt).abs() <= np.maximum(0.01, 0.005 * g.amount.abs())
print('amount acc', round(ok.mean(), 4), '| exception exact', round((g.gexc == g.pexc).mean(), 4))
# confusion of the errors
err = g[g.gexc != g.pexc]
print(pd.crosstab(err.gexc.replace('', '-'), err.pexc.replace('', '-')).to_string())
print('amount errors by golden exc:'); print(g[~ok].gexc.replace('', '-').value_counts().to_string())
g.to_pickle('matching/out/cache/exc_dev/last_oracle.pkl')
x = g[g.pexc == 'PARTIAL']
x = x.assign(nc=x.ncand.clip(-1, 3))
print('PARTIAL precision by #candidates:'); print(x.groupby('nc').apply(lambda d: pd.Series({'rows': len(d), 'prec': (d.gexc == 'PARTIAL').mean().round(3), 'dispute': (d.gexc == 'SHORT_DISPUTE').mean().round(3)})).to_string())
nn = g.groupby('bank_txn_id').item_id.transform('size')
x = g[g.pexc == 'PARTIAL'].assign(nb=pd.cut(nn[g.pexc == 'PARTIAL'], [0, 1, 5, 22, 2000]), nc=lambda d: d.ncand.clip(-1, 2))
print(x.groupby(['nc', 'nb'], observed=True).apply(lambda d: pd.Series({'rows': len(d), 'prec': (d.gexc == 'PARTIAL').mean().round(3), 'gold_empty': (d.gexc == '').mean().round(3),'dispute': (d.gexc == 'SHORT_DISPUTE').mean().round(3)})).to_string())
print(x[(x.gexc == '') & (x.nb.astype(str) == '(0, 1]')].segment.value_counts().head(8))
