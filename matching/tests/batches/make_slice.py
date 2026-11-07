"""Build a pseudo-phase cache `slice23` (train 2023-04..06) under matching/out/cache/slice23 so that
`matchers.batches.match('slice23')` and `evalutil.report(..., 'slice23')` work unchanged. Second-period check only.
Usage: PYTHONUTF8=1 .venv/Scripts/python.exe matching/tests/batches/make_slice.py [name start end]"""
import sys, os, glob
sys.path.insert(0, '.')
import pandas as pd
from matching.lib import data

name = sys.argv[1] if len(sys.argv) > 1 else 'slice23'
START = sys.argv[2] if len(sys.argv) > 2 else '2023-04-01'
END = sys.argv[3] if len(sys.argv) > 3 else '2023-06-30'
out = os.path.join(data.CACHE, name); os.makedirs(out, exist_ok=True)
tr = data.phase_dir('train')
months = {d.strftime('%Y%m') for d in pd.date_range(START, END, freq='D')}

def files(sub):
    return [f for f in sorted(glob.glob(os.path.join(tr, sub, '*.csv'))) if not os.path.basename(f).startswith('._')
            and os.path.basename(f).rsplit('_', 1)[-1][:6] in months]

# bank
b = pd.read_csv(os.path.join(tr, 'bank_transactions.csv.gz'), dtype={"bank_code": str, "bank_subcode": str, "ref1": str, "ref2": str})
b['booking_date'] = pd.to_datetime(b.booking_date); b['value_date'] = pd.to_datetime(b.value_date)
b = b[(b.booking_date >= START) & (b.booking_date <= END)].copy()
b['direction'] = b.amount.gt(0).map({True: 'in', False: 'out'}); b['amount_abs'] = b.amount.abs().round(2); b['entity'] = b.entity.astype(str)
b.to_parquet(f'{out}/bank.parquet', index=False)
g = pd.read_csv(os.path.join(tr, 'golden_allocations.csv.gz'), dtype={"exception": str, "category": str})
g = g[g.bank_txn_id.isin(set(b.bank_txn_id))]; g.to_parquet(f'{out}/golden.parquet', index=False)
print('bank', len(b), 'golden', len(g))
# items: AR invoices issued up to END (open_amount = total_amount), + credit notes
ar = pd.read_csv(os.path.join(tr, 'ar_invoices.csv.gz'), dtype={"document_number": str})
ar = ar[ar.issue_date <= END]
a = pd.DataFrame({"item_id": ar.item_id, "item_type": "ARI", "side": "AR", "entity": ar.entity, "party_id": ar.customer_id,
                  "party_name": ar.customer_name, "document_number": ar.document_number, "issue_date": pd.to_datetime(ar.issue_date),
                  "due_date": pd.to_datetime(ar.due_date), "currency": ar.currency, "total_amount": ar.total_amount,
                  "open_amount": ar.total_amount, "payment_terms": ar.payment_terms, "payment_method": ar.payment_method,
                  "po_number": ar.customer_po, "related_document": None})
a.to_parquet(f'{out}/items.parquet', index=False); print('items', len(a)); del ar, a
# batch files (reuse the shared loaders by temporarily pointing at the filtered list)
import matching.lib.data as D
def load(sub, f):
    fs = files(sub)
    return pd.concat([pd.read_csv(x, dtype=str).assign(_file=os.path.basename(x)) for x in fs], ignore_index=True)
d = load('direct_debit_batches', None)
d['amount'] = d.amount.astype(float).round(2); d['batch_total'] = d.batch_total.astype(float).round(2)
d['collection_date'] = pd.to_datetime(d.collection_date); d['entity'] = d.account_id.str[:4]; d['re_presentation'] = d.re_presentation.eq('Y')
d.to_parquet(f'{out}/dd.parquet', index=False)
c = load('card_settlements', None)
for k in ('gross', 'fee', 'net'): c[k] = c[k].astype(float).round(2)
c['settlement_date'] = pd.to_datetime(c.settlement_date); c['entity'] = c.account_id.str[:4]; c.to_parquet(f'{out}/card.parquet', index=False)
q = load('cheque_deposits', None)
q['amount'] = q.amount.astype(float).round(2); q['deposit_total'] = q.deposit_total.astype(float).round(2)
q['deposit_date'] = pd.to_datetime(q.deposit_date); q['entity'] = q.account_id.str[:4]; q.to_parquet(f'{out}/cheque.parquet', index=False)
print('dd', len(d), 'card', len(c), 'chq', len(q))
