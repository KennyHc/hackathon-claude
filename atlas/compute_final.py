"""FINAL-phase facts for the atlas (inputs only: FINAL has no answer key). Reads data/final/, writes atlas/agg_final.json.

    python3 atlas/compute_final.py     # ~10 s; build.py merges the result into the page
"""
import glob
import json
import os

import pandas as pd

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'final')
OUT = os.path.dirname(os.path.abspath(__file__)) + '/agg_final.json'


def nrows(p):
    return len(pd.read_csv(p, usecols=[0]))


def nfiles(sub):
    return len([f for f in glob.glob(f'{D}/{sub}/**/*', recursive=True) if os.path.isfile(f) and not f.endswith('index.csv')])


b = pd.read_csv(f'{D}/bank_transactions.csv.gz', usecols=['booking_date', 'entity', 'currency', 'statement_format', 'account_id'])
m = D + '/masters_final/'
ba = pd.read_csv(m + 'bank_accounts.csv')
za = b[b.entity == 'ZA01']
mon = b[b.account_id.isin(ba[ba.bank_name.str.contains('Moncayo')].account_id)]
agg = dict(
    bank_lines=len(b), first_date=b.booking_date.min(), last_date=b.booking_date.max(),
    monthly=[[k, int(v)] for k, v in b.booking_date.str[:7].value_counts().sort_index().items()],
    entity=[[k, int(v)] for k, v in b.entity.value_counts().items()],
    currency=[[k, int(v)] for k, v in b.currency.value_counts().items()],
    formats={k: int(v) for k, v in b.statement_format.value_counts().items()},
    n_entities=nrows(m + 'entities.csv'), n_accounts=len(ba), n_banks=int(ba.bank_name.nunique()),
    n_customers=nrows(m + 'customers.csv'), n_suppliers=nrows(m + 'suppliers.csv'),
    za_lines=len(za), za_formats={k: int(v) for k, v in za.statement_format.value_counts().items()},
    za_banks=sorted(ba[ba.entity == 'ZA01'].bank_name.unique().tolist()),
    za_currencies=sorted(ba[ba.entity == 'ZA01'].currency.unique().tolist()),
    moncayo_formats={k: int(v) for k, v in mon.statement_format.value_counts().items()},
    open_items_at_start=nrows(f'{D}/open_items_at_start.csv.gz'),
    ar_invoices=nrows(f'{D}/ar_invoices.csv.gz'), ap_invoices=nrows(f'{D}/ap_invoices.csv.gz'),
    credit_notes=nrows(f'{D}/credit_notes.csv.gz'), pdfs=nfiles('pdfs'), remittances=nfiles('remittance_advices'),
)
json.dump(agg, open(OUT, 'w'), indent=1)
print(json.dumps({k: v for k, v in agg.items() if k not in ('entity', 'currency')}, indent=1))
