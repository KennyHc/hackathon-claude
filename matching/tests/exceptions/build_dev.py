"""Build the dev slice (train golden booked 2023-2024 only) used to fit/evaluate exception_rules. Output: matching/out/cache/exc_dev/*.parquet"""
import sys, os
sys.path.insert(0, '.')
import pandas as pd
from matching.lib import data
OUT = os.path.join(data.CACHE, 'exc_dev'); os.makedirs(OUT, exist_ok=True)
LO, HI = '2023-01-01', '2025-01-01'
tr = data.phase_dir('train')
b = pd.read_csv(os.path.join(tr, 'bank_transactions.csv.gz'), dtype={"bank_code": str, "bank_subcode": str, "ref1": str, "ref2": str})
b['booking_date'] = pd.to_datetime(b.booking_date); b['value_date'] = pd.to_datetime(b.value_date)
b = b[(b.booking_date >= LO) & (b.booking_date < HI)].copy()
b['direction'] = b.amount.gt(0).map({True: 'in', False: 'out'}); b['amount_abs'] = b.amount.abs().round(2); b['entity'] = b.entity.astype(str)
b.to_parquet(f'{OUT}/bank.parquet', index=False)
ids = set(b.bank_txn_id)
g = pd.read_csv(os.path.join(tr, 'golden_allocations.csv.gz'), dtype={"exception": str, "category": str})
g = g[g.bank_txn_id.isin(ids)].copy(); g.to_parquet(f'{OUT}/golden.parquet', index=False)
need = set(g.item_id.dropna())
ar = pd.read_csv(os.path.join(tr, 'ar_invoices.csv.gz'), dtype={"document_number": str})
ar = ar[ar.item_id.isin(need)]
a = pd.DataFrame({"item_id": ar.item_id, "item_type": "ARI", "side": "AR", "entity": ar.entity, "party_id": ar.customer_id, "party_name": ar.customer_name,
    "document_number": ar.document_number, "issue_date": ar.issue_date, "due_date": ar.due_date, "currency": ar.currency, "total_amount": ar.total_amount,
    "open_amount": ar.total_amount, "payment_terms": ar.payment_terms, "payment_method": ar.payment_method, "po_number": ar.customer_po, "related_document": None}); del ar
ap = pd.read_csv(os.path.join(tr, 'ap_invoices.csv.gz'), dtype={"supplier_invoice_number": str, "our_po": str})
ap = ap[ap.item_id.isin(need)]
p = pd.DataFrame({"item_id": ap.item_id, "item_type": "API", "side": "AP", "entity": ap.entity, "party_id": ap.supplier_id, "party_name": ap.supplier_name,
    "document_number": ap.supplier_invoice_number, "issue_date": ap.issue_date, "due_date": ap.due_date, "currency": ap.currency, "total_amount": ap.total_amount,
    "open_amount": ap.total_amount, "payment_terms": ap.payment_terms, "payment_method": ap.payment_method, "po_number": ap.our_po, "related_document": None}); del ap
cn = pd.read_csv(os.path.join(tr, 'credit_notes.csv.gz'), dtype={"document_number": str}); cn = cn[cn.item_id.isin(need)]
c = pd.DataFrame({"item_id": cn.item_id, "item_type": cn.item_id.str[:3], "side": cn.side, "entity": cn.entity, "party_id": cn.party_id, "party_name": cn.party_name,
    "document_number": cn.document_number, "issue_date": cn.issue_date, "due_date": None, "currency": cn.currency, "total_amount": cn.total_amount,
    "open_amount": cn.total_amount, "payment_terms": None, "payment_method": None, "po_number": None, "related_document": cn.related_document})
it = pd.concat([a, p, c], ignore_index=True).drop_duplicates('item_id')
for col in ('issue_date', 'due_date'): it[col] = pd.to_datetime(it[col])
it.to_parquet(f'{OUT}/items.parquet', index=False)
print(len(b), len(g), len(it), g.exception.value_counts().to_dict())
