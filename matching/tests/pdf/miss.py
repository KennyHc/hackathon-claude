"""Why are golden lines of a segment not claimed? (dev only: sample / train slices)"""
import sys
sys.path.insert(0, '.')
import pandas as pd
from matching.lib import data, pdf_clauses
ph, seg = sys.argv[1], sys.argv[2]
out = pd.read_parquet(f'matching/out/{ph}/pdf_cases.parquet')
g = data.golden(ph); g = g[(g.segment == seg) & ~g.bank_txn_id.isin(out.bank_txn_id)]
it = data.items(ph).set_index('item_id')
cl = pdf_clauses.clauses(ph, verbose=False)
b = data.bank(ph).set_index('bank_txn_id')
for r in g.head(int(sys.argv[3]) if len(sys.argv) > 3 else 15).itertuples():
    i = it.loc[r.item_id] if r.item_id in it.index else None
    c = cl[(cl.entity == i.entity) & (cl.document_number == i.document_number)] if i is not None else pd.DataFrame()
    bl = b.loc[r.bank_txn_id]
    print(r.bank_txn_id, r.item_id, r.amount, r.exception, '| bank', bl.amount, bl.currency, '| item', None if i is None else (i.document_number, i.total_amount, i.open_amount, str(i.issue_date)[:10], i.currency))
    print('    clauses:', c[['clause_type', 'amount', 'payable', 'amounts', 'ref_document']].to_dict('records'))
    print('    text:', str(bl.text)[:150], '|', bl.ref2, bl.counterparty_name)
