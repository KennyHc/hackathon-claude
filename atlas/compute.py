"""Precompute aggregates for the dataset atlas. Reads data/ (incl. data/cache parquet copies), writes atlas/agg.json."""
import json, os, glob
import pandas as pd
import numpy as np

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUT = os.path.dirname(os.path.abspath(__file__)) + '/agg.json'
C = D + '/cache/'

g = pd.read_parquet(C + 'train__golden_allocations.csv.gz.parquet')
g['amount'] = pd.to_numeric(g['amount'], errors='coerce')
b = pd.read_parquet(C + 'train__bank_transactions.csv.gz.parquet')
lb = pd.read_parquet(C + 'live__bank_transactions.csv.gz.parquet')

agg = {}

# ---------- phases / volumes ----------
def nrows(p):
    return len(pd.read_csv(p, usecols=[0]))
def nfiles(p, sub):
    return len([f for f in glob.glob(f'{D}/{p}/{sub}/**/*', recursive=True) if os.path.isfile(f) and not f.endswith('index.csv')])

phases = {}
for p, bank in [('train', b), ('live', lb)]:
    ph = dict(
        bank_lines=len(bank),
        ar_invoices=nrows(f'{D}/{p}/ar_invoices.csv.gz'),
        ap_invoices=nrows(f'{D}/{p}/ap_invoices.csv.gz'),
        credit_notes=nrows(f'{D}/{p}/credit_notes.csv.gz'),
        pdfs=nfiles(p, 'pdfs'),
        remittances=nfiles(p, 'remittance_advices'),
        dd_files=nfiles(p, 'direct_debit_batches'),
        card_files=nfiles(p, 'card_settlements'),
        cheque_files=nfiles(p, 'cheque_deposits'),
        run_files=nfiles(p, 'payment_runs'),
        statements=nfiles(p, 'bank_statements'),
    )
    phases[p] = ph
phases['train']['golden_rows'] = len(g)
phases['live']['open_items_at_start'] = len(pd.read_parquet(C + 'live__open_items_at_start.csv.gz.parquet'))
agg['phases'] = phases

# monthly bank lines across train+live
m = pd.concat([b.booking_date.str[:7], lb.booking_date.str[:7]]).value_counts().sort_index()
agg['monthly'] = [[k, int(v)] for k, v in m.items()]

# ---------- LIVE-like window: TRAIN 2025-01..06 ----------
win = b[b.booking_date >= '2025-01-01']
wid = set(win.bank_txn_id)
gw = g[g.bank_txn_id.isin(wid)].copy()
gw['is_item'] = gw.item_id.notna() & (gw.item_id != '')
nlines = gw.bank_txn_id.nunique()
npairs = int(gw.is_item.sum())
agg['window'] = dict(lines=nlines, pairs=npairs, rows=len(gw), bank_lines=len(win))

seg = gw.groupby('segment').agg(lines=('bank_txn_id', 'nunique'), pairs=('is_item', 'sum')).reset_index()
FAM = {
 'receipts': ['ar_ref_noise','ar_1to1','ar_1toN_text','ar_1toN_credit_note','ar_n_to_m','ar_same_amount_fifo','ar_same_amount_month',
              'ar_third_party_parent','ar_third_party_agency','ar_third_party_confirming','ar_fx'],
 'payments': ['ap_payment_multi','ap_payment_single','ap_fx','ap_credit_note_netted','ap_paid_by_holding','ap_confirming','ap_factored',
              'ap_direct_debit','ap_payment_run_aggregated','ap_partial','ap_supplier_refund'],
 'batches': ['ar_dd_batch','ar_card_settlement','ar_cheque_deposit'],
 'remittances': ['ar_1toN_remittance','ar_short_dispute','ar_1to1_remittance','ar_dispute_settlement'],
 'exceptions': ['ar_short_discount','ar_discount_late','ar_short_withholding','ar_short_bank_charges','ar_partial','ar_overpayment',
                'ar_duplicate_payment','ap_duplicate_payment','ar_dd_returned','ar_cheque_returned','ar_card_chargeback',
                'ap_returned_transfer','ar_dd_represented_transfer'],
 'pdf': ['ar_instalment','ar_retention','ar_retention_release','ar_advance_deduction','ar_replaced_invoice','ar_credit_note_paid_out','ar_new_iban'],
}
def fam(s):
    for f, l in FAM.items():
        if s in l: return f
    if s.startswith('nonitem') or s in ('customer_refund','customer_advance','ap_duplicate_refund','unidentified','unidentified_lookalike'):
        return 'nonitem'
    raise ValueError(s)
seg['family'] = seg.segment.map(fam)
seg['line_pct'] = seg.lines / nlines
seg['pair_pct'] = seg.pairs / npairs
seg['points'] = 30 * seg.line_pct + 50 * seg.pair_pct
seg = seg.sort_values('points', ascending=False)
agg['segments'] = seg.round(5).to_dict('records')
fam_t = seg.groupby('family').agg(lines=('lines','sum'), pairs=('pairs','sum'), n=('segment','count')).reset_index()
fam_t['line_pct'] = fam_t.lines / nlines
fam_t['pair_pct'] = fam_t.pairs / npairs
fam_t['points'] = 30 * fam_t.line_pct + 50 * fam_t.pair_pct
agg['families'] = fam_t.round(5).to_dict('records')

# ---------- exceptions (window) ----------
ex = gw[gw.exception.notna() & (gw.exception != '')]
exc = ex.groupby('exception').agg(rows=('bank_txn_id','size'), lines=('bank_txn_id','nunique')).reset_index().sort_values('rows', ascending=False)
agg['exceptions'] = exc.to_dict('records')
agg['exc_rows_total'] = len(ex)
agg['lines_with_exc'] = int(ex.bank_txn_id.nunique())

# ---------- items per line ----------
ipl = gw.groupby('bank_txn_id').is_item.sum()
bins = [(0,0,'No item'),(1,1,'1'),(2,5,'2–5'),(6,20,'6–20'),(21,100,'21–100'),(101,10**9,'100+')]
rows = []
for lo, hi, lab in bins:
    sel = ipl[(ipl >= lo) & (ipl <= hi)]
    rows.append(dict(bucket=lab, lines=int(len(sel)), pairs=int(sel.sum())))
agg['items_per_line'] = rows
agg['max_items'] = int(ipl.max())
top = ipl.sort_values(ascending=False).head(1)
agg['max_items_segment'] = gw[gw.bank_txn_id == top.index[0]].segment.iloc[0]

# ---------- categories ----------
cat = gw[gw.category.notna() & (gw.category != '')].category.value_counts()
agg['categories'] = [[k, int(v)] for k, v in cat.items()]

# ---------- difficulty ----------
dl = gw.groupby('bank_txn_id').difficulty.first().value_counts().sort_index()
agg['difficulty'] = {str(k): int(v) for k, v in dl.items()}

# ---------- world (LIVE) ----------
agg['live_formats'] = {k: int(v) for k, v in lb.statement_format.value_counts().items()}
agg['live_formats_monthly'] = lb.groupby([lb.booking_date.str[:7], 'statement_format']).size().unstack(fill_value=0).reset_index().to_dict('records')
agg['live_currency'] = [[k, int(v)] for k, v in lb.currency.value_counts().items()]
agg['live_entity'] = [[k, int(v)] for k, v in lb.entity.value_counts().items()]
ent = pd.read_csv(D + '/masters/entities.csv')
agg['entities'] = ent[['entity','country','functional_currency','role']].to_dict('records')
ba = pd.read_csv(D + '/masters/bank_accounts.csv')
agg['n_banks'] = int(ba.bank.nunique()); agg['n_accounts'] = len(ba)
agg['n_customers'] = nrows(D + '/masters/customers.csv'); agg['n_suppliers'] = nrows(D + '/masters/suppliers.csv')
za = lb[lb.entity == 'ZA01']
agg['za_live'] = za[['booking_date','amount','currency']].to_dict('records')
# Moncayo
mon = lb[lb.bank_name.str.contains('Moncayo', na=False)] if 'bank_name' in lb else None
mon_acc = ba[ba.bank_name.str.contains('Moncayo')].account_id.tolist()
mm = lb[lb.account_id.isin(mon_acc)]
agg['moncayo'] = mm.groupby([mm.booking_date.str[:7], 'statement_format']).size().unstack(fill_value=0).reset_index().to_dict('records')

# remittance formats live / train
agg['rem_formats_live'] = pd.read_csv(D + '/live/remittance_advices/index.csv').format.value_counts().to_dict()
agg['pdf_types_live'] = pd.read_csv(D + '/live/pdfs/index.csv').document_type.value_counts().to_dict()
agg['open_items_types'] = pd.read_parquet(C + 'live__open_items_at_start.csv.gz.parquet').item_type.value_counts().to_dict()

json.dump(agg, open(OUT, 'w'), default=lambda o: o.item() if hasattr(o, 'item') else str(o), indent=1)
print(json.dumps({k: agg[k] for k in ['phases','window','families','exceptions','items_per_line','max_items','max_items_segment','difficulty','n_banks','n_accounts','za_live','moncayo','categories']}, default=str, indent=1))
