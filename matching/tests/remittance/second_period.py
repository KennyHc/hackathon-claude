"""Second-period check of the `remittance` matcher on a quarter of TRAIN (default 2023-Q1): parses only that quarter's advices,
loads only the bank lines / AR items / golden rows needed (never the whole of train).

    PYTHONUTF8=1 .venv/Scripts/python.exe matching/tests/remittance/second_period.py --start 2023-01-01 --end 2023-03-31
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from matching.lib import data, evalutil, remit_parse as rp  # noqa: E402
from matching.matchers import remittance as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2023-01-01")
ap.add_argument("--end", default="2023-03-31")
a = ap.parse_args()
start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
D = os.path.join(data.DATA, "train")

# bank slice
b = pd.read_csv(os.path.join(D, "bank_transactions.csv.gz"), dtype={"bank_code": str, "bank_subcode": str, "ref1": str, "ref2": str})
b["booking_date"] = pd.to_datetime(b.booking_date); b["value_date"] = pd.to_datetime(b.value_date)
b = b[(b.booking_date >= start) & (b.booking_date <= end)].copy()
b["direction"] = b.amount.gt(0).map({True: "in", False: "out"}); b["amount_abs"] = b.amount.abs().round(2); b["entity"] = b.entity.astype(str)
print("bank slice", len(b))

# AR items issued up to the end of the slice (train has no open_items_at_start: all invoices since 2021-03 are in the file)
parts = []
cols = ["item_id", "entity", "customer_id", "customer_name", "document_number", "issue_date", "currency", "total_amount", "payment_terms"]
for ch in pd.read_csv(os.path.join(D, "ar_invoices.csv.gz"), usecols=cols, dtype={"document_number": str}, chunksize=500_000):
    ch = ch[pd.to_datetime(ch.issue_date) <= end]
    parts.append(pd.DataFrame({"item_id": ch.item_id, "item_type": "ARI", "entity": ch.entity.astype(str), "party_id": ch.customer_id, "party_name": ch.customer_name,
                               "document_number": ch.document_number, "issue_date": pd.to_datetime(ch.issue_date), "currency": ch.currency,
                               "total_amount": ch.total_amount, "open_amount": ch.total_amount, "payment_terms": ch.payment_terms}))
cn = pd.read_csv(os.path.join(D, "credit_notes.csv.gz"), dtype={"document_number": str})
cn = cn[(cn.side == "AR") & (pd.to_datetime(cn.issue_date) <= end)]
parts.append(pd.DataFrame({"item_id": cn.item_id, "item_type": "ARC", "entity": cn.entity.astype(str), "party_id": cn.party_id, "party_name": cn.party_name,
                           "document_number": cn.document_number, "issue_date": pd.to_datetime(cn.issue_date), "currency": cn.currency,
                           "total_amount": cn.total_amount, "open_amount": cn.total_amount, "payment_terms": None}))
it = pd.concat(parts, ignore_index=True); del parts
it["fk"] = it.document_number.map(rp.norm_doc); it["dk"] = it.document_number.map(rp.digits_key)
print("AR items", len(it))

# advices (slice + 35 days before)
st = {}
L = rp.lines("train", date_from=start - pd.Timedelta(days=R.PREV_DAYS), date_to=end + pd.Timedelta(days=R.WIN_AFTER), cache=False, stats=st, quiet=True)
print("advice lines", len(L), "advices", L.ra_file.nunique()); rp.print_stats(st)

stats = {}
out = R.match_frames(L, b, it, data.customers(), data.payment_terms(), None, stats)
print(stats)

# golden of the slice only
ids = set(b.bank_txn_id)
g = pd.concat([ch[ch.bank_txn_id.isin(ids)] for ch in pd.read_csv(os.path.join(D, "golden_allocations.csv.gz"), dtype={"exception": str, "category": str}, chunksize=500_000)])
data.golden = lambda phase: g  # evalutil.report reads golden through data.golden
evalutil.report(out, "train", name=f"remittance on train {a.start}..{a.end}")
# precision by confidence and by segment with the slice gold
gi = g[g.item_id.notna()][["bank_txn_id", "item_id", "amount", "exception"]].rename(columns={"amount": "ga", "exception": "gexc"})
m = out.merge(gi, on=["bank_txn_id", "item_id"], how="left")
m["ok"] = m.ga.notna() & ((m.amount - m.ga).abs() <= (0.005 * m.ga.abs()).clip(lower=0.01))
print("pair precision", round(m.ok.mean(), 4), "| exception agree on correct pairs", round((m[m.ok].exception.fillna("") == m[m.ok].gexc.fillna("")).mean(), 4))
print(m.groupby(m.confidence.round(2)).ok.agg(["size", "mean"]).to_string())
print("exception confusion on correct pairs:")
print(m[m.ok].groupby([m.exception.fillna("-"), m.gexc.fillna("-")]).size().to_string())

# dump for rule analysis (train slice only)
os.makedirs(os.path.join(data.CACHE, "remit_slice"), exist_ok=True)
dm = m.merge(it[["item_id", "total_amount", "issue_date", "payment_terms", "party_id", "currency"]], on="item_id", how="left").merge(
    b[["bank_txn_id", "booking_date", "entity", "currency"]].rename(columns={"currency": "bank_cur"}), on="bank_txn_id", how="left")
dm.to_parquet(os.path.join(data.CACHE, "remit_slice", f"m_{a.start}.parquet"), index=False)
