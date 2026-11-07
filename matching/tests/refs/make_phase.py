"""Build a LIVE-shaped pseudo phase (cache only) from a slice of TRAIN, for second-period checks of matchers.

    PYTHONUTF8=1 .venv/Scripts/python.exe matching/tests/refs/make_phase.py --name v23 --start 2023-01-01 --end 2023-03-31

Writes matching/out/cache/<name>/{bank,items,golden,dd,card,cheque,runs}.parquet (what matching.lib.data reads through its cache).
AR side only (invoices + credit notes). open_amount at `start` = total - sum(golden before start), same rule as integration/make_val.py.
Dev tool only: the golden of the slice is for scoring.
"""
import argparse, os, sys
sys.path.insert(0, ".")
import pandas as pd
from matching.lib import data

TR = os.path.join(data.DATA, "train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    a = ap.parse_args()
    out = os.path.join(data.CACHE, a.name); os.makedirs(out, exist_ok=True)
    bt = pd.read_csv(os.path.join(TR, "bank_transactions.csv.gz"), dtype={"bank_code": str, "bank_subcode": str, "ref1": str, "ref2": str})
    bt["booking_date"] = pd.to_datetime(bt.booking_date); bt["value_date"] = pd.to_datetime(bt.value_date)
    bt["direction"] = bt.amount.gt(0).map({True: "in", False: "out"}); bt["amount_abs"] = bt.amount.abs().round(2); bt["entity"] = bt.entity.astype(str)
    booked = bt.set_index("bank_txn_id").booking_date
    win = bt[(bt.booking_date >= a.start) & (bt.booking_date <= a.end)].copy()
    win.to_parquet(os.path.join(out, "bank.parquet"), index=False)
    gold = pd.read_csv(os.path.join(TR, "golden_allocations.csv.gz"), dtype={"exception": str, "category": str})
    gold["bd"] = gold.bank_txn_id.map(booked)
    gwin = gold[gold.bank_txn_id.isin(set(win.bank_txn_id))].drop(columns="bd")
    gwin.to_parquet(os.path.join(out, "golden.parquet"), index=False)
    g = gold[gold.item_id.notna() & (gold.bd < a.start)]
    sign = g.exception.map({"RETURNED": -1.0, "DUPLICATE_PAYMENT": 0.0}).fillna(1.0)
    settled = (g.amount * sign).groupby(g.item_id).sum()
    touched = set(gold.item_id.dropna())
    del gold, g, bt
    ar = pd.read_csv(os.path.join(TR, "ar_invoices.csv.gz"), dtype={"document_number": str},
                     usecols=["item_id", "entity", "document_number", "issue_date", "due_date", "customer_id", "customer_name", "currency", "total_amount", "payment_terms", "customer_po", "payment_method"])
    ar = ar[ar.issue_date <= a.end]
    cn = pd.read_csv(os.path.join(TR, "credit_notes.csv.gz"), dtype={"document_number": str}); cn = cn[(cn.side == "AR") & (cn.issue_date <= a.end)]
    canc = cn[cn.reason.str.startswith("Anulaci")]
    cancelled = set(canc.item_id) | set(ar.item_id[pd.MultiIndex.from_arrays([ar.entity, ar.document_number]).isin(list(zip(canc.entity, canc.related_document)))])
    ar = ar[~ar.item_id.isin(cancelled)]; cn = cn[~cn.item_id.isin(cancelled)]
    a1 = pd.DataFrame({"item_id": ar.item_id, "item_type": "ARI", "side": "AR", "entity": ar.entity, "party_id": ar.customer_id, "party_name": ar.customer_name,
                       "document_number": ar.document_number, "issue_date": ar.issue_date, "due_date": ar.due_date, "currency": ar.currency,
                       "total_amount": ar.total_amount, "payment_terms": ar.payment_terms, "payment_method": ar.payment_method, "po_number": ar.customer_po, "related_document": None})
    c1 = pd.DataFrame({"item_id": cn.item_id, "item_type": "ARC", "side": "AR", "entity": cn.entity, "party_id": cn.party_id, "party_name": cn.party_name,
                       "document_number": cn.document_number, "issue_date": cn.issue_date, "due_date": None, "currency": cn.currency,
                       "total_amount": cn.total_amount, "payment_terms": None, "payment_method": None, "po_number": None, "related_document": cn.related_document})
    it = pd.concat([a1, c1], ignore_index=True)
    it["issue_date"] = pd.to_datetime(it.issue_date); it["due_date"] = pd.to_datetime(it.due_date)
    it = it[(it.issue_date >= "2021-07-01") | it.item_id.isin(touched)]
    it["open_amount"] = (it.total_amount - it.item_id.map(settled).fillna(0.0)).clip(lower=0, upper=it.total_amount).round(2)
    it = it[(it.issue_date >= a.start) | (it.open_amount > 0.005)]
    it.loc[it.issue_date >= a.start, "open_amount"] = it.total_amount
    it.to_parquet(os.path.join(out, "items.parquet"), index=False)
    for n in ("dd", "card", "cheque", "runs"):
        pd.DataFrame({"x": []}).to_parquet(os.path.join(out, n + ".parquet"), index=False)
    print(a.name, len(win), len(gwin), len(it))


main()
