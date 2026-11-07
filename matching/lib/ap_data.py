"""AP matcher data access, memory-light: reads the shared caches of matching.lib.data (matching/out/cache/<phase>/) with only
the columns the AP matcher needs. Never loads data/train (team rule after the memory incident).

Optional `use_prior=True` (off by default) also loads the last months of the previous phase's payment runs and AP invoices
(late confirming debits, duplicates of items settled before the phase). It reads the previous phase folder, so it is
heavier; only use it on live/final if memory allows.
"""
from __future__ import annotations
import glob
import os
import pandas as pd
from . import data

BANK_COLS = ["bank_txn_id", "account_id", "entity", "statement_format", "booking_date", "amount", "currency", "ref1", "ref2", "text",
             "counterparty_name", "end_to_end_id", "remittance_info", "orig_amount", "orig_currency", "direction", "amount_abs"]
ITEM_COLS = ["item_id", "item_type", "side", "party_id", "party_name", "document_number", "issue_date", "due_date", "currency",
             "total_amount", "open_amount", "payment_method"]
RUN_COLS = ["run_id", "run_type", "execution_date", "debtor_account_id", "end_to_end_id", "beneficiary", "beneficiary_iban", "currency",
            "transfer_amount", "document_number", "document_type", "document_amount"]
PRIOR_DAYS = 300
LOOKBACK_MONTHS = 4
PHASE_START = {"val": "2025-01-01", "live": "2025-07-01", "final": "2026-01-01"}
PREV = {"val": "train", "live": "train", "final": "live"}


def _cache(phase, name, loader, cols):
    p = os.path.join(data.CACHE, phase, name + ".parquet")
    if not os.path.exists(p):
        loader(phase)  # builds the shared cache once
    import pyarrow.parquet as pq
    have = set(pq.read_schema(p).names)
    return pd.read_parquet(p, columns=[c for c in cols if c in have])


def bank_slice(phase, start=None, end=None):
    b = _cache(phase, "bank", data.bank, BANK_COLS)
    if start is not None:
        b = b[(b.booking_date >= start) & (b.booking_date <= end)]
    return b


def _read_runs_files(phase_dir_, months):
    fs = [f for f in glob.glob(os.path.join(phase_dir_, "payment_runs", "*.csv"))
          if not os.path.basename(f).startswith("._") and os.path.basename(f).split("_")[1] in months]
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f, dtype=str, usecols=RUN_COLS) for f in fs], ignore_index=True)
    for c in ("transfer_amount", "document_amount"):
        d[c] = d[c].astype(float).round(2)
    d["execution_date"] = pd.to_datetime(d.execution_date)
    return d


def runs_slice(phase, start=None, end=None, use_prior=False):
    r = _cache(phase, "runs", data.runs, RUN_COLS)
    if use_prior and phase in PREV:
        s = pd.Period(PHASE_START[phase], "M")
        months = {str(m).replace("-", "") for m in pd.period_range(s - LOOKBACK_MONTHS, s - 1, freq="M")}
        prev = _read_runs_files(data.phase_dir(PREV[phase]), months)
        prev = prev[prev.execution_date < PHASE_START[phase]]
        r = pd.concat([r, prev], ignore_index=True).drop_duplicates()
    r["entity"] = r.debtor_account_id.str[:4]
    return r


def ap_items(phase, start=None, end=None, use_prior=False):
    it = _cache(phase, "items", data.items, ITEM_COLS)
    it = it[it.side == "AP"].drop(columns=["side"]).copy()
    if "payment_method" not in it:
        it["payment_method"] = None
    it["prior"] = False
    # open_items_at_start carries no party name / payment method: take them from the supplier master
    try:
        sup = pd.read_csv(os.path.join(data.MASTERS, "suppliers.csv"), dtype=str, usecols=["supplier_id", "name", "payment_method"]).set_index("supplier_id")
        it["party_name"] = it.party_name.fillna(it.party_id.map(sup.name))
        it["payment_method"] = it.payment_method.fillna(it.party_id.map(sup.payment_method))
    except Exception as e:  # never fail on masters drift
        print("ap_data: supplier master fill failed:", e)
    if use_prior and phase in PREV:
        ph = pd.Timestamp(PHASE_START[phase])
        lo = ph - pd.Timedelta(days=PRIOR_DAYS)
        d = data.phase_dir(PREV[phase])
        ap = pd.read_csv(os.path.join(d, "ap_invoices.csv.gz"), dtype={"supplier_invoice_number": str},
                         usecols=["item_id", "supplier_id", "supplier_name", "supplier_invoice_number", "issue_date", "due_date",
                                  "currency", "total_amount", "payment_method"])
        ap["issue_date"] = pd.to_datetime(ap.issue_date)
        ap = ap[(ap.issue_date >= lo) & (ap.issue_date < ph) & ~ap.item_id.isin(it.item_id)]
        pr = pd.DataFrame({"item_id": ap.item_id, "item_type": "API", "party_id": ap.supplier_id, "party_name": ap.supplier_name,
                           "document_number": ap.supplier_invoice_number, "issue_date": ap.issue_date,
                           "due_date": pd.to_datetime(ap.due_date), "currency": ap.currency, "total_amount": ap.total_amount,
                           "open_amount": ap.total_amount, "payment_method": ap.payment_method, "prior": True})
        it = pd.concat([it, pr], ignore_index=True)
    return it
