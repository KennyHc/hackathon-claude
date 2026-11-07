"""Shared loaders for the matching piece. Reads data/<phase>/ directly (ingestion v1 is not available) and returns tables
with the column names of docs/CONTRACTS.md, cached as parquet under matching/out/cache/<phase>/.

    from matching.lib import data
    bank  = data.bank("val")          # one row per bank line
    items = data.items("val")         # open items: open_items_at_start + items issued in the phase
    dd, card, chq, runs = data.dd("val"), data.card("val"), data.cheque("val"), data.runs("val")
    gold  = data.golden("val")        # only for scoring / learning from TRAIN before the cut. NEVER learn from val golden.

No-leakage (CLAUDE.md 2b): for `val` learn only from data.golden("train") rows booked before 2025-01-01.
"""
from __future__ import annotations

import glob
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "matching", "out", "cache")
MASTERS = os.path.join(DATA, "masters")
CUT = "2025-01-01"  # val no-leakage cut: learn only from train booked before this date


def phase_dir(phase: str) -> str:
    if phase == "train_lite":
        return os.path.join(DATA, "train-lite", "train")
    return os.path.join(DATA, phase)


def _cached(phase: str, name: str, build):
    p = os.path.join(CACHE, phase, name + ".parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    df = build()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_parquet(p, index=False)
    return df


def _csv_dir(phase: str, sub: str, usecols=None) -> pd.DataFrame:
    fs = sorted(f for f in glob.glob(os.path.join(phase_dir(phase), sub, "*.csv")) if not os.path.basename(f).startswith("._"))
    if not fs:
        return pd.DataFrame()
    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(lambda f: pd.read_csv(f, dtype=str, usecols=usecols).assign(_file=os.path.basename(f)), fs))
    return pd.concat(parts, ignore_index=True)


def bank(phase: str) -> pd.DataFrame:
    """Bank lines + direction, amount_abs. booking_date/value_date are datetime64."""
    def build():
        df = pd.read_csv(os.path.join(phase_dir(phase), "bank_transactions.csv.gz"), dtype={"bank_code": str, "bank_subcode": str, "ref1": str, "ref2": str})
        df["booking_date"] = pd.to_datetime(df.booking_date)
        df["value_date"] = pd.to_datetime(df.value_date)
        df["direction"] = df.amount.gt(0).map({True: "in", False: "out"})
        df["amount_abs"] = df.amount.abs().round(2)
        df["entity"] = df.entity.astype(str)
        return df
    return _cached(phase, "bank", build)


def items(phase: str) -> pd.DataFrame:
    """All items usable in `phase`: open_items_at_start (when the phase has it) + AR/AP invoices and credit notes issued in the
    phase. Columns follow CONTRACTS open_items. total_amount/open_amount are positive for credit notes too (item_type tells)."""
    def build():
        d = phase_dir(phase)
        parts = []
        p0 = os.path.join(d, "open_items_at_start.csv.gz")
        if os.path.exists(p0):
            o = pd.read_csv(p0, dtype={"document_number": str})
            o["side"] = o.item_type.str[:2].map({"AR": "AR", "AP": "AP"})
            parts.append(o)
        ar = pd.read_csv(os.path.join(d, "ar_invoices.csv.gz"), dtype={"document_number": str})
        a = pd.DataFrame({"item_id": ar.item_id, "item_type": "ARI", "side": "AR", "entity": ar.entity, "party_id": ar.customer_id,
                          "party_name": ar.customer_name, "document_number": ar.document_number, "issue_date": ar.issue_date,
                          "due_date": ar.due_date, "currency": ar.currency, "total_amount": ar.total_amount,
                          "open_amount": ar.total_amount, "payment_terms": ar.payment_terms, "payment_method": ar.payment_method,
                          "po_number": ar.customer_po, "related_document": None})
        ap = pd.read_csv(os.path.join(d, "ap_invoices.csv.gz"), dtype={"supplier_invoice_number": str, "our_po": str})
        p = pd.DataFrame({"item_id": ap.item_id, "item_type": "API", "side": "AP", "entity": ap.entity, "party_id": ap.supplier_id,
                          "party_name": ap.supplier_name, "document_number": ap.supplier_invoice_number, "issue_date": ap.issue_date,
                          "due_date": ap.due_date, "currency": ap.currency, "total_amount": ap.total_amount,
                          "open_amount": ap.total_amount, "payment_terms": ap.payment_terms, "payment_method": ap.payment_method,
                          "po_number": ap.our_po, "related_document": None})
        cn = pd.read_csv(os.path.join(d, "credit_notes.csv.gz"), dtype={"document_number": str})
        c = pd.DataFrame({"item_id": cn.item_id, "item_type": cn.item_id.str[:3], "side": cn.side, "entity": cn.entity, "party_id": cn.party_id,
                          "party_name": cn.party_name, "document_number": cn.document_number, "issue_date": cn.issue_date,
                          "due_date": None, "currency": cn.currency, "total_amount": cn.total_amount, "open_amount": cn.total_amount,
                          "payment_terms": None, "payment_method": None, "po_number": None, "related_document": cn.related_document})
        parts += [a, p, c]
        df = pd.concat(parts, ignore_index=True).drop_duplicates("item_id", keep="first")  # at_start wins: it carries the real open_amount
        for col in ("issue_date", "due_date"):
            df[col] = pd.to_datetime(df[col])
        return df
    return _cached(phase, "items", build)


def dd(phase: str) -> pd.DataFrame:
    def build():
        d = _csv_dir(phase, "direct_debit_batches")
        if d.empty:
            return d
        d["amount"] = d.amount.astype(float).round(2)
        d["batch_total"] = d.batch_total.astype(float).round(2)
        d["collection_date"] = pd.to_datetime(d.collection_date)
        d["entity"] = d.account_id.str[:4]
        d["re_presentation"] = d.re_presentation.eq("Y")
        return d
    return _cached(phase, "dd", build)


def card(phase: str) -> pd.DataFrame:
    def build():
        d = _csv_dir(phase, "card_settlements")
        if d.empty:
            return d
        for c in ("gross", "fee", "net"):
            d[c] = d[c].astype(float).round(2)
        d["settlement_date"] = pd.to_datetime(d.settlement_date)
        d["entity"] = d.account_id.str[:4]
        return d
    return _cached(phase, "card", build)


def cheque(phase: str) -> pd.DataFrame:
    def build():
        d = _csv_dir(phase, "cheque_deposits")
        if d.empty:
            return d
        d["amount"] = d.amount.astype(float).round(2)
        d["deposit_total"] = d.deposit_total.astype(float).round(2)
        d["deposit_date"] = pd.to_datetime(d.deposit_date)
        d["entity"] = d.account_id.str[:4]
        return d
    return _cached(phase, "cheque", build)


def runs(phase: str) -> pd.DataFrame:
    """Payment-run rows (one per document paid). `run_id` = our payment file (PR…/CF…); `end_to_end_id` = one transfer."""
    def build():
        d = _csv_dir(phase, "payment_runs")
        if d.empty:
            return d
        for c in ("transfer_amount", "document_amount"):
            d[c] = d[c].astype(float).round(2)
        d["execution_date"] = pd.to_datetime(d.execution_date)
        d["entity"] = d.debtor_account_id.str[:4]
        return d
    return _cached(phase, "runs", build)


def golden(phase: str) -> pd.DataFrame:
    """Labels. Only train/sample/val have them. Use for SCORING (val) or LEARNING (train before CUT) only."""
    def build():
        g = pd.read_csv(os.path.join(phase_dir(phase), "golden_allocations.csv.gz"), dtype={"exception": str, "category": str})
        return g
    return _cached(phase, "golden", build)


def suppliers() -> pd.DataFrame:
    return pd.read_csv(os.path.join(MASTERS, "suppliers.csv"), dtype=str)


def customers() -> pd.DataFrame:
    return pd.read_csv(os.path.join(MASTERS, "customers.csv"), dtype=str)


def payment_terms() -> pd.DataFrame:
    return pd.read_csv(os.path.join(MASTERS, "payment_terms.csv"))


def fx_rates() -> pd.DataFrame:
    return pd.read_csv(os.path.join(MASTERS, "fx_rates.csv"), parse_dates=["date"])
