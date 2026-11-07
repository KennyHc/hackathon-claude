"""Load one phase of the hackathon data into pandas.

A *phase* is what we reconcile: its bank lines plus every item that can be settled during it, with the amount still
open at the start of the phase.

    ph = load_phase("live")                       # LIVE: open_items_at_start + items issued during LIVE
    ph = load_phase("train", start="2025-01-01")  # TRAIN window that mimics LIVE (open amounts rebuilt from golden)
    ph = load_phase("sample")                     # one month (2025-06), quick tests

Big CSVs are cached as parquet under data/cache/ the first time they are read.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "cache")

ITEM_COLS = ["item_id", "item_type", "side", "entity", "party_id", "party_name", "document_number", "issue_date",
             "due_date", "currency", "total_amount", "payment_terms", "po", "payment_method", "related_document", "reason"]


def read_csv(path: str, **kw) -> pd.DataFrame:
    """Read a csv(.gz) as strings (no NaN), cached as parquet for the big ones."""
    big = os.path.getsize(path) > 2_000_000
    cp = os.path.join(CACHE, os.path.relpath(path, DATA).replace(os.sep, "__") + ".parquet")
    if big and os.path.exists(cp) and os.path.getmtime(cp) >= os.path.getmtime(path):
        return pd.read_parquet(cp)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, **kw)
    if big:
        os.makedirs(CACHE, exist_ok=True)
        df.to_parquet(cp, index=False)
    return df


def _items_of(folder: str) -> pd.DataFrame:
    """All AR/AP invoices and credit notes of one data folder, in one common shape."""
    d = os.path.join(DATA, folder)
    ar = read_csv(os.path.join(d, "ar_invoices.csv.gz"))
    ar = pd.DataFrame({
        "item_id": ar.item_id, "item_type": "ARI", "side": "AR", "entity": ar.entity, "party_id": ar.customer_id,
        "party_name": ar.customer_name, "document_number": ar.document_number, "issue_date": ar.issue_date,
        "due_date": ar.due_date, "currency": ar.currency, "total_amount": ar.total_amount,
        "payment_terms": ar.payment_terms, "po": ar.customer_po, "payment_method": ar.payment_method,
        "related_document": "", "reason": ""})
    ap = read_csv(os.path.join(d, "ap_invoices.csv.gz"))
    ap = pd.DataFrame({
        "item_id": ap.item_id, "item_type": "API", "side": "AP", "entity": ap.entity, "party_id": ap.supplier_id,
        "party_name": ap.supplier_name, "document_number": ap.supplier_invoice_number, "issue_date": ap.issue_date,
        "due_date": ap.due_date, "currency": ap.currency, "total_amount": ap.total_amount,
        "payment_terms": ap.payment_terms, "po": ap.our_po, "payment_method": ap.payment_method,
        "related_document": "", "reason": ""})
    cn = read_csv(os.path.join(d, "credit_notes.csv.gz"))
    cn = pd.DataFrame({
        "item_id": cn.item_id, "item_type": np.where(cn.side == "AR", "ARC", "APC"), "side": cn.side,
        "entity": cn.entity, "party_id": cn.party_id, "party_name": cn.party_name,
        "document_number": cn.document_number, "issue_date": cn.issue_date, "due_date": "", "currency": cn.currency,
        "total_amount": cn.total_amount, "payment_terms": "", "po": "", "payment_method": "",
        "related_document": cn.related_document, "reason": cn.reason})
    out = pd.concat([ar, ap, cn], ignore_index=True)[ITEM_COLS]
    out["total_amount"] = out.total_amount.astype(float)
    return out


@dataclass
class Phase:
    name: str                       # sample / train / live / final
    folder: str                     # data sub-folder holding the bank lines and side files
    start: str                      # first booking date reconciled
    end: str                        # last booking date reconciled
    bank: pd.DataFrame              # bank lines to reconcile, sorted by booking date
    items: pd.DataFrame             # every item that may be settled, with `open_amount` at the start
    masters: dict = field(default_factory=dict)
    golden: pd.DataFrame | None = None   # labels for this window (TRAIN / sample only)

    def path(self, *p: str) -> str:
        return os.path.join(DATA, self.folder, *p)


def load_masters(folder: str | None = None) -> dict[str, pd.DataFrame]:
    """masters/ (FINAL brings final/masters_final/ which overrides the base files)."""
    m = {}
    for f in sorted(os.listdir(os.path.join(DATA, "masters"))):
        if f.endswith(".csv"):
            m[f[:-4]] = read_csv(os.path.join(DATA, "masters", f))
    if folder:
        upd = os.path.join(DATA, folder, "masters_final")
        if os.path.isdir(upd):
            for f in sorted(os.listdir(upd)):
                if f.endswith(".csv"):
                    m[f[:-4]] = read_csv(os.path.join(upd, f))
    return m


def _bank(folder: str) -> pd.DataFrame:
    b = read_csv(os.path.join(DATA, folder, "bank_transactions.csv.gz"))
    b["amount"] = b.amount.astype(float)
    b["orig_amount"] = pd.to_numeric(b.orig_amount, errors="coerce")
    return b.sort_values(["booking_date", "bank_txn_id"], kind="stable").reset_index(drop=True)


def _golden(folder: str) -> pd.DataFrame:
    g = read_csv(os.path.join(DATA, folder, "golden_allocations.csv.gz"))
    g["amount"] = pd.to_numeric(g.amount, errors="coerce")
    return g


def _open_from_golden(items: pd.DataFrame, golden: pd.DataFrame, bank_dates: pd.Series, cutoff: str) -> pd.Series:
    """Open amount of each item at `cutoff`, replaying the golden allocations booked before it."""
    g = golden[golden.item_id != ""]
    g = g[g.bank_txn_id.map(bank_dates) < cutoff]
    signed = np.where(g.exception == "RETURNED", g.amount, -g.amount)
    delta = pd.Series(signed, index=g.item_id.values).groupby(level=0).sum()
    total = items.set_index("item_id").total_amount
    open_ = (total + delta.reindex(total.index).fillna(0.0)).clip(lower=0.0)
    return open_.where(open_ <= total + 0.005, total).round(2)


def load_phase(name: str, start: str | None = None, end: str | None = None, with_golden: bool = True) -> Phase:
    if name in ("sample", "train"):
        folder = name
        bank_all = _bank(folder)
        golden_all = _golden(folder) if with_golden else None
        start = start or bank_all.booking_date.min()
        end = end or bank_all.booking_date.max()
        # item metadata: always from TRAIN (the sample is a slice of TRAIN; its item files are partial)
        items = _items_of("train").drop_duplicates("item_id")
        if name == "sample":
            gold_tr = _golden("train")
            bank_tr = _bank("train")
            dates = pd.Series(bank_tr.booking_date.values, index=bank_tr.bank_txn_id.values)
        else:
            gold_tr, dates = golden_all, pd.Series(bank_all.booking_date.values, index=bank_all.bank_txn_id.values)
        items["open_amount"] = items.item_id.map(_open_from_golden(items, gold_tr, dates, start))
        items = items[(items.open_amount > 0.005) | (items.issue_date >= start)]
        items = items[items.issue_date <= end]
        bank = bank_all[(bank_all.booking_date >= start) & (bank_all.booking_date <= end)].reset_index(drop=True)
        golden = None
        if with_golden:
            golden = golden_all[golden_all.bank_txn_id.isin(set(bank.bank_txn_id))].reset_index(drop=True)
        return Phase(name, folder, start, end, bank, items.reset_index(drop=True), load_masters(), golden)

    # LIVE / FINAL: open items at start + items issued during the phase; metadata looked up in earlier phases
    folder = name
    bank = _bank(folder)
    oi = read_csv(os.path.join(DATA, folder, "open_items_at_start.csv.gz"))
    oi["open_amount"] = oi.open_amount.astype(float)
    prev = {"live": ["train"], "final": ["train", "live"]}[name]
    meta = pd.concat([_items_of(p) for p in prev], ignore_index=True).drop_duplicates("item_id", keep="last")
    old = oi[["item_id", "open_amount"]].merge(meta, on="item_id", how="left")
    missing = old.entity.isna()
    if missing.any():   # items without metadata in earlier phases: rebuild from open_items_at_start
        o = oi[oi.item_id.isin(old.item_id[missing])]
        old.loc[missing, ITEM_COLS] = pd.DataFrame({
            "item_id": o.item_id, "item_type": o.item_type, "side": np.where(o.item_type.str.startswith("AR"), "AR", "AP"),
            "entity": o.entity, "party_id": o.party_id, "party_name": "", "document_number": o.document_number,
            "issue_date": o.issue_date, "due_date": o.due_date, "currency": o.currency,
            "total_amount": o.total_amount.astype(float), "payment_terms": "", "po": "", "payment_method": "",
            "related_document": "", "reason": ""}).set_index(old.index[missing])
        old = old.fillna("")
    new = _items_of(folder)
    new["open_amount"] = new.total_amount
    items = pd.concat([old[ITEM_COLS + ["open_amount"]], new[~new.item_id.isin(set(old.item_id))]], ignore_index=True)
    return Phase(name, folder, bank.booking_date.min(), bank.booking_date.max(), bank, items, load_masters(folder), None)


if __name__ == "__main__":
    import sys
    import time
    t = time.time()
    ph = load_phase(*sys.argv[1:2] or ["sample"], start=(sys.argv[2] if len(sys.argv) > 2 else None))
    print(ph.name, ph.start, ph.end, "bank", len(ph.bank), "items", len(ph.items),
          "open>0", int((ph.items.open_amount > 0.005).sum()), "golden", None if ph.golden is None else len(ph.golden),
          f"{time.time() - t:.1f}s")
