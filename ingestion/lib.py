"""Shared loading helpers for ingestion (and read-only use by other pieces).

Phases
  sample  one TRAIN month (2025-06) with labels — seconds, for development
  val     TRAIN 2025-01-01 → 2025-06-30 with labels — LIVE-shaped validation window (~200k lines)
  live    LIVE 2025-07 → 2025-12 (no labels)
  final   FINAL 2026-01 → 2026-06 (no labels, released 18:00)

For sample/val the open amount of each item at the window start is rebuilt from the TRAIN golden, so the window looks
exactly like LIVE/FINAL (open_items_at_start + items issued during the phase).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("RECON_DATA") or os.path.join(ROOT, "data")   # RECON_DATA: dress-rehearsal data root
CACHE = os.path.join(ROOT, "ingestion", "out", "_cache")

PHASES = {
    #        bank/side-file folder, window start, window end, item folders (metadata), labelled
    "sample": ("train", "2025-06-01", "2025-06-30", ["train"], True),
    "val": ("train", "2025-01-01", "2025-06-30", ["train"], True),
    # untouched holdout: no rule or threshold was tuned on this window; learned pieces cut at 2024-01-01
    "hold": ("train", "2024-01-01", "2024-06-30", ["train"], True),
    "live": ("live", None, None, ["train", "live"], False),
    "final": ("final", None, None, ["train", "live", "final"], False),
}


def out_dir(piece: str, phase: str) -> str:
    d = os.path.join(ROOT, piece, "out", phase)
    os.makedirs(d, exist_ok=True)
    return d


def read_csv(path: str) -> pd.DataFrame:
    """CSV(.gz) as strings, no NaN; big files cached as parquet under ingestion/out/_cache."""
    big = os.path.getsize(path) > 2_000_000
    cp = os.path.join(CACHE, os.path.relpath(path, DATA).replace(os.sep, "__") + ".parquet")
    if big and os.path.exists(cp) and os.path.getmtime(cp) >= os.path.getmtime(path):
        return pd.read_parquet(cp)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if big:
        os.makedirs(CACHE, exist_ok=True)
        df.to_parquet(cp, index=False)
    return df


def masters(folder: str | None = None) -> dict[str, pd.DataFrame]:
    m = {f[:-4]: read_csv(os.path.join(DATA, "masters", f))
         for f in sorted(os.listdir(os.path.join(DATA, "masters"))) if f.endswith(".csv")}
    if folder:
        upd = os.path.join(DATA, folder, "masters_final")
        if os.path.isdir(upd):
            for f in sorted(os.listdir(upd)):
                if f.endswith(".csv"):
                    m[f[:-4]] = read_csv(os.path.join(upd, f))
    return m


def items_of(folder: str) -> pd.DataFrame:
    d = os.path.join(DATA, folder)
    if not os.path.exists(os.path.join(d, "ar_invoices.csv.gz")):
        return pd.DataFrame()
    ar = read_csv(os.path.join(d, "ar_invoices.csv.gz"))
    ap = read_csv(os.path.join(d, "ap_invoices.csv.gz"))
    cn = read_csv(os.path.join(d, "credit_notes.csv.gz"))
    out = pd.concat([
        pd.DataFrame({"item_id": ar.item_id, "item_type": "ARI", "side": "AR", "entity": ar.entity,
                      "party_id": ar.customer_id, "party_name": ar.customer_name, "document_number": ar.document_number,
                      "issue_date": ar.issue_date, "due_date": ar.due_date, "currency": ar.currency,
                      "total_amount": ar.total_amount, "payment_terms": ar.payment_terms,
                      "payment_method": ar.payment_method, "po_number": ar.customer_po, "related_document": "",
                      "reason": ""}),
        pd.DataFrame({"item_id": ap.item_id, "item_type": "API", "side": "AP", "entity": ap.entity,
                      "party_id": ap.supplier_id, "party_name": ap.supplier_name,
                      "document_number": ap.supplier_invoice_number, "issue_date": ap.issue_date,
                      "due_date": ap.due_date, "currency": ap.currency, "total_amount": ap.total_amount,
                      "payment_terms": ap.payment_terms, "payment_method": ap.payment_method, "po_number": ap.our_po,
                      "related_document": "", "reason": ""}),
        pd.DataFrame({"item_id": cn.item_id, "item_type": np.where(cn.side == "AR", "ARC", "APC"), "side": cn.side,
                      "entity": cn.entity, "party_id": cn.party_id, "party_name": cn.party_name,
                      "document_number": cn.document_number, "issue_date": cn.issue_date, "due_date": "",
                      "currency": cn.currency, "total_amount": cn.total_amount, "payment_terms": "",
                      "payment_method": "", "po_number": "", "related_document": cn.related_document,
                      "reason": cn.reason}),
    ], ignore_index=True)
    out["total_amount"] = out.total_amount.astype(float)
    return out


def bank_of(folder: str) -> pd.DataFrame:
    b = read_csv(os.path.join(DATA, folder, "bank_transactions.csv.gz"))
    b["amount"] = b.amount.astype(float)
    b["orig_amount"] = pd.to_numeric(b.orig_amount, errors="coerce")
    return b.sort_values(["booking_date", "bank_txn_id"], kind="stable").reset_index(drop=True)


def golden_train() -> pd.DataFrame:
    g = read_csv(os.path.join(DATA, "train", "golden_allocations.csv.gz"))
    g["amount"] = pd.to_numeric(g.amount, errors="coerce")
    return g


def open_at(items: pd.DataFrame, golden: pd.DataFrame, txn_dates: pd.Series, cutoff: str) -> pd.Series:
    """Open amount per item at `cutoff`, replaying golden allocations booked before it (RETURNED re-opens)."""
    g = golden[(golden.item_id != "") & (golden.exception != "DUPLICATE_PAYMENT")]
    g = g[g.bank_txn_id.map(txn_dates) < cutoff]
    delta = pd.Series(np.where(g.exception == "RETURNED", g.amount, -g.amount), index=g.item_id.values).groupby(level=0).sum()
    total = items.set_index("item_id").total_amount
    op = (total + delta.reindex(total.index).fillna(0.0)).clip(lower=0.0)
    return op.where(op <= total + 0.005, total).round(2)


def open_at_backward(items: pd.DataFrame, golden: pd.DataFrame, txn_dates: pd.Series, cutoff: str) -> pd.Series:
    """Open amount per item at `cutoff` (inside TRAIN): start from the official LIVE open_items_at_start (= state at
    2025-07-01) and undo the golden allocations booked in [cutoff, 2025-07-01). Exact, unlike a forward replay
    (items paid before TRAIN starts, or never settled by bank, are not open)."""
    oi = read_csv(os.path.join(DATA, "live", "open_items_at_start.csv.gz"))
    base = pd.Series(oi.open_amount.astype(float).values, index=oi.item_id.values)
    # a DUPLICATE_PAYMENT never reduced the open amount (the item was already settled): don't undo it
    g = golden[(golden.item_id != "") & (golden.exception != "DUPLICATE_PAYMENT")]
    d = g.bank_txn_id.map(txn_dates)
    g = g[(d >= cutoff) & (d < "2025-07-01")]
    delta = pd.Series(np.where(g.exception == "RETURNED", -g.amount, g.amount), index=g.item_id.values).groupby(level=0).sum()
    total = items.set_index("item_id").total_amount
    op = (base.reindex(total.index).fillna(0.0) + delta.reindex(total.index).fillna(0.0)).clip(lower=0.0)
    return op.where(op <= total + 0.005, total).round(2)


def load(phase: str):
    """-> (bank, items_with_open_amount, masters, golden_or_None, folder)"""
    folder, start, end, meta_folders, labelled = PHASES[phase]
    m = masters(folder)
    if labelled:
        bank_all = bank_of("train")
        golden = golden_train()
        items = items_of("train").drop_duplicates("item_id")
        dates = pd.Series(bank_all.booking_date.values, index=bank_all.bank_txn_id.values)
        # Items issued inside TRAIN: forward replay is exact (all their bank settlements are in the golden) and keeps
        # items that later disappear WITHOUT a bank payment (cancelled / netted) — LIVE's open_items_at_start has those
        # too, so they must stay as distractors. Older items (issued before TRAIN, unknown history): backward replay
        # from LIVE's official open list. Only labels booked BEFORE `start` are used for the forward part.
        # Known residual (measured for val): the backward part uses window labels for pre-TRAIN items; 5,092 such
        # items are open at 2025-01-01 and only 26 of them are included solely because they get paid in the window.
        fwd = open_at(items, golden, dates, start)
        bwd = open_at_backward(items, golden, dates, start)
        in_train = items.set_index("item_id").issue_date >= "2021-07-01"
        items["open_amount"] = items.item_id.map(fwd.where(in_train, bwd))
        new = items.issue_date >= start
        items.loc[new, "open_amount"] = items.loc[new, "total_amount"]   # issued during the window: fully open
        items = items[((items.open_amount > 0.005) | new) & (items.issue_date <= end)]
        bank = bank_all[(bank_all.booking_date >= start) & (bank_all.booking_date <= end)].reset_index(drop=True)
        golden = golden[golden.bank_txn_id.isin(set(bank.bank_txn_id))].reset_index(drop=True)
        return bank, items.reset_index(drop=True), m, golden, folder
    bank = bank_of(folder)
    oi = read_csv(os.path.join(DATA, folder, "open_items_at_start.csv.gz"))
    oi["open_amount"] = oi.open_amount.astype(float)
    meta = pd.concat([items_of(f) for f in meta_folders], ignore_index=True).drop_duplicates("item_id", keep="last")
    old = oi[["item_id", "open_amount"]].merge(meta, on="item_id", how="left")
    miss = old.entity.isna()
    if miss.any():
        o = oi.set_index("item_id").loc[old.item_id[miss]]
        fill = pd.DataFrame({"item_type": o.item_type.values,
                             "side": np.where(o.item_type.str.startswith("AR"), "AR", "AP"),
                             "entity": o.entity.values, "party_id": o.party_id.values, "party_name": "",
                             "document_number": o.document_number.values, "issue_date": o.issue_date.values,
                             "due_date": o.due_date.values, "currency": o.currency.values,
                             "total_amount": o.total_amount.astype(float).values, "payment_terms": "",
                             "payment_method": "", "po_number": "", "related_document": "", "reason": ""},
                            index=old.index[miss])
        for c in fill.columns:
            old.loc[miss, c] = fill[c]
    old = old.fillna("")
    new = items_of(folder)
    new["open_amount"] = new.total_amount
    items = pd.concat([old, new[~new.item_id.isin(set(old.item_id))]], ignore_index=True)
    items["total_amount"] = items.total_amount.astype(float)
    return bank, items, m, None, folder
