#!/usr/bin/env python3
"""Build data/val/: a LIVE-shaped phase cut from the last half-year of TRAIN (2025-01-01 → 2025-06-30).

Same layout as data/live/ (so every piece just runs `--phase val`), plus golden_allocations.csv.gz for scoring:

  bank_transactions.csv.gz        bank lines booked in the window
  ar_invoices / ap_invoices / credit_notes.csv.gz   items issued in the window
  open_items_at_start.csv.gz      items issued before the window still open on 2025-01-01, rebuilt from the TRAIN golden
  bank_statements/ direct_debit_batches/ card_settlements/ cheque_deposits/ payment_runs/
  remittance_advices/ pdfs/       only the window's files (symlinks into data/train/) + filtered index.csv
  golden_allocations.csv.gz       labels of the window's bank lines (for integration/score.py only — never train on it)

Open amount at a cut date = total_amount − Σ golden amounts booked before the cut, where RETURNED rows re-open (+)
and DUPLICATE_PAYMENT rows don't reduce. Cancelled invoices ("Anulación de factura" credit note + the invoice it cancels)
and items issued before 2021-07 that the golden never settles are left out, like LIVE does.
Self-check: the same rebuild at 2025-07-01 vs the official data/live/open_items_at_start.csv.gz gives 325,639 vs 325,419
items, 321,460 in common (98.8 %), open_amount equal on 99.98 % of them. Known residue: ~4k pre-2021-07 untouched items
LIVE keeps (no visible rule), ~4k items LIVE treats as closed, APC due_date differs on 0.6 %.

    .venv/bin/python integration/make_val.py            # ~1-2 min
"""
import argparse
import glob
import os
import shutil
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TRAIN = os.path.join(DATA, "train")
START, END = "2025-01-01", "2025-06-30"
MONTHS = {f"2025{m:02d}" for m in range(1, 7)}
OPEN_COLS = ["item_id", "item_type", "entity", "party_id", "document_number", "issue_date", "due_date", "currency",
             "total_amount", "open_amount"]
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def read(path, **kw):
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kw)


def write(df, path):
    df.to_csv(path, index=False, compression="gzip")


def items_table():
    """All TRAIN items in the open_items_at_start schema."""
    ar = read(os.path.join(TRAIN, "ar_invoices.csv.gz"))
    ap = read(os.path.join(TRAIN, "ap_invoices.csv.gz"))
    cn = read(os.path.join(TRAIN, "credit_notes.csv.gz"))
    a = pd.DataFrame({"item_id": ar.item_id, "entity": ar.entity, "party_id": ar.customer_id,
                      "document_number": ar.document_number, "issue_date": ar.issue_date, "due_date": ar.due_date,
                      "currency": ar.currency, "total_amount": ar.total_amount})
    p = pd.DataFrame({"item_id": ap.item_id, "entity": ap.entity, "party_id": ap.supplier_id,
                      "document_number": ap.supplier_invoice_number, "issue_date": ap.issue_date,
                      "due_date": ap.due_date, "currency": ap.currency, "total_amount": ap.total_amount})
    c = pd.DataFrame({"item_id": cn.item_id, "entity": cn.entity, "party_id": cn.party_id,
                      "document_number": cn.document_number, "issue_date": cn.issue_date,
                      "due_date": cn.issue_date,  # LIVE uses issue_date as due_date for credit notes
                      "currency": cn.currency, "total_amount": cn.total_amount})
    it = pd.concat([a, p, c], ignore_index=True)
    it["item_type"] = it.item_id.str[:3]
    # Cancelled invoices ("Anulación de factura" credit note + the invoice it cancels) are never settled by bank
    # and LIVE's open_items_at_start leaves both out.
    canc = cn[cn.reason.str.startswith("Anulaci")]
    cancelled = set(canc.item_id) | set(
        it.item_id[it.set_index(["entity", "document_number"]).index.isin(
            list(zip(canc.entity, canc.related_document)))])
    it["cancelled"] = it.item_id.isin(cancelled)
    return it, ar, ap, cn


def open_at(items, golden, booked, cut):
    """Items issued before `cut` with an open balance on `cut` (rules reverse-engineered from LIVE)."""
    touched = set(golden.item_id[golden.item_id != ""])
    items = items[~items.cancelled]
    # items issued before TRAIN starts are context only unless the golden ever settles them
    items = items[(items.issue_date >= "2021-07-01") | items.item_id.isin(touched)]
    g = golden[golden.item_id != ""].copy()
    g["booking_date"] = g.bank_txn_id.map(booked)
    g = g[g.booking_date < cut]
    amt = g.amount.astype(float)
    sign = g.exception.map({"RETURNED": -1.0, "DUPLICATE_PAYMENT": 0.0}).fillna(1.0)
    settled = (amt * sign).groupby(g.item_id).sum()
    o = items[items.issue_date < cut].copy()
    tot = o.total_amount.astype(float)
    o["open"] = (tot - o.item_id.map(settled).fillna(0.0)).clip(lower=0).clip(upper=tot)
    o = o[o.open > 0.005]
    o["open_amount"] = o.open.map(lambda x: f"{x:.2f}")
    return o[OPEN_COLS]


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default=os.path.join(DATA, "val"))
    ap_.add_argument("--skip-check", action="store_true", help="skip the comparison with LIVE's open_items_at_start")
    a = ap_.parse_args()
    out = a.out
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    log("bank lines + golden")
    bt = read(os.path.join(TRAIN, "bank_transactions.csv.gz"))
    booked = bt.set_index("bank_txn_id").booking_date
    win = bt[(bt.booking_date >= START) & (bt.booking_date <= END)]
    write(win, os.path.join(out, "bank_transactions.csv.gz"))
    golden = read(os.path.join(TRAIN, "golden_allocations.csv.gz"))
    ids = set(win.bank_txn_id)
    gwin = golden[golden.bank_txn_id.isin(ids)]
    write(gwin, os.path.join(out, "golden_allocations.csv.gz"))
    log(f"  {len(win):,} bank lines, {len(gwin):,} golden rows")

    log("items issued in the window")
    items, ar, ap, cn = items_table()
    for name, df in [("ar_invoices", ar), ("ap_invoices", ap), ("credit_notes", cn)]:
        w = df[(df.issue_date >= START) & (df.issue_date <= END)]
        write(w, os.path.join(out, f"{name}.csv.gz"))
        log(f"  {name}: {len(w):,}")

    log("open_items_at_start (2025-01-01)")
    oi = open_at(items, golden, booked, START)
    write(oi, os.path.join(out, "open_items_at_start.csv.gz"))
    log(f"  {len(oi):,} open items " + str(oi.item_type.value_counts().to_dict()))

    gi = set(gwin.item_id[gwin.item_id != ""])
    avail = set(oi.item_id) | set(items.item_id[(items.issue_date >= START) & (items.issue_date <= END)])
    miss = gi - avail
    log(f"  coverage: {len(gi) - len(miss):,}/{len(gi):,} golden items of the window are available "
        f"({100 * (1 - len(miss) / max(len(gi), 1)):.2f}%)")

    log("files (symlinks)")
    n = 0
    for f in glob.glob(os.path.join(TRAIN, "bank_statements", "*", "*")):
        if os.path.basename(f).rsplit("_", 1)[-1][:6] in MONTHS:
            link(f, os.path.join(out, "bank_statements", os.path.basename(os.path.dirname(f)), os.path.basename(f)))
            n += 1
    for d in ["direct_debit_batches", "card_settlements", "cheque_deposits", "payment_runs"]:
        for f in glob.glob(os.path.join(TRAIN, d, "*.csv")):
            parts = os.path.basename(f)[:-4].split("_")
            if any(p in MONTHS for p in parts):
                link(f, os.path.join(out, d, os.path.basename(f)))
                n += 1
    for d, col in [("remittance_advices", "received_date"), ("pdfs", "document_date")]:
        idx = read(os.path.join(TRAIN, d, "index.csv"))
        w = idx[(idx[col] >= START) & (idx[col] <= END)]
        os.makedirs(os.path.join(out, d), exist_ok=True)
        w.to_csv(os.path.join(out, d, "index.csv"), index=False)
        for rel in w.file:
            src = os.path.join(TRAIN, d, rel)
            if os.path.exists(src):
                link(src, os.path.join(out, d, rel))
                n += 1
        log(f"  {d}: {len(w):,} of {len(idx):,}")
    log(f"  {n:,} files linked")

    with open(os.path.join(out, "README.txt"), "w") as fh:
        fh.write(f"Generated by integration/make_val.py: LIVE-shaped validation phase {START} → {END} cut from data/train.\n"
                 "golden_allocations.csv.gz is for scoring only (integration/score.py). Never train on this window.\n")

    if not a.skip_check:
        log("self-check: rebuild open items at 2025-07-01 vs official LIVE open_items_at_start")
        mine = open_at(items, golden, booked, "2025-07-01").set_index("item_id")
        off = read(os.path.join(DATA, "live", "open_items_at_start.csv.gz")).set_index("item_id")
        both = mine.index.intersection(off.index)
        same_amt = (mine.loc[both, "open_amount"].astype(float) - off.loc[both, "open_amount"].astype(float)).abs() <= 0.01
        same_cols = all((mine.loc[both, c] == off.loc[both, c]).mean() > 0.999 for c in OPEN_COLS[1:-1])
        log(f"  official {len(off):,} · rebuilt {len(mine):,} · in both {len(both):,} · "
            f"only official {len(off.index.difference(mine.index)):,} · only rebuilt {len(mine.index.difference(off.index)):,}")
        log(f"  open_amount equal on {100 * same_amt.mean():.2f}% of common items · other columns equal: {same_cols}")
    log(f"done → {out}")


if __name__ == "__main__":
    main()
