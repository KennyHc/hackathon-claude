"""ingestion v1: bank_lines.parquet + open_items.parquet (+ golden.parquet for labelled phases).

    python ingestion/run.py --phase sample|val|live|final
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion import lib, text  # noqa: E402

TEXT_COLS = ["text", "ref1", "ref2", "remittance_info", "end_to_end_id"]


def fx_eur(bank: pd.DataFrame, rates: pd.DataFrame) -> pd.Series:
    """amount in EUR at booking date (last available ECB/BAM rate)."""
    r = rates.copy()
    r["units_per_eur"] = r.units_per_eur.astype(float)
    r["date"] = pd.to_datetime(r.date)
    out = pd.Series(np.nan, index=bank.index)
    eur = bank.currency == "EUR"
    out[eur] = bank.amount[eur]
    b = bank.loc[~eur, ["currency", "booking_date", "amount"]].copy()
    b["date"] = pd.to_datetime(b.booking_date)
    b["_i"] = b.index
    m = pd.merge_asof(b.sort_values("date"), r.sort_values("date"), on="date", by="currency", direction="backward")
    out[m._i.values] = (m.amount / m.units_per_eur).values
    return out.round(2)


def bank_lines(bank: pd.DataFrame, m: dict) -> pd.DataFrame:
    b = bank.copy()
    b["direction"] = np.where(b.amount > 0, "in", "out")
    b["amount_abs"] = b.amount.abs().round(2)
    b["amount_eur"] = fx_eur(b, m["fx_rates"])
    b["party_name_raw"] = b.counterparty_name
    b["party_name_norm"] = text.norm_name_series(b.party_name_raw)
    joined = b[TEXT_COLS].agg(" ".join, axis=1)
    toks = joined.map(text.doc_tokens)
    b["doc_tokens"] = toks
    b["doc_refs"] = toks.map(lambda ts: sorted({d for d in (text.digits_norm(t) for t in ts) if len(d) >= 3}))
    b["batch_ref"] = None
    return b


def open_items(items: pd.DataFrame) -> pd.DataFrame:
    it = items.copy()
    it["party_name_norm"] = text.norm_name_series(it.party_name.fillna(""))
    it["document_number_norm"] = text.doc_number_norm_series(it.document_number.fillna(""))
    it["document_alnum"] = text.alnum_series(it.document_number.fillna(""))
    it["po_norm"] = text.alnum_series(it.po_number.fillna(""))
    for c in ("total_amount", "open_amount"):
        it[c] = it[c].astype(float).round(2)
    return it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="sample", choices=list(lib.PHASES))
    a = ap.parse_args()
    t = time.time()
    bank, items, m, golden, folder = lib.load(a.phase)
    od = lib.out_dir("ingestion", a.phase)
    bl = bank_lines(bank, m)
    bl.to_parquet(os.path.join(od, "bank_lines.parquet"), index=False)
    oi = open_items(items)
    oi.to_parquet(os.path.join(od, "open_items.parquet"), index=False)
    if golden is not None:
        golden.to_parquet(os.path.join(od, "golden.parquet"), index=False)
    print(f"[ingestion] {a.phase}: {len(bl):,} bank lines, {len(oi):,} items "
          f"({int((oi.open_amount > 0.005).sum()):,} open) -> {od}  {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
