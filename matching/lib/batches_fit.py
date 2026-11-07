"""Fit the only learned part of the `batches` matcher: per reason-code histogram of the delay between a direct-debit collection and
its return (7-day bins). Uses TRAIN labels booked before the cut only (default: a pseudo-phase cache built by
matching/tests/batches/make_slice.py from train 2022-H2, i.e. < 2025-01-01). Re-runnable with another slice / cut:

    PYTHONUTF8=1 .venv/Scripts/python.exe -m matching.lib.batches_fit --phase slice22 [--cut 2025-01-01]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from matching.lib import data, batches_core as bc  # noqa: E402


def fit(phase="slice22", cut="2025-01-01", out=None):
    from matching.matchers import batches as B
    bank = data.bank(phase)
    bank = bank[bank.booking_date < cut]
    g = data.golden(phase)
    g = g[g.bank_txn_id.isin(set(bank.bank_txn_id)) & g.item_id.notna() & g.exception.eq("RETURNED")][["bank_txn_id", "item_id"]]
    start = bank.booking_date.min()
    dd = B._cat(data.dd(phase), B._read_hist(phase, "direct_debit_batches", start, B._fix_dd))
    db = bank[bank.bank_txn_id.isin(set(g.bank_txn_id)) & (bank.amount < 0)]
    m = B._dd_candidates(db, dd)
    A = bc.prep_items(data.items(phase))
    R = pd.DataFrame({"rid": np.arange(len(m)), "entity": m.entity.values, "ccy": m.currency.values, "ref": m.remittance_info.values,
                      "amount": m.amount.values, "name": m.debtor_name.values, "adate": m.collection_date.values})
    res = bc.resolve(R, A)
    m = m.reset_index(drop=True); m["rid"] = np.arange(len(m))
    m = m.merge(res[["rid", "item_id"]], on="rid").merge(g, on=["bank_txn_id", "item_id"], suffixes=("_pred", ""))
    m = m.sort_values(["bank_txn_id", "lag"]).drop_duplicates("bank_txn_id")
    model = {}
    for code, x in m.groupby("code"):
        if len(x) >= 20:
            h = np.bincount((x.lag.values // B.LAG_BIN).clip(0, B.NBIN - 1), minlength=B.NBIN).astype(float) + 0.2
            model[code] = (h / h.sum()).round(5).tolist()
    h = np.bincount((m.lag.values // B.LAG_BIN).clip(0, B.NBIN - 1), minlength=B.NBIN).astype(float) + 0.2
    model["_all"] = (h / h.sum()).round(5).tolist()
    out = out or os.path.join(data.ROOT, "matching", "lib", "batches_lag_model.json")
    json.dump(model, open(out, "w"))
    print("fitted on", len(m), "returns before", cut, "codes", {k: round(float(np.dot(np.arange(B.NBIN) * B.LAG_BIN + 3, v)), 1) for k, v in model.items()})
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="slice22"); ap.add_argument("--cut", default="2025-01-01"); a = ap.parse_args()
    fit(a.phase, a.cut)
