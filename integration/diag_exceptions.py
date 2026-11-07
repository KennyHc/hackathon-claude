#!/usr/bin/env python3
"""Where do exception codes go wrong? Confusion of predicted vs golden exception on submission item rows.

    .venv/bin/python integration/diag_exceptions.py SUB.csv.gz [--phase val] [--code OVERPAYMENT] [--n 10]

For each predicted code: how many of its rows hit a golden (line, item) pair, and the golden code there; rows on
pairs not in the golden = wrong item. With --code, prints example rows (pred and golden amounts, cash, methods).
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from integration.scoring import golden_path, txns_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sub")
    ap.add_argument("--phase", default="val")
    ap.add_argument("--code", default=None)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--alloc", default=None, help="allocations.parquet to show the method of each row")
    a = ap.parse_args()
    s = pd.read_csv(a.sub, dtype=str, keep_default_na=False)
    g = pd.read_csv(golden_path(a.phase), dtype=str, keep_default_na=False)
    b = pd.read_csv(txns_path(a.phase), dtype=str, keep_default_na=False,
                    usecols=["bank_txn_id", "amount", "currency", "text", "booking_date"])
    si, gi = s[s.item_id != ""], g[g.item_id != ""]
    m = si.merge(gi[["bank_txn_id", "item_id", "amount", "exception", "segment"]], on=["bank_txn_id", "item_id"],
                 how="left", suffixes=("", "_g"), indicator=True)
    m["gold"] = m.exception_g.where(m._merge == "both", "<wrong pair>").replace("", "(none)")
    m["pred"] = m.exception.replace("", "(none)")
    print("predicted code → golden code on the same (line, item):")
    t = pd.crosstab(m.pred, m.gold, margins=True)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.to_string())
    # golden rows we never emitted, per code
    mg = gi.merge(si[["bank_txn_id", "item_id"]].assign(hit=1), on=["bank_txn_id", "item_id"], how="left")
    lines_cov = set(si.bank_txn_id)
    mg["state"] = mg.hit.notna().map({True: "pair emitted", False: ""})
    mg.loc[mg.hit.isna() & mg.bank_txn_id.isin(lines_cov), "state"] = "line covered, pair missing"
    mg.loc[mg.hit.isna() & ~mg.bank_txn_id.isin(lines_cov), "state"] = "line not covered"
    print("\ngolden code → coverage of its rows:")
    print(pd.crosstab(mg.exception.replace("", "(none)"), mg.state).to_string())
    if a.code:
        x = m[(m.pred == a.code) | (m.gold == a.code)].merge(b, on="bank_txn_id", suffixes=("", "_bank"))
        if a.alloc:
            al = pd.read_parquet(a.alloc)[["bank_txn_id", "item_id", "method"]]
            x = x.merge(al, on=["bank_txn_id", "item_id"], how="left")
        nrows = s[s.item_id != ""].groupby("bank_txn_id").size()
        x["rows"] = x.bank_txn_id.map(nrows)
        cols = ["bank_txn_id", "item_id", "amount", "amount_g", "amount_bank", "currency", "rows", "pred", "gold",
                "segment"] + (["method"] if a.alloc else []) + ["text"]
        with pd.option_context("display.width", 250, "display.max_colwidth", 60, "display.max_columns", 30):
            for (p, gcode), grp in x.groupby(["pred", "gold"]):
                if p == gcode:
                    continue
                print(f"\n--- pred {p} / gold {gcode}: {len(grp)} rows")
                print(grp[cols].head(a.n).to_string(index=False))


if __name__ == "__main__":
    main()
