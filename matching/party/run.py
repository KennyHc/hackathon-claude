#!/usr/bin/env python3
"""K3 party stage, development entry point.

    .venv/bin/python matching/party/run.py --phase val        # → matching/out/val/alloc_party.parquet
    .venv/bin/python integration/run.py --phase val --b matching/out/val/alloc_party.parquet \
        --out matching/out/val/sub_party.csv.gz               # score it alone
    .venv/bin/python matching/party/eval.py --phase val       # per-segment accuracy of the K3 segments

Earlier stages' matching/out/<phase>/alloc_{batch,refs}.parquet are passed as `claimed` (their lines are dropped).
"""
import argparse
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from integration.lib.ledger import load_items  # noqa: E402
from integration.lib.masters import masters_dir  # noqa: E402
from integration.scoring import txns_path  # noqa: E402
import matcher  # noqa: E402

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--no-claimed", action="store_true", help="ignore earlier stages' outputs")
    ap.add_argument("--anywhere", action="store_true", help="history-free: master names anywhere in the text")
    ap.add_argument("--harvest", action="store_true", help="aliases from claimed lines of the phase + 2nd pass")
    ap.add_argument("--loeo", default="", help="leave this entity out of alias history (ZA01-like test)")
    ap.add_argument("--out", default="", help="output parquet (default matching/out/<phase>/alloc_party.parquet)")
    a = ap.parse_args()
    lines = pd.read_csv(txns_path(a.phase), dtype=str, keep_default_na=False)
    log(f"{len(lines):,} bank lines")
    items = load_items(a.phase)
    log(f"{len(items):,} items")
    outd = os.path.join(ROOT, "matching", "out", a.phase)
    cl = []
    if not a.no_claimed:
        for st in ("batch", "refs"):
            f = os.path.join(outd, f"cascade_{st}.parquet")  # what the cascade kept (preferred)
            if not os.path.exists(f):
                f = os.path.join(outd, f"alloc_{st}.parquet")
            if os.path.exists(f):
                df = pd.read_parquet(f)
                log(f"claimed by {st}: {df.bank_txn_id.nunique():,} lines")
                cl.append(df)
    claimed = pd.concat(cl, ignore_index=True) if cl else pd.DataFrame(
        columns=["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"])
    claimed = claimed.merge(lines[["bank_txn_id", "booking_date"]], on="bank_txn_id", how="left")
    free = lines[~lines.bank_txn_id.isin(set(claimed.bank_txn_id))]
    lab = os.path.join(ROOT, "classification", "out", a.phase, "line_labels.parquet")
    labels = pd.read_parquet(lab) if os.path.exists(lab) else None
    pp = os.path.join(outd, "pdf_clauses.parquet")
    pdf = pd.read_parquet(pp) if os.path.exists(pp) else None
    res = matcher.match(a.phase, free, items, claimed, line_labels=labels, pdf_clauses=pdf, masters_dir=masters_dir(a.phase), log=log,
                          party_anywhere=a.anywhere or None, party_harvest=a.harvest or None, loeo=a.loeo)
    os.makedirs(outd, exist_ok=True)
    f = a.out or os.path.join(outd, "alloc_party.parquet")
    os.makedirs(os.path.dirname(os.path.abspath(f)), exist_ok=True)
    res.to_parquet(f, index=False)
    log(f"wrote {len(res):,} rows / {res.bank_txn_id.nunique():,} lines → {os.path.relpath(f, ROOT)}")


if __name__ == "__main__":
    main()
