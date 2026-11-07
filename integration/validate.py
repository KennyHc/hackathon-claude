#!/usr/bin/env python3
"""Check a submission before it is uploaded (any phase, labelled or not). Exit code 0 = safe to upload.

    .venv/bin/python integration/validate.py integration/out/live/submission.csv.gz --phase live
    .venv/bin/python integration/validate.py SUB --phase live --repair integration/out/live/submission.fixed.csv.gz

Checks: the official scorer's format checks (with a label-less golden of the phase), size/row limits, every bank line of
the phase present, item ids known in the phase (open_items_at_start + items issued in the phase; warning only).
--repair writes a fixed copy and logs every fix: drops rows of unknown lines / duplicates / invalid ids, clears category
on item rows, keeps item rows when a line has both, blanks unknown exceptions, adds missing lines as UNIDENTIFIED.
"""
import argparse
import gzip
import io
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integration.scoring import EXC, SUB_COLS, S, blank_golden, categories, phase_dir, score_bytes, txns_path  # noqa: E402


def read_sub(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def known_items(phase):
    d = phase_dir(phase)
    ids = set()
    for f in ["open_items_at_start", "ar_invoices", "ap_invoices", "credit_notes"]:
        p = os.path.join(d, f + ".csv.gz")
        if os.path.exists(p):
            ids |= set(pd.read_csv(p, dtype=str, keep_default_na=False, usecols=["item_id"]).item_id)
    return ids


def repair(sub, lines, cats, log):
    n0 = len(sub)
    sub = sub[SUB_COLS].copy()
    for c in SUB_COLS:
        sub[c] = sub[c].astype(str).str.strip()
    m = ~sub.bank_txn_id.isin(lines)
    if m.any():
        log(f"drop {m.sum()} rows whose bank_txn_id is not a line of the phase")
        sub = sub[~m]
    item = sub.item_id != ""
    bad = item & ~sub.item_id.str.fullmatch(S.ITEM_RE.pattern)
    if bad.any():
        log(f"drop {bad.sum()} rows with an invalid item_id")
        sub, item = sub[~bad], item[~bad]
    amt = pd.to_numeric(sub.amount, errors="coerce")
    bad = item & ~(amt >= 0)
    if bad.any():
        log(f"drop {bad.sum()} item rows with a missing/negative/non-numeric amount")
        sub, item, amt = sub[~bad], item[~bad], amt[~bad]
    sub.loc[item, "amount"] = amt[item].map(lambda x: f"{x:.2f}")
    sub.loc[~item, "amount"] = ""
    m = item & (sub.category != "")
    if m.any():
        log(f"clear category on {m.sum()} item rows")
        sub.loc[m, "category"] = ""
    m = ~sub.exception.isin(EXC + [""])
    if m.any():
        log(f"blank {m.sum()} unknown exception codes")
        sub.loc[m, "exception"] = ""
    m = ~item & ~sub.category.isin(cats)
    if m.any():
        log(f"set UNIDENTIFIED on {m.sum()} no-item rows with an unknown/empty category")
        sub.loc[m, ["category", "exception"]] = ["UNIDENTIFIED", "UNIDENTIFIED"]
    has_items = set(sub.bank_txn_id[item])
    m = ~item & sub.bank_txn_id.isin(has_items)
    if m.any():
        log(f"drop {m.sum()} category rows of lines that also have item rows (items kept)")
        sub = sub[~m]
    d = sub.duplicated(["bank_txn_id", "item_id"])
    if d.any():
        log(f"drop {d.sum()} duplicate (bank_txn_id, item_id) rows (first kept)")
        sub = sub[~d]
    miss = sorted(set(lines) - set(sub.bank_txn_id))
    if miss:
        log(f"add {len(miss)} missing lines as UNIDENTIFIED")
        sub = pd.concat([sub, pd.DataFrame({"bank_txn_id": miss, "item_id": "", "amount": "", "category": "UNIDENTIFIED",
                                            "exception": "UNIDENTIFIED"})], ignore_index=True)
    log(f"rows {n0:,} -> {len(sub):,}")
    return sub


def to_bytes(sub):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(sub[SUB_COLS].to_csv(index=False).encode())
    return buf.getvalue()


def check(path, phase, data_bytes=None, quiet=False):
    """Returns (ok, problems, warnings)."""
    data = data_bytes if data_bytes is not None else open(path, "rb").read()
    probs, warns = [], []
    res = score_bytes(blank_golden(phase), data, path)
    if not res["ok"]:
        probs += [f"scorer: {e}" for e in res["errors"]]
    raw = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
    if len(raw) > S.MAX_UNCOMPRESSED:
        probs.append(f"{len(raw) / 2**20:.1f} MB uncompressed > 60 MB limit")
    sub = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    if list(sub.columns[:5]) != SUB_COLS:
        probs.append(f"columns {list(sub.columns)} != {SUB_COLS}")
        return False, probs, warns
    if len(sub) > S.MAX_ROWS:
        probs.append(f"{len(sub):,} rows > {S.MAX_ROWS:,}")
    lines = set(pd.read_csv(txns_path(phase), dtype=str, usecols=["bank_txn_id"]).bank_txn_id)
    miss = lines - set(sub.bank_txn_id)
    if miss:
        probs.append(f"{len(miss):,} of {len(lines):,} bank lines missing (they would score as wrong), e.g. {sorted(miss)[:3]}")
    ids = known_items(phase)
    unk = sub.item_id[(sub.item_id != "") & ~sub.item_id.isin(ids)]
    if len(unk):
        warns.append(f"{len(unk):,} item rows reference an item not open/issued in {phase}, e.g. {list(unk[:3])}")
    if not quiet:
        n_item_lines = sub.bank_txn_id[sub.item_id != ""].nunique()
        print(f"{path}: {len(sub):,} rows · {sub.bank_txn_id.nunique():,}/{len(lines):,} lines · "
              f"{n_item_lines:,} with items · exceptions {sub.exception[sub.exception != ''].value_counts().to_dict()}")
    return not probs, probs, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--phase", required=True)
    ap.add_argument("--repair", metavar="OUT", help="write a repaired copy to OUT (.csv.gz) and validate it")
    a = ap.parse_args()
    if a.repair:
        lines = pd.read_csv(txns_path(a.phase), dtype=str, usecols=["bank_txn_id"]).bank_txn_id
        fixed = repair(read_sub(a.submission), set(lines), categories(), lambda m: print("  repair:", m))
        os.makedirs(os.path.dirname(os.path.abspath(a.repair)), exist_ok=True)
        open(a.repair, "wb").write(to_bytes(fixed))
        print(f"  repaired copy → {a.repair}")
        path = a.repair
    else:
        path = a.submission
    ok, probs, warns = check(path, a.phase)
    for w in warns:
        print("  WARN ", w)
    for p in probs:
        print("  ERROR", p)
    print("VALID ✓" if ok else "INVALID ✗ (do not upload; try --repair)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
