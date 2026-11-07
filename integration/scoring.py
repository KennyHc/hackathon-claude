"""Shared helpers for validate.py / score.py: wrap the official scorer (docs/starter/bankrec_score.py).

    from integration.scoring import golden_for, blank_golden, score_bytes, per_exception
"""
import csv
import gzip
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "docs", "starter"))
import bankrec_score as S  # noqa: E402

EXC = ["PARTIAL", "OVERPAYMENT", "SHORT_DISCOUNT", "SHORT_BANK_CHARGES", "SHORT_WITHHOLDING", "SHORT_DISPUTE",
       "DUPLICATE_PAYMENT", "UNIDENTIFIED", "RETURNED", "FX_DIFFERENCE"]
SUB_COLS = ["bank_txn_id", "item_id", "amount", "category", "exception"]


def phase_dir(phase, data="data"):
    return os.path.join(ROOT, data, phase)


def categories(data="data"):
    with open(os.path.join(ROOT, data, "masters", "categories.csv"), encoding="utf-8") as fh:
        return [r["category"] for r in csv.DictReader(fh)]


def txns_path(phase, data="data"):
    return os.path.join(phase_dir(phase, data), "bank_transactions.csv.gz")


def golden_path(phase, data="data"):
    return os.path.join(phase_dir(phase, data), "golden_allocations.csv.gz")


def has_golden(phase, data="data"):
    return os.path.exists(golden_path(phase, data))


def golden_for(phase, data="data"):
    """Official Golden for a labelled phase (train, train_lite, sample, val)."""
    return S.load_golden(golden_path(phase, data), txns_path(phase, data), EXC, categories(data))


def blank_golden(phase, data="data"):
    """Label-less Golden (every line of the phase, no labels): enough for the scorer's format checks on live/final."""
    with tempfile.NamedTemporaryFile("wb", suffix=".csv.gz", delete=False) as tmp:
        pass
    with gzip.open(tmp.name, "wt") as fh:
        fh.write("bank_txn_id,item_id,amount,category,exception\n")
    try:
        return S.load_golden(tmp.name, txns_path(phase, data), EXC, categories(data))
    finally:
        os.unlink(tmp.name)


def score_bytes(golden, data: bytes, filename: str) -> dict:
    return S.score_submission(golden, data, filename)


def per_exception(golden, sub):
    """Per-code precision/recall/F1 exactly as the scorer counts them. `sub`: DataFrame with SUB_COLS (str)."""
    gold = {c: 0 for c in EXC}
    for gitems, _, gexc, _ in golden.txns.values():
        if gitems:
            for _, ge in gitems.values():
                if ge in gold:
                    gold[ge] += 1
        elif gexc in gold:
            gold[gexc] += 1
    tp = {c: 0 for c in EXC}
    fp = {c: 0 for c in EXC}
    for t, item, exc in zip(sub.bank_txn_id, sub.item_id, sub.exception):
        if not exc:
            continue
        rec = golden.txns.get(t)
        if rec is None:
            continue
        if item:
            gi = rec[0].get(item)
            ok = gi is not None and gi[1] == exc
        else:
            ok = not rec[0] and rec[2] == exc
        if exc in tp:
            (tp if ok else fp)[exc] += 1
    rows = []
    for c in EXC:
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / gold[c] if gold[c] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"exception": c, "gold": gold[c], "pred": tp[c] + fp[c], "tp": tp[c], "precision": round(p, 4),
                     "recall": round(r, 4), "f1": round(f, 4), "counts": bool(gold[c])})
    return rows
