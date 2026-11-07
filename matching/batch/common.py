"""Shared helpers for the batch stage (K1): data loading with train look-back, amount parsing, item indexes."""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PHASE_START = {"sample": "2025-06-01", "val": "2025-01-01", "train": "2021-07-01", "train_lite": "2021-07-01",
               "live": "2025-07-01", "final": "2026-01-01"}
PAYFACTORY = {"ES00": ["ES00", "ES01", "ES02", "PT01"]}  # ES00 pays supplier invoices of ES01, ES02, PT01

OUT_COLS = ["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"]


def phase_dir(phase):
    return os.path.join(DATA, phase)


def _month_key(fname):
    m = re.search(r"_(\d{6})(?:_\d+)?\.csv$", fname)
    return m.group(1) if m else None


def load_detail(phase, sub, lookback_months=3):
    """All CSVs of data/<phase>/<sub>/ plus those of data/train/<sub>/ for the `lookback_months` months before the
    phase start (detail files issued before the cash lands). Input documents, not labels → no leakage."""
    files = sorted(glob.glob(os.path.join(phase_dir(phase), sub, "*.csv")))
    start = PHASE_START.get(phase)
    if phase not in ("train", "train_lite") and start:
        y, m = int(start[:4]), int(start[5:7])
        months = []
        for _ in range(lookback_months):
            m -= 1
            if m == 0:
                y, m = y - 1, 12
            months.append(f"{y}{m:02d}")
        have = {os.path.basename(f) for f in files}
        prev = {"final": ["live", "train"]}.get(phase, ["train"])   # FINAL looks back into LIVE (2025-10..12)
        for src in prev:
            for f in sorted(glob.glob(os.path.join(DATA, src, sub, "*.csv"))):
                if _month_key(os.path.basename(f)) in months and os.path.basename(f) not in have:
                    files.append(f)
                    have.add(os.path.basename(f))
        # val is inside train: also the phase months themselves come from the val folder already
    dfs = [pd.read_csv(f, dtype=str, keep_default_na=False) for f in files]
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    return df


def load_lines(phase):
    p = os.path.join(phase_dir(phase), "bank_transactions.csv.gz")
    return pd.read_csv(p, dtype=str, keep_default_na=False)


_AMT_RE = re.compile(r"[^\d,.\-]")


def parse_amount(s):
    """'1,234.56' '1.234,56' '1234,56' '-1.111,25' '(1,111.25)' → float (nan if empty)."""
    if s is None:
        return np.nan
    s = str(s).strip()
    if not s:
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = _AMT_RE.sub("", s)
    if not s:
        return np.nan
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = head.replace(",", "") + "." + tail if len(tail) in (1, 2) else s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -abs(v) if neg else v


def norm_doc(s):
    return re.sub(r"[^0-9A-Z]", "", str(s).upper())


def days_between(a, b):
    """b - a in days for ISO strings (vectorised)."""
    return (pd.to_datetime(b, errors="coerce") - pd.to_datetime(a, errors="coerce")).dt.days


def out_frame(rows):
    df = pd.DataFrame(rows, columns=OUT_COLS)
    df["amount"] = df["amount"].astype(float).round(2)
    df["confidence"] = df["confidence"].astype(float)
    return df


def item_index(items, types):
    """(entity, document_number) → list of row dicts for items of the given item_type prefixes."""
    it = items[items.item_type.isin(types)]
    idx = {}
    for r in it[["item_id", "entity", "document_number", "party_id", "total_amount", "open_amount", "currency",
                 "issue_date", "due_date", "item_type"]].itertuples(index=False):
        idx.setdefault((r.entity, r.document_number), []).append(r)
    return idx
