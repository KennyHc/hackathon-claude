"""K1 batch stage: bank lines explained by a detail file (payment runs, DD batches, card settlements, cheque
deposits) or a remittance advice, plus returns of those operations. Interface: docs/PLAN_LUIGI.md."""
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from matching.batch import cards_cheques, dd, pr  # noqa: E402
from matching.batch.common import OUT_COLS, out_frame  # noqa: E402

try:
    from matching.batch import remit  # noqa: E402
except Exception:  # parser not landed yet
    remit = None

SUBMATCHERS = [("pr", pr), ("dd", dd), ("cards_cheques", cards_cheques), ("remit", remit)]


def match(phase: str, lines: pd.DataFrame, items: pd.DataFrame, claimed: pd.DataFrame, **ctx) -> pd.DataFrame:
    """Rows only for lines this stage claims; each line claimed by at most one sub-matcher (first wins)."""
    taken = set(claimed.bank_txn_id) if claimed is not None and len(claimed) else set()
    out = []
    verbose = ctx.get("verbose", True)
    for name, mod in SUBMATCHERS:
        if mod is None:
            continue
        t0 = time.time()
        L = lines[~lines.bank_txn_id.isin(taken)]
        rows = mod.match(L, items, {"phase": phase, **ctx})
        df = out_frame(rows)
        df = df[~df.bank_txn_id.isin(taken)]
        # a line must not list the same item twice
        df = df[~df.bank_txn_id.isin(df[df.duplicated(["bank_txn_id", "item_id"])].bank_txn_id)]
        taken |= set(df.bank_txn_id)
        out.append(df)
        if verbose:
            print(f"  [batch/{name}] {df.bank_txn_id.nunique():,} lines, {len(df):,} rows, {time.time() - t0:.1f}s",
                  flush=True)
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=OUT_COLS)
    res = res[res.amount > 0]
    return res[OUT_COLS].reset_index(drop=True)
