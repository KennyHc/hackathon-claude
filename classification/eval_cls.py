"""Classifier-only evaluation + fast re-merge (matching output is fixed, so it is reused as is).

    python classification/eval_cls.py --phase val                       # classification/out/val/line_labels.parquet
    python classification/eval_cls.py --phase val --labels <other.parquet> [--labels <b.parquet> ...]
    python classification/eval_cls.py --phase val --labels a.parquet --unid-p 0.95 --p-min 0.2 --conf 0.9

For each labels file:
  * p_items vs "line has item rows" (golden): log loss, AUC
  * category accuracy on golden no-item lines (all of them, and the unclaimed ones that reach the submission)
  * UNIDENTIFIED precision / recall / F1 exactly as the official scorer counts it after the integration gate
  * veto: lines dropped from matching, and how many of them really had item rows (false drops)
  * FULL official score of the re-merged submission (integration.run.merge, unchanged, with this labels file)

The labels file is swapped into classification/out/<phase>/line_labels.parquet for the merge and the original is put
back afterwards. Thresholds can be overridden (monkeypatched on integration.run) to test a proposed change.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "starter"))
import bankrec_score as S  # noqa: E402

from integration import run as IR  # noqa: E402
from integration import score as SC  # noqa: E402


def _logloss_auc(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score
    q = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    return ll, float(roc_auc_score(y, p))


_GOLD = {}


def _golden(phase, bank_ids, golden):
    """integration.score.official_golden, but its temp files go to classification/out/zoo/."""
    if phase in _GOLD:
        return _GOLD[phase]
    import gzip
    d = os.path.join(ROOT, "classification", "out", "zoo", f"_gold_{phase}")
    os.makedirs(d, exist_ok=True)
    gp, tp = os.path.join(d, "_golden.csv.gz"), os.path.join(d, "_txns.csv.gz")
    with gzip.open(tp, "wt") as f:
        f.write("bank_txn_id\n" + "\n".join(bank_ids) + "\n")
    cols = ["bank_txn_id", "item_id", "amount", "category", "exception", "segment", "difficulty"]
    g = golden.copy()
    g["amount"] = g.amount.map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    g[cols].to_csv(gp, index=False, compression="gzip")
    _GOLD[phase] = S.load_golden(gp, tp, SC.EXC, SC.categories())
    return _GOLD[phase]


def evaluate(phase: str, labels_path: str, unid_p=None, p_min=None, conf=None, verbose=True) -> dict:
    t0 = time.time()
    ing = os.path.join(ROOT, "ingestion", "out", phase)
    bank = pd.read_parquet(os.path.join(ing, "bank_lines.parquet"))
    gold = pd.read_parquet(os.path.join(ing, "golden.parquet"))
    alloc = pd.read_parquet(os.path.join(ROOT, "matching", "out", phase, "allocations.parquet"))
    lab = pd.read_parquet(labels_path).set_index("bank_txn_id").reindex(bank.bank_txn_id)

    has_item = gold.item_id.ne("").groupby(gold.bank_txn_id).any().reindex(bank.bank_txn_id).fillna(False).values
    gcat = gold[gold.item_id == ""].drop_duplicates("bank_txn_id").set_index("bank_txn_id").category
    res = {"labels": os.path.relpath(labels_path, ROOT)}
    res["logloss"], res["auc"] = _logloss_auc(lab.p_items.values.astype(float), has_item.astype(float))
    noitem = ~has_item
    gc = gcat.reindex(bank.bank_txn_id).values
    res["cat_acc_noitem"] = float((lab.category.values[noitem] == gc[noitem]).mean())
    res["cat_wrong_noitem"] = int((lab.category.values[noitem] != gc[noitem]).sum())

    # veto as integration does it
    pm = IR.P_MIN if p_min is None else p_min
    co = IR.CONF_OVERRIDE if conf is None else conf
    la = lab.reindex(alloc.bank_txn_id.drop_duplicates())
    veto_ids = la.index[(la.p_items.values < pm) & (la.confidence.values >= co)]
    item_lines = set(bank.bank_txn_id[has_item])
    res["veto_lines"] = int(len(veto_ids))
    res["veto_false_drop"] = int(sum(1 for t in veto_ids if t in item_lines))

    # full re-merge with the unchanged integration merge; the labels file is redirected in memory (no file swap)
    dst = os.path.join(ROOT, "classification", "out", phase, "line_labels.parquet")
    old = (IR.P_MIN, IR.CONF_OVERRIDE, IR.UNID_P, IR.pd)

    class _PD:
        def __getattr__(self, k):
            return getattr(pd, k)

        def read_parquet(self, path, *a, **k):
            if os.path.abspath(path) == os.path.abspath(dst):
                path = labels_path
            return pd.read_parquet(path, *a, **k)

    try:
        IR.P_MIN, IR.CONF_OVERRIDE, IR.pd = pm, co, _PD()
        if unid_p is not None:
            IR.UNID_P = unid_p
        sub, _ = IR.merge(phase)
    finally:
        IR.P_MIN, IR.CONF_OVERRIDE, IR.UNID_P, IR.pd = old
    buf = io.BytesIO()
    sub.to_csv(buf, index=False, compression={"method": "gzip", "mtime": 0})
    sc = S.score_submission(_golden(phase, bank.bank_txn_id, gold), buf.getvalue(), "mem.csv.gz")
    assert sc["ok"], sc["errors"][:5]
    m = sc["metrics"]
    res.update(score4=round(100 * (0.5 * m["allocation_f1"] + 0.3 * m["txn_accuracy"] + 0.2 * m["exception_macro_f1"]), 4),
               score=round(sc["score"], 4), allocF1=round(m["allocation_f1"], 5), txnAcc=round(m["txn_accuracy"], 5),
               excF1=round(m["exception_macro_f1"], 5))
    sub = sub.astype({"amount": str})
    et = SC.exception_table(sub.fillna(""), gold).set_index("exception")
    u = et.loc["UNIDENTIFIED"]
    res.update(unid_tp=int(u.tp), unid_fp=int(u.fp), unid_gold=int(u.golden), unid_p=float(u.precision),
               unid_r=float(u.recall), unid_f1=float(u.f1))
    # unclaimed golden no-item lines: category right?
    cat_rows = sub[sub.item_id == ""].drop_duplicates("bank_txn_id").set_index("bank_txn_id").category
    un = cat_rows.index[cat_rows.index.isin(gcat.index)]
    res["cat_acc_submitted"] = float((cat_rows.loc[un] == gcat.loc[un]).mean())
    res["secs"] = round(time.time() - t0, 1)
    if verbose:
        print(" ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="val", choices=["sample", "val", "hold"])
    ap.add_argument("--labels", action="append", default=[])
    ap.add_argument("--unid-p", type=float, default=None)
    ap.add_argument("--p-min", type=float, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--csv", default="", help="append results to this CSV")
    a = ap.parse_args()
    paths = a.labels or [os.path.join(ROOT, "classification", "out", a.phase, "line_labels.parquet")]
    rows = [dict(phase=a.phase, **evaluate(a.phase, p, a.unid_p, a.p_min, a.conf)) for p in paths]
    if a.csv:
        df = pd.DataFrame(rows)
        df.to_csv(a.csv, mode="a", header=not os.path.exists(a.csv), index=False)


if __name__ == "__main__":
    main()
