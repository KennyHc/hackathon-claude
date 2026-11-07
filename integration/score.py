"""Score a submission on a labelled phase (sample / val) with the OFFICIAL scorer + diagnostics.

    python integration/score.py --phase sample            # integration/out/sample/submission.csv.gz
    python integration/score.py --phase val --top 30
    python integration/score.py --phase val --dump ar_ref_noise   # wrong lines of a segment -> integration/out/val/dump_<seg>.csv

Prints: official score + components; segments sorted by points lost (≈ 30·lines_wrong/N + 50·pairs_missed/P);
per-method precision (share of claimed lines fully correct); per-exception P/R/F1.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "starter"))
import bankrec_score as S  # noqa: E402

from ingestion import lib  # noqa: E402

EXC = ["PARTIAL", "OVERPAYMENT", "SHORT_DISCOUNT", "SHORT_BANK_CHARGES", "SHORT_WITHHOLDING", "SHORT_DISPUTE",
       "DUPLICATE_PAYMENT", "UNIDENTIFIED", "RETURNED", "FX_DIFFERENCE"]


def categories() -> list[str]:
    return [r["category"] for r in csv.DictReader(open(os.path.join(lib.DATA, "masters", "categories.csv"), encoding="utf-8"))]


def official_golden(phase: str, bank_ids: pd.Series, golden: pd.DataFrame | None):
    """Golden object of the official scorer (empty labels for unlabelled phases -> validation only)."""
    d = lib.out_dir("integration", phase)
    gp, tp = os.path.join(d, "_golden.csv.gz"), os.path.join(d, "_txns.csv.gz")
    with gzip.open(tp, "wt") as f:
        f.write("bank_txn_id\n" + "\n".join(bank_ids) + "\n")
    cols = ["bank_txn_id", "item_id", "amount", "category", "exception", "segment", "difficulty"]
    if golden is None:
        golden = pd.DataFrame(columns=cols)
    g = golden.copy()
    g["amount"] = g.amount.map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    g[cols].to_csv(gp, index=False, compression="gzip")
    return S.load_golden(gp, tp, EXC, categories())


def line_correct(sub: pd.DataFrame, gold: pd.DataFrame) -> pd.Series:
    """bank_txn_id -> bool, same rule as the official transaction accuracy."""
    sub = sub.copy()
    sub["amount"] = pd.to_numeric(sub.amount, errors="coerce")
    gi, si = gold[gold.item_id != ""], sub[sub.item_id != ""]
    m = gi[["bank_txn_id", "item_id", "amount", "exception"]].merge(
        si[["bank_txn_id", "item_id", "amount", "exception"]], on=["bank_txn_id", "item_id"], how="outer",
        suffixes=("_g", "_s"), indicator=True)
    ok = (m._merge == "both") & ((m.amount_s - m.amount_g).abs() <= np.maximum(0.01, 0.005 * m.amount_g.abs()) + 1e-9) \
        & (m.exception_g.fillna("") == m.exception_s.fillna(""))
    item_ok = ok.groupby(m.bank_txn_id).all()
    gn, sn = gold[gold.item_id == ""].set_index("bank_txn_id"), sub[sub.item_id == ""].set_index("bank_txn_id")
    sn = sn[~sn.index.duplicated()]
    noitem_ok = (sn.reindex(gn.index).category.fillna("#") == gn.category) & \
                (sn.reindex(gn.index).exception.fillna("#") == gn.exception)
    # a no-item golden line is wrong if the submission has item rows for it
    noitem_ok &= ~gn.index.isin(set(si.bank_txn_id))
    # an item golden line is wrong if the submission has a no-item row for it
    item_ok = item_ok & ~item_ok.index.isin(set(sn.index))
    res = pd.concat([item_ok, noitem_ok])
    return res.groupby(level=0).all()   # a line can appear in both (mixed golden / submission rows)


def exception_table(sub: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    gk = gold.assign(k=gold.bank_txn_id + "|" + gold.item_id)
    noitem_lines = set(gold.bank_txn_id) - set(gold.bank_txn_id[gold.item_id != ""])
    gexc = dict(zip(gk.k, gk.exception))
    sk = sub.assign(k=sub.bank_txn_id + "|" + sub.item_id)
    rows = []
    for c in EXC:
        s = sk[sk.exception == c]
        hit = s.k.map(gexc).eq(c) & ((s.item_id != "") | s.bank_txn_id.isin(noitem_lines))
        tp, fp = int(hit.sum()), int((~hit).sum())
        n = int((gold.exception == c).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / n if n else 0.0
        rows.append((c, n, tp, fp, round(p, 3), round(r, 3), round(2 * p * r / (p + r), 3) if p + r else 0.0))
    return pd.DataFrame(rows, columns=["exception", "golden", "tp", "fp", "precision", "recall", "f1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="sample", choices=["sample", "val"])
    ap.add_argument("--file", default="")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--dump", default="", help="segment name: write its wrong lines to a CSV")
    a = ap.parse_args()
    t0 = time.time()
    od = lib.out_dir("integration", a.phase)
    path = a.file or os.path.join(od, "submission.csv.gz")
    bank = pd.read_parquet(os.path.join(ROOT, "ingestion", "out", a.phase, "bank_lines.parquet"))
    gold = pd.read_parquet(os.path.join(ROOT, "ingestion", "out", a.phase, "golden.parquet"))
    res = S.score_submission(official_golden(a.phase, bank.bank_txn_id, gold), open(path, "rb").read(), path)
    if not res["ok"]:
        print(json.dumps(res["errors"][:10], indent=1))
        sys.exit(1)
    m = res["metrics"]
    print(f"\n=== {a.phase}: SCORE {res['score']:.2f}   allocF1 {m['allocation_f1']:.3f} (P {m['precision']:.3f} "
          f"R {m['recall']:.3f})   txnAcc {m['txn_accuracy']:.3f}   excMacroF1 {m['exception_macro_f1']:.3f}")

    sub = pd.read_csv(path, dtype=str, keep_default_na=False)
    ok = line_correct(sub, gold).reindex(bank.bank_txn_id).fillna(False)
    N = len(bank)
    P = int((gold.item_id != "").sum())
    seg = gold.groupby("bank_txn_id").segment.first()
    pairs = gold[gold.item_id != ""].groupby("bank_txn_id").size().reindex(seg.index).fillna(0)
    df = pd.DataFrame({"segment": seg, "ok": ok.reindex(seg.index).values, "pairs": pairs})
    st = df.groupby("segment").agg(lines=("ok", "size"), acc=("ok", "mean"), pairs=("pairs", "sum"),
                                   pairs_wrong=("pairs", lambda x: x[~df.loc[x.index, "ok"]].sum()))
    st["pts_lost"] = 30 * st.lines * (1 - st.acc) / N + 50 * st.pairs_wrong / P
    st = st.sort_values("pts_lost", ascending=False)
    print(f"\n--- segments by points lost (top {a.top}) ---")
    print(st.head(a.top).assign(acc=lambda x: x.acc.round(3), pts_lost=lambda x: x.pts_lost.round(2)).to_string())

    mp = os.path.join(od, "methods.parquet")
    if os.path.exists(mp) and not a.file:
        meth = pd.read_parquet(mp).set_index("bank_txn_id").method.reindex(bank.bank_txn_id).fillna("?")
        mt = pd.DataFrame({"method": meth.values, "ok": ok.values}).groupby("method").agg(lines=("ok", "size"), precision=("ok", "mean"))
        print("\n--- per method (precision = share of its lines fully correct) ---")
        print(mt.sort_values("lines", ascending=False).round(3).to_string())

    print("\n--- exceptions ---")
    print(exception_table(sub, gold).to_string(index=False))
    if a.dump:
        ids = st.index  # noqa
        bad = df[(df.segment == a.dump) & ~df.ok].index
        b = bank.set_index("bank_txn_id").loc[bad, ["booking_date", "entity", "amount", "currency", "text", "counterparty_name", "remittance_info"]]
        g = gold[gold.bank_txn_id.isin(bad)].groupby("bank_txn_id").apply(
            lambda x: "; ".join(f"{i or x.category.iloc[0]}:{am}:{e}" for i, am, e in zip(x.item_id, x.amount, x.exception)))
        s = sub[sub.bank_txn_id.isin(bad)].groupby("bank_txn_id").apply(
            lambda x: "; ".join(f"{i or c}:{am}:{e}" for i, am, c, e in zip(x.item_id, x.amount, x.category, x.exception)))
        b["golden"], b["ours"] = g, s
        p = os.path.join(od, f"dump_{a.dump}.csv")
        b.to_csv(p)
        print(f"\nwrote {len(b)} wrong lines -> {p}")
    print(f"\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
