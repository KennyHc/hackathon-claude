#!/usr/bin/env python3
"""Score a submission locally on a labelled phase (val = the team's official local score; also sample, train).

    .venv/bin/python integration/score.py integration/out/val/submission.csv.gz --phase val --label "starter"

Prints score + components, per-exception P/R/F1 (as the scorer counts them), the segments losing the most txn-accuracy
points, and appends a row to integration/out/scores.csv (with the change vs the previous run of the same phase).
"""
import argparse
import datetime as dt
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integration.scoring import ROOT, golden_for, has_golden, per_exception, score_bytes  # noqa: E402

LOG = os.path.join(ROOT, "integration", "out", "scores.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--phase", default="val")
    ap.add_argument("--label", default="", help="short name for the log")
    ap.add_argument("--segments", type=int, default=15)
    a = ap.parse_args()
    if not has_golden(a.phase):
        sys.exit(f"phase {a.phase} has no golden_allocations.csv.gz (val: run integration/make_val.py)")
    g = golden_for(a.phase)
    data = open(a.submission, "rb").read()
    res = score_bytes(g, data, a.submission)
    if not res["ok"]:
        print("INVALID:", *res["errors"][:20], sep="\n  ")
        sys.exit(1)
    m = res["metrics"]
    print(f"SCORE {res['score']:.2f}  =  50·allocF1 {m['allocation_f1']:.4f} (P {m['precision']:.4f} R {m['recall']:.4f})"
          f" + 30·txnAcc {m['txn_accuracy']:.4f} + 20·excMacroF1 {m['exception_macro_f1']:.4f}"
          f"   [{m['txns_covered']:,}/{m['txns_total']:,} lines, {m['rows']:,} rows]")
    print(f"      points: alloc {50 * m['allocation_f1']:.2f} · txn {30 * m['txn_accuracy']:.2f} · "
          f"exc {20 * m['exception_macro_f1']:.2f}")

    sub = pd.read_csv(a.submission, dtype=str, keep_default_na=False)
    ex = pd.DataFrame(per_exception(g, sub))
    n_codes = max(int(ex.counts.sum()), 1)
    ex["pts_lost"] = ((1 - ex.f1) * 20 / n_codes).where(ex.counts, 0).round(2)
    print("\nper exception (each code present in the golden is worth 20/n points of macro-F1):")
    print(ex.drop(columns="counts").to_string(index=False))

    seg = pd.DataFrame([{"segment": k, "n": v["n"], "acc": v["accuracy"]} for k, v in res["segments"].items()])
    seg["txn_pts_lost"] = (30 * seg.n * (1 - seg.acc) / m["txns_total"]).round(2)
    print(f"\nsegments losing the most txn-accuracy points (top {a.segments}):")
    print(seg.sort_values("txn_pts_lost", ascending=False).head(a.segments).to_string(index=False))

    row = {"time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "phase": a.phase, "label": a.label,
           "file": os.path.relpath(os.path.abspath(a.submission), ROOT), "score": res["score"],
           "alloc_f1": m["allocation_f1"], "txn_acc": m["txn_accuracy"], "exc_f1": m["exception_macro_f1"]}
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=list(row))
    prev = old[old.phase == a.phase]
    if len(prev):
        p = prev.iloc[-1]
        print(f"\nvs previous {a.phase} run ({p['label'] or p['file']}, {p['time']}): score {res['score'] - p['score']:+.2f} · "
              f"alloc {m['allocation_f1'] - p['alloc_f1']:+.4f} · txn {m['txn_accuracy'] - p['txn_acc']:+.4f} · "
              f"exc {m['exception_macro_f1'] - p['exc_f1']:+.4f}")
    (pd.concat([old, pd.DataFrame([row])], ignore_index=True) if len(old) else pd.DataFrame([row])).to_csv(LOG, index=False)
    print(f"logged → {os.path.relpath(LOG, ROOT)}")


if __name__ == "__main__":
    main()
