import sys, os, json, re
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, R + "/integration")
import pandas as pd, numpy as np
import score as SC
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

def seg_table(phase):
    bank = pd.read_parquet(f"{R}/ingestion/out/{phase}/bank_lines.parquet")
    gold = pd.read_parquet(f"{R}/ingestion/out/{phase}/golden.parquet")
    sub = pd.read_csv(f"{R}/integration/out/{phase}/submission.csv.gz", dtype=str, keep_default_na=False)
    ok = SC.line_correct(sub, gold).reindex(bank.bank_txn_id).fillna(False)
    N, P = len(bank), int((gold.item_id != "").sum())
    seg = gold.groupby("bank_txn_id").segment.first()
    pairs = gold[gold.item_id != ""].groupby("bank_txn_id").size().reindex(seg.index).fillna(0)
    df = pd.DataFrame({"segment": seg, "ok": ok.reindex(seg.index).values, "pairs": pairs})
    st = df.groupby("segment").agg(lines=("ok", "size"), acc=("ok", "mean"), pairs=("pairs", "sum"),
                                   pairs_wrong=("pairs", lambda x: x[~df.loc[x.index, "ok"]].sum()))
    st["pts_lost"] = 30 * st.lines * (1 - st.acc) / N + 50 * st.pairs_wrong / P
    st["max_pts"] = 30 * st.lines / N + 50 * st.pairs / P
    meth = pd.read_parquet(f"{R}/integration/out/{phase}/methods.parquet").set_index("bank_txn_id").method.reindex(bank.bank_txn_id).fillna("?")
    fam = meth.str.replace(r"^cls:.*", "classifier", regex=True).str.split(":").str[0] \
        .str.replace(r"^(cards_cheques|dd_batches|remittance|duplicates)_.*", r"\1", regex=True)
    mt = pd.DataFrame({"m": fam.values, "ok": ok.values}).groupby("m").agg(lines=("ok", "size"), precision=("ok", "mean"))
    ex = SC.exception_table(sub, gold)
    return st, mt, ex

# official totals from logs
def totals(log):
    t = open(log).read()
    m = re.search(r"SCORE ([\d.]+)\s+allocF1 ([\d.]+) \(P ([\d.]+) R ([\d.]+)\)\s+txnAcc ([\d.]+)\s+excMacroF1 ([\d.]+)", t)
    return dict(zip(["score", "allocation_f1", "precision", "recall", "txn_accuracy", "exception_macro_f1"], map(float, m.groups())))

v, h = totals(f"{R}/integration/out/val_latest.log"), totals(f"{R}/integration/out/hold_latest.log")
runs = [
    dict(run="Validation", window="TRAIN Jan–Jun 2025", role="Tuned on", lines=199987, **v),
    dict(run="Holdout", window="TRAIN Jan–Jun 2024", role="Never tuned on", lines=190169, **h),
    dict(run="LIVE v1", window="LIVE Jul–Dec 2025", role="Leaderboard (older code)", lines=203083, score=93.73,
         allocation_f1=0.9443, precision=0.9982, recall=0.8959, txn_accuracy=0.964, exception_macro_f1=0.8801),
    dict(run="LIVE v2", window="LIVE Jul–Dec 2025", role="Leaderboard (previous code)", lines=203083, score=95.58,
         allocation_f1=0.9487, precision=0.9983, recall=0.9038, txn_accuracy=0.9734, exception_macro_f1=0.947),
    dict(run="LIVE v3", window="LIVE Jul–Dec 2025", role="Leaderboard (current code)", lines=203083, score=95.62,
         allocation_f1=0.9487, precision=0.9983, recall=0.9038, txn_accuracy=0.9741, exception_macro_f1=0.9481),
]
json.dump(runs, open(f"{OUT}/runs.json", "w"), indent=1)

prog = [
    dict(step=1, label="Organisers' starter", score=11.28, phase="sample", note="Starter script as shipped"),
    dict(step=2, label="Our baseline", score=12.72, phase="val", note="Starter logic on our ingestion"),
    dict(step=3, label="First full pipeline", score=90.04, phase="val", note="8 matcher agents, categories, exception rules"),
    dict(step=4, label="UNIDENTIFIED threshold", score=90.81, phase="val", note="Flag only when p_items < 0.9; bad fallback removed"),
    dict(step=5, label="Second-wave matchers", score=96.40, phase="val", note="Duplicates, refunds, refs, subset-sum, rounding"),
    dict(step=6, label="Start-state leak fixed", score=96.36, phase="val", note="Honest re-measurement"),
    dict(step=7, label="UNIDENTIFIED threshold 0.97", score=v["score"], phase="val", note="Flat plateau 0.8–0.99; best in both windows"),
    dict(step=8, label="Untouched holdout", score=h["score"], phase="hold", note="Jan–Jun 2024, never tuned on"),
    dict(step=9, label="LIVE v1 (leaderboard)", score=93.73, phase="live", note="Older file, uploaded 12:38"),
    dict(step=10, label="LIVE v2 (leaderboard)", score=95.58, phase="live", note="Second-wave matchers, uploaded 13:14"),
    dict(step=11, label="LIVE v3 (leaderboard)", score=95.62, phase="live", note="Threshold 0.97, uploaded 14:18"),
]
json.dump(prog, open(f"{OUT}/progression.json", "w"), indent=1)

sv, mv, ev = seg_table("val")
sh, mh, eh = seg_table("hold")
live_v1 = json.load(open(f"{OUT}/live_v1_segments.json"))
segs = sv.join(sh, lsuffix="_val", rsuffix="_hold", how="outer").reset_index()
segs["acc_live_v1"] = segs.segment.map(live_v1)
segs["accuracy_live"] = segs.segment.map(json.load(open(f"{OUT}/live_v3_segments.json")))   # latest LIVE upload (v3)
segs = segs.rename(columns={"lines_val": "lines", "pairs_val": "pairs", "max_pts_val": "max_points", "acc_val": "accuracy_val",
                            "acc_hold": "accuracy_hold", "pts_lost_val": "points_lost_val", "pts_lost_hold": "points_lost_hold"})
segs = segs[["segment", "lines", "pairs", "max_points", "accuracy_val", "accuracy_hold", "acc_live_v1", "accuracy_live", "points_lost_val", "points_lost_hold"]]
segs = segs.rename(columns={"acc_live_v1": "accuracy_live_v1"}).sort_values("max_points", ascending=False)
segs = segs[segs.lines.notna()]
for c in segs.columns[1:]:
    segs[c] = segs[c].astype(float).round(4)
segs["lines"] = segs.lines.astype(int); segs["pairs"] = segs.pairs.astype(int)
segs.to_json(f"{OUT}/segments.json", orient="records", indent=1)

meth = mv.join(mh, lsuffix="_val", rsuffix="_hold", how="outer").reset_index().rename(columns={"m": "matcher"})
order = ["payment_runs", "dd_batches", "cards_cheques", "remittance", "pdf_cases", "refs", "subset_sum", "same_amount",
         "short_payments", "ap_refunds", "duplicates", "classifier"]
meth["order"] = meth.matcher.map({m: i + 1 for i, m in enumerate(order)})
meth = meth[meth.order.notna()].sort_values("order")
for c in ["precision_val", "precision_hold"]:
    meth[c] = meth[c].round(4)
meth["order"] = meth.order.astype(int); meth["lines_val"] = meth.lines_val.astype(int); meth["lines_hold"] = meth.lines_hold.fillna(0).astype(int)
meth.to_json(f"{OUT}/matchers.json", orient="records", indent=1)

ex = ev.merge(eh, on="exception", suffixes=("_val", "_hold"))[["exception", "golden_val", "precision_val", "recall_val", "f1_val", "golden_hold", "precision_hold", "recall_hold", "f1_hold"]]
ex.to_json(f"{OUT}/exceptions.json", orient="records", indent=1)
print(segs.head(12).to_string()); print(meth.to_string()); print(ex.to_string())
