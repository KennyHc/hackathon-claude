"""Zoo experiments for a phase: member + ensemble line_labels candidates -> classification/out/<phase>/cand/*.parquet

    python classification/zoo_run.py --phase val --members lgb,xgb,chr          # writes candidates (+ timings)
    python classification/eval_cls.py --phase val --labels classification/out/val/cand/<name>.parquet ...

Candidates: lr (reference, recomputed), <m>_raw (member P(ITEM) as is), <m>_cal (member Platt-calibrated on the
stacker slice), lr+<m> (stacker of LR and the member), all (stacker of LR and every member), avg (mean probability).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from classification import categories as C  # noqa: E402
from classification import zoo as Z  # noqa: E402
from ingestion import lib  # noqa: E402


def finish(P: np.ndarray, classes: list, bank: pd.DataFrame, method: str) -> pd.DataFrame:
    """Same post-processing as categories.label (keyword rules, reversal pairs) on a 22-class matrix."""
    ji = classes.index(C.ITEM)
    p_items = P[:, ji]
    Pn = P.copy()
    Pn[:, ji] = -1
    best = Pn.argmax(axis=1)
    cat = np.array(classes)[best]
    conf = P[np.arange(len(P)), best]
    meth = np.full(len(bank), method, dtype=object)
    kw = C.keyword_rules(bank).values
    hit = kw != ""
    cat = np.where(hit, kw, cat)
    p_items = np.where(hit, np.minimum(p_items, 0.05), p_items)
    conf = np.where(hit, np.maximum(conf, 0.95), conf)
    meth[hit] = "rule_kw"
    rev = bank.bank_txn_id.isin(C.reversal_pairs(bank)).values
    cat = np.where(rev, "BANK_REVERSAL", cat)
    p_items = np.where(rev, 0.01, p_items)
    conf = np.where(rev, 0.99, conf)
    meth[rev] = "reversal_pair"
    return pd.DataFrame({"bank_txn_id": bank.bank_txn_id.values, "p_items": np.round(p_items, 4), "category": cat,
                         "exception": np.where(cat == "UNIDENTIFIED", "UNIDENTIFIED", ""),
                         "confidence": np.round(conf, 4), "method": meth})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="val")
    ap.add_argument("--members", default="lgb,xgb,chr")
    a = ap.parse_args()
    members = [m for m in a.members.split(",") if m]
    cutoff = C.CUTOFF[a.phase]
    t0 = time.time()
    tc = Z.train_cache()
    bank = pd.read_parquet(os.path.join(ROOT, "ingestion", "out", a.phase, "bank_lines.parquet"))
    lk = C.Lookups(lib.masters(lib.PHASES[a.phase][0]))
    vpos = {x: j for j, x in enumerate(tc["vocab"])}
    Xs = C._matrix(C._features(bank, lk), vpos)
    tim = {"features": time.time() - t0}
    lr = C.load_model(cutoff)
    classes = list(lr["classes"])
    P_lr = Z.lr_proba(lr, Xs, tc["vocab"])
    t = time.time()
    pm = Z.member_probs(cutoff, members, Xs, tc["vocab"], bank, lr, tc)
    tim["members_fit_predict"] = time.time() - t
    t = time.time()
    od = os.path.join(Z.ZDIR, f"cand_{a.phase}")
    os.makedirs(od, exist_ok=True)
    cands = {"lr": pm["lr"]}
    pmask = Z.pop_mask(P_lr, classes, bank.amount.values)

    def qm(st, p, pop):
        """quantile-map the stacker output onto the LR's distribution on the stacker slice (same population)."""
        k = st["slice_pop"] if pop else np.ones(len(st["slice_y"]), bool)
        src = Z.apply_stacker(st, st["slice_probs"])[k]
        return Z.qmap_apply(Z.qmap_fit(src, st["slice_probs"]["lr"][k]), p)

    for m in [None, *members]:            # population-only stackers (UNIDENTIFIED gate lines), LR elsewhere
        ms = [m] if m else []
        st = Z.fit_stacker(cutoff, ms, tc, pop=True)
        s_ = Z.apply_stacker(st, pm)
        nm = f"pop_lr{'+' + m if m else ''}"
        q = pm["lr"].copy()
        q[pmask] = s_[pmask]
        cands[nm] = q
        if m:
            q = pm["lr"].copy()
            q[pmask] = qm(st, s_, True)[pmask]
            cands[nm + "_qm"] = q
    for m in members:
        if m == "lgbr":
            continue
        cands[f"{m}_raw"] = pm[m]
        st = Z.fit_stacker(cutoff, [m], tc)
        s_ = Z.apply_stacker(st, pm)
        cands[f"lr+{m}"] = s_
        cands[f"lr+{m}_qm"] = qm(st, s_, False)
    full = [m for m in members if m != "lgbr"]
    if len(full) > 1:
        st = Z.fit_stacker(cutoff, full, tc)
        s_ = Z.apply_stacker(st, pm)
        tag = "all" if full == ["lgb", "xgb", "chr"] else "stack_" + "-".join(full)
        cands[tag] = s_
        cands[tag + "_qm"] = qm(st, s_, False)
        cands["avg" if tag == "all" else "avg_" + "-".join(full)] = np.mean([pm[n] for n in ["lr", *full]], axis=0)
    tim["stackers"] = time.time() - t
    for name, p in cands.items():
        P = Z.compose(P_lr, classes, p)
        finish(P, classes, bank, f"zoo:{name}").to_parquet(os.path.join(od, f"{name.replace('+', '_')}.parquet"), index=False)
    pd.DataFrame({"bank_txn_id": bank.bank_txn_id.values, **{k: v for k, v in pm.items()}}).to_parquet(
        os.path.join(od, "_member_probs.parquet"), index=False)
    tim["total"] = time.time() - t0
    json.dump(tim, open(os.path.join(od, "_timing.json"), "w"), indent=1)
    Z.log(f"{a.phase}: {len(cands)} candidates -> {od}  timing {json.dumps({k: round(v) for k, v in tim.items()})}")


def _single(cutoff, m, tc):
    """Platt calibration of one member alone on the stacker slice (re-uses the lr+m stacker's slice predictions)."""
    from sklearn.linear_model import LogisticRegression
    st = Z.fit_stacker(cutoff, [m], tc)
    z = Z._logit(st["slice_probs"][m])[:, None]
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(z, st["slice_y"])
    return {"names": [m], "coef": lr.coef_.ravel(), "intercept": float(lr.intercept_[0])}


if __name__ == "__main__":
    main()
