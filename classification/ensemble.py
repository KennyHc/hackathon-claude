"""Production path of the p_items ensemble: pre-trained models only, falls back to the plain LR on any failure.

    label(bank, phase) -> same frame as categories.label

Config (CONFIG below): which members feed the stacker. Models are loaded from classification/out/zoo/ for the phase's
cutoff (categories.CUTOFF); if one is missing it is trained (members ~minutes; the stacker needs the TRAIN feature
cache). Any exception -> categories.label (the LR alone), logged.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from classification import categories as C  # noqa: E402
from classification import zoo as Z  # noqa: E402
from ingestion import lib  # noqa: E402

# kind: "full" (stacker on all lines), "pop" (stacker on the UNIDENTIFIED-gate population only, LR elsewhere);
# "_qm" suffix: quantile-mapped onto the LR's slice distribution. Set from the zoo results (see the report).
CONFIG = {"members": ["lgb"], "kind": "full_qm"}


def _hist(cutoff: str) -> dict:
    p = os.path.join(Z.ZDIR, f"hist_v{Z.ZOO_VERSION}_{cutoff[:10]}.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    bank, y = Z._labels_light()
    bank = Z._plain(bank[["booking_date", "counterparty_name", "counterparty_account"]])
    t = Z._hist_tables(bank, y, cutoff)
    os.makedirs(Z.ZDIR, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(t, f, protocol=4)
    return t


def _ensemble(bank: pd.DataFrame, phase: str) -> pd.DataFrame:
    from classification.zoo_run import finish
    t0 = time.time()
    cutoff = C.CUTOFF[phase]
    members = list(CONFIG["members"])
    lr = C.load_model(cutoff)
    lk = C.Lookups(lib.masters(lib.PHASES[phase][0]))
    feats = C._features(bank, lk)
    vocab = sorted({x for f in feats for x in f})
    vpos = {x: j for j, x in enumerate(vocab)}
    Xs = C._matrix(feats, vpos)
    vocab = np.array(vocab)
    classes = list(lr["classes"])
    P_lr = Z.lr_proba(lr, Xs, vocab)
    pm = {"lr": P_lr[:, classes.index(C.ITEM)]}
    mods = {}
    for m in members:
        path = Z._mpath(cutoff, m)
        if not os.path.exists(path):
            print(f"[ensemble] {m} for {cutoff} not pre-trained: training now", flush=True)
            Z.fit_members(cutoff, [m])
        with open(path, "rb") as f:
            mods[m] = pickle.load(f)
    if members:
        D = Z.dense(bank, _hist(cutoff), cutoff)
        for m in members:
            pm[m] = Z.predict_member(mods[m], Xs, vocab, D, bank)
    kind = CONFIG["kind"]
    pop = kind.startswith("pop")
    sp = os.path.join(Z.ZDIR, f"stack_v{Z.ZOO_VERSION}_{cutoff[:10]}_{'-'.join(members)}{'_pop' if pop else ''}.pkl")
    if not os.path.exists(sp):
        print(f"[ensemble] stacker {os.path.basename(sp)} not pre-trained: fitting now", flush=True)
        Z.fit_stacker(cutoff, members, Z.train_cache(), pop=pop)
    with open(sp, "rb") as f:
        st = pickle.load(f)
    s_ = Z.apply_stacker(st, pm)
    if kind.endswith("_qm"):
        k = st["slice_pop"] if pop else np.ones(len(st["slice_y"]), bool)
        src = Z.apply_stacker(st, st["slice_probs"])[k]
        s_ = Z.qmap_apply(Z.qmap_fit(src, st["slice_probs"]["lr"][k]), s_)
    if pop:
        mask = Z.pop_mask(P_lr, classes, bank.amount.values)
        p = pm["lr"].copy()
        p[mask] = s_[mask]
    else:
        p = s_
    for k, v in pm.items():
        if not np.isfinite(v).all():
            raise ValueError(f"non-finite probabilities from {k}")
    out = finish(Z.compose(P_lr, classes, p), classes, bank, "ens:" + kind + ":" + "+".join(st["names"]))
    print(f"[ensemble] {phase}: {len(bank):,} lines, members {st['names']} ({time.time() - t0:.0f}s)", flush=True)
    return out


def label(bank: pd.DataFrame, phase: str) -> pd.DataFrame:
    try:
        return _ensemble(bank, phase)
    except Exception:
        print(f"[ensemble] FAILED, falling back to the plain LR\n{traceback.format_exc()}", flush=True)
        return C.label(bank, phase)
