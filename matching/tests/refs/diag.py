"""Diagnose refs on val by cause (uses the val golden for COUNTING only; never to fit constants).
    PYTHONUTF8=1 timeout 900 .venv/Scripts/python.exe matching/tests/refs/diag.py [--rerun]
Re-runs refs with tracing (prior = batches+ap+remittance+pdf_cases outputs, p_items) unless a trace exists, then prints
per-segment line accuracy and cause counts."""
import os, pickle, sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from matching.lib import data
from matching.matchers import refs as R

PH = "val"
OUT = os.path.join("matching", "out", PH)
TR = os.path.join(OUT, "refs_diag.pkl")
SEGS = ["ar_ref_noise", "ar_1toN_text", "ar_same_amount_fifo", "ar_short_discount", "ar_discount_late", "ar_1toN_credit_note",
        "ar_n_to_m", "ar_partial", "ar_third_party_agency", "ar_new_iban", "ar_1to1", "ar_same_amount_month"]
GATE = 0.65

if "--rerun" in sys.argv or not os.path.exists(TR):
    claimed, priors = set(), []
    for f in ("batches", "ap", "remittance", "pdf_cases"):
        fp = os.path.join(OUT, f + ".parquet")
        if os.path.exists(fp):
            d = pd.read_parquet(fp, columns=["bank_txn_id", "item_id", "amount", "exception", "confidence"])
            claimed |= set(d.bank_txn_id); priors.append(d)
    pi = pd.read_parquet(os.path.join("classification", "out", PH, "line_labels.parquet"), columns=["bank_txn_id", "p_items"]).set_index("bank_txn_id").p_items
    diag = {}
    out = R.match(PH, claimed=claimed, prior=pd.concat(priors), p_items=pi, diag=diag, verbose=True)
    out.to_parquet(os.path.join(OUT, "refs.parquet"), index=False)
    with open(TR, "wb") as fh:
        pickle.dump((diag, claimed, set(pi.index[pi.values < 0.1])), fh)

with open(TR, "rb") as fh:
    diag, claimed, lowp = pickle.load(fh)
a = pd.read_parquet(os.path.join(OUT, "refs.parquet"))
g = data.golden(PH)
g = g[g.segment.isin(SEGS)]
it = pd.read_parquet(os.path.join(data.CACHE, PH, "items.parquet"), columns=["item_id", "party_id", "currency"]).set_index("item_id")
bank = data.bank(PH).set_index("bank_txn_id")
gi = g[g.item_id.notna()].copy()
gi["party"] = gi.item_id.map(it.party_id)
tol = lambda x: np.maximum(0.01, 0.005 * np.abs(x))
A = a[a.confidence >= GATE]
set_A = set(A.bank_txn_id); set_a = set(a.bank_txn_id)
A = A.set_index("bank_txn_id", drop=False); a2 = a.set_index("bank_txn_id", drop=False)
rows = []
for tx, gg in gi.groupby("bank_txn_id"):
    seg = gg.segment.iloc[0]
    gold = dict(zip(gg.item_id, gg.amount)); gexc = dict(zip(gg.item_id, gg.exception.fillna("")))
    if tx in claimed:
        cause = "other_matcher"
    elif tx in lowp:
        cause = "p_items<0.1"
    else:
        pr = A.loc[[tx]] if tx in set_A else None
        if pr is not None and len(pr):
            pred = dict(zip(pr.item_id, pr.amount)); pexc = dict(zip(pr.item_id, pr.exception.fillna("")))
            if set(pred) != set(gold):
                cause = "wrong_items"
            elif any(abs(pred[k] - gold[k]) > tol(gold[k]) for k in gold):
                cause = "wrong_amount"
            elif any(pexc[k] != gexc[k] for k in gold):
                cause = "ok_items_exc_differs"
            else:
                cause = "correct"
            cause += ":" + pr.method.iloc[0]
        else:
            st, cands, trace = diag.get(tx, ("not_seen", [], []))
            parties = set(gg.party.dropna())
            cur = bank.loc[tx, "currency"]
            if any(it.currency.get(i) != cur for i in gg.item_id):
                cause = "fx_currency"
            elif tx in set_a:
                cause = "gated<0.65:" + a2.loc[[tx]].method.iloc[0]
            elif st == "no_party":
                cause = "no_payer"
            elif len(parties) > 1:
                cause = "multi_party_gold"
            elif not parties & set(cands):
                cause = "payer_not_in_cands"
            else:
                tt = [t for t in trace if t[0] in parties]
                n = max((t[2] for t in tt), default=0)
                meth = [t[3] for t in tt if t[3]]
                if st == "ambiguous_payers":
                    cause = "ambiguous_payers"
                elif meth and any(x in meth[0] for x in R.DROP):
                    cause = "dropped:" + meth[0]
                elif n > 28:
                    cause = "pool>28"
                else:
                    cause = "no_unique_subset"
    rows.append((tx, seg, cause))

d = pd.DataFrame(rows, columns=["tx", "seg", "cause"])
d.to_pickle(os.path.join(OUT, "refs_diag_causes.pkl"))
d["c0"] = d.cause.str.split(":").str[0]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
acc = d.groupby("seg").agg(lines=("tx", "size"), acc=("c0", lambda x: (x == "correct").mean())).sort_values("lines", ascending=False)
print(acc.round(3).to_string())
print(pd.crosstab(d.seg, d.c0).to_string())
print(d[~d.c0.isin(["correct"])].groupby(["seg", "cause"]).size().sort_values(ascending=False).head(60).to_string())
