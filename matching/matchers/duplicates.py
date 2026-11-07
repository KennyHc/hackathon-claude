"""duplicates matcher: a customer pays again an invoice that was already settled (POLICIES s3 DUPLICATE_PAYMENT).

    python -m matching.matchers.duplicates --phase val

Runs last, on incoming lines nobody claimed.  An AR invoice is "already settled" at the booking date of the line when
  (a) in this phase: earlier allocations (`prior`, every matcher before this one) brought its balance to 0 on a bank line
      booked BEFORE this line and no RETURNED re-opened it afterwards (ledger: RETURNED re-opens, so a re-payment after a
      bounce is not a duplicate), or
  (b) before the phase: the invoice was issued before the phase start but is absent from open_items_at_start (it was closed
      in the previous phase).  Only for lines booked in the first PREV_DAYS days of the phase (duplicates arrive days after
      the first payment, measured on train < 2025: 1-14 days).
The line must come from the invoice's payer (matching.lib.party, or the owner of the quoted number), pay exactly the invoice
total, and the payer must have no OPEN item of that amount (then it would be a normal payment):
  dup_ref   the line quotes the settled invoice (refs_text.find_refs)                              -> conf 0.95
  dup_amt   no quote; the only settled invoice of the payer with that total in the window          -> conf 0.9 (100 % on val)
  dup_*_prev  same against invoices closed before the phase; dup_amt_prev measured 93 % -> conf 0.6 (not emitted)
Rows: amount = cash (what was paid again, item currency), exception = DUPLICATE_PAYMENT (passed through by the engine).
Refunds of duplicates are categories (CUSTOMER_REFUND), never claimed here.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from matching.lib import data, evalutil, party as partylib, refs_text as rt  # noqa: E402
from matching.lib import shortpay_core as sc  # noqa: E402

CUT = {"val": "2025-01-01", "sample": "2025-01-01"}
DEFAULT_CUT = "2025-07-01"
CONF_MIN = 0.65
WINDOW_DAYS = 30   # first payment booked at most this many days before the duplicate (train < 2025: max 14)
PREV_DAYS = 21     # lines this close to the phase start may duplicate a payment of the previous phase
COLS = ["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"]
PREV_OF = {"val": ["train"], "live": ["train"], "final": ["train", "live"]}


def _prev_items(phase: str, start: pd.Timestamp) -> pd.DataFrame:
    """AR invoices issued in [start - 400d, start) in the previous phase folders and NOT open at the phase start (inputs only).
    Cached under matching/out/cache/<phase>/shortpay_prev_ar.parquet."""
    cp = os.path.join(data.CACHE, phase, "shortpay_prev_ar.parquet")
    if os.path.exists(cp):
        return pd.read_parquet(cp)
    lo = (start - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    hi = start.strftime("%Y-%m-%d")
    parts = []
    for ph in PREV_OF.get(phase, []):
        f = os.path.join(data.phase_dir(ph), "ar_invoices.csv.gz")
        if not os.path.exists(f):
            continue
        for ch in pd.read_csv(f, usecols=["item_id", "entity", "document_number", "issue_date", "customer_id", "currency", "total_amount",
                                          "customer_po"], dtype={"document_number": str, "customer_po": str}, chunksize=300_000):
            parts.append(ch[(ch.issue_date >= lo) & (ch.issue_date < hi)])
    if not parts:
        out = pd.DataFrame(columns=["item_id", "entity", "document_number", "issue_date", "customer_id", "currency", "total_amount", "customer_po"])
    else:
        out = pd.concat(parts, ignore_index=True).drop_duplicates("item_id")
        p0 = os.path.join(data.phase_dir(phase), "open_items_at_start.csv.gz")
        if os.path.exists(p0):
            o = pd.read_csv(p0, usecols=["item_id", "open_amount"])
            still = set(o.item_id[o.open_amount > 0.004])
            out = out[~out.item_id.isin(still)]
        else:
            out = out.iloc[0:0]  # no opening balances: cannot tell settled from open
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    out.to_parquet(cp, index=False)
    return out


def match(phase: str, claimed: set | None = None, prior: pd.DataFrame | None = None, p_items: pd.Series | None = None,
          cut: str | None = None, keep_weak: bool = False, verbose: bool = False, p_min: float = 0.1) -> pd.DataFrame:
    t0 = time.time()
    claimed = set(claimed or ())
    if p_items is not None:
        claimed |= set(p_items.index[p_items.values < p_min])
    cut = cut or CUT.get(phase, DEFAULT_CUT)
    bank = data.bank(phase)
    start = bank.booking_date.min()
    bdate = bank.set_index("bank_txn_id").booking_date
    b = bank[(bank.amount > 0) & ~bank.bank_txn_id.isin(claimed)].sort_values(["booking_date", "bank_txn_id"])
    st = sc.ARState(phase, prior, keep_closed=True)
    # settlement date of the items closed by earlier allocations (last consuming allocation; None if re-opened afterwards)
    settled_at: dict[int, np.datetime64] = {}
    if prior is not None and len(prior):
        pos = pd.Series(np.arange(len(st.item_id)), index=st.item_id)
        p = prior[prior.item_id.isin(pos.index)][["bank_txn_id", "item_id", "exception"]].copy()
        p["d"] = p.bank_txn_id.map(bdate)
        p["ret"] = p.exception.fillna("").eq("RETURNED")
        p = p[~p.exception.fillna("").eq("DUPLICATE_PAYMENT")].sort_values("d")
        last = p.groupby("item_id").agg(d=("d", "last"), ret=("ret", "last"))
        for it, d, ret in zip(last.index, last.d, last.ret):
            i = pos[it]
            if not ret and st.c[i] == 0:
                settled_at[i] = np.datetime64(d)
    prev = _prev_items(phase, start) if phase in PREV_OF else pd.DataFrame()
    prev_by_party = defaultdict(list)
    prev_ref = defaultdict(list)
    if len(prev):
        prev = prev[~prev.item_id.isin(set(st.item_id))].reset_index(drop=True)
        prev["cents"] = np.round(prev.total_amount.astype(float) * 100).astype("int64")
        for k, r in enumerate(prev.itertuples(index=False)):
            prev_by_party[r.customer_id].append(k)
            kk = rt.doc_keys(r.document_number)
            if len(kk["full"]) >= 8:
                prev_ref[(r.entity, kk["full"])].append(k)
    px = partylib.PartyIndex(phase, cut=cut)
    if verbose:
        print(f"[duplicates] setup {time.time() - t0:.0f}s; {len(b)} lines, {len(settled_at)} items settled in-phase, {len(prev)} prev-phase settled")
    rows, stats = [], defaultdict(int)
    win = np.timedelta64(WINDOW_DAYS, "D")
    for r in b.itertuples(index=False):
        try:
            row = r._asdict()
            bd = np.datetime64(r.booking_date)
            cash = sc.cents(r.amount)
            toks = rt.text_tokens(rt.bank_text(row))
            cands = px.candidates(row, r.entity)
            known = {c.customer_id for c in cands}
            owners = st.ref_owners(toks, r.entity, typ="ARI", open_only=False)
            for t in toks:
                for k in prev_ref.get((r.entity, t), ()):
                    owners[prev.customer_id[k]] = owners.get(prev.customer_id[k], 0) + 1
            for p_ in owners:
                if p_ not in known:
                    cands.append(partylib.Cand(p_, 0.6, "ref", "doc"))
            found = []
            for c in sorted(cands, key=lambda c: -c.score)[:30]:
                its = st.by_party.get(c.customer_id, ())
                if any(st.c[i] == cash and st.cur[i] == r.currency and st.typ[i] == "ARI" and st.issue[i] <= bd for i in its):
                    continue  # an open invoice of that amount exists: not a duplicate
                cand_items = []
                for i in its:
                    if st.typ[i] == "ARI" and st.full[i] == cash and st.cur[i] == r.currency and i in settled_at:
                        sa = settled_at[i]
                        if sa < bd and bd - sa <= win:
                            cand_items.append(("cur", i, st.docs[i], st.pos[i]))
                if (bd - np.datetime64(start)) <= np.timedelta64(PREV_DAYS, "D"):
                    for k in prev_by_party.get(c.customer_id, ()):
                        if prev.cents[k] == cash and prev.currency[k] == r.currency:
                            po = prev.customer_po[k]
                            cand_items.append(("prev", k, prev.document_number[k], po if isinstance(po, str) else ""))
                if not cand_items:
                    continue
                hits = rt.find_refs(toks, [x[2] for x in cand_items])
                for k2, lv in rt.find_refs(toks, [x[3] for x in cand_items], typo=False).items():
                    hits[k2] = max(hits.get(k2, 0), lv)
                good = [cand_items[k2] for k2, lv in hits.items() if lv >= 2]
                if len(good) == 1:
                    found.append((c, good[0], "dup_ref" + ("_prev" if good[0][0] == "prev" else ""), 0.95))
                elif not good and len(cand_items) == 1 and not hits:
                    found.append((c, cand_items[0], "dup_amt" + ("_prev" if cand_items[0][0] == "prev" else ""), 0.6 if cand_items[0][0] == "prev" else 0.9))
            if not found:
                continue
            found.sort(key=lambda x: -(x[3] * (0.5 + 0.5 * x[0].score)))
            best = found[0]
            if any(x[1][:2] != best[1][:2] and x[3] >= best[3] for x in found[1:]):
                stats["ambiguous"] += 1
                continue
            kind, k, *_ = best[1]
            iid = st.item_id[k] if kind == "cur" else prev.item_id[k]
            rows.append((r.bank_txn_id, iid, cash / 100.0, "DUPLICATE_PAYMENT", best[3], best[2]))
            stats["m_" + best[2]] += 1
        except Exception:
            stats["error"] += 1
    out = pd.DataFrame(rows, columns=COLS)
    if not keep_weak:
        out = out[out.confidence >= CONF_MIN]
    if verbose:
        print("[duplicates] stats", dict(stats), f"{time.time() - t0:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="val")
    ap.add_argument("--prior", default=None, help="allocations of the earlier matchers (default matching/out/<phase>/allocations.parquet)")
    ap.add_argument("--keep-weak", action="store_true")
    a = ap.parse_args()
    fp = a.prior or os.path.join("matching", "out", a.phase, "allocations.parquet")
    prior = pd.read_parquet(fp, columns=COLS) if os.path.exists(fp) else None
    if prior is not None:
        prior = prior[~prior.method.fillna("").str.startswith("dup_")]
    claimed = set(prior.bank_txn_id) if prior is not None else set()
    pi = None
    fpc = os.path.join("classification", "out", a.phase, "line_labels.parquet")
    if os.path.exists(fpc):
        pi = pd.read_parquet(fpc, columns=["bank_txn_id", "p_items"]).set_index("bank_txn_id").p_items
    out = match(a.phase, claimed, prior=prior, p_items=pi, keep_weak=a.keep_weak, verbose=True)
    os.makedirs(os.path.join("matching", "out", a.phase), exist_ok=True)
    out.to_parquet(os.path.join("matching", "out", a.phase, "duplicates.parquet"), index=False)
    if len(out) and a.phase in ("sample", "val", "train"):
        evalutil.report(out, a.phase, name="duplicates", top=15)
        from matching.tests.shortpay.method_report import by_method
        by_method(out, a.phase)


if __name__ == "__main__":
    main()
