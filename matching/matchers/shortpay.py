"""shortpay matcher: customer receipts whose cash does NOT equal the documents, for one identified payer.

    python -m matching.matchers.shortpay --phase val        (uses matching/out/<phase>/allocations.parquet of the earlier
                                                             matchers as `prior`/`claimed` when run stand-alone)

Runs after the refs matcher on the lines nobody claimed (date order, own ledger on top of `prior`):
  FX (POLICIES s3 FX_DIFFERENCE): line currency != item currency and the bank shows the instructed amount (`orig_amount`,
     `orig_currency`): the refs evidence ladder on the instructed amount in the item currency
       fx_ref_exact / fx_ref_fill / fx_party_subset, fx_ref_short_* (withholding/discount on the instructed amount)
  same currency, one payer (matching.lib.party + owners of quoted document numbers):
       ref_short_wht / ref_short_disc(_late)   quoted invoices paid short by an exact withholding rate / their discount
       ref_over / ref_chg                      one quoted invoice, cash above (<= 30 %) / below (<= 60 EUR and <= 3 %, payer
                                               outside SEPA)
       wht_1 / disc_1 / disc_late_1            one unquoted invoice, exact rate relationship (unique among the payer's items)
       wht_multi / disc_multi                  unique subset of the payer's invoices, all reduced by the same rate
       over_1 / chg_1                          one unquoted invoice slightly above / below; tier by n*|diff|/cash
  cn_paid_out: outgoing transfer quoting one of OUR credit notes (rappel paid out) for exactly its open amount.
Amounts: full open amount (POLICIES s2) except late discounts (PARTIAL at the cash). Exceptions are left to the engine.
Methods below CONF_MIN are never emitted (measured below the 97 % pair-precision gate on val, or not measured yet).
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
from matching.lib import shortpay_core as sc, shortpay_solve as ss  # noqa: E402

CUT = {"val": "2025-01-01", "sample": "2025-01-01"}
DEFAULT_CUT = "2025-07-01"
CONF_MIN = 0.65
COLS = ["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"]


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def match(phase: str, claimed: set | None = None, prior: pd.DataFrame | None = None, p_items: pd.Series | None = None,
          cut: str | None = None, keep_weak: bool = False, verbose: bool = False, p_min: float = 0.1) -> pd.DataFrame:
    t0 = time.time()
    claimed = set(claimed or ())
    if p_items is not None:
        claimed |= set(p_items.index[p_items.values < p_min])
    cut = cut or CUT.get(phase, DEFAULT_CUT)
    bank = data.bank(phase)
    b = bank[~bank.bank_txn_id.isin(claimed)].sort_values(["booking_date", "bank_txn_id"])
    st = sc.ARState(phase, prior)
    st.fees = sc.fee_grid(cut)  # learned bank-fee amounts (train labels before the cut)
    px = partylib.PartyIndex(phase, cut=cut)
    from integration.lib import fx
    if verbose:
        print(f"[shortpay] setup {time.time() - t0:.0f}s; {len(b)} unclaimed lines, {len(st.items)} open AR items")
    rows, stats = [], defaultdict(int)
    # FX lines announced by a remittance advice: the advice total is in the invoice currency (= the instructed amount)
    fxr = _fx_remit(phase, b, st, stats)
    if len(fxr):
        rows += list(fxr.itertuples(index=False, name=None))
        b = b[~b.bank_txn_id.isin(set(fxr.bank_txn_id))]
    for r in b.itertuples(index=False):
        row = r._asdict()
        try:
            if r.amount > 0:
                res = _one_in(st, px, fx, row, stats)
            else:
                res = _one_out(st, row)
        except Exception:  # never let one odd line kill the matcher
            stats["error"] += 1
            continue
        if res is None:
            continue
        stats["m_" + res["method"]] += 1
        for i, amt, exc in zip(res["idx"], res["amts"], res["excs"]):
            rows.append((r.bank_txn_id, st.item_id[i], amt / 100.0, exc, res["conf"], res["method"]))
            if res["conf"] >= CONF_MIN:
                st.c[i] = max(0, st.c[i] - amt)
    out = pd.DataFrame(rows, columns=COLS)
    if not keep_weak:
        out = out[out.confidence >= CONF_MIN]
    if verbose:
        print("[shortpay] stats", dict(stats), f"{time.time() - t0:.0f}s")
    return out


def _fx_remit(phase, b, st, stats) -> pd.DataFrame:
    """Run the remittance matcher's core on the unclaimed FX receipts with amount := orig_amount, currency := orig_currency.
    Consumes the items it settles in the shortpay ledger."""
    try:
        from matching.matchers import remittance as rm
        from matching.lib import remit_parse as rp
        oa = pd.to_numeric(b.orig_amount, errors="coerce")
        fxl = b[(b.amount > 0) & b.orig_currency.notna() & (b.orig_currency != b.currency) & (oa > 0)].copy()
        if fxl.empty:
            return pd.DataFrame(columns=COLS)
        fxl["amount"] = pd.to_numeric(fxl.orig_amount, errors="coerce").round(2)
        fxl["amount_abs"] = fxl.amount
        fxl["currency"] = fxl.orig_currency
        L = rp.lines(phase)
        P = rm.prev_lines(phase, data.bank(phase).booking_date.min())
        if len(P):
            L = pd.concat([P[~P.ra_id.isin(set(L.ra_id))], L], ignore_index=True)
        items = rm._ar_items(phase)
        items = items[items.item_id.isin(set(st.item_id))].copy()  # only items still open after the earlier matchers
        cur_open = pd.Series(st.c / 100.0, index=st.item_id)
        items["open_amount"] = items.item_id.map(cur_open).values
        out = rm.match_frames(L, fxl, items, data.customers(), data.payment_terms())
    except Exception as e:  # remittance internals changed / no advices: no claim
        stats["fx_remit_error"] += 1
        print("[shortpay] fx_remit skipped:", repr(e)[:200])
        return pd.DataFrame(columns=COLS)
    if out is None or out.empty:
        return pd.DataFrame(columns=COLS)
    out = out[COLS].copy()
    out["method"] = "fx_remit_" + out.method.astype(str)
    out["exception"] = out.exception.where(out.exception.isin(["SHORT_DISPUTE"]), None)  # the engine sets FX / PARTIAL
    pos = pd.Series(np.arange(len(st.item_id)), index=st.item_id)
    for it, amt in zip(out.item_id, out.amount):
        i = pos.get(it)
        if i is not None:
            st.c[i] = max(0, st.c[i] - sc.cents(amt))
    stats["fx_remit_lines"] += out.bank_txn_id.nunique()
    return out


def _one_out(st, row):
    """our transfer paying out one of our credit notes (POLICIES s5.9: rappel paid out settles the ARC item)."""
    toks = rt.text_tokens(rt.bank_text(row))
    if not toks:
        return None
    cash = sc.cents(abs(row["amount"]))
    cand = set()
    for t in toks:
        for key in {t, rt.digits(t)}:
            for i in st.ref_idx.get((row["entity"], key), ()):
                if st.typ[i] == "ARC" and st.c[i] == cash and st.cur[i] == row["currency"]:
                    cand.add(i)
    if len(cand) == 1:
        return ss._pack(st, list(cand), "cn_paid_out", 0.95)
    return None


def _one_in(st, px, fx, row, stats):
    ent, ccy = row["entity"], row["currency"]
    bd = np.datetime64(row["booking_date"])
    text = rt.bank_text(row)
    toks = rt.text_tokens(text)
    oc, oa = row.get("orig_currency"), _f(row.get("orig_amount"))
    is_fx = isinstance(oc, str) and oc and oc != ccy and oa is not None and oa > 0
    pay_ccy = oc if is_fx else ccy
    cash = sc.cents(oa if is_fx else row["amount"])
    cands = px.candidates(row, ent)
    known = {c.customer_id for c in cands}
    for p in st.ref_owners(toks, ent):
        if p not in known:
            cands.append(partylib.Cand(p, 0.6, "ref", "doc"))
    if not cands:
        stats["no_party"] += 1
        return None
    cands = sorted(cands, key=lambda c: -c.score)[:30]
    date = str(row["booking_date"])[:10]
    eur = (lambda c: fx.convert(c / 100.0, ccy, "EUR", date)) if not is_fx else None
    results = []
    for c in cands:
        country = st.country.get(c.customer_id, "")
        for tier, members in (("own", [c.customer_id]), ("grp", px.group_members(c.customer_id))):
            if tier == "grp" and len(members) <= 1:
                continue
            pool = st.pool(members, pay_ccy, bd)
            if not pool:
                continue
            hits_l = rt.find_refs(toks, [st.docs[i] for i in pool])
            for k, lv in rt.find_refs(toks, [st.pos[i] for i in pool], typo=False).items():
                hits_l[k] = max(hits_l.get(k, 0), min(lv, 3))
            hits = {pool[k]: lv for k, lv in hits_l.items()}
            if tier == "grp" and not hits:
                continue
            if is_fx:
                res = ss.solve_fx(st, pool, cash, hits, bd, country)
            else:
                res = ss.solve_short(st, pool, cash, hits, bd, country, eur)
            if res is None:
                continue
            if tier == "grp":
                if not any(i in hits for i in res["idx"]):
                    continue
                res["method"] += "_grp"
                res["conf"] = round(res["conf"] * 0.9, 3)
            results.append((c, res))
            break
    if not results:
        stats["no_solution"] += 1
        return None
    w = lambda x: x[1]["conf"] * (0.5 + 0.5 * x[0].score)
    results.sort(key=lambda x: -w(x))
    best = results[0]
    sig = lambda res: tuple(sorted(res["idx"]))
    others = [x for x in results[1:] if sig(x[1]) != sig(best[1])]
    if others and w(others[0]) > 0.85 * w(best):
        stats["ambiguous_payers"] += 1
        return None
    return best[1]


def _load_prior(phase):
    fp = os.path.join("matching", "out", phase, "allocations.parquet")
    if not os.path.exists(fp):
        return None
    d = pd.read_parquet(fp, columns=COLS)
    return d[~d.method.fillna("").str.match(r"^(fx_)?(ref_short_wht|ref_short_disc|ref_over|ref_chg|wht_|disc_|over_|chg_|cn_paid_out|dup_)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="val")
    ap.add_argument("--prior", default=None, help="allocations parquet of the earlier matchers (default matching/out/<phase>/allocations.parquet)")
    ap.add_argument("--keep-weak", action="store_true")
    a = ap.parse_args()
    prior = pd.read_parquet(a.prior, columns=COLS) if a.prior else _load_prior(a.phase)
    claimed = set(prior.bank_txn_id) if prior is not None else set()
    pi = None
    fp = os.path.join("classification", "out", a.phase, "line_labels.parquet")
    if os.path.exists(fp):
        pi = pd.read_parquet(fp, columns=["bank_txn_id", "p_items"]).set_index("bank_txn_id").p_items
    out = match(a.phase, claimed, prior=prior, p_items=pi, keep_weak=a.keep_weak, verbose=True)
    os.makedirs(os.path.join("matching", "out", a.phase), exist_ok=True)
    out.to_parquet(os.path.join("matching", "out", a.phase, "shortpay.parquet"), index=False)
    if len(out) and a.phase in ("sample", "val", "train"):
        evalutil.report(out, a.phase, name="shortpay", top=20)
        try:
            from matching.tests.shortpay.method_report import by_method
            by_method(out, a.phase)
        except ImportError:
            pass


if __name__ == "__main__":
    main()
