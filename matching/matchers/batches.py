"""Matcher `batches`: bank lines that summarise a batch file (SEPA/BACS/ACH direct-debit batch, card settlement, cheque/lockbox
deposit) plus their reversals (returned DD, returned cheque, card chargeback).  Structural rules only, no label learning.

    PYTHONUTF8=1 .venv/Scripts/python.exe -m matching.matchers.batches --phase val
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from matching.lib import data  # noqa: E402
from matching.lib import batches_core as bc  # noqa: E402

NAME = "batches"
COLS = ["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"]
HIST_MONTHS = 5  # how far back (months before the phase start) batch files are read for reversals / boundary batches
STATS: dict = {}
RET_MAX_LAG = 120
RET_PICK_NEAREST = True


# ----------------------------------------------------------------------------------------------------------- loading
def _month_of(path):
    return os.path.basename(path).rsplit("_", 1)[-1][:6]


def _hist_files(phase, sub, start):
    """Batch files of other phases (train/live/final dirs) in the HIST_MONTHS months before `start`."""
    own = os.path.abspath(data.phase_dir(phase))
    lo = (start - pd.DateOffset(months=HIST_MONTHS)).strftime("%Y%m")
    hi = start.strftime("%Y%m")
    out = []
    for ph in ("train", "live", "final"):
        d = data.phase_dir(ph)
        if os.path.abspath(d) == own or not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, sub, "*.csv"))):
            if os.path.basename(f).startswith("._"):
                continue
            if lo <= _month_of(f) < hi:
                out.append(f)
    return out


def _read_hist(phase, sub, start, fix):
    fs = _hist_files(phase, sub, start)
    if not fs:
        return pd.DataFrame()
    return fix(pd.concat([pd.read_csv(f, dtype=str) for f in fs], ignore_index=True))


def _fix_dd(d):
    d["amount"] = d.amount.astype(float).round(2); d["batch_total"] = d.batch_total.astype(float).round(2)
    d["collection_date"] = pd.to_datetime(d.collection_date); d["entity"] = d.account_id.str[:4]
    d["re_presentation"] = d.re_presentation.eq("Y") if d.re_presentation.dtype == object else d.re_presentation
    return d


def _fix_card(d):
    for c in ("gross", "fee", "net"):
        d[c] = d[c].astype(float).round(2)
    d["settlement_date"] = pd.to_datetime(d.settlement_date); d["entity"] = d.account_id.str[:4]
    return d


def _fix_chq(d):
    d["amount"] = d.amount.astype(float).round(2); d["deposit_total"] = d.deposit_total.astype(float).round(2)
    d["deposit_date"] = pd.to_datetime(d.deposit_date); d["entity"] = d.account_id.str[:4]
    return d


def _hist_items(phase, start, exclude_ids, days=800):
    """AR invoices of other phases issued before `start` (items settled before the phase are not in its open-item pool but can be
    returned / re-presented inside it)."""
    own = os.path.abspath(data.phase_dir(phase))
    lo = start - pd.Timedelta(days=days)
    parts = []
    for ph in ("train", "live", "final"):
        d = data.phase_dir(ph)
        f = os.path.join(d, "ar_invoices.csv.gz")
        if os.path.abspath(d) == own or not os.path.exists(f):
            continue
        keep = []
        for ch in pd.read_csv(f, dtype=str, usecols=["item_id", "entity", "customer_id", "customer_name", "document_number", "issue_date", "currency", "total_amount"],
                              chunksize=400_000):  # chunked: the train file is ~3M rows
            keep.append(ch[(ch.issue_date >= str(lo.date())) & (ch.issue_date < str(start.date())) & ~ch.item_id.isin(exclude_ids)])
        ar = pd.concat(keep, ignore_index=True)
        parts.append(ar)
    if not parts:
        return pd.DataFrame()
    ar = pd.concat(parts, ignore_index=True).drop_duplicates("item_id")
    return pd.DataFrame({"item_id": ar.item_id, "item_type": "ARI", "side": "AR", "entity": ar.entity, "party_id": ar.customer_id,
                         "party_name": ar.customer_name, "document_number": ar.document_number, "issue_date": pd.to_datetime(ar.issue_date),
                         "currency": ar.currency, "total_amount": ar.total_amount.astype(float), "open_amount": ar.total_amount.astype(float)})


def _cat(*frames):
    fr = [f for f in frames if f is not None and len(f)]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


# --------------------------------------------------------------------------------------------------- batch credit lines
def _link_batches(bank_cr, T, idcol, acc="account_id", tot="tot", date="date", lo=0, hi=3, hi_id=10):
    """T: one row per batch (idcol, acc, tot, date). Returns DataFrame[bank_txn_id, <idcol>, conf] 1:1 links.
    Pass 1 keys on the account; pass 2 (account renamed / new IBAN) keys on the entity and needs the id in the text or uniqueness."""
    out = []
    bk = bank_cr[["bank_txn_id", "account_id", "amount", "booking_date", "text"]].copy()
    bk["ent"] = bk.account_id.str[:4]
    T = T.copy(); T["ent"] = T[acc].str[:4]
    for key_b, key_t, pen in (("account_id", acc, 0.0), ("ent", "ent", 0.04)):
        if len(out):
            done_l = set(pd.concat(out).bank_txn_id); done_b = set(pd.concat(out)[idcol])
            bk_ = bk[~bk.bank_txn_id.isin(done_l)]; T_ = T[~T[idcol].isin(done_b)]
        else:
            bk_, T_ = bk, T
        if bk_.empty or T_.empty:
            continue
        m = T_.merge(bk_, left_on=[key_t, tot], right_on=[key_b, "amount"], suffixes=("", "_b"))
        if m.empty:
            continue
        m["dt"] = (m.booking_date - m[date]).dt.days
        m["idin"] = [str(i) in str(t) for i, t in zip(m[idcol], m.text)]
        m = m[((m.dt >= lo) & (m.dt <= hi)) | (m.idin & (m.dt >= -1) & (m.dt <= hi_id))]
        # a batch must have a single candidate line and the line a single batch, unless the id is quoted in the text
        nb = m.groupby(idcol).bank_txn_id.transform("nunique"); nl = m.groupby("bank_txn_id")[idcol].transform("nunique")
        m = m[m.idin | ((nb == 1) & (nl == 1))].copy()
        m = m.sort_values(["idin", "dt"], ascending=[False, True]).drop_duplicates("bank_txn_id").drop_duplicates(idcol)
        m["conf"] = np.where(m.idin, 0.99, 0.93) - pen
        out.append(m[["bank_txn_id", idcol, "conf"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["bank_txn_id", idcol, "conf"])


# ------------------------------------------------------------------------------------------------------------ reversals
_MID = re.compile(r"MID\s*(\d{6,})")
_L4 = re.compile(r"\*{4}\s?(\d{4})")
_CODE = re.compile(r"(?<![A-Z0-9])([A-Z]{2}[0-9]{2})(?![A-Z0-9])")
_TOK = re.compile(r"[A-Z0-9][A-Z0-9\-]{5,}")


def _dd_candidates(bank_db, ddall):
    """Debit lines naming a known mandate id x receipts of that mandate with the same amount collected before the booking."""
    mand = set(ddall.mandate_id.dropna().unique())
    rows = []
    for r in bank_db[["bank_txn_id", "account_id", "amount_abs", "booking_date", "text", "remittance_info"]].itertuples(index=False):
        s = f"{r.text} {r.remittance_info}".upper()
        hit = [t for t in _TOK.findall(s) if t in mand]
        if hit:
            rows.append((r.bank_txn_id, hit[0], r.amount_abs, r.booking_date, r.account_id, s))
    if not rows:
        return pd.DataFrame()
    L = pd.DataFrame(rows, columns=["bank_txn_id", "mandate_id", "amount_abs", "booking_date", "bacc", "text"])
    m = L.merge(ddall[["mandate_id", "amount", "collection_date", "batch_id", "end_to_end_id", "entity", "remittance_info", "debtor_name",
                       "re_presentation", "currency"]], on="mandate_id", suffixes=("", "_r"))
    m = m[((m.amount - m.amount_abs).abs() <= 0.01)].copy()
    m["lag"] = (m.booking_date - m.collection_date).dt.days
    tx = dict(zip(L.bank_txn_id, L.text))
    m["bidin"] = [str(b) in tx.get(i, "") for b, i in zip(m.batch_id, m.bank_txn_id)]
    m["code"] = [(_CODE.search(tx.get(i, "")) or [None, "NA"])[1] if _CODE.search(tx.get(i, "")) else "NA" for i in m.bank_txn_id]
    m = m[(m.bidin & (m.lag >= 0) & (m.lag <= 120)) | ((m.lag >= 1) & (m.lag <= RET_MAX_LAG))]
    m["ncand"] = m.groupby("bank_txn_id").bank_txn_id.transform("size")
    return m


LAG_BIN = 7
NBIN = 31


def load_lag_model():
    """Per reason-code histogram of (return date - collection date) in 7-day bins, fitted on TRAIN before the cut by
    matching/lib/batches_fit.py (labels only from rows booked before 2025-01-01). None if the file is missing."""
    import json
    p = os.path.join(data.ROOT, "matching", "lib", "batches_lag_model.json")
    if os.path.exists(p):
        return json.load(open(p))
    return None


def _dd_returns(bank_db, ddall, lag_model=None):
    """Pick the returned receipt among same-mandate/same-amount candidates. Batch id quoted in the text is decisive; otherwise the
    reason-code specific lag histogram (fitted, else self-calibrated on the unambiguous returns) chooses."""
    m = _dd_candidates(bank_db, ddall)
    if m.empty:
        return m
    u = m[m.ncand == 1]
    allh = np.bincount((u.lag.values // LAG_BIN).clip(0, NBIN - 1), minlength=NBIN).astype(float) + 0.2
    allh = allh / allh.sum()
    hist = {k: np.asarray(v) for k, v in (lag_model or {}).items() if k != "_all"}
    if lag_model and "_all" in lag_model:
        allh = np.asarray(lag_model["_all"])
    m["dens"] = [hist.get(c, allh)[min(int(l // LAG_BIN), NBIN - 1)] for c, l in zip(m.code, m.lag)]
    # a returned receipt is often re-presented later (re_presentation=Y, same mandate, amount and reference, after the return)
    rp = ddall[ddall.re_presentation][["mandate_id", "amount", "remittance_info", "collection_date"]].rename(columns={"collection_date": "rp_date"})
    if len(rp):
        j = m[["bank_txn_id", "mandate_id", "amount", "remittance_info", "booking_date"]].reset_index().merge(rp, on=["mandate_id", "amount", "remittance_info"])
        j = j[(j.rp_date > j.booking_date) & ((j.rp_date - j.booking_date).dt.days <= 60)]
        m["rep"] = m.index.isin(j["index"])
    else:
        m["rep"] = False
    m = m.sort_values(["bank_txn_id", "bidin", "rep", "dens", "lag"], ascending=[True, False, False, False, True])
    # every receipt is returned once: lines are served in booking order, a receipt already taken by an earlier return is skipped
    taken, keep = set(), []
    first = m.drop_duplicates("bank_txn_id").sort_values("booking_date").bank_txn_id
    grp = {k: v for k, v in m.groupby("bank_txn_id", sort=False)}
    for t in first:
        for ix, r in zip(grp[t].index, grp[t].itertuples()):
            key = (r.batch_id, r.end_to_end_id)
            if key not in taken:
                taken.add(key); keep.append(ix); break
    m = m.loc[keep].copy()
    m["dens2"] = [grp[t].dens.iloc[1] if len(grp[t]) > 1 else 0.0 for t in m.bank_txn_id]
    gap = m.dens / (m.dens + m.dens2)
    # confidence: certain when single candidate / batch id / re-presentation; otherwise calibrated on the train-2023 slice
    m["conf"] = np.where((m.ncand == 1) | m.bidin | m.rep, 0.97, np.clip(0.45 + 0.5 * (gap - 0.5) / 0.5 * 1.0, 0.45, 0.95))
    return m


def _chq_returns(bank_db, chqall):
    num = {}
    for cn, a, dt, ref, dr, ccy, ent in zip(chqall.cheque_number, chqall.amount, chqall.deposit_date, chqall.stub_reference, chqall.drawer,
                                            chqall.currency, chqall.entity):
        num.setdefault(str(int(cn)) if str(cn).isdigit() else cn, []).append((a, dt, ref, dr, ccy, ent))
    rows = []
    for r in bank_db[["bank_txn_id", "amount_abs", "booking_date", "text", "remittance_info"]].itertuples(index=False):
        s = f"{r.text} {r.remittance_info}".upper()
        for t in re.findall(r"\d{5,10}", s):
            k = str(int(t))
            for a, dt, ref, dr, ccy, ent in num.get(k, []):
                if abs(a - r.amount_abs) <= 0.01 and dt <= r.booking_date and (r.booking_date - dt).days <= 150:
                    rows.append((r.bank_txn_id, k, a, dt, ref, dr, ccy, ent))
    if not rows:
        return pd.DataFrame()
    m = pd.DataFrame(rows, columns=["bank_txn_id", "chq", "amount", "deposit_date", "stub_reference", "drawer", "currency", "entity"])
    m = m.sort_values(["bank_txn_id", "deposit_date"], ascending=[True, False])
    return m.drop_duplicates("bank_txn_id")


def _chargebacks(bank_db, cardall):
    ms = set(cardall.merchant_id.dropna().unique())
    rows = []
    for r in bank_db[["bank_txn_id", "amount_abs", "booking_date", "text", "remittance_info"]].itertuples(index=False):
        s = f"{r.text} {r.remittance_info}"
        l4 = _L4.search(s)
        mid = [t for t in re.findall(r"\d{6,}", s) if t in ms]
        if l4 and mid:
            rows.append((r.bank_txn_id, f"****{l4.group(1)}", mid[0], r.amount_abs, r.booking_date))
    if not rows:
        return pd.DataFrame()
    L = pd.DataFrame(rows, columns=["bank_txn_id", "card", "merchant_id", "amount_abs", "booking_date"])
    m = L.merge(cardall[["card", "merchant_id", "gross", "settlement_date", "order_number", "entity", "currency"]], on=["card", "merchant_id"])
    m = m[((m.gross - m.amount_abs).abs() <= 0.01) & (m.settlement_date <= m.booking_date) & ((m.booking_date - m.settlement_date).dt.days <= 150)]
    m = m.sort_values(["bank_txn_id", "settlement_date"], ascending=[True, False])
    return m.drop_duplicates("bank_txn_id")


def _represented(cr, ret):
    """Customer re-pays a returned receipt by credit transfer: credit line, same entity, amount == returned receipt amount, within
    90 days after the return, and the text quotes the receipt reference or the payer name matches the debtor."""
    if ret is None or ret.empty or cr.empty:
        return pd.DataFrame()
    r = ret[["bank_txn_id", "booking_date", "entity", "amount", "remittance_info", "debtor_name", "collection_date", "currency", "mandate_id"]].rename(
        columns={"bank_txn_id": "ret_txn", "booking_date": "ret_date"})
    c = cr[["bank_txn_id", "entity", "amount_abs", "booking_date", "text", "remittance_info", "counterparty_name"]].copy()
    c["ak"] = (c.amount_abs * 100).round().astype("int64"); r["ak"] = (r.amount * 100).round().astype("int64")
    m = c.merge(r, on=["entity", "ak"], suffixes=("", "_r"))
    m["dt"] = (m.booking_date - m.ret_date).dt.days
    m = m[(m.dt >= 0) & (m.dt <= 90)].copy()
    if m.empty:
        return m
    txt = (m.text.fillna("") + " " + m.remittance_info.fillna("")).str.upper()
    rd = m.remittance_info_r.map(bc.digits)
    m["refhit"] = [len(d) >= 5 and any(g.endswith(d[-6:]) or d.endswith(g) for g in re.findall(r"\d{5,}", t)) for d, t in zip(rd, txt)]
    m["nsim"] = [max(bc.name_sim(a, b), bc.name_sim(t, b)) for a, b, t in zip(m.counterparty_name, m.debtor_name, txt)]
    m = m[m.refhit | (m.nsim >= 0.6)].copy()
    m["conf"] = np.where(m.refhit, 0.93, 0.8)
    m = m.sort_values(["bank_txn_id", "refhit", "nsim", "dt"], ascending=[True, False, False, True])
    # unique: one receipt per credit line, one credit line per returned receipt; otherwise drop (a wrong pair costs more)
    m["n1"] = m.groupby("bank_txn_id").ret_txn.transform("nunique"); m["n2"] = m.groupby("ret_txn").bank_txn_id.transform("nunique")
    m = m[(m.n1 == 1) & (m.n2 == 1)].drop_duplicates("bank_txn_id")
    return m


# ------------------------------------------------------------------------------------------------------------- match
def match(phase: str, claimed: set | None = None) -> pd.DataFrame:
    claimed = set(claimed or ())
    bank = data.bank(phase)
    bank = bank[~bank.bank_txn_id.isin(claimed)]
    start = bank.booking_date.min()
    dd = data.dd(phase); card = data.card(phase); chq = data.cheque(phase)
    ddh = _read_hist(phase, "direct_debit_batches", start, _fix_dd)
    cardh = _read_hist(phase, "card_settlements", start, _fix_card)
    chqh = _read_hist(phase, "cheque_deposits", start, _fix_chq)
    ddall = _cat(dd, ddh).drop_duplicates(["batch_id", "end_to_end_id"]) if len(dd) or len(ddh) else pd.DataFrame()
    cardall = _cat(card, cardh).drop_duplicates(["settlement_id", "order_number", "gross", "card"]) if len(card) or len(cardh) else pd.DataFrame()
    chqall = _cat(chq, chqh).drop_duplicates(["deposit_id", "cheque_number"]) if len(chq) or len(chqh) else pd.DataFrame()

    items = data.items(phase)
    cn = None
    try:
        c = data.customers(); cn = dict(zip(c.customer_id, c.name))
    except Exception:
        pass
    A = bc.prep_items(items, cn)
    del items

    cr = bank[bank.amount > 0]; db = bank[bank.amount < 0]
    dd_ret = None
    recs = []  # receipt table: bank_txn_id, entity, ccy, ref, amount, name, asof, exception, method, conf
    # 1. DD batches
    if len(ddall):
        T = ddall.groupby("batch_id").agg(account_id=("account_id", "first"), tot=("batch_total", "first"), date=("collection_date", "first")).reset_index()
        L = _link_batches(cr, T, "batch_id")
        STATS["dd_lines"] = len(L)
        r = ddall.merge(L, on="batch_id")
        recs.append(pd.DataFrame({"bank_txn_id": r.bank_txn_id, "entity": r.entity, "ccy": r.currency, "ref": r.remittance_info, "amount": r.amount,
                                  "name": r.debtor_name, "adate": r.collection_date, "exception": None, "method": "dd_batch", "conf": r.conf}))
    # 2. card settlements
    if len(cardall):
        T = cardall.groupby("settlement_id").agg(account_id=("account_id", "first"), tot=("net", "sum"), date=("settlement_date", "first")).reset_index()
        T["tot"] = T.tot.round(2)
        L = _link_batches(cr, T, "settlement_id", hi=3)
        STATS["card_lines"] = len(L)
        r = cardall.merge(L, on="settlement_id")
        recs.append(pd.DataFrame({"bank_txn_id": r.bank_txn_id, "entity": r.entity, "ccy": r.currency, "ref": r.order_number, "amount": r.gross,
                                  "name": None, "adate": r.settlement_date, "exception": None, "method": "card_settlement", "conf": r.conf}))
    # 3. cheque deposits
    if len(chqall):
        T = chqall.groupby("deposit_id").agg(account_id=("account_id", "first"), tot=("deposit_total", "first"), date=("deposit_date", "first")).reset_index()
        L = _link_batches(cr, T, "deposit_id", hi=3)
        STATS["chq_lines"] = len(L)
        r = chqall.merge(L, on="deposit_id")
        recs.append(pd.DataFrame({"bank_txn_id": r.bank_txn_id, "entity": r.entity, "ccy": r.currency, "ref": r.stub_reference, "amount": r.amount,
                                  "name": r.drawer, "adate": r.deposit_date, "exception": None, "method": "cheque_deposit", "conf": r.conf}))
    # 4. reversals
    if len(ddall) and len(db):
        m = _dd_returns(db, ddall, load_lag_model()); STATS["dd_returns"] = len(m)
        dd_ret = m
        if len(m):
            recs.append(pd.DataFrame({"bank_txn_id": m.bank_txn_id, "entity": m.entity, "ccy": m.currency, "ref": m.remittance_info, "amount": m.amount,
                                      "name": m.debtor_name, "adate": m.collection_date, "exception": "RETURNED", "method": "dd_return", "conf": m.conf.values}))
    if dd_ret is not None and len(dd_ret):
        used = set(pd.concat([r["bank_txn_id"] for r in recs])) if recs else set()
        m = _represented(cr[~cr.bank_txn_id.isin(used)], dd_ret); STATS["represented"] = len(m)
        if len(m):
            recs.append(pd.DataFrame({"bank_txn_id": m.bank_txn_id.values, "entity": m.entity.values, "ccy": m.currency.values, "ref": m.remittance_info_r.values,
                                      "amount": m.amount.values, "name": m.debtor_name.values, "adate": m.collection_date.values, "exception": None,
                                      "method": "dd_represented", "conf": m.conf.values}))
    if len(chqall) and len(db):
        m = _chq_returns(db, chqall); STATS["chq_returns"] = len(m)
        if len(m):
            recs.append(pd.DataFrame({"bank_txn_id": m.bank_txn_id, "entity": m.entity, "ccy": m.currency, "ref": m.stub_reference, "amount": m.amount,
                                      "name": m.drawer, "adate": m.deposit_date, "exception": "RETURNED", "method": "cheque_return", "conf": 0.95}))
    if len(cardall) and len(db):
        m = _chargebacks(db, cardall); STATS["chargebacks"] = len(m)
        if len(m):
            recs.append(pd.DataFrame({"bank_txn_id": m.bank_txn_id, "entity": m.entity, "ccy": m.currency, "ref": m.order_number, "amount": m.gross,
                                      "name": None, "adate": m.settlement_date, "exception": "RETURNED", "method": "chargeback", "conf": 0.95}))
    if not recs:
        return pd.DataFrame(columns=COLS)
    R = pd.concat(recs, ignore_index=True)
    R["rid"] = np.arange(len(R))
    cols = ["rid", "entity", "ccy", "ref", "amount", "name", "adate"]
    res = bc.resolve(R[cols], A)
    res["pool"] = "open"
    A2 = None
    for step in ("hist", "name"):  # fall-backs for entries still unresolved: invoices settled before the phase; payer name + amount
        miss = R[~R.rid.isin(res.rid)]
        if step == "name":
            miss = miss[miss.method.str.startswith("cheque")]  # only cheques carry a payer name that is a reliable key
        if miss.empty:
            continue
        try:
            if A2 is None:
                H = _hist_items(phase, start, set(A.item_id))
                A2 = bc.prep_items(H, cn) if len(H) else A.iloc[:0]
            if step == "hist":
                r2 = bc.resolve(miss[cols], A2) if len(A2) else res.iloc[:0]
            else:
                r2 = bc.resolve_by_name(miss[cols], A)
                m2 = miss[~miss.rid.isin(r2.rid)]
                if len(m2) and len(A2):
                    r2 = pd.concat([r2, bc.resolve_by_name(m2[cols], A2)], ignore_index=True)
            r2["pool"] = step
            STATS[f"resolved_{step}"] = len(r2)
            res = pd.concat([res, r2], ignore_index=True)
        except Exception as e:  # never crash: degrade to no claim
            STATS[f"{step}_error"] = repr(e)
    STATS["receipts"] = len(R); STATS["resolved"] = len(res)
    out = R.merge(res, on="rid")
    out = out.sort_values(["bank_txn_id", "item_id", "score"], ascending=[True, True, False]).drop_duplicates(["bank_txn_id", "item_id"])
    out["confidence"] = out.conf
    return out[["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"]].reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="val"); a = ap.parse_args()
    from matching.lib import evalutil
    res = match(a.phase)
    os.makedirs(f"matching/out/{a.phase}", exist_ok=True)
    res.to_parquet(f"matching/out/{a.phase}/{NAME}.parquet", index=False)
    print(STATS)
    evalutil.report(res, a.phase, name=NAME, top=15)
