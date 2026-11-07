"""Matcher `remittance`: bank line <-> remittance advice (payment ref / payer name / total) -> advice document lines -> items.

Targets golden segments ar_1toN_remittance, ar_short_dispute, ar_1to1_remittance (and every other segment whose cash was
announced by an advice).  Rules:
  link   bank line <-> advice: same entity, advice total == bank amount (to the cent), advice received in [-10d, +45d] of the
         booking date, and (payment ref found in the bank text/refs, or payer name found).  One advice per bank line and one bank
         line per advice, else no claim (duplicate payments are left to other matchers).
  items  each advice line -> AR item of the entity: full normalised number, then digits-only (prefix-less / zero-stripped),
         then trailing-digits (our number without prefix) + gross amount, then (issue date, gross) unique within the customer.
         The customer = majority party of the resolved lines (names never needed, so renamed customers still work).
  rows   one row per item.  no deduction -> amount = paid.  Credit note netted -> its full amount.  Deduction with a stated
         dispute reason -> SHORT_DISPUTE, amount = paid (only on that line).  RET -> SHORT_WITHHOLDING (full amount).
         DISCOUNT / silent shortfall -> SHORT_DISCOUNT (full amount) when the invoice terms allow it, else PARTIAL (amount = paid).
         Item currency != bank currency -> FX_DIFFERENCE on fully cleared invoices.
Run:  python -m matching.matchers.remittance --phase val
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from matching.lib import data, evalutil, remit_parse as rp  # noqa: E402

NAME = "remittance"
WIN_BEFORE, WIN_AFTER = -10, 45          # advice received - booking date, days
DISCOUNT_WORDS = ("DISCOUNT", "DTO", "PPD", "SKONTO", "ESCOMPTE", "SCONTO", "DESCUENTO", "DESCONTO", "PRONTO")
WITHHOLD_WORDS = ("RET", "WHT", "WITHHOLD", "RETENC", "RETENUE", "RITENUTA", "QUELLENST")
WITHHOLD_RATES = (1.25, 1.5, 10.0)             # POLICIES s3: some customers in Mexico, Brazil, Morocco
WITHHOLD_COUNTRIES = ("MX", "BR", "MA")
LEGAL = {"SA", "SL", "SAS", "SARL", "GMBH", "LTD", "LLC", "INC", "PTY", "BV", "SRL", "SPA", "AG", "LDA", "CO", "CORP", "LIMITED",
         "EURL", "SCOOP", "SLU", "SAU", "EK", "KG", "SP", "ZOO", "DE", "CV", "THE"}


def _norm_name(s) -> str:
    if s is None or (isinstance(s, float) and s != s):
        return ""
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    toks = [w for w in re.sub(r"[^A-Z0-9 ]", " ", t).split() if w not in LEGAL]
    return "".join(toks)


def _compact(s) -> str:
    if s is None or (isinstance(s, float) and s != s):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _ref_hit(ref: str, fields: list[str]) -> bool:
    """payment ref present in one of the bank fields (full, or truncated: a field ends with a >=7-char prefix of the ref)."""
    r = _compact(ref)
    if len(r) < 6:
        return False
    for f in fields:
        c = _compact(f)
        if not c:
            continue
        if r in c:
            return True
        for k in range(len(r) - 1, 6, -1):
            if c.endswith(r[:k]):
                return True
    return False


def _name_hit(payer: str, cp, text) -> bool:
    p = _norm_name(payer)
    if len(p) < 6:
        return False
    k = p[: min(len(p), 12)]
    return k in _compact(text) or (_norm_name(cp)[:8] == p[:8] and len(_norm_name(cp)) >= 6) or k in _norm_name(text)


# ---------------------------------------------------------------------------------------------------------------- linking
def link(phase: str, bank: pd.DataFrame, H: pd.DataFrame) -> pd.DataFrame:
    """Candidate (bank line, advice) pairs with evidence, filtered to unique ones. Columns: bank_txn_id, ra_id, ref_hit, name_hit."""
    bi = bank[bank.direction == "in"].copy()
    bi["k"] = (bi.amount * 100).round().astype("int64")
    h = H.copy()
    h["k"] = (h.tot * 100).round().astype("int64")
    m = bi.merge(h, on=["entity", "k"], suffixes=("", "_h"))
    m["dd"] = (m.rec - m.booking_date).dt.days
    m = m[(m.dd >= WIN_BEFORE) & (m.dd <= WIN_AFTER)]
    if m.empty:
        return m.assign(ref_hit=False, name_hit=False)[["bank_txn_id", "ra_id"]].assign(ref_hit=False, name_hit=False)
    fields = ["text", "ref1", "ref2", "end_to_end_id"]
    for f in fields:
        if f not in m:
            m[f] = None
    m["ref_hit"] = [_ref_hit(r, [a, b, c, d]) for r, a, b, c, d in zip(m.ref, m.text, m.ref1, m.ref2, m.end_to_end_id)]
    m["name_hit"] = [_name_hit(p, cp, t) for p, cp, t in zip(m.payer, m.counterparty_name, m.text)]
    m = m[m.ref_hit | m.name_hit].copy()
    m["score"] = m.ref_hit.astype(int) * 2 + m.name_hit.astype(int)
    # keep best candidate per bank line / per advice when unambiguous
    def _best(df, by):
        top = df.groupby(by).score.transform("max")
        df = df[df.score == top]
        return df[df.groupby(by).score.transform("size") == 1]
    m = _best(m, "bank_txn_id")
    m = _best(m, "ra_id")
    return m[["bank_txn_id", "ra_id", "ref_hit", "name_hit", "dd"]].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------- items
def _ar_items(phase: str) -> pd.DataFrame:
    it = data.items(phase)
    it = it[it.side == "AR"][["item_id", "item_type", "entity", "party_id", "party_name", "document_number", "issue_date", "currency",
                              "total_amount", "open_amount", "payment_terms"]].copy()
    it["entity"] = it.entity.astype(str)
    it["fk"] = it.document_number.map(rp.norm_doc)
    it["dk"] = it.document_number.map(rp.digits_key)
    return it.reset_index(drop=True)


def _amt_ok(gross, total, open_):
    g = np.abs(gross)
    return ((g - total).abs() <= np.maximum(0.011, 0.005 * total)) | ((g - open_).abs() <= np.maximum(0.011, 0.005 * open_))


def resolve_lines(Ll: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Ll: advice lines (columns ra_id, line_no, entity, doc_ref_raw, gross, doc_date). Returns Ll with item_id + tier + amt_ok."""
    Ll = Ll.copy()
    Ll["fk"] = Ll.doc_ref_raw.map(rp.norm_doc)
    Ll["dk"] = Ll.doc_ref_raw.map(rp.digits_key)
    Ll["dd_date"] = pd.to_datetime(Ll.doc_date, errors="coerce", dayfirst=True, format="mixed")
    cols = ["item_id", "party_id", "total_amount", "open_amount", "currency", "item_type", "issue_date", "payment_terms", "party_name"]
    out = []
    left = Ll
    tiers = []
    # tier 1/2: exact keys
    for tier, key in ((1, "fk"), (2, "dk")):
        if left.empty:
            break
        it = items[items[key].str.len() >= 3]
        m = left.merge(it[["entity", key] + cols], on=["entity", key], how="inner")
        m["tier"] = tier
        m["amt_ok"] = _amt_ok(m.gross, m.total_amount, m.open_amount)
        out.append(m)
        left = left[~left.set_index(["ra_id", "line_no"]).index.isin(m.set_index(["ra_id", "line_no"]).index)]
    # tier 3: advice number is the trailing digits of ours (prefix-less), gross amount must agree
    if not left.empty:
        left = left[left.dk.str.len() >= 4]
        parts = []
        for (ent, L), g in left.groupby(["entity", left.dk.str.len()], observed=True):
            it = items[(items.entity == ent) & (items.dk.str.len() > L)].copy()
            it["suf"] = it.dk.str[-L:]
            m = g.merge(it[["suf"] + cols], left_on="dk", right_on="suf", how="inner")
            m["amt_ok"] = _amt_ok(m.gross, m.total_amount, m.open_amount)
            parts.append(m[m.amt_ok])
        if parts:
            m = pd.concat(parts)
            m["tier"] = 3
            out.append(m)
            left = left[~left.set_index(["ra_id", "line_no"]).index.isin(m.set_index(["ra_id", "line_no"]).index)]
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return res, left


def _discount_terms(pt: pd.DataFrame) -> dict:
    return {r.code: (r.early_payment_discount_pct, r.discount_days) for r in pt.itertuples() if r.early_payment_discount_pct == r.early_payment_discount_pct}


def _classify(reason, ded, gross, paid, item_type, terms, issue, booking, bank_cur, item_cur, pay_terms, country):
    """-> (amount, exception) for one advice line (POLICIES s2/s3)."""
    full = round(abs(gross), 2)
    amt = round(abs(paid), 2)
    exc = None
    if item_type == "ARC":
        return full, None
    short = round(abs(gross) - abs(paid), 2)
    if short > 0.004:
        r = (reason or "").upper()
        pct = 100.0 * short / full if full else 0.0
        wh = (country in WITHHOLD_COUNTRIES) and any(abs(short - round(full * w / 100.0, 2)) <= 0.011 for w in WITHHOLD_RATES)
        disc = False
        pd_ = terms.get(pay_terms) if pay_terms else None
        cands = [pd_] if pd_ is not None else [(1.0, 15), (2.0, 10), (3.0, 8)]
        for dp, dd in cands:
            if abs(paid - round(gross * (1 - dp / 100.0), 2)) <= 0.011 and pd.notna(issue) and pd.notna(booking) and booking <= issue + pd.Timedelta(days=int(dd) + 3):
                disc = True
        if r and not any(w in r for w in DISCOUNT_WORDS):
            if wh and any(w in r for w in WITHHOLD_WORDS):
                amt, exc = full, "SHORT_WITHHOLDING"
            else:
                exc = "SHORT_DISPUTE"
        elif disc:
            amt, exc = full, "SHORT_DISCOUNT"
        elif wh:
            amt, exc = full, "SHORT_WITHHOLDING"
        else:
            exc = "PARTIAL"
    if bank_cur is not None and item_cur is not None and bank_cur != item_cur and exc in (None, "SHORT_DISCOUNT", "SHORT_WITHHOLDING"):
        exc, amt = "FX_DIFFERENCE", full
    return amt, exc


PARTIAL_RATIOS = tuple(r / 100 for r in range(5, 100, 5))   # unplanned partial = a round fraction of the invoice (checked on a 2023 train slice)
MAX_SUBSET = 18


def _explain_shortfall(rows: pd.DataFrame, D: float, terms: dict, st: dict):
    """Advice total is D below the sum of listed lines. Unique explanation among: (a) one line paid (1-r) of its total (partial),
    (b) a subset of lines that took the customer's early-payment discount. Returns {row index: (amount, exception)} or None."""
    if D <= 0.011:
        return None
    sols = []
    for i, r in rows.iterrows():
        if r.item_type == "ARC":
            continue
        for rt in PARTIAL_RATIOS:
            if abs(round(abs(r.gross) * rt, 2) - D) <= 0.011:
                sols.append({i: (round(abs(r.gross) - D, 2), "PARTIAL")})
    el = [(i, r) for i, r in rows.iterrows() if r.item_type != "ARC" and terms.get(r.payment_terms) is not None]
    if el and len(el) <= MAX_SUBSET:
        pct, days = terms[el[0][1].payment_terms]
        cents = np.array([int(round(round(abs(r.gross) * pct / 100.0, 2) * 100)) for _, r in el])
        n = len(el)
        target = int(round(D * 100))
        sums = np.zeros(1, dtype=np.int64)
        for c in cents:
            sums = np.concatenate([sums, sums + c])
        hit = np.nonzero(np.abs(sums - target) <= 1)[0]
        for h in hit:
            if h == 0:
                continue
            sol = {}
            for b_, (i, r) in enumerate(el):
                if (h >> b_) & 1:
                    pc, dd = terms[r.payment_terms]
                    inw = r.booking_date <= r.issue_date + pd.Timedelta(days=int(dd) + 3)
                    sol[i] = (round(abs(r.gross), 2), "SHORT_DISCOUNT") if inw else (round(abs(r.gross) - round(abs(r.gross) * pc / 100.0, 2), 2), "PARTIAL")
            sols.append(sol)
    if len(sols) == 1:
        st["shortfall_solved"] = st.get("shortfall_solved", 0) + 1
        return sols[0]
    st["shortfall_ambiguous"] = st.get("shortfall_ambiguous", 0) + 1
    return None


# ---------------------------------------------------------------------------------------------------------------- match
PREV_PHASE = {"val": "train", "live": "val", "final": "live"}
PREV_DAYS = 35


def prev_lines(phase: str, start: pd.Timestamp) -> pd.DataFrame:
    """Advices received shortly BEFORE the phase (cash booked in the first days of the phase), read from the previous phase's
    folder (they are past documents, no labels involved). Cached next to the phase cache."""
    src = PREV_PHASE.get(phase)
    if src is None or not os.path.isdir(rp._ra_dir(src)):
        return pd.DataFrame()
    cpath = os.path.join(rp.CACHE, phase, "remittance_lines_prev.parquet")
    if os.path.exists(cpath):
        return pd.read_parquet(cpath)
    df = rp.lines(src, date_from=start - pd.Timedelta(days=PREV_DAYS), date_to=start - pd.Timedelta(days=1), cache=False, quiet=True)
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    df.to_parquet(cpath, index=False)
    return df


def match(phase: str, claimed: set[str] | None = None, stats: dict | None = None) -> pd.DataFrame:
    L = rp.lines(phase)
    bank = data.bank(phase)
    P = prev_lines(phase, bank.booking_date.min())
    if len(P):
        L = pd.concat([P[~P.ra_id.isin(set(L.ra_id))], L], ignore_index=True)
    return match_frames(L, bank, _ar_items(phase), data.customers(), data.payment_terms(), claimed, stats)


def match_frames(L: pd.DataFrame, bank: pd.DataFrame, items: pd.DataFrame, cu: pd.DataFrame, pterms: pd.DataFrame,
                 claimed: set[str] | None = None, stats: dict | None = None) -> pd.DataFrame:
    """Core of the matcher on explicit frames (used by `match` and by the train-slice check)."""
    claimed = claimed or set()
    L = L[L.ra_id.notna()].copy()
    L["entity"] = L.entity.astype(str)
    L["line_no"] = L.groupby("ra_id", observed=True).cumcount()
    H = (L.groupby("ra_id", observed=True)
         .agg(entity=("entity", "first"), payer=("payer_name", "first"), rec=("received_date", "first"), ref=("payment_ref", "first"),
              cur=("currency", "first"), tot=("header_total", "first"), n=("amount_paid", "size"), total_ok=("total_ok", "first"))
         .reset_index())
    H = H[H.tot.notna()]
    bank = bank[~bank.bank_txn_id.isin(claimed)]
    lk = link("", bank, H)
    if lk.empty:
        return pd.DataFrame(columns=["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"])
    Ll = lk.merge(L[["ra_id", "line_no", "entity", "doc_ref_raw", "gross", "deduction", "deduction_reason", "amount_paid", "doc_date"]], on="ra_id")
    res, left = resolve_lines(Ll[["ra_id", "line_no", "entity", "doc_ref_raw", "gross", "doc_date"]], items)
    if res.empty:
        return pd.DataFrame(columns=["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"])
    # customer = majority party over amount-consistent lines of the advice
    good = res[res.amt_ok]
    maj = good.groupby("ra_id").party_id.agg(lambda s: s.value_counts().index[0] if len(s) else None)
    res["party_maj"] = res.ra_id.map(maj)
    res = res[(res.party_id == res.party_maj) & res.amt_ok].copy()
    # tier 4: unresolved lines -> (customer, issue date, gross) unique among items of the majority party
    if not left.empty:
        lf = left.copy()
        lf["party_id"] = lf.ra_id.map(maj)
        lf = lf[lf.party_id.notna() & lf.dd_date.notna()]
        it4 = items[["entity", "party_id", "issue_date", "total_amount", "open_amount", "item_id", "currency", "item_type", "payment_terms", "party_name"]]
        lf["tk"] = (lf.gross.abs() * 100).round().astype("int64")
        it4 = it4.assign(tk=(it4.total_amount * 100).round().astype("int64"))
        m4 = lf.merge(it4, left_on=["entity", "party_id", "dd_date", "tk"], right_on=["entity", "party_id", "issue_date", "tk"], how="inner")
        m4 = m4[m4.groupby(["ra_id", "line_no"]).item_id.transform("size") == 1]
        m4["tier"] = 4
        m4["amt_ok"] = True
        res = pd.concat([res, m4], ignore_index=True)
    # one item per line; an item only once per advice
    res = res.drop_duplicates(["ra_id", "line_no"])
    res = res[res.groupby(["ra_id", "item_id"]).line_no.transform("size") == 1]
    res = res.merge(Ll[["ra_id", "line_no", "deduction", "deduction_reason", "amount_paid", "bank_txn_id", "ref_hit", "name_hit"]], on=["ra_id", "line_no"])
    res = res.merge(bank[["bank_txn_id", "booking_date", "currency", "amount"]].rename(columns={"currency": "bank_cur", "amount": "bank_amt"}), on="bank_txn_id")
    terms = _discount_terms(pterms)
    res["country"] = res.party_id.map(cu.drop_duplicates("customer_id").set_index("customer_id").country)
    cust_terms = cu.drop_duplicates("customer_id").set_index("customer_id").payment_terms
    res["payment_terms"] = res.payment_terms.where(res.payment_terms.notna(), res.party_id.map(cust_terms))
    am, ex = [], []
    for r in res.itertuples():
        a, e = _classify(r.deduction_reason, r.deduction, r.gross, r.amount_paid, r.item_type, terms, r.issue_date, r.booking_date,
                         r.bank_cur, r.currency, r.payment_terms, r.country)
        am.append(a); ex.append(e)
    res["amount"] = am
    res["exception"] = ex
    # advices whose listed paid amounts do not add up to the header total (shortfall not itemised): explain it if unique
    res["unexplained"] = False
    tot_by = H.set_index("ra_id").tot
    ra_bad = set(L.loc[~L.total_ok.astype(bool), "ra_id"]) & set(res.ra_id)
    nl0 = L.groupby("ra_id", observed=True).size()
    for ra in ra_bad:
        idx = res.index[res.ra_id == ra]
        if len(idx) != nl0[ra]:
            res.loc[idx, "unexplained"] = True
            continue
        D = round(res.loc[idx, "amount_paid"].sum() - tot_by[ra], 2)
        sol = _explain_shortfall(res.loc[idx], D, terms, (stats or {}))
        if sol is None:
            res.loc[idx, "unexplained"] = True
        else:
            for i, (a_, e_) in sol.items():
                res.at[i, "amount"], res.at[i, "exception"] = a_, e_
    # confidence per line: link evidence x fraction of advice lines resolved x amount consistency
    nl = L.groupby("ra_id", observed=True).size()
    res["frac"] = res.groupby("ra_id").item_id.transform("size") / res.ra_id.map(nl)
    ev = res.groupby("bank_txn_id").agg(ref=("ref_hit", "first"), name=("name_hit", "first"), frac=("frac", "first"), n=("item_id", "size"))
    conf = np.where(ev.ref & ev.name, 0.98, np.where(ev.ref, 0.96, 0.93))
    conf = conf * np.where(ev.frac >= 0.999, 1.0, np.where(ev.frac >= 0.8, 0.92, 0.8))
    un = res.groupby("bank_txn_id").unexplained.first()
    conf = conf * np.where(un.reindex(ev.index).fillna(False).astype(bool), 0.6, 1.0)
    res["confidence"] = res.bank_txn_id.map(pd.Series(conf, index=ev.index)).round(3)
    res["method"] = NAME + "_t" + res.tier.astype(str)
    if stats is not None:
        stats["links"] = len(lk); stats["lines_resolved"] = len(res); stats["tier_counts"] = res.tier.value_counts().to_dict()
    return res[["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"]].reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="val")
    a = ap.parse_args()
    st: dict = {}
    out = match(a.phase, stats=st)
    print(st)
    od = os.path.join(data.ROOT, "matching", "out", a.phase)
    os.makedirs(od, exist_ok=True)
    out.to_parquet(os.path.join(od, NAME + ".parquet"), index=False)
    if a.phase in ("val", "sample", "train", "train_lite"):
        evalutil.report(out, a.phase, name=NAME)
