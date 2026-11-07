"""Supplier payments (AP debits) settled through our payment-run files (payment_runs/).

A bank debit is linked to a "unit" of a payment file (run ids are only unique per debtor account):
  - transfers run  -> one unit per end_to_end_id (one debit per beneficiary)
  - aggregated run -> one unit for the whole run (text "BULK PAYMENT PR…", "TRANSF. EMITIDAS FICHERO C34 PR…")
  - confirming run -> one debit per maturity date ("PAGO CONFIRMING | REMESA CF… n PROVEEDORES"): the beneficiaries of
                      the run whose invoices fall due on the booking date.
Link keys, most certain first: end_to_end_id / run id in the bank line; else same account + amount (orig amount for
FX) + booking date shortly after execution (+ N43 transfer sequence number, + beneficiary name).
Each document of the unit becomes a row: amount = |document_amount| (item currency), PARTIAL when the document is paid
for less than its open amount, FX_DIFFERENCE on fully-paid invoices when the account currency differs (POLICIES §2-3).
An end_to_end_id with a trailing "D" is a duplicate transfer of an old run -> DUPLICATE_PAYMENT.
Items of ES01/ES02/PT01 paid from ES00 accounts belong to the subsidiary (payment factory, POLICIES §5.2).
"""
from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from matching import ap_helpers as H
from matching.common import Ctx, alloc_frame

NAME = "payment_runs"

RE_E2E = re.compile(r"((?:PR|CF)\d{12})-(\d{4})(D?)")
RE_RUN = re.compile(r"((?:PR|CF)\d{12})")
RE_N43_SEQ = re.compile(r"PAGO TRANSFERENCIAS \| \S+ \d{6} (\d{6})")
RE_RETURN = r"^(?:DEVOLUCION TRANSFERENCIA|RETURN [A-Z]{2}\d{2}) "
RE_NPROV = re.compile(r"(\d+) PROVEEDORES")
LAST_HOW: dict[str, str] = {}   # bank_txn_id -> link kind (debugging)
CLAIM_RETURNS = True          # ap_returned_transfer lines are claimed by another matcher
MAX_LAG = 10                    # booking date - execution date, days (implicit matches)


def _compact(s: str) -> str:
    return s.replace(" | ", "").replace("|", "")


def _days(a: str, b: str) -> int:
    return int((np.datetime64(a) - np.datetime64(b)).astype(int))


def _name_hit(benef: str, text: str) -> bool:
    toks = [w for w in benef.split() if len(w) >= 3][:2]
    return bool(toks) and all(w in text for w in toks)


class _Runs:
    def __init__(self, ctx: Ctx):
        r = H.load_runs(ctx, months_before=6, months_after=3)
        self.empty = r.empty
        if self.empty:
            return
        r["item_id"], self.closed = H.resolve_items(ctx, r, ctx.items)
        self.open_ids = set(ctx.items.item_id)
        r["ekey"] = r.debtor_account_id + "|" + r.end_to_end_id
        r["rkey"] = r.debtor_account_id + "|" + r.run_id
        self.r = r
        self.rows = {k: g for k, g in r.groupby("ekey", sort=False)}
        first = r.drop_duplicates("ekey")
        self.info = first.set_index("ekey")
        self.amt = dict(zip(first.ekey, first.transfer_amount.round(2)))
        self.by_e2e = first.groupby("end_to_end_id").ekey.agg(list).to_dict()
        self.run_ekeys = first.groupby("rkey").ekey.agg(list).to_dict()
        self.by_run = first.groupby("run_id").rkey.agg(lambda s: sorted(set(s))).to_dict()
        self.run_type = dict(zip(first.rkey, first.run_type))
        self.by_doc_rows = defaultdict(list)
        for k, d, e, a, it, al in zip(r.ekey, r.execution_date, r.debtor_entity, r.document_amount.abs(), r.item_id,
                                      r.doc_alnum):
            self.by_doc_rows[al].append((k, d, e, a, it))
        tr = first[first.run_type == "transfers"]
        self.by_acc_amt = defaultdict(list)
        for k, acc, a in zip(tr.ekey, tr.debtor_account_id, tr.transfer_amount):
            self.by_acc_amt[(acc, round(a * 100))].append(k)
        self.by_acc_cur_amt = defaultdict(list)
        for k, acc, cur, a in zip(tr.ekey, tr.debtor_account_id, tr.currency, tr.transfer_amount):
            self.by_acc_cur_amt[(acc, cur, round(a * 100))].append(k)

    @staticmethod
    def pick_account(keys: list[str], t) -> list[str]:
        """keys 'ACC|…' -> those of the line's account, else of its entity (payment factory: ES00 accounts)."""
        k1 = [k for k in keys if k.split("|")[0] == t.account_id]
        if k1:
            return k1
        return [k for k in keys if k[:4] == t.entity]


def run(ctx: Ctx) -> pd.DataFrame:
    R = _Runs(ctx)
    if R.empty:
        return alloc_frame([])
    lines = ctx.unclaimed()
    lines = lines[lines.amount < 0].sort_values(["booking_date", "bank_txn_id"])
    links: dict[str, tuple[list[str], str, float, bool]] = {}   # bt -> (ekeys, how, confidence, duplicate)
    used: set[str] = set()

    # ---- pass 1: explicit references (end_to_end_id / run id in the line) -------------------------------------
    implicit = []
    for t in lines.itertuples(index=False):
        txt = " ".join(x for x in (t.text, t.remittance_info, t.end_to_end_id, t.ref1, t.ref2) if isinstance(x, str))
        ctxt = _compact(txt)
        m = RE_E2E.search(ctxt)
        if m:
            e2e, dup = m.group(1) + "-" + m.group(2), bool(m.group(3))
            ks = [k for k in R.pick_account(R.by_e2e.get(e2e, []), t) if dup or k not in used]
            if len(ks) == 1 and R.info.run_type[ks[0]] == "transfers" and _amount_ok(t, R.amt[ks[0]], R.info.currency[ks[0]]):
                links[t.bank_txn_id] = (ks, "e2e_dup" if dup else "e2e", 0.99, dup)
                if not dup:
                    used.add(ks[0])
                continue
            implicit.append(t)
            continue
        m = RE_RUN.search(ctxt)
        if m:
            rks = R.pick_account(R.by_run.get(m.group(1), []), t)
            if len(rks) == 1:
                rk = rks[0]
                typ = R.run_type[rk]
                if typ == "aggregated":
                    ks = R.run_ekeys[rk]
                    if not any(k in used for k in ks):
                        links[t.bank_txn_id] = (ks, "agg", 0.97, False)
                        used.update(ks)
                    continue
                if typ == "confirming":
                    ks = _confirming(t, ctxt, R, R.run_ekeys[rk], ctx.items_by_id, used)
                    if ks:
                        links[t.bank_txn_id] = (ks, "confirming", 0.95, False)
                        used.update(ks)
                    continue
        implicit.append(t)

    # ---- pass 2: implicit (account + amount + date [+ N43 sequence] [+ name]) -----------------------------------
    for t in implicit:
        cands = list(R.by_acc_amt.get((t.account_id, round(abs(t.amount) * 100)), ()))
        if isinstance(t.orig_currency, str) and t.orig_currency and t.orig_currency != t.currency and pd.notna(t.orig_amount):
            cands += R.by_acc_cur_amt.get((t.account_id, t.orig_currency, round(abs(t.orig_amount) * 100)), [])
        cands = [k for k in dict.fromkeys(cands)
                 if k not in used and 0 <= _days(t.booking_date, R.info.execution_date[k]) <= MAX_LAG]
        if not cands:
            continue
        seq = RE_N43_SEQ.search(t.text or "")
        if seq:
            cands = [k for k in cands if k.endswith("-" + seq.group(1)[-4:])]
        if len(cands) > 1:
            nt = H.norm_name(t.text)
            c2 = [k for k in cands if _name_hit(R.info.benef_norm[k], nt)]
            if c2:
                cands = c2
        if len(cands) > 1:
            lag = {k: _days(t.booking_date, R.info.execution_date[k]) for k in cands}
            best = min(lag.values())
            cands = [k for k in cands if lag[k] == best]
        if len(cands) != 1:
            continue
        links[t.bank_txn_id] = (cands, "implicit", 0.9, False)
        used.add(cands[0])

    # ---- pass 3: our transfers returned by the beneficiary bank (credit lines): they re-open the items. The lines
    # themselves belong to another matcher (CLAIM_RETURNS = False); we only use them to keep open amounts right. ----
    credits = ctx.bank
    credits = credits[(credits.amount > 0) & credits.text.fillna("").str.contains(RE_RETURN)]
    for t in credits.itertuples(index=False):
        txt = " ".join(x for x in (t.text, t.remittance_info, t.end_to_end_id, t.ref1, t.ref2) if isinstance(x, str))
        m = RE_E2E.search(_compact(txt))
        if m:
            ks = R.pick_account(R.by_e2e.get(m.group(1) + "-" + m.group(2), []), t)
            if len(ks) == 1 and _amount_ok(t, R.amt[ks[0]], R.info.currency[ks[0]]):
                links[t.bank_txn_id] = (ks, "returned", 0.97, False)
            continue
        nt = H.norm_name(t.text)
        ks = [k for k in R.by_acc_amt.get((t.account_id, round(t.amount * 100)), ())
              if 0 <= _days(t.booking_date, R.info.execution_date[k]) <= 45 and _name_hit(R.info.benef_norm[k], nt)]
        if len(ks) == 1:
            links[t.bank_txn_id] = (ks, "returned_implicit", 0.9, False)

    # ---- rows, in booking order, tracking the open amount of each item -----------------------------------------
    items = ctx.items_by_id
    open_amt = items.open_amount.to_dict()
    cur_of = items.currency.to_dict()
    cur_of.update(zip(R.closed.item_id, R.closed.currency))
    bank = ctx.bank.set_index("bank_txn_id")
    order = sorted(links, key=lambda bt: (bank.booking_date[bt], bt))
    linked_e2e: set[str] = set()   # units actually emitted so far (chronological)
    start = ctx.bank.booking_date.min()
    out = []
    LAST_HOW.clear()
    for bt in order:
        ks, how, conf, dup = links[bt]
        df = pd.concat([R.rows[k] for k in ks])
        bank_cur = bank.currency[bt]
        if how == "agg":   # a whole file: keep the documents we can resolve (pairs count even if one is missing)
            df = df[df.item_id != ""]
        rr = []
        df_doc = dict(zip(df.item_id, zip(df.doc_alnum, df.execution_date, df.debtor_entity)))
        for iid, dt, da in zip(df.item_id, df.document_type, df.document_amount):
            if not iid:
                rr = None
                break
            amt = round(abs(da), 2)
            op = open_amt.get(iid, 0.0)
            exc = ""
            if how.startswith("returned"):
                exc = "RETURNED"
            elif dup:
                exc = "DUPLICATE_PAYMENT"
            elif dt != "CREDIT_NOTE":
                if amt < op - max(0.01, 0.005 * op) and not _remainder(R, iid, df_doc[iid], amt, op, linked_e2e, start):
                    exc = "PARTIAL"
                elif cur_of[iid] != bank_cur:
                    exc = "FX_DIFFERENCE"
            rr.append((iid, amt, exc))
        if not rr or len({x[0] for x in rr}) != len(rr):
            continue
        returned = how.startswith("returned")
        for iid, amt, exc in rr:
            if returned:
                open_amt[iid] = open_amt.get(iid, 0.0) + amt
            elif not dup:
                open_amt[iid] = max(0.0, open_amt.get(iid, 0.0) - amt)
            if CLAIM_RETURNS or not returned:
                out.append((bt, iid, amt, exc, conf, NAME))
        if returned and not CLAIM_RETURNS:
            continue
        LAST_HOW[bt] = how
        linked_e2e.update(ks)
    out += _direct_debits(ctx, lines[~lines.bank_txn_id.isin(set(LAST_HOW))], open_amt)
    return alloc_frame(out)


def _direct_debits(ctx: Ctx, lines: pd.DataFrame, open_amt: dict) -> list[tuple]:
    """Supplier direct debits (no run file): the supplier pulls exactly one open 'dd' invoice. Candidates: open AP
    invoices with payment_method dd, same entity (or payment-factory subsidiary) and currency, open amount = debit.
    Confirmed by the document number or the supplier name in the bank text; else FIFO is not trusted."""
    it = ctx.open_items
    dd = it[(it.item_type == "API") & (it.payment_method == "dd")]
    idx = defaultdict(list)
    for t in dd.itertuples(index=False):
        idx[(t.entity, t.currency, round(t.open_amount * 100))].append(t)
    out = []
    used: set[str] = set()
    for t in lines.sort_values(["booking_date", "bank_txn_id"]).itertuples(index=False):
        cents = round(abs(t.amount) * 100)
        cands = []
        for e in H.FACTORY_SUBS.get(t.entity, {t.entity}):
            cands += idx.get((e, t.currency, cents), [])
        cands = [c for c in cands if c.item_id not in used and c.issue_date <= t.booking_date
                 and abs(open_amt.get(c.item_id, 0.0) - abs(t.amount)) < 0.005]
        if not cands:
            continue
        toks = set(t.doc_tokens) if t.doc_tokens is not None else set()
        txt = H.norm_name(" ".join(x for x in (t.text, t.counterparty_name, t.remittance_info) if isinstance(x, str)))
        by_doc = [c for c in cands if c.document_alnum and (c.document_alnum in toks or c.document_alnum in txt.replace(" ", ""))]
        by_name = [c for c in cands if _name_in(c.party_name, txt)]
        how = None
        if len(by_doc) == 1:
            pick, how = by_doc[0], "dd_doc"
        elif len(by_name) == 1:
            pick, how = by_name[0], "dd_name"
        elif len(by_name) > 1:
            pick, how = sorted(by_name, key=lambda c: (c.due_date, c.issue_date, c.item_id))[0], "dd_name_fifo"
        if how is None:
            continue
        used.add(pick.item_id)
        open_amt[pick.item_id] = 0.0
        out.append((t.bank_txn_id, pick.item_id, round(abs(t.amount), 2), "", 0.85, NAME))
        LAST_HOW[t.bank_txn_id] = how
    return out


_STOP = {"DE", "DEL", "LA", "EL", "LOS", "LAS", "Y", "E", "DA", "DO", "DI", "THE", "AND", "OF", "VAN", "DER"}
_LEGAL_W = {"SL", "SLU", "SA", "SAU", "COOP", "SCOOP", "SARL", "SAS", "EURL", "GMBH", "LTD", "LLC", "INC", "CORP", "CO",
            "BV", "SPA", "SRL", "SNC", "PTY", "LIMITED", "LTDA", "EIRELI", "LDA", "EK", "KG", "AG", "NV", "LLP", "PLC"}


def _sig_words(s: str) -> list[str]:
    return [w for w in H.norm_name(s).split() if w not in _STOP]


def _name_in(party_name: str, text: str) -> bool:
    """Supplier name (first two significant words) visible in a bank text, the second word possibly truncated."""
    toks = [w for w in _sig_words(party_name) if w not in _LEGAL_W and len(w) >= 2][:2]
    if not toks:
        return False
    words = _sig_words(text)
    for i, w in enumerate(words):
        if w != toks[0]:
            continue
        if len(toks) == 1:
            return True
        if i + 1 < len(words) and len(words[i + 1]) >= 2 and toks[1].startswith(words[i + 1]):
            return True
    return False


def _remainder(R: _Runs, iid: str, doc: tuple, amt: float, op: float, linked_e2e: set, start: str) -> bool:
    """True when an earlier run row of this phase that we could not link to a bank line (e.g. a partial payment to a
    factor whose document was ambiguous) paid exactly the difference: this payment is the remainder, not a partial."""
    alnum_, exec_date, ent = doc
    ents = H.FACTORY_SUBS.get(ent, {ent})
    prev = R.by_doc_rows.get(alnum_, [])
    paid = [a for k, d, e, a, it in prev
            if k not in linked_e2e and start <= d < exec_date and e in ents and it in ("", iid)]
    return any(abs(op - amt - a) <= max(0.01, 0.005 * op) for a in paid)


def _amount_ok(t, unit_amount: float, unit_cur: str) -> bool:
    if unit_cur == t.currency:
        return abs(abs(t.amount) - unit_amount) <= 0.011
    if pd.notna(t.orig_amount):
        return abs(abs(t.orig_amount) - unit_amount) <= 0.011
    return True


def _pick(sel, target: int, nprov: int | None, max_drop: int = 3) -> list[str] | None:
    """The e2e set of `sel` (dropping at most `max_drop`, or choosing `nprov` of ≤ 25) that sums to `target`, if unique."""
    if not sel:
        return None
    if sum(a for _, a, _ in sel) == target and (nprov is None or len(sel) == nprov):
        return [k for k, _, _ in sel]
    if nprov is None or nprov >= len(sel):
        return None
    drop = len(sel) - nprov
    if drop <= max_drop and len(sel) <= 80:
        tot = sum(a for _, a, _ in sel)
        sols = [c for c in combinations(range(len(sel)), drop) if tot - sum(sel[i][1] for i in c) == target]
        if len(sols) == 1:
            return [k for i, (k, _, _) in enumerate(sel) if i not in sols[0]]
        return None
    if len(sel) <= 25 and nprov <= 4:
        sols = [c for c in combinations(sel, nprov) if sum(a for _, a, _ in c) == target]
        if len(sols) == 1:
            return [k for k, _, _ in sols[0]]
    return None


def _paydate(exec_date: str, maturity: str) -> str:
    d = np.datetime64(maturity)
    while not np.is_busday(d):
        d += np.timedelta64(1, "D")
    return str(d)


def _confirming(t, ctxt, R: _Runs, ekeys: list[str], items, used) -> list[str] | None:
    """ekeys of a confirming run paid by this debit (one debit per maturity date)."""
    m = RE_NPROV.search(ctxt)
    nprov = int(m.group(1)) if m else None
    target = round(abs(t.amount) * 100)
    open_ids = R.open_ids
    cand = []
    for k in ekeys:
        g = R.rows[k]
        if k in used or not all(i and i in open_ids for i in g.item_id):
            continue   # unresolved, or already settled before the phase
        inv = g[g.document_type != "CREDIT_NOTE"]
        due = items.due_date.reindex(inv.item_id).dropna()
        mat = due.max() if len(due) else ""
        cand.append((k, round(R.amt[k] * 100), mat))
    bd = t.booking_date
    # 1) by maturity: an e2e is paid on the first business day >= the latest due date of its invoices
    sel = [(k, a, mat) for k, a, mat in cand if mat and _paydate(R.info.execution_date[k], mat) == bd]
    hit = _pick(sel, target, nprov)
    if hit:
        return hit
    sel = [(k, a, mat) for k, a, mat in cand if mat and 0 <= _days(bd, _paydate(R.info.execution_date[k], mat)) <= 3]
    hit = _pick(sel, target, nprov)
    if hit:
        return hit
    # 2) everything still unpaid that has fallen due by the booking date (the run's own execution day)
    sel = [(k, a, mat) for k, a, mat in cand if mat and _paydate(R.info.execution_date[k], mat) <= bd]
    if sel and sum(a for _, a, _ in sel) == target and (nprov is None or len(sel) == nprov):
        return [k for k, _, _ in sel]
    # 3) subset sum of the open beneficiaries with the stated count (unique solution only)
    if nprov is not None and nprov <= 3 and len(cand) <= 60:
        sols = [c for c in combinations(cand, nprov) if sum(a for _, a, _ in c) == target]
        if len(sols) == 1:
            return [k for k, _, _ in sols[0]]
    return None
