"""Payment runs (our outgoing supplier payments): transfers (one debit per e2e), aggregated (one debit per run),
confirming (one debit per maturity date). Bank line → transfer(s) → documents → AP items.
e2e / run ids are NOT unique across entities: transfers are keyed by (debtor account, e2e), runs by (entity, run_id)."""
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from matching.batch.common import PAYFACTORY, item_index, load_detail, parse_amount

RUN_RE = re.compile(r"((?:PR|CF)\d{12})(?:-(\d{1,4}))?")
OCMT_RE = re.compile(r"/OCMT/([A-Z]{3})([\d.,]+)")


def _nm(s):
    return re.sub(r"[^0-9A-Z]", "", str(s).upper())


def resolve_docs(pr, items, phase=""):
    """Add item_id per payment-run row: entity-group + document_number, then supplier (beneficiary IBAN / name),
    then |amount|, then currency; unique hit or ''."""
    from integration.lib.masters import read_master
    idx = item_index(items, ["API", "APC"])
    pname = dict(zip(items.item_id, items.party_name.map(_nm)))
    sup = read_master("suppliers.csv", phase)
    iban2sup = defaultdict(set)
    for sid, ib in zip(sup.supplier_id, sup.iban):
        if ib:
            iban2sup[ib].add(sid)
    ids = []
    for ent, doc, dtype, amt, ccy, ben, bib in zip(pr.debtor_account_id.str[:4], pr.document_number,
                                                   pr.document_type, pr.document_amount.astype(float).abs(),
                                                   pr.currency, pr.beneficiary.map(_nm), pr.beneficiary_iban):
        want = "APC" if dtype == "CREDIT_NOTE" else "API"
        cands = [r for e in PAYFACTORY.get(ent, [ent]) for r in idx.get((e, doc), []) if r.item_type == want]
        if len(cands) > 1 and bib and bib in iban2sup:
            c2 = [r for r in cands if r.party_id in iban2sup[bib]]
            cands = c2 if c2 else cands
        if len(cands) > 1:
            c2 = [r for r in cands if pname.get(r.item_id) == ben]
            cands = c2 if c2 else cands
        if len(cands) > 1:
            c2 = [r for r in cands if abs(r.total_amount - amt) < 0.01 or abs(r.open_amount - amt) < 0.01]
            cands = c2 if c2 else cands
        if len(cands) > 1:
            c2 = [r for r in cands if r.currency == ccy]
            cands = c2 if c2 else cands
        ids.append(cands[0].item_id if len(cands) == 1 else "")
    pr = pr.copy()
    pr["item_id"] = ids
    return pr


class Tr:
    __slots__ = ("key", "e2e", "run_id", "run_type", "exec_date", "account", "ccy", "amt", "beneficiary", "docs")


def build_transfers(pr):
    T = {}
    for r in pr.itertuples(index=False):
        k = (r.debtor_account_id, r.end_to_end_id)
        t = T.get(k)
        if t is None:
            t = Tr()
            t.key, t.e2e, t.run_id, t.run_type, t.exec_date = k, r.end_to_end_id, r.run_id, r.run_type, r.execution_date
            t.account, t.ccy, t.amt, t.beneficiary, t.docs = (r.debtor_account_id, r.currency,
                                                              float(r.transfer_amount), r.beneficiary, [])
            T[k] = t
        t.docs.append((r.item_id, abs(float(r.document_amount)), r.document_type))
    return T


def match(lines, items, ctx=None):
    """Returns list of output rows (bank_txn_id,item_id,amount,exception,confidence,method)."""
    phase = (ctx or {}).get("phase")
    pr = load_detail(phase, "payment_runs")
    if pr.empty:
        return []
    pr = resolve_docs(pr, items, phase)
    T = build_transfers(pr)
    by_acc_e2e = T
    by_ent_e2e = defaultdict(list)
    by_prefix = defaultdict(list)       # (entity, e2e[:16]) → transfers
    runs = defaultdict(list)            # (entity, run_id) → transfers
    for t in T.values():
        ent = t.account[:4]
        by_ent_e2e[(ent, t.e2e)].append(t)
        by_prefix[(ent, t.e2e[:16])].append(t)
        runs[(ent, t.run_id)].append(t)
    item_ccy = dict(zip(items.item_id, items.currency))
    item_due = dict(zip(items.item_id, items.due_date))
    item_open = dict(zip(items.item_id, items.open_amount))

    rows = []
    used = set()

    def emit(btid, trs, method, conf, line_ccy):
        out = []
        for t in trs:
            for it, a, _ in t.docs:
                if not it:
                    return False
                if item_ccy.get(it, t.ccy) != t.ccy:   # document listed in another currency: full open amount
                    a = item_open.get(it, a)
                out.append((it, a))
        if len({it for it, _ in out}) != len(out):
            return False
        for it, a in out:
            exc = "FX_DIFFERENCE" if item_ccy.get(it, line_ccy) != line_ccy and not it.startswith("APC") else ""
            if not it.startswith("APC") and a < item_open.get(it, a) - 0.01 and not exc:
                exc = "PARTIAL"
            item_open[it] = max(0.0, item_open.get(it, a) - a)    # local ledger, lines come in booking order
            rows.append((btid, it, a, exc, conf, method))
        used.update(t.key for t in trs)
        return True

    # returned supplier transfers (credit): e2e in text / remittance_info → items of that transfer, RETURNED
    cre = lines[(lines.amount.astype(float) > 0)
                & lines.text.str.upper().str.contains("RETURN|DEVOLUCION|RUECK|RETOUR|REJET|STORNO", regex=True)]
    for r in cre.itertuples(index=False):
        amt = float(r.amount)
        blob = " ".join([r.end_to_end_id, r.ref1, r.ref2, r.text.replace(" | ", " "), r.remittance_info])
        m = re.search(r"((?:PR|CF)\d{12}-\d{4})", blob)
        cands = []
        if m:
            cands = [t for t in by_ent_e2e.get((r.account_id[:4], m.group(1)), []) if _amount_ok(t, amt, r)]
        if len(cands) == 1:
            t = cands[0]
            if all(it for it, _, _ in t.docs) and len({it for it, _, _ in t.docs}) == len(t.docs):
                for it, a, _ in t.docs:
                    rows.append((r.bank_txn_id, it, a, "RETURNED", 0.9, "pr_return"))
    deb = lines[lines.amount.astype(float) < 0]
    for r in deb.itertuples(index=False):
        amt = -float(r.amount)
        ent = r.account_id[:4]
        # 1. exact e2e (camt)
        if r.end_to_end_id:
            t = by_acc_e2e.get((r.account_id, r.end_to_end_id))
            cands = [t] if t is not None else by_ent_e2e.get((ent, r.end_to_end_id), [])
            cands = [t for t in cands if t.key not in used and _amount_ok(t, amt, r)]
            if len(cands) == 1:
                emit(r.bank_txn_id, cands, "pr_e2e", 0.99, r.currency)
                continue
        blob = " ".join([r.end_to_end_id, r.ref1, r.ref2, r.text.replace(" | ", ""), r.remittance_info])
        m = RUN_RE.search(blob)
        if not m:
            continue
        run_id = m.group(1)
        R = [t for t in runs.get((ent, run_id), []) if t.key not in used]
        rtype = R[0].run_type if R else ""
        # 2. aggregated run: one debit for the whole file
        if rtype == "aggregated":
            nm = re.search(r"\(?(\d+)\s*(?:ORD|TRANSFER|VIREMENT|OPERA|ÜBERWEIS|UEBERWEIS|BONIFIC|PAGAMENT|TRF)", blob.upper())
            n = int(nm.group(1)) if nm else None
            tot = sum(_in_ccy(t.amt, t.ccy, r.currency, r.booking_date) for t in R)
            if abs(tot - amt) <= 0.01 or (n == len(R) and abs(tot - amt) <= 0.01 * amt):
                emit(r.bank_txn_id, R, "pr_agg", 0.97, r.currency)
            else:
                sub = _drop_one(R, amt)
                if sub is not None:
                    emit(r.bank_txn_id, sub, "pr_agg_sub", 0.85, r.currency)
            continue
        # 3. confirming: one debit per maturity date (= due date of the transfer's items)
        if rtype == "confirming":
            nm = re.search(r"(\d+)\s+PROVEEDOR", blob)
            n = int(nm.group(1)) if nm else None
            cand = [t for t in R if max((item_due.get(it, "") for it, _, _ in t.docs), default="") == r.booking_date]
            if cand and abs(sum(t.amt for t in cand) - amt) <= 0.01 and (n is None or n == len(cand)):
                emit(r.bank_txn_id, cand, "pr_conf_due", 0.95, r.currency)
                continue
            sub = _subset_n(R, amt, n)
            if sub is not None:
                emit(r.bank_txn_id, sub, "pr_conf_sub", 0.8, r.currency)
            continue
        # 4. transfer with truncated e2e (N43 / MT940): prefix + account + amount
        pref = m.group(0)[:16]
        hits = [t for t in by_prefix.get((ent, pref), [])
                if t.key not in used and t.account == r.account_id and _amount_ok(t, amt, r)]
        if len(hits) == 1:
            emit(r.bank_txn_id, hits, "pr_prefix", 0.95, r.currency)
        elif len(hits) > 1:
            txt = r.text.upper()
            h2 = [t for t in hits if t.beneficiary.upper()[:20] in txt]
            if len(h2) == 1:
                emit(r.bank_txn_id, h2, "pr_prefix_name", 0.9, r.currency)
    return rows


def _in_ccy(a, frm, to, date):
    if frm == to:
        return a
    try:
        from integration.lib.fx import convert
        return convert(a, frm, to, date)
    except Exception:
        return float("nan")


def _amount_ok(t, amt, r):
    if t.ccy == r.currency:
        return abs(t.amt - amt) <= 0.01
    if r.orig_amount and r.orig_currency == t.ccy:
        return abs(float(r.orig_amount) - t.amt) <= 0.01
    m = OCMT_RE.search(r.text)
    if m and m.group(1) == t.ccy:
        return abs(parse_amount(m.group(2)) - t.amt) <= 0.01
    return False


def _drop_one(R, amt):
    """Aggregated run whose debit misses exactly one transfer (rejected): all but the unique transfer = the gap."""
    diff = round(sum(t.amt for t in R) - amt, 2)
    hits = [i for i, t in enumerate(R) if abs(t.amt - diff) <= 0.01]
    if len(hits) == 1:
        return [t for i, t in enumerate(R) if i != hits[0]]
    return None


def _subset_n(R, amt, n):
    """Confirming fallback: n (1 or 2) transfers of the run summing to amt, unique."""
    if n == 1:
        hits = [t for t in R if abs(t.amt - amt) <= 0.01]
        return hits if len(hits) == 1 else None
    if n == 2 and len(R) < 3000:
        pos = defaultdict(list)
        for i, t in enumerate(R):
            pos[round(t.amt, 2)].append(i)
        sols = set()
        for i, t in enumerate(R):
            for j in pos.get(round(amt - t.amt, 2), []):
                if j > i:
                    sols.add((i, j))
        if len(sols) == 1:
            i, j = sols.pop()
            return [R[i], R[j]]
    return None
