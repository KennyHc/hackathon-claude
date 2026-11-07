"""PDF-clause cases (POLICIES §3, §5.7–5.9): instalments, guarantee retentions (+ release), advances to deduct,
cancelled-and-replaced invoices, rappel credit notes paid out by transfer.

Clauses come from matching/pdf_clauses.py (matching/out/<phase>/doc_clauses.parquet, rebuilt when missing).

Bank lines are walked in booking order, keeping a running open amount per clause item (lines claimed by an earlier
matcher still update it, so instalment / retention state stays right):
  1. candidate items = clause items referenced in the text (exact document number, or a suffix / truncated prefix of
     it, or a cancelled invoice X -> its replacement Y), else clause items of the same entity + currency whose expected
     amount equals the cash, confirmed by the payer's name or IBAN (and no other open item of that payer has that amount);
  2. the cash must fit the clause exactly (to the cent):
       INSTALMENTS     cash = an unpaid instalment of the plan (or the remaining open amount). Which instalment:
                       text "k/n" if present, else the only unpaid instalment with that amount, else (identical amounts)
                       the one whose due date (invoice date + days) passed most recently. "" if it is the LAST
                       instalment of the plan, PARTIAL otherwise; amount = cash.
       RETENTION       open = total and cash = total − retention -> PARTIAL; open = cash = retention -> "" (release)
       ADVANCE_DEDUCT  cash = net amount after the advance -> PARTIAL (amount = cash); not a payment of the proforma
       REPLACES        a payment citing X settles the replacement(s): cash = sum of open amounts -> ""
       RAPPEL_PAYOUT   outgoing transfer = open amount of a recent ARC -> ""
  3. claim the line only when exactly one reading fits.
"""
from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from itertools import combinations

import pandas as pd

from ingestion.text import norm_name
from matching import pdf_clauses
from matching.common import Ctx, alloc_frame

try:  # rules library (POLICIES §2-3) for replaced-invoice lines whose cash is not the plain sum
    from classification.exceptions import explain as _explain
except Exception:  # noqa: BLE001
    _explain = None

NAME = "pdf_cases"
EPS = 0.011            # cash vs an amount written in the PDF: exact to the cent (rounding of the last cent)
FRAC = re.compile(r"(?<![\d/])(\d{1,2})\s*/\s*(\d{1,2})(?![\d/])")
RAPPEL_MAX_DAYS = 150  # a rappel credit note is paid out within a few weeks of its issue
FX_REL = 0.02          # cross-currency: bank corporate rate vs ECB reference (spread ≤ 1.5 %, POLICIES §3)
ORIG_REL = 0.002       # cross-currency with the original (item-currency) amount given in the bank line


class _FX:
    def __init__(self, rates: pd.DataFrame):
        self.t: dict[str, tuple[list[str], list[float]]] = {}
        r = rates.sort_values("date")
        for c, g in r.groupby("currency"):
            self.t[c] = (list(g.date), [float(x) for x in g.units_per_eur])

    def per_eur(self, ccy: str, d: str) -> float | None:
        if ccy == "EUR":
            return 1.0
        if ccy not in self.t:
            return None
        ds, vs = self.t[ccy]
        i = bisect_right(ds, d) - 1
        return vs[max(i, 0)]

    def convert(self, amount: float, src: str, dst: str, d: str) -> float | None:
        a, b = self.per_eur(src, d), self.per_eur(dst, d)
        return None if a is None or b is None else amount / a * b


def _close(a, b, eps: float = EPS) -> bool:
    return a is not None and b is not None and a == a and b == b and abs(a - b) <= eps


class _Names:
    """party_id -> name keys / IBANs, to confirm an amount-only candidate."""

    def __init__(self, ctx: Ctx, clauses: pd.DataFrame):
        self.keys: dict[str, set[str]] = defaultdict(set)
        self.ibans: dict[str, set[str]] = defaultdict(set)
        for pid, n in zip(ctx.items.party_id, ctx.items.party_name_norm):
            self._add(pid, n)
        for mname, idc in (("customers", "customer_id"), ("suppliers", "supplier_id")):
            m = ctx.masters.get(mname)
            if m is None or m.empty or idc not in m.columns:
                continue
            for col in ("name", "trade_name"):
                if col in m.columns:
                    for pid, n in zip(m[idc], m[col]):
                        if n:
                            self._add(pid, norm_name(n))
            if "iban" in m.columns:
                for pid, ib in zip(m[idc], m.iban):
                    if ib:
                        self.ibans[pid].add(ib.replace(" ", ""))
        nib = clauses[clauses.clause_type == "NEW_IBAN"]
        for pid, a, b in zip(nib.party_id, nib.iban, nib.old_iban):
            for ib in (a, b):
                if ib:
                    self.ibans[pid].add(ib)

        # competitors: parties of the same entity whose name starts with the same 6 characters
        self.prefix6: dict[tuple[str, str], set[str]] = defaultdict(set)
        ent_of = dict(zip(ctx.items.party_id, ctx.items.entity))
        for pid, ns in self.keys.items():
            if pid in ent_of:
                for n in ns:
                    self.prefix6[(ent_of[pid], n[:6])].add(pid)

    def _add(self, pid: str, n: str):
        if n and len(n) >= 3:
            self.keys[pid].add(n)

    def score(self, pid: str, cp_norm: str, text_norm: str) -> int:
        """How well the bank names match the party: counterparty name equal (100+) or starting like the party's name
        (matched length, ≥ 8, truncation allowed); text containing the name (100+ if whole, else its first 16 chars)."""
        best = 0
        for n in self.keys.get(pid, ()):
            if cp_norm:
                if cp_norm == n:
                    best = max(best, 100 + len(n))
                else:
                    L = min(len(n), len(cp_norm), 16)
                    if L >= 8 and cp_norm[:L] == n[:L]:
                        best = max(best, L)
            if f" {n} " in text_norm:
                best = max(best, 100 + len(n))
            elif len(n) > 16 and f" {n[:16]}" in text_norm:
                best = max(best, 16)
        return best

    def match(self, pid: str, cp_norm: str, text_norm: str, iban: str) -> bool:
        if iban and iban in self.ibans.get(pid, ()):
            return True
        return self.score(pid, cp_norm, text_norm) > 0

    def rivals(self, pid: str, ent: str, cp_norm: str, text_norm: str, iban: str) -> set[str] | None:
        """None if the bank names / IBAN do not point to the party; else the other parties of the entity that match
        the bank names at least as well (same or truncated names: several customers share a name)."""
        if iban and iban in self.ibans.get(pid, ()):
            return set()
        sc = self.score(pid, cp_norm, text_norm)
        if sc <= 0:
            return None
        cand = set().union(*[self.prefix6.get((ent, n[:6]), set()) for n in self.keys.get(pid, ())]) - {pid}
        return {q for q in cand if self.score(q, cp_norm, text_norm) >= sc}


def run(ctx: Ctx):
    cl = pdf_clauses.load_or_build(ctx)
    items = ctx.items_by_id
    cl = cl[(cl.item_id != "") & cl.item_id.isin(items.index)]
    by_type: dict[str, dict[str, object]] = defaultdict(dict)
    for r in cl.itertuples(index=False):
        if r.clause_type in ("INSTALMENTS", "RETENTION", "ADVANCE_DEDUCT", "RAPPEL_PAYOUT", "REPLACES"):
            by_type[r.clause_type].setdefault(r.item_id, r)
    inst, ret, adv = by_type["INSTALMENTS"], by_type["RETENTION"], by_type["ADVANCE_DEDUCT"]
    rappel, repl = by_type["RAPPEL_PAYOUT"], by_type["REPLACES"]
    rappel = {i: r for i, r in rappel.items() if items.at[i, "item_type"] == "ARC"}
    pay_items = set(inst) | set(ret) | set(adv)          # incoming payments with a clause-specific amount
    clause_items = pay_items | set(rappel) | set(repl)
    names = _Names(ctx, cl)
    fx = _FX(ctx.masters["fx_rates"])

    # cancelled invoice X (docnorm / alnum, entity) -> replacement item Y
    repl_map: dict[tuple[str, str], str] = {}
    for y, r in repl.items():
        x = r.ref_document
        for k in {re.sub(r"\D", "", x).lstrip("0"), re.sub(r"[^A-Z0-9]", "", x.upper())}:
            if len(k) >= 5:
                repl_map[(r.entity, k)] = y
    cancelled = {r.ref_item_id for r in repl.values() if r.ref_item_id}

    # expected amounts (amount-only candidates), suffix / prefix keys of the document numbers (noisy references)
    amt_lists: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
    affix: dict[tuple[str, str], set[str]] = defaultdict(set)
    full_dn: dict[tuple[str, str], set[str]] = defaultdict(set)
    for iid in pay_items | set(rappel):
        it = items.loc[iid]
        exp = []
        if iid in inst:
            exp += [float(a) for a in inst[iid].amounts]
        if iid in ret:
            exp += [ret[iid].net_amount, ret[iid].amount]
        if iid in adv:
            exp += [adv[iid].net_amount]
        if iid in rappel:
            exp += [it.open_amount]
        for a in exp:
            if a is not None and a == a:
                amt_lists[(it.entity, it.currency)].append((float(a), iid))
        dn = it.document_number_norm
        if len(dn) >= 8:
            full_dn[(it.entity, dn)].add(iid)
        for j in range(1, len(dn) - 4):
            affix[(it.entity, dn[j:])].add(iid)
        for j in range(7, len(dn)):
            affix[(it.entity, dn[:j])].add(iid)
    amt_index = {k: (sorted(v), [a for a, _ in sorted(v)]) for k, v in amt_lists.items()}
    entity_ccys: dict[str, set[str]] = defaultdict(set)
    for ent, ccy in amt_index:
        entity_ccys[ent].add(ccy)

    def amount_cands(ent: str, ccy: str, value: float, tol: float) -> set[str]:
        if (ent, ccy) not in amt_index:
            return set()
        lst, keys = amt_index[(ent, ccy)]
        i, j = bisect_left(keys, value - tol), bisect_right(keys, value + tol)
        return {iid for _, iid in lst[i:j]}

    def cash_in(t, ccy: str) -> tuple[float | None, float]:
        """The line's cash expressed in currency `ccy`, and the matching tolerance."""
        cash = round(float(t.amount_abs), 2)
        if ccy == t.currency:
            return cash, EPS
        if t.orig_currency == ccy and t.orig_amount == t.orig_amount and t.orig_amount:
            v = abs(float(t.orig_amount))
            return v, max(EPS, ORIG_REL * v)
        v = fx.convert(cash, t.currency, ccy, t.booking_date)
        return v, (max(EPS, FX_REL * v) if v is not None else 0.0)

    open_run: dict[str, float] = {}

    def open_of(iid: str) -> float:
        return open_run.get(iid, float(items.at[iid, "open_amount"]))

    def ref_items(t, side: str) -> tuple[set[str], set[str], set[str]]:
        """-> (exact items, fuzzy clause items, replacement items) referenced by the line."""
        exact, fuzzy, rep = set(), set(), set()
        for tok in t.doc_tokens:
            exact.update(ctx.by_alnum.get(tok, ()))
            y = repl_map.get((t.entity, tok))
            if y:
                rep.add(y)
        for d in t.doc_refs:
            exact.update(ctx.by_docnorm.get(d, ()))
            y = repl_map.get((t.entity, d))
            if y:
                rep.add(y)
            if len(d) >= 5:
                fuzzy |= affix.get((t.entity, d), set())
            for j in range(8, len(d)):          # number glued to the next token: FA250209976/EREF/PAY-2025-...
                fuzzy |= full_dn.get((t.entity, d[:j]), set())
        keep = lambda s: {i for i in s if i in items.index and items.at[i, "side"] == side  # noqa: E731
                          and items.at[i, "entity"] == t.entity}
        return keep(exact), keep(fuzzy) - exact, keep(rep)

    def k_of_n(text: str, n: int):
        for m in FRAC.finditer(text):
            k, nn = int(m.group(1)), int(m.group(2))
            if nn == n and 1 <= k <= n:
                return k - 1
        return None

    inst_unpaid: dict[str, set[int] | None] = {}

    def unpaid_of(iid: str) -> set[int] | None:
        """Instalments (0-based) still unpaid at the phase start: the subset of the plan whose sum = open amount
        (closest to the cent; ties -> the latest instalments)."""
        if iid not in inst_unpaid:
            plan = [float(a) for a in inst[iid].amounts]
            o = float(items.at[iid, "open_amount"])
            best = None
            if len(plan) <= 12:
                for m in range(1, len(plan) + 1):
                    for sub in combinations(range(len(plan)), m):
                        e = round(abs(sum(plan[i] for i in sub) - o), 2)
                        if e <= EPS * m and (best is None or (-e, sum(sub)) > best[0]):
                            best = ((-e, sum(sub)), set(sub))
            inst_unpaid[iid] = best[1] if best else None
        return inst_unpaid[iid]

    def evaluate(iid: str, t, text: str, out: bool = False, via_rep: bool = False):
        """-> (exception, amount, tag, instalment index) if the cash fits the item's clause, else None."""
        it = items.loc[iid]
        o = open_of(iid)
        if o <= 0.005 or t.booking_date < it.issue_date:
            return None
        cash, tol = cash_in(t, it.currency)
        if cash is None:
            return None
        is_fx = it.currency != t.currency
        close = lambda a, b: a is not None and b is not None and a == a and b == b and abs(a - b) <= tol  # noqa: E731
        full = "FX_DIFFERENCE" if is_fx else ""          # exception of a row that clears the item in full
        if out:
            if iid in rappel and not is_fx and close(cash, o) and \
                    (pd.Timestamp(t.booking_date) - pd.Timestamp(it.issue_date)).days <= RAPPEL_MAX_DAYS:
                return ("", round(o, 2), "rappel", None)
            return None
        if iid in inst:
            plan = [float(a) for a in inst[iid].amounts]
            n = len(plan)
            if cash > o + tol:
                return None
            unpaid = unpaid_of(iid)
            k = k_of_n(text, n)
            cand = []
            if unpaid is not None:
                cand = [i for i in unpaid if abs(cash - plan[i]) <= min(tol, 0.005)] or \
                       [i for i in unpaid if close(cash, plan[i])]
                if k is None and cand:
                    if len(cand) == 1:
                        k = cand[0]
                    else:   # identical amounts: the instalment whose due date passed most recently
                        iss = pd.Timestamp(it.issue_date)
                        days = [int(d) for d in inst[iid].instalment_days]
                        if len(days) == n:
                            due = [iss + pd.Timedelta(days=d) for d in days]
                            past = [i for i in cand if due[i] <= pd.Timestamp(t.booking_date)]
                            k = max(past) if past else min(cand)
                if k is None and not cand and not close(cash, o):
                    return None
            elif not any(close(cash, a) for a in plan) and not close(cash, o):
                return None
            last = (k == n - 1) if k is not None else close(cash, o)
            if not is_fx:
                return ("" if last else "PARTIAL", cash, "inst", k)
            if last:
                return (full, plan[k] if k is not None and plan[k] <= o + EPS else round(o, 2), "inst", k)
            if k is None:
                return None
            return ("PARTIAL", plan[k], "inst", k)
        if iid in ret:
            r = ret[iid]
            if close(cash, r.net_amount) and _close(o, it.total_amount):
                return ("PARTIAL", r.net_amount if is_fx else cash, "ret", None)
            if close(cash, r.amount) and _close(o, r.amount):
                return (full, round(o, 2) if is_fx else cash, "rel", None)
            return None
        if iid in adv:
            a = adv[iid]
            pf = re.sub(r"[^A-Z0-9]", "", a.ref_document.upper())
            if "PROFORMA" in text.upper() or (pf and pf in t.doc_tokens):
                return None   # the proforma itself being paid (CUSTOMER_ADVANCE)
            if close(cash, a.net_amount) and o >= a.net_amount - EPS:
                return ("PARTIAL", a.net_amount if is_fx else cash, "adv", None)
            return None
        if iid in repl and via_rep and not is_fx and close(cash, o):
            return ("", round(o, 2), "repl", None)
        return None

    def commit(iid: str, res):
        exc, amt, tag, k = res
        open_run[iid] = open_of(iid) - amt
        if k is not None and inst_unpaid.get(iid) is not None:
            inst_unpaid[iid].discard(k)

    def other_same_amount(pid: str, iid: str, value: float, tol: float) -> bool:
        po = ctx.by_party.get(pid)
        if po is None:
            return False
        same = ((po.open_amount - value).abs() <= tol) & (po.item_id != iid) & ~po.item_id.isin(clause_items)
        return bool(same.any())

    def replaced_line(t, exact: set[str], rep: set[str]):
        """-> [(item_id, amount, exception)] for a line citing cancelled invoice(s), or None."""
        sel = set(rep) | {i for i in exact if i not in cancelled}
        pids = {items.at[i, "party_id"] for i in rep}
        sel = {i for i in sel if items.at[i, "party_id"] in pids}
        for d in t.doc_refs:   # short / truncated references (RMR*IV*506502, "0107207 0111917") of the same payer
            if len(d) < 4 or (t.entity, d) in repl_map:
                continue
            for pid in pids:
                po = ctx.by_party.get(pid)
                if po is None:
                    continue
                m = po[(po.document_number_norm == d) | po.document_number_norm.str.endswith(d)]
                m = m[(m.side == "AR") & (m.currency == t.currency) & ~m.item_id.isin(cancelled)]
                if len(m) == 1:
                    sel.add(m.item_id.iloc[0])
        sel = {i for i in sel if items.at[i, "currency"] == t.currency and items.at[i, "item_type"] in ("ARI", "ARC")
               and t.booking_date >= items.at[i, "issue_date"] and open_of(i) > 0.005}
        if not sel or not any(i in rep for i in sel):
            return None
        ids = sorted(sel)
        cash = round(float(t.amount_abs), 2)
        sign = {i: (-1 if items.at[i, "item_type"] == "ARC" else 1) for i in ids}
        tot = sum(sign[i] * open_of(i) for i in ids)
        if _close(tot, cash):
            return [(i, round(open_of(i), 2), "") for i in ids]
        if _explain is None or len(ids) != 1:
            return None   # several documents: only an exact sum is trusted
        its = items.loc[ids].reset_index(drop=True).copy()
        its["open_amount"] = [round(open_of(i), 2) for i in ids]
        try:
            ex = _explain(its, cash, t._asdict(), ctx.masters)
        except Exception:  # noqa: BLE001 - shared rules library in flux: never break the matcher
            return None
        if not ex or len(ex) != len(ids) or any(e == "OVERPAYMENT" for _, _, e in ex):
            return None
        return [(i, float(a), e or "") for i, a, e in ex]

    def party_has_ccy(pid: str, ccy: str) -> bool:
        po = ctx.by_party.get(pid)
        return po is not None and bool((po.currency == ccy).any())

    bank = ctx.bank.sort_values(["booking_date", "bank_txn_id"], kind="stable")
    rows = []
    for t in bank.itertuples(index=False):
        if t.direction not in ("in", "out"):
            continue
        cash = round(float(t.amount_abs), 2)
        text = f"{t.text} {t.remittance_info or ''}"

        if t.direction == "out":
            # rappel credit note paid out: our outgoing transfer settles the ARC
            cands = amount_cands(t.entity, t.currency, cash, EPS) & set(rappel)
            if not cands:
                continue
            exact, fuzzy, _ = ref_items(t, "AR")
            hit, via = [i for i in cands if i in exact or i in fuzzy], "ref"
            if not hit:
                cp, tx = norm_name(t.counterparty_name or ""), f" {norm_name(t.text or '')} "
                ib = (t.counterparty_account or "").replace(" ", "")
                hit, via = [i for i in cands if names.rivals(items.at[i, "party_id"], t.entity, cp, tx, ib) == set()], "name"
            res = [(i, r) for i in hit for r in [evaluate(i, t, text, out=True)] if r]
            if len(res) == 1:
                i, r = res[0]
                commit(i, r)
                if t.bank_txn_id not in ctx.claimed:
                    rows.append((t.bank_txn_id, i, r[1], r[0], 0.95, f"{NAME}:{r[2]}_{via}"))
            continue

        exact, fuzzy, rep = ref_items(t, "AR")
        # a referenced non-clause item that the cash pays exactly: not ours
        if any(i not in clause_items and i not in cancelled and _close(open_of(i), cash) for i in exact):
            continue

        # (a) replaced invoice(s): a payment citing a cancelled invoice X settles its replacement Y, plus the other
        #     documents of the same payer it cites (credit notes netted); the cash must be explained by them
        if rep:
            r = replaced_line(t, exact, rep)
            if r:
                for i, amt, exc in r:
                    commit(i, (exc, amt, "repl", None))
                    if t.bank_txn_id not in ctx.claimed:
                        rows.append((t.bank_txn_id, i, round(amt, 2), exc, 0.95, f"{NAME}:repl"))
                continue

        # (b) a clause item referenced in the text
        res = []
        for i in (exact & pay_items) | (fuzzy & pay_items):
            if i in fuzzy and items.at[i, "currency"] != t.currency:
                continue   # a noisy reference plus an FX-tolerant amount is too weak
            r = evaluate(i, t, text)
            if r:
                res.append((i, r, "ref" if i in exact else "fuzzy"))
        if not res:
            # (c) amount-only: clause items whose expected amount = cash, confirmed by payer name / IBAN, and the line
            #     cites no other document of that payer
            ref_parties = {items.at[i, "party_id"] for i in exact | fuzzy}
            cp, tx = norm_name(t.counterparty_name or ""), f" {norm_name(t.text or '')} "
            ib = (t.counterparty_account or "").replace(" ", "")
            if any(names.match(q, cp, tx, ib) for q in ref_parties):
                ref_parties.add("*")   # the payer cites its own other documents: no amount-only guess
            cands = set()
            for ccy in entity_ccys.get(t.entity, ()):
                v, tol = cash_in(t, ccy)
                if v is not None:
                    cands |= {(i, v, tol) for i in amount_cands(t.entity, ccy, v, tol)}
            cands = {c for c in cands if c[0] in pay_items}
            if cands and "*" not in ref_parties:
                for i, v, tol in cands:
                    pid = items.at[i, "party_id"]
                    if pid in ref_parties or other_same_amount(pid, i, v, tol):
                        continue
                    riv = names.rivals(pid, t.entity, cp, tx, ib)
                    if riv is None:
                        continue
                    if items.at[i, "currency"] != t.currency:
                        if riv or party_has_ccy(pid, t.currency):
                            continue   # cross-currency guess: only for an unambiguous payer without invoices in that currency
                    elif any(other_same_amount(q, "", v, tol) for q in riv):
                        continue       # a namesake customer has an open item of that amount
                    r = evaluate(i, t, text)
                    if r:
                        res.append((i, r, "name"))
        if len({i for i, _, _ in res}) != 1:
            continue
        i, r, via = res[0]
        commit(i, r)
        if t.bank_txn_id not in ctx.claimed:
            rows.append((t.bank_txn_id, i, r[1], r[0], 0.95 if via != "name" else 0.9,
                         f"{NAME}:{r[2]}{'_fx' if items.at[i, 'currency'] != t.currency else ''}_{via}"))
    return alloc_frame(rows)
