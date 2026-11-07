"""Remittance advices (#18): a customer pays dozens/hundreds of documents with one transfer and sends an advice.

1. Parse every advice (matching/remittance_parse.py, cached per phase).
2. Resolve each advice line to one of our AR items: our number (alnum / digits, also without prefix), the last 6
   digits in the 2025-09 "Voucher" XLSX layout, else amount + date inside the payer's customer group.
3. Link a bank receipt to an advice by the payment reference found in the bank fields (+ amount check), else by
   entity + amount == advice total around the payment date (the advice may arrive a few days after the cash).
4. Rows (POLICIES §2/§3): credit notes full amount, no exception; deduction with a dispute reason -> SHORT_DISPUTE
   (amount = paid); DISCOUNT -> SHORT_DISCOUNT within terms (full amount) else PARTIAL (paid); RET -> SHORT_WITHHOLDING;
   bank currency != item currency -> FX_DIFFERENCE on fully cleared invoice rows.
5. Separate pass: "SETTLEMENT <doc>" / "REGULARIZACION <doc>" pays the disputed remainder (no exception).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from matching import remittance_parse as rp
from matching.common import Ctx, alloc_frame

NAME = "remittance"

DISCOUNT_REASONS = {"DISCOUNT", "DTO", "DTO. PP", "DTO.PP", "SKONTO", "ESCOMPTE", "PRONTO PAGO"}
WITHHOLDING_REASONS = {"RET", "RETENCION", "RETENCIÓN", "IRRF", "WHT", "RAS"}
WHT_RATES = (0.0125, 0.015, 0.10)
WHT_ENTITIES = {"MX01", "BR01", "MA01"}
REF_FIELDS = ["text", "remittance_info", "ref1", "ref2", "end_to_end_id"]
SETTLE_RE = re.compile(r"\b(SETTLEMENT|REGULARIZACION|REGULARISATION|REGULARIZACAO)\b")


def _alnum(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


def _norm(s) -> str:
    return re.sub(r"\D", "", str(s or "")).lstrip("0")


def _tol(x: float) -> float:
    return max(0.01, 0.005 * abs(x))


class Fx:
    """EUR reference rates (units per EUR), last available on/before a date."""

    def __init__(self, rates: pd.DataFrame):
        r = rates.copy()
        r["date"] = pd.to_datetime(r.date)
        self.by = {c: g.sort_values("date").set_index("date").units_per_eur for c, g in r.groupby("currency")}

    def rate(self, ccy: str, d) -> float | None:
        if ccy == "EUR":
            return 1.0
        s = self.by.get(ccy)
        if s is None:
            return None
        i = s.index.searchsorted(pd.Timestamp(d), side="right") - 1
        return float(s.iloc[max(i, 0)])

    def conv(self, amt: float, frm: str, to: str, d) -> float | None:
        a, b = self.rate(frm, d), self.rate(to, d)
        if a is None or b is None:
            return None
        return amt / a * b


# ---------------------------------------------------------------------------------------------------------------------
def _item_indexes(ctx: Ctx, entities: set[str]):
    it = ctx.items[(ctx.items.side == "AR") & ctx.items.entity.isin(entities)].copy()
    cust = ctx.masters["customers"]
    gid = cust.group_id.fillna("").astype(str)
    grp = dict(zip(cust.customer_id, np.where(gid != "", gid, cust.customer_id)))
    it["grp"] = it.party_id.map(grp).fillna(it.party_id)
    it["last6"] = it.document_number_norm.str[-6:]
    by_alnum, by_norm, by_grp = defaultdict(list), defaultdict(list), defaultdict(list)
    recs = it.to_dict("records")
    for r in recs:
        by_alnum[(r["entity"], r["document_alnum"])].append(r)
        if r["document_number_norm"]:
            by_norm[(r["entity"], r["document_number_norm"])].append(r)
            by_norm[(r["entity"], "#" + r["last6"])].append(r)   # 2025-09 Voucher layout: last 6 digits only
        by_grp[(r["entity"], r["grp"])].append(r)
    # customer name -> groups (to know the payer even when no document resolves)
    cn = defaultdict(set)
    for c in cust.itertuples(index=False):
        g = grp.get(c.customer_id, c.customer_id)
        for n in (c.name, c.trade_name):
            if isinstance(n, str) and n:
                cn[(c.entity, n.strip().upper())].add(g)
    return by_alnum, by_norm, by_grp, cn


def _pick(cands: list[dict], gross: float, grp: str | None) -> dict | None:
    if not cands:
        return None
    want = "ARC" if gross < 0 else "ARI"
    c = [r for r in cands if r["item_type"] == want] or cands
    if grp is not None:
        c2 = [r for r in c if r["grp"] == grp]
        if c2:
            c = c2
    if len(c) > 1:
        c2 = [r for r in c if abs(r["total_amount"] - abs(gross)) <= 0.011]
        if c2:
            c = c2
    if len(c) > 1:
        c2 = [r for r in c if r["open_amount"] > 0.005]
        if c2:
            c = c2
    return c[0] if len(c) == 1 else None


def resolve_advice(lines: pd.DataFrame, idx, payer: str) -> tuple[list[dict | None], str | None]:
    by_alnum, by_norm, by_grp, cn = idx
    ent = lines.entity.iloc[0]
    cands = []
    for l in lines.itertuples(index=False):
        a, n = _alnum(l.doc), _norm(l.doc)
        c = by_alnum.get((ent, a), [])
        if not c and n:
            c = by_norm.get((ent, n), [])
        if not c and re.fullmatch(r"\d{6}", str(l.doc).strip()):
            gross = abs(float(l.gross)) if l.gross == l.gross else -1
            c = [r for r in by_norm.get((ent, "#" + str(l.doc).strip()), []) if abs(r["total_amount"] - gross) <= 0.011]
        cands.append(c)
    # payer group = majority of uniquely resolved documents, else from the customer master by name
    votes = Counter(c[0]["grp"] for c in cands if len({r["grp"] for r in c}) == 1 and c)
    grp = votes.most_common(1)[0][0] if votes else None
    if grp is None:
        gs = cn.get((ent, str(payer).strip().upper()), set())
        grp = next(iter(gs)) if len(gs) == 1 else None
    out = []
    used = set()
    pool = by_grp.get((ent, grp), []) if grp else []
    for l, c in zip(lines.itertuples(index=False), cands):
        gross = float(l.gross) if l.gross == l.gross else 0.0
        r = _pick(c, gross, grp)
        if r is None and grp:
            d = str(l.doc).strip()
            want = "ARC" if gross < 0 else "ARI"
            if re.fullmatch(r"\d{4,8}", d):   # our number without prefix / truncated (Voucher layout: last 6 digits)
                k = [x for x in pool if x["item_type"] == want and x["document_number_norm"].endswith(d.lstrip("0") or d)]
                if len(k) > 1:
                    k = [x for x in k if abs(x["total_amount"] - abs(gross)) <= 0.011] or k
                r = k[0] if len(k) == 1 else None
            if r is None:                     # customer's own voucher number: amount (+ date) inside the group
                k = [x for x in pool if x["item_type"] == want and abs(x["total_amount"] - abs(gross)) <= 0.005
                     and x["item_id"] not in used]
                if len(k) > 1 and l.doc_date:
                    k = [x for x in k if x["issue_date"] == l.doc_date] or k
                r = k[0] if len(k) == 1 else None
        if r is not None:
            if r["item_id"] in used:
                r = None
            else:
                used.add(r["item_id"])
        out.append(r)
    return out, grp


# ---------------------------------------------------------------------------------------------------------------------
def _bank_tokens(t) -> tuple[set[str], str]:
    u = " ".join(str(getattr(t, f) or "") for f in REF_FIELDS).upper()
    toks = set(re.findall(r"[A-Z0-9]+", u))
    toks |= {_alnum(m) for m in re.findall(r"(?:REM/\d{4}/\d+|PAY-\d{4}-\d+|[A-Z]{2,4}[-/]\d[\d\-/]+)", u)}
    return {x for x in toks if len(x) >= 6}, u


def _terms(ctx: Ctx) -> dict[str, tuple[float, int]]:
    pt = ctx.masters["payment_terms"]
    out = {}
    for r in pt.itertuples(index=False):
        pct = pd.to_numeric(r.early_payment_discount_pct, errors="coerce")
        days = pd.to_numeric(r.discount_days, errors="coerce")
        if pct == pct and pct and days == days:
            out[r.code] = (float(pct) / 100.0, int(days))
    return out


def run(ctx: Ctx):
    L = rp.load(ctx.phase)
    L = L[L.line_no >= 0].copy()
    if L.empty:
        return alloc_frame([])
    bank = ctx.unclaimed()
    bank = bank[bank.amount > 0]
    idx = _item_indexes(ctx, set(L.entity) | set(bank.entity))
    fx = Fx(ctx.masters["fx_rates"])
    terms = _terms(ctx)

    # ---- advices ----------------------------------------------------------------------------------------------------
    adv = {}
    for f, g in L.groupby("file", sort=False):
        h = g.iloc[0]
        paid = round(float(g.paid.fillna(0).sum()), 2)
        adv[f] = dict(file=f, entity=h.entity, payer=h.payer_name, ref=_alnum(h.payment_ref), ccy=h.currency,
                      paid=paid, stated=h.stated_total if h.stated_total == h.stated_total else None,
                      pdate=h.payment_date or h.received_date, rdate=h.received_date, layout=h.layout, lines=g)
    by_ref = defaultdict(list)
    by_amt = defaultdict(list)
    for a in adv.values():
        if a["ref"]:
            by_ref[a["ref"]].append(a)
        for v in {a["paid"], a["stated"]}:
            if v is not None:
                by_amt[(a["entity"], round(v, 2))].append(a)

    # ---- link bank lines -> advices ---------------------------------------------------------------------------------
    links = []  # (prio, bank row, advice, how)
    for t in bank.itertuples(index=False):
        toks, _ = _bank_tokens(t)
        bd = pd.Timestamp(t.booking_date)
        got = []
        for k in toks:
            for a in by_ref.get(k, ()):
                if a["entity"] == t.entity:
                    got.append(a)
        got = list({a["file"]: a for a in got}.values())
        if len(got) > 1:  # same ref reused: keep the ones whose amount fits best / nearest date
            fits = {a["file"]: _amount_fits(a, t, fx) for a in got}
            best = [a for a in got if fits[a["file"]] in ("exact", "fx")] or \
                [a for a in got if fits[a["file"]] is not None]
            got = best or got
            if len(got) > 1:
                got.sort(key=lambda a: abs((pd.Timestamp(a["pdate"]) - bd).days))
                got = got[:1]
        if got:
            links.append((0, t, got[0], "ref"))
            continue
        cand = [a for a in by_amt.get((t.entity, round(t.amount, 2)), ()) if a["ccy"] in ("", t.currency)]
        cand = [a for a in cand if -20 <= (bd - pd.Timestamp(a["pdate"])).days <= 20
                or -5 <= (pd.Timestamp(a["rdate"]) - bd).days <= 25]
        cand = list({a["file"]: a for a in cand}.values())
        if len(cand) == 1:
            links.append((1, t, cand[0], "amt"))
    links.sort(key=lambda x: x[0])
    used_adv, used_bank = set(), set()
    rows = []
    for prio, t, a, how in links:
        if a["file"] in used_adv or t.bank_txn_id in used_bank:
            continue
        r = _rows_for(t, a, idx, fx, terms, how)
        if r:
            used_adv.add(a["file"])
            used_bank.add(t.bank_txn_id)
            rows += r

    # ---- dispute settlements ----------------------------------------------------------------------------------------
    rows += _settlements(ctx, L, bank, used_bank, idx, fx)
    return alloc_frame(rows)


def _settlements(ctx: Ctx, L: pd.DataFrame, bank: pd.DataFrame, used_bank: set, idx, fx: Fx) -> list[tuple]:
    """'SETTLEMENT <doc>' / 'REGULARIZACION <doc>': the customer pays the remainder it had disputed (no exception)."""
    disputed = {}
    for r in L[L.deduction.fillna(0) < 0].itertuples(index=False):
        disputed[(r.entity, _alnum(r.doc))] = -float(r.deduction)
    by_alnum, by_norm, _, _ = idx
    # invoices of each entity that are partly open, or that had a deduction in an advice (prefix search pool)
    pool = defaultdict(list)
    for (ent, _a), lst in by_alnum.items():
        for r in lst:
            if r["item_type"] == "ARI" and (r["open_amount"] < r["total_amount"] - 0.01
                                            or (ent, r["document_alnum"]) in disputed):
                pool[ent].append(r)
    out = []
    for t in bank.itertuples(index=False):
        if t.bank_txn_id in used_bank:
            continue
        u = " ".join(str(getattr(t, f) or "") for f in REF_FIELDS).upper()
        m = SETTLE_RE.search(u)
        if not m:
            continue
        tok = re.match(r"\s*([A-Z0-9][A-Z0-9\-]*)", u[m.end():])
        tok = _alnum(tok.group(1)) if tok else ""
        orig = float(t.orig_amount) if t.orig_amount == t.orig_amount and t.orig_amount else None

        def target(r):  # remainder expected, in item currency
            if r["currency"] == t.currency:
                return float(t.amount), 0.011
            if orig and t.orig_currency == r["currency"]:
                return orig, 0.011
            c = fx.conv(float(t.amount), t.currency, r["currency"], t.booking_date)
            return (c, 0.03 * c) if c else (None, 0)

        def remainder(r):
            d = disputed.get((t.entity, r["document_alnum"]))
            opts = [x for x in (d, r["open_amount"] if r["open_amount"] < r["total_amount"] - 0.01 else None) if x]
            return opts

        hits = {}
        if tok:
            for r in by_alnum.get((t.entity, tok), []) + by_norm.get((t.entity, _norm(tok)), []):
                if r["item_type"] == "ARI":
                    hits[r["item_id"]] = r
        if not hits:
            for k in t.doc_tokens:
                for r in by_alnum.get((t.entity, k), []):
                    if r["item_type"] == "ARI":
                        hits[r["item_id"]] = r
        if len(hits) > 1:
            continue
        if not hits and len(tok) >= 4:   # truncated number: prefix + remainder amount among partly open invoices
            k = []
            for r in pool.get(t.entity, ()):
                if not r["document_alnum"].startswith(tok):
                    continue
                tg, tl = target(r)
                if tg and any(abs(x - tg) <= tl for x in remainder(r)):
                    k.append(r)
            if len(k) != 1:
                continue
            hits = {k[0]["item_id"]: k[0]}
        if len(hits) != 1:
            continue
        r = next(iter(hits.values()))
        tg, tl = target(r)
        if tg is None or tg > r["total_amount"] + 0.01:
            continue
        if r["currency"] == t.currency:
            amt, exc = float(t.amount), ""
        else:
            rem = [x for x in remainder(r) if abs(x - tg) <= max(tl, 0.03 * tg)]
            amt, exc = (rem[0] if rem else round(tg, 2)), "FX_DIFFERENCE"
        out.append((t.bank_txn_id, r["item_id"], round(amt, 2), exc, 0.85, "remittance_settle"))
    return out


def _amount_fits(a: dict, t, fx: Fx):
    """None if the advice cannot be this bank line; else 'exact' / 'short' (cash below the listed total) / 'fx'."""
    amt = float(t.amount)
    if a["ccy"] in ("", t.currency):
        if abs(a["paid"] - amt) <= 0.011:
            return "exact"
        if amt < a["paid"]:
            return "short"
        return None
    # the bank's own original amount, when it is in the advice's currency: exact and independent of the rates table
    # (fx_rates.csv may stop months before the bank lines, e.g. FINAL 2026-H1 vs rates to 2025-12-31)
    oc, oa = getattr(t, "orig_currency", None) or "", getattr(t, "orig_amount", None)
    if oc == a["ccy"] and oa is not None and oa == oa and abs(abs(float(oa)) - a["paid"]) <= 0.011:
        return "fx"
    conv = fx.conv(a["paid"], a["ccy"], t.currency, t.booking_date)
    if conv and abs(conv - amt) <= 0.03 * amt:
        return "fx"
    return None


def _late(it: dict, terms, bd: pd.Timestamp) -> int | None:
    tr = terms.get(it["payment_terms"])
    if not tr:
        return None
    return (bd - pd.Timestamp(it["issue_date"])).days - tr[1]


def _disc_amount(it: dict, terms) -> float | None:
    tr = terms.get(it["payment_terms"])
    return round(it["total_amount"] * tr[0], 2) if tr else None


# Golden behaviour (TRAIN): a discount the advice DECLARES is accepted up to 3 days after the discount deadline;
# a discount taken silently (paid < gross, or cash below the advice total) is accepted up to 14 days after it.
GRACE_DECLARED, GRACE_SILENT = 3, 14
PART_FRACTIONS = (0.4, 0.5, 0.6)
DISPUTE_RATES = (0.03, 0.05, 0.10, 0.20, 0.30)


def _rows_for(t, a: dict, idx, fx: Fx, terms, how: str) -> list[tuple]:
    # XLSX advices carry no currency: their amounts are in the invoices' currency (e.g. CAD invoices paid into a USD
    # account). If the advice's own items are in another currency than the bank line, read it in that currency.
    if not a["ccy"]:
        its, _ = resolve_advice(a["lines"], idx, a["payer"])
        cc = Counter(it["currency"] for it in its if it)
        if cc:
            c = cc.most_common(1)[0][0]
            if c != t.currency and _amount_fits(dict(a, ccy=c), t, fx) == "fx":
                a = dict(a, ccy=c)
    fit = _amount_fits(a, t, fx)
    if fit is None:
        return []
    lines = a["lines"]
    items, grp = resolve_advice(lines, idx, a["payer"])
    if not any(items):
        return []
    bd = pd.Timestamp(t.booking_date)
    out = []
    for l, it in zip(lines.itertuples(index=False), items):
        if it is None:
            continue
        gross = float(l.gross) if l.gross == l.gross else 0.0
        paid = float(l.paid) if l.paid == l.paid else gross
        ded = float(l.deduction) if l.deduction == l.deduction else 0.0
        reason = (l.reason or "").upper().strip()
        if it["item_type"] == "ARC" or gross < 0:
            out.append([it, abs(gross), ""])
            continue
        silent = False
        if abs(ded) < 0.005 and paid < gross - 0.005:   # shortfall without a deduction entry
            ded, silent = round(paid - gross, 2), True
        if abs(ded) < 0.005:
            exc = ""
            if paid < it["open_amount"] - _tol(it["open_amount"]):
                exc = "PARTIAL"
            out.append([it, paid, exc])
            continue
        kind = None
        if reason in DISCOUNT_REASONS or reason.startswith(("DTO", "DISC", "SKONTO", "ESCOMPTE")):
            kind = "disc"
        elif reason in WITHHOLDING_REASONS:
            kind = "wht"
        elif reason:
            kind = "dispute"
        else:  # no reason given: infer from the size of the deduction
            da = _disc_amount(it, terms)
            if da is not None and abs(da + ded) <= 0.011:
                kind = "disc"
            elif it["entity"] in WHT_ENTITIES and any(abs(round(gross * w, 2) + ded) <= 0.011 for w in WHT_RATES):
                kind = "wht"
            else:
                kind = "partial" if silent else "dispute"
        if kind == "partial":
            out.append([it, paid, "PARTIAL"])
        elif kind == "disc":
            late = _late(it, terms, bd)
            # the 2025-09 Voucher layout lists every deduction but has no reason column: still a DECLARED deduction
            declared = (not silent) and (bool(reason) or a.get("layout") == "xlsx_voucher")
            grace = GRACE_DECLARED if declared else GRACE_SILENT
            ok = late is not None and late <= grace
            out.append([it, it["open_amount"], "SHORT_DISCOUNT"] if ok else [it, paid, "PARTIAL"])
        elif kind == "wht":
            out.append([it, it["open_amount"], "SHORT_WITHHOLDING"])
        else:
            out.append([it, paid, "SHORT_DISPUTE"])
    if fit == "short":
        diff = round(a["paid"] - float(t.amount), 2)
        if not _explain_short(out, diff, t, terms, bd):
            return []
    res = []
    conf = 0.95 if how == "ref" else 0.9
    for it, amt, exc in out:
        if it["currency"] != t.currency and it["item_type"] == "ARI" and exc in ("", "SHORT_DISCOUNT", "SHORT_WITHHOLDING"):
            exc = "FX_DIFFERENCE"
        res.append((t.bank_txn_id, it["item_id"], round(amt, 2), exc, conf, f"remittance_{how}"))
    return res


def _explain_short(out: list, diff: float, t, terms, bd) -> bool:
    """The cash is `diff` below what the advice lists: explain it (in place) or return False."""
    free = [r for r in out if not r[2] and r[0]["item_type"] == "ARI"]
    if not free or diff <= 0:
        return False
    # (a) tax withheld on every invoice at one rate
    if t.entity in WHT_ENTITIES:
        base = sum(r[1] for r in free)
        for w in WHT_RATES:
            if abs(sum(round(r[1] * w, 2) for r in free) - diff) <= 0.011 * max(1, len(free) / 10) \
                    or abs(round(base * w, 2) - diff) <= 0.011:
                for r in free:
                    r[1], r[2] = r[0]["open_amount"], "SHORT_WITHHOLDING"
                return True
    # (b) undeclared early-payment discounts on 1..3 invoices
    cands = [(r, _disc_amount(r[0], terms)) for r in free]
    cands = [(r, d) for r, d in cands if d]
    cents = int(round(diff * 100))
    found = []
    from itertools import combinations
    for k in (1, 2, 3):
        if len(cands) < k or (k == 3 and len(cands) > 120):
            break
        for combo in combinations(range(len(cands)), k):
            if abs(sum(int(round(cands[j][1] * 100)) for j in combo) - cents) <= 1:
                found.append(combo)
                if len(found) > 1:
                    break
        if found:
            break
    if len(found) == 1:
        for j in found[0]:
            r, d = cands[j]
            late = _late(r[0], terms, bd)
            if late is not None and late <= GRACE_SILENT:
                r[1], r[2] = r[0]["open_amount"], "SHORT_DISCOUNT"
            else:
                r[1], r[2] = round(r[1] - d, 2), "PARTIAL"
        return True
    if found:
        return False
    # (c) one invoice paid only in part (40/50/60 %) -> PARTIAL; or an undeclared deduction of a typical dispute
    #     size (3/5/10/20/30 %) -> SHORT_DISPUTE. Must be unique.
    hits = []
    for r in free:
        if any(abs(round(r[1] * (1 - f), 2) - diff) <= 0.011 for f in PART_FRACTIONS):
            hits.append((r, "PARTIAL"))
        elif any(abs(round(r[1] * f, 2) - diff) <= 0.011 for f in DISPUTE_RATES):
            hits.append((r, "SHORT_DISPUTE"))
    if len(hits) == 1:
        r, code = hits[0]
        r[1], r[2] = round(r[1] - diff, 2), code
        return True
    return False
