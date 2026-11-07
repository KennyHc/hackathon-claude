"""Exception rules library (POLICIES §2–§3).

Public API
----------
    explain(items, cash, line, masters) -> list[(item_id, amount_in_item_ccy, exception)] | None
        Is `cash` (|bank amount|, bank currency) explained by exactly these `items` under POLICIES? If yes, one tuple
        per item (same order as `items`), else None. `items` = DataFrame rows of ctx.items (or list of dicts) whose
        `open_amount` must be the open amount JUST BEFORE this line. Handles: full payment, credit notes netted,
        credit-note-only refunds, SHORT_DISCOUNT (within terms: booking <= issue + discount days + 3) and the late
        discount (PARTIAL, amount = cash), SHORT_WITHHOLDING (MX/BR/MA customers, 1.25/1.5/10 % of total),
        card orders settled net of the acquirer fee (empty), SHORT_BANK_CHARGES (non-SEPA payer, <= 60 EUR and
        <= 2 %), OVERPAYMENT (single invoice, plausible excess), PARTIAL (1-3 invoices at a round fraction, else one invoice absorbs
        the shortfall), FX_DIFFERENCE (bank ccy != item ccy; exact via orig_amount, else within 3 % of the ECB rate).
        Not detected: SHORT_DISPUTE / DUPLICATE_PAYMENT / RETURNED (need remittance advices / history).

    match_short_payment(payer_open_items, cash, line, masters) -> (item_id, amount, exception) | None
        For a payer matcher: the unique open invoice explained by SHORT_BANK_CHARGES / OVERPAYMENT / SHORT_WITHHOLDING.
    overpayment_plausible(open, cash) -> bool   (excess = round amount 1..200 or exactly 1 % / 10 %)

    explain_codes(items, amounts, cash, line, masters) -> list[str] | None
        Same rules when the amounts per item are already known (remittance advice, payment run).

    to_ccy(amount, from_ccy, to_ccy, date, fx_rates) -> float      (fx_rates = units per EUR, last available rate)
    running_open(allocations, bank_lines, open_items) -> pd.Series (open amount of each row's item before its line)

    assign_exceptions(allocations, bank_lines, open_items, **ctx) -> pd.Series aligned with allocations.index
        Exception per matcher row (None = keep the matcher's exception). Never changes items or amounts.
"""
from __future__ import annotations

import bisect
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

WITHHOLDING_COUNTRIES = {"MX", "BR", "MA"}
WITHHOLDING_RATES = (0.0125, 0.015, 0.10)
DISCOUNT_GRACE_DAYS = 3
BANK_CHARGES_MAX_EUR = 60.0
BANK_CHARGES_MAX_PCT = 0.02
FX_TOL = 0.03                       # POLICIES: FX difference always well within 3 %
# euro-area SEPA countries: a transfer from one of these is not "cross-border from outside SEPA"
SEPA_EURO = {"ES", "PT", "FR", "IT", "DE", "NL", "LU", "BE", "AT", "IE", "MT", "FI", "GR", "SK", "SI", "EE", "LV",
             "LT", "CY", "HR", "MC", "SM", "AD", "VA"}
CREDIT_TYPES = {"ARC", "APC"}
FULL_CLEAR_CODES = {"", "SHORT_DISCOUNT", "SHORT_WITHHOLDING", "SHORT_BANK_CHARGES", "FX_DIFFERENCE", "OVERPAYMENT"}
PARTIAL_FRACTIONS = (0.5, 0.3, 0.7, 0.25, 0.4, 0.6, 1 / 3, 2 / 3, 1 / 6, 0.2, 0.75, 0.8, 0.9, 0.95)
CARD_FEE_MAX = 0.03                 # card acquirer fee (net settlement) tolerated on card orders
ROUND_FRACTIONS = (0.5, 0.3, 0.7, 0.25, 0.75, 1 / 3, 2 / 3, 0.4, 0.6, 0.2, 0.8, 0.1, 0.9, 0.05, 0.95, 0.15, 0.85)


# ------------------------------------------------------------------------------------------------------------- FX
_FX_CACHE: dict[int, tuple[list[str], dict[str, np.ndarray]]] = {}


def _fx_table(fx_rates: pd.DataFrame):
    key = id(fx_rates)
    t = _FX_CACHE.get(key)
    if t is None:
        f = fx_rates[["date", "currency", "units_per_eur"]].copy()
        f["units_per_eur"] = pd.to_numeric(f.units_per_eur, errors="coerce")
        p = f.pivot_table(index="date", columns="currency", values="units_per_eur", aggfunc="last").sort_index().ffill()
        t = (list(p.index.astype(str)), {c: p[c].to_numpy(dtype=float) for c in p.columns})
        _FX_CACHE[key] = t
    return t


def rate(ccy: str, date: str, fx_rates: pd.DataFrame) -> float:
    """Units of `ccy` per EUR on `date` (last available rate on or before it)."""
    if ccy == "EUR" or not ccy:
        return 1.0
    dates, cols = _fx_table(fx_rates)
    col = cols.get(ccy)
    if col is None:
        return float("nan")
    i = max(bisect.bisect_right(dates, str(date)[:10]) - 1, 0)
    return float(col[i])


def to_ccy(amount: float, from_ccy: str, to_ccy_: str, date: str, fx_rates: pd.DataFrame) -> float:
    if from_ccy == to_ccy_:
        return float(amount)
    return float(amount) / rate(from_ccy, date, fx_rates) * rate(to_ccy_, date, fx_rates)


# ------------------------------------------------------------------------------------------------------------- masters
_TERMS_CACHE: dict[int, dict[str, tuple[float, int]]] = {}
_COUNTRY_CACHE: dict[int, dict[str, str]] = {}
_ENTITY_CACHE: dict[int, dict[str, str]] = {}


def _terms(masters) -> dict[str, tuple[float, int]]:
    pt = masters["payment_terms"]
    t = _TERMS_CACHE.get(id(pt))
    if t is None:
        t = {}
        for code, pct, dd in zip(pt.code, pt.early_payment_discount_pct, pt.discount_days):
            pct, dd = pd.to_numeric(pct, errors="coerce"), pd.to_numeric(dd, errors="coerce")
            if pd.notna(pct) and pd.notna(dd) and float(pct) > 0:
                t[str(code)] = (float(pct) / 100.0, int(dd))
        _TERMS_CACHE[id(pt)] = t
    return t


def _party_country(masters) -> dict[str, str]:
    c = masters["customers"]
    t = _COUNTRY_CACHE.get(id(c))
    if t is None:
        t = dict(zip(c.customer_id, c.country))
        s = masters.get("suppliers")
        if s is not None:
            for k, v in zip(s.supplier_id, s.country):
                t.setdefault(k, v)
        _COUNTRY_CACHE[id(c)] = t
    return t


def _entity_country(masters) -> dict[str, str]:
    e = masters["entities"]
    t = _ENTITY_CACHE.get(id(e))
    if t is None:
        t = dict(zip(e.entity, e.country))
        t.setdefault("ZA01", "ZA")
        _ENTITY_CACHE[id(e)] = t
    return t


# ------------------------------------------------------------------------------------------------------------- helpers
def _f(x, default=0.0) -> float:
    try:
        v = float(x)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _get(line, key, default=""):
    try:
        v = line.get(key, default)
    except AttributeError:
        return default
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v


def _days_between(d0: str, d1: str) -> int:
    return (pd.Timestamp(str(d1)[:10]) - pd.Timestamp(str(d0)[:10])).days


def _cents(x: float) -> int:
    return int(round(x * 100))


def _roundness(frac: float) -> float:
    return min(abs(frac - f) for f in ROUND_FRACTIONS)


def is_cross_border_non_sepa(party_id: str, entity: str, masters, line=None) -> bool:
    """Payer outside the euro SEPA area paying an entity of another country (or the bank text shows SHA charges)."""
    if line is not None:
        txt = str(_get(line, "text", "")).upper()
        if "/CHGS/" in txt or "GASTOS SHA" in txt:
            return True
    pc = _party_country(masters).get(party_id, "")
    ec = _entity_country(masters).get(entity, "")
    if not pc or not ec or pc == ec:
        return False
    return pc not in SEPA_EURO


def _days_late(issue_date: str, booking: str, dd: int) -> int:
    """Days past the discount deadline (issue + discount days + grace); <= 0 means within terms."""
    return _days_between(issue_date, booking) - dd - DISCOUNT_GRACE_DAYS


def _alt_options(it: dict, booking: str, masters) -> list[tuple[float, str, bool]]:
    """Deductions an AR invoice may carry besides being paid at its open amount:
    (cash_for_item, exception, fully_cleared). Discount -> SHORT_DISCOUNT (within terms) or PARTIAL (late, the
    discount stays open); withholding at 1.25 / 1.5 / 10 % for MX/BR/MA customers."""
    out = []
    if it["item_type"] != "ARI":
        return out
    total, open_ = it["total"], it["open"]
    if abs(total - open_) > 0.005:          # discounts / withholding are on the total of an untouched invoice
        return out
    t = _terms(masters).get(it["payment_terms"])
    if t:
        pct, dd = t
        in_time = bool(booking) and _days_late(it["issue_date"], booking, dd) <= 0
        for cc in sorted({round(total - round(pct * total, 2), 2), round(total - pct * total, 2)}):
            out.append((cc, "SHORT_DISCOUNT" if in_time else "PARTIAL", in_time))
    if _party_country(masters).get(it["party_id"], "") in WITHHOLDING_COUNTRIES:
        for r in WITHHOLDING_RATES:
            out.append((round(total - round(r * total, 2), 2), "SHORT_WITHHOLDING", True))
    return out


def _dp(alts: list[list[tuple[int, str]]], need: int, tol: int, kinds=None, max_states=60_000, budget=2_000_000):
    """Smallest set of (item, alternative) whose (negative) deltas sum to need (± tol). Only alternatives whose kind is
    in `kinds` (None = all). -> {item: alt} or None (also None when the search exceeds its work budget)."""
    if abs(need) <= tol:
        return {}
    floor = need - tol
    states: dict[int, tuple] = {0: ()}
    work = 0
    for i, a in enumerate(alts):
        opts = [(j, d) for j, (d, k) in enumerate(a) if (kinds is None or k in kinds) and d < 0]
        if not opts:
            continue
        work += len(states) * len(opts)
        if work > budget:
            return None
        new = dict(states)
        for s, path in states.items():
            for j, d in opts:
                ns = s + d
                if ns < floor:
                    continue
                old = new.get(ns)
                if old is None or len(old) > len(path) + 1:
                    new[ns] = path + ((i, j),)
        if len(new) > max_states:
            return None
        states = new
    for dv in sorted(range(-tol, tol + 1), key=abs):
        p = states.get(need + dv)
        if p:
            return dict(p)
    return None


def _subset_match(alts: list[list[tuple[int, str]]], need: int, tol: int):
    """Pick deductions (alternatives) so that their deltas sum to `need` cents (± tol). -> {item: alt} or None.
      1. every eligible item takes the same kind/variant (all discounted, all withheld at 1.5 %, ...)
      2. every invoice within its discount window takes the discount; smallest subset of the rest (late discount
         first, then withholding, then any)
      3. smallest subset with late discounts only / withholding only / anything."""
    if abs(need) <= tol:
        return {}
    variants: dict[tuple[str, int], dict[int, tuple[int, int]]] = {}
    for i, a in enumerate(alts):
        seen: dict[str, int] = {}
        for j, (d, k) in enumerate(a):
            v = seen.get(k, 0)
            seen[k] = v + 1
            variants.setdefault((k, v), {})[i] = (j, d)
    for best in variants.values():
        # per-item rounding: the payer may compute the deduction on the sum -> allow ~0.5 cent per item
        if abs(sum(d for _, d in best.values()) - need) <= max(tol, (len(best) + 1) // 2):
            return {i: j for i, (j, _) in best.items()}
    forced: dict[int, int] = {}
    for i, a in enumerate(alts):
        for j, (d, k) in enumerate(a):
            if k == "SHORT_DISCOUNT":
                forced[i] = j
                break
    passes = ({"PARTIAL"}, {"SHORT_WITHHOLDING"}, None)
    if forced:
        rest = [[] if i in forced else a for i, a in enumerate(alts)]
        need2 = need - sum(alts[i][j][0] for i, j in forced.items())
        if abs(need2) <= tol:
            return forced
        for kinds in passes:
            got = _dp(rest, need2, tol, kinds)
            if got is not None:
                return {**forced, **got}
    for kinds in passes:
        got = _dp(alts, need, tol, kinds)
        if got is not None:
            return got
    return None


def _fraction_alts(rows, inv_idx) -> list[list[tuple[int, int, float]]]:
    """Per invoice: candidate partial payments round(f × open | total) -> (delta_cents, alt_cents, amount)."""
    out = [[] for _ in rows]
    for i in inv_idx:
        op, tot = rows[i]["open"], rows[i]["total"]
        seen = set()
        for f in PARTIAL_FRACTIONS:
            for base in (op, tot):
                amt = round(f * base, 2)
                if 0.005 < amt < op - 0.005 and amt not in seen:
                    seen.add(amt)
                    out[i].append((_cents(amt) - _cents(op), amt))
    return out


def _small_subset(fr, need: int, tol: int, kmax: int = 3):
    """Up to kmax distinct items each paying a fraction so that deltas sum to need (± tol). -> {item: amount}|None"""
    flat = [(i, d, amt) for i, lst in enumerate(fr) for d, amt in lst]
    if not flat:
        return None
    for i, d, amt in flat:                                 # k = 1
        if abs(d - need) <= tol:
            return {i: amt}
    if kmax < 2 or len(flat) > 4000:
        return None
    by_d: dict[int, list[tuple[int, float]]] = {}
    for i, d, amt in flat:
        by_d.setdefault(d, []).append((i, amt))
    for i, d, amt in flat:                                 # k = 2
        for dv in range(-tol, tol + 1):
            for i2, amt2 in by_d.get(need - d + dv, ()):
                if i2 != i:
                    return {i: amt, i2: amt2}
    if kmax < 3 or len(flat) > 600:
        return None
    for x in range(len(flat)):                             # k = 3
        i, d, amt = flat[x]
        for y in range(x + 1, len(flat)):
            i2, d2, amt2 = flat[y]
            if i2 == i:
                continue
            for dv in range(-tol, tol + 1):
                for i3, amt3 in by_d.get(need - d - d2 + dv, ()):
                    if i3 != i and i3 != i2:
                        return {i: amt, i2: amt2, i3: amt3}
    return None


def _pick_partial(rows, inv_idx, R) -> int | None:
    """Item that absorbs a shortfall R (<0): its applied amount open+R must stay in (0, open)."""
    cands = []
    for i in inv_idx:
        op = rows[i]["open"]
        a = op + R
        if 0.005 < a < op - 0.005:
            cands.append((_roundness(a / op), -op, i))
    if not cands:
        return None
    cands.sort()
    return cands[0][2]


def _snap(amount: float, open_: float, total: float) -> float:
    """FX-converted partial amount: snap to a round fraction of the open / total amount when within 2 %."""
    best = amount
    err = 0.02
    for f in PARTIAL_FRACTIONS:
        for base in (open_, total):
            c = round(f * base, 2)
            if c > 0 and abs(c / amount - 1) < err:
                best, err = c, abs(c / amount - 1)
    return best


OVERPAY_ROUND_CENTS = {100, 200, 500, 1000, 2000, 5000, 10000, 20000}
OVERPAY_PCTS = (0.01, 0.10)


def overpayment_plausible(open_: float, cash: float) -> bool:
    """Overpayments in the data are the invoice + a round amount (1, 2, 5 ... 200) or + exactly 1 % / 10 %. Any other
    excess on one invoice is almost always a wrong item set (the line pays more documents): val 0 / 55 such matcher
    OVERPAYMENT rows were right, vs 310 / 314 for the patterns."""
    exc = _cents(cash) - _cents(open_)
    if exc <= 0:
        return False
    if exc in OVERPAY_ROUND_CENTS and exc <= 0.35 * _cents(open_):
        return True
    return any(abs(_cents(cash) - _cents(round(open_ * (1 + p), 2))) <= 1 for p in OVERPAY_PCTS)


SHORT_PAYMENT_CODES = ("SHORT_BANK_CHARGES", "OVERPAYMENT", "SHORT_WITHHOLDING")


def match_short_payment(cands, cash: float, line, masters, codes=SHORT_PAYMENT_CODES):
    """One open AR invoice of the (resolved) payer explains a receipt that matches no document exactly:
    SHORT_BANK_CHARGES / OVERPAYMENT (plausible excess) / SHORT_WITHHOLDING. `cands` = the payer's open items (DataFrame
    rows of ctx.items or list of dicts, open_amount = open now). -> (item_id, amount, exception) when exactly one
    invoice qualifies, else None. Val simulation on the receipts left unclaimed by the full pipeline (real payer
    resolver): 117 lines claimed at 96.6 % (SBC 113/117), OVERPAYMENT 81/85, SHORT_WITHHOLDING 18/18."""
    recs = cands if isinstance(cands, list) else cands.to_dict("records")
    bccy = str(_get(line, "currency", ""))
    hits = []
    for r in recs:
        if str(r.get("item_type", "")) != "ARI" or str(r.get("currency", "")) != bccy:
            continue
        if _f(r.get("open_amount"), 0.0) <= 0.005:
            continue
        ex = explain([r], cash, line, masters)
        if ex and ex[0][2] in codes:
            hits.append(ex[0])
            if len(hits) > 1:
                return None
    return hits[0] if hits else None


# ------------------------------------------------------------------------------------------------------------- explain
def explain(items: pd.DataFrame, cash: float, line, masters: dict) -> list[tuple[str, float, str]] | None:
    """[(item_id, amount_in_item_ccy, exception)] in the order of `items` if `cash` (|bank amount|, bank ccy) is
    explained by exactly these items under POLICIES, else None. `items` needs item_id, item_type, total_amount,
    open_amount (= open just before this line), currency, issue_date, payment_terms, party_id, entity
    (+ payment_method, optional: card orders are cleared gross)."""
    if items is None or len(items) == 0:
        return None
    rows = []
    recs = items if isinstance(items, list) else items.to_dict("records")
    for r in recs:
        total = _f(r.get("total_amount"))
        op = _f(r.get("open_amount"), total)
        rows.append({"item_id": r["item_id"], "item_type": str(r.get("item_type", "")), "total": total,
                     "open": round(op, 2), "ccy": str(r.get("currency", "")), "issue_date": str(r.get("issue_date", "")),
                     "payment_terms": str(r.get("payment_terms", "") or ""), "party_id": str(r.get("party_id", "")),
                     "entity": str(r.get("entity", "")), "pm": str(r.get("payment_method", "") or "")})
    ccys = {r["ccy"] for r in rows}
    if len(ccys) != 1:
        return None
    iccy = ccys.pop()
    bccy = str(_get(line, "currency", iccy)) or iccy
    booking = str(_get(line, "booking_date", ""))[:10]
    cash = abs(_f(cash))
    cross = bccy != iccy
    oa = _f(_get(line, "orig_amount", np.nan), np.nan)
    oc = str(_get(line, "orig_currency", ""))
    fx = masters.get("fx_rates") if isinstance(masters, dict) else None
    exact = True
    if cross:
        if oc == iccy and not np.isnan(oa) and oa != 0:
            cash_i = abs(oa)
        else:
            if fx is None:
                return None
            cash_i = to_ccy(cash, bccy, iccy, booking, fx)
            if not np.isfinite(cash_i):
                return None
            exact = False
    else:
        cash_i = cash

    sign = [(-1 if r["item_type"] in CREDIT_TYPES else 1) for r in rows]
    inv_idx = [i for i, r in enumerate(rows) if r["item_type"] not in CREDIT_TYPES]
    inv_set = set(inv_idx)
    S = sum(s * r["open"] for s, r in zip(sign, rows))
    if not inv_idx:
        # only credit notes: a refund of them (supplier refunds its APC / we pay out an ARC) at their full amount
        tot = sum(r["open"] for r in rows)
        if not cross and abs(tot - cash) <= 0.01 * len(rows):
            return [(r["item_id"], r["open"], "") for r in rows]
        return None
    if S <= 0:
        return None
    amounts = [r["open"] for r in rows]
    codes = [""] * len(rows)

    def finish():
        out = []
        for i, r in enumerate(rows):
            code = codes[i]
            if cross and i in inv_set and code in FULL_CLEAR_CODES:
                code = "FX_DIFFERENCE"
            out.append((r["item_id"], round(amounts[i], 2), code))
        return out

    if not exact:
        # the bank converted at its own rate: a full clearing is within the FX tolerance of the documents
        if abs(cash_i / S - 1) <= FX_TOL:
            return finish()
        R = cash_i - S
        if R < 0:
            j = _pick_partial(rows, inv_idx, R)
            if j is not None:
                amounts[j] = _snap(rows[j]["open"] + R, rows[j]["open"], rows[j]["total"])
                codes[j] = "PARTIAL"
                return finish()
        return None

    # exact cash in item currency ------------------------------------------------------------------------------
    tol = 1
    base = [s * _cents(r["open"]) for s, r in zip(sign, rows)]
    target = _cents(cash_i)
    need = target - sum(base)
    if abs(need) <= tol:
        return finish()
    if need > 0:
        if len(rows) == 1 and rows[0]["item_type"] == "ARI" and overpayment_plausible(rows[0]["open"], cash_i):
            codes[0] = "OVERPAYMENT"
            return finish()
        return None
    # shortfall: deductions (discount / withholding) that clear the invoices
    alts_full, alts_c = [], []
    for i, r in enumerate(rows):
        opts = _alt_options(r, booking, masters) if sign[i] > 0 else []
        alts_full.append(opts)
        alts_c.append([(_cents(c) - base[i], exc) for c, exc, _ in opts])
    if any(alts_c):
        ch = _subset_match(alts_c, need, tol)
        if ch is not None:
            for i, j in ch.items():
                c, exc, full = alts_full[i][j]
                codes[i] = exc
                amounts[i] = rows[i]["open"] if full else c
            return finish()
    short = -need / 100.0
    # card acquirer settlements clear the card orders at gross (the fee is contractual, not an exception)
    if all(r["pm"] == "card" for r in rows) and short <= CARD_FEE_MAX * S:
        return finish()
    # intermediary bank fee on a cross-border transfer from outside SEPA
    if all(r["item_type"] in ("ARI", "ARC") for r in rows):
        j = max(inv_idx, key=lambda i: rows[i]["open"])
        r = rows[j]
        short_eur = short if iccy == "EUR" else (to_ccy(short, iccy, "EUR", booking, fx) if fx is not None else np.nan)
        if (short <= BANK_CHARGES_MAX_PCT * r["open"] + 0.005 and np.isfinite(short_eur)
                and short_eur <= BANK_CHARGES_MAX_EUR + 0.005
                and is_cross_border_non_sepa(r["party_id"], r["entity"], masters, line)):
            codes[j] = "SHORT_BANK_CHARGES"
            return finish()
    # not cleared in full: one or a few invoices paid at a round fraction, else one invoice absorbs the shortfall
    got = _small_subset(_fraction_alts(rows, inv_idx), need, tol, kmax=3 if len(inv_idx) <= 40 else 2)
    if got:
        for i, amt in got.items():
            amounts[i] = amt
            codes[i] = "PARTIAL"
        return finish()
    j = _pick_partial(rows, inv_idx, -short)
    if j is None:
        return None
    amounts[j] = round(rows[j]["open"] - short, 2)
    codes[j] = "PARTIAL"
    return finish()


# ------------------------------------------------------------------------------------------------------------- assign
_MASTERS_CACHE: dict[str, dict] = {}


def _load_masters(bank_lines: pd.DataFrame) -> dict:
    from ingestion import lib
    folder = None
    if "statement_file" in bank_lines.columns and len(bank_lines):
        f = str(bank_lines.statement_file.iloc[0])
        folder = f.split("/")[0] if "/" in f else None
    key = folder or ""
    if key not in _MASTERS_CACHE:
        _MASTERS_CACHE[key] = lib.masters(folder)
    return _MASTERS_CACHE[key]


def explain_codes(items: list[dict], amounts: list[float], cash: float, line, masters) -> list[str] | None:
    """Exceptions for rows whose amounts are already known (e.g. from a remittance advice / payment run).
    Rows applied for less than their open amount -> PARTIAL; the cash shortfall against the applied amounts must be
    explained by deductions on the fully applied invoices (in-time discount / withholding), else None."""
    if not items:
        return None
    ccys = {str(r.get("currency", "")) for r in items}
    if len(ccys) != 1:
        return None
    iccy = ccys.pop()
    bccy = str(_get(line, "currency", iccy)) or iccy
    cross = bccy != iccy
    booking = str(_get(line, "booking_date", ""))[:10]
    if cross:
        oa = _f(_get(line, "orig_amount", np.nan), np.nan)
        if str(_get(line, "orig_currency", "")) != iccy or np.isnan(oa) or oa == 0:
            return None
        cash = abs(oa)
    codes, alts = [], []
    applied = 0
    for r, a in zip(items, amounts):
        op = round(_f(r.get("open_amount"), _f(r.get("total_amount"))), 2)
        credit = str(r.get("item_type", "")) in CREDIT_TYPES
        applied += (-1 if credit else 1) * _cents(a)
        if credit:
            codes.append("")
            alts.append([])
            continue
        if a < op - max(0.01, 0.005 * op):
            codes.append("PARTIAL")
            alts.append([])
            continue
        codes.append("")
        row = {"item_type": str(r.get("item_type", "")), "total": _f(r.get("total_amount")), "open": op,
               "issue_date": str(r.get("issue_date", "")), "payment_terms": str(r.get("payment_terms", "") or ""),
               "party_id": str(r.get("party_id", ""))}
        alts.append([(_cents(c) - _cents(op), e) for c, e, full in _alt_options(row, booking, masters) if full])
    need = _cents(cash) - applied
    if abs(need) > 1:
        if need > 0:
            return None
        ch = _subset_match(alts, need, 1)
        if ch is None:
            return None
        for i, j in ch.items():
            codes[i] = alts[i][j][1]
    if cross:
        codes = ["FX_DIFFERENCE" if (c in FULL_CLEAR_CODES and str(r.get("item_type", "")) not in CREDIT_TYPES) else c
                 for c, r in zip(codes, items)]
    return codes


def running_open(allocations: pd.DataFrame, bank_lines: pd.DataFrame, open_items: pd.DataFrame) -> pd.Series:
    """Open amount of each allocation row's item just BEFORE its bank line: open at the phase start minus what earlier
    lines (booking date, then bank_txn_id order) allocated to it (RETURNED re-opens). Aligned with allocations.index."""
    order = pd.Series(np.arange(len(bank_lines)), index=bank_lines.bank_txn_id.values)
    order = order[~order.index.duplicated()]
    al = allocations[["bank_txn_id", "item_id", "amount", "exception"]].copy()
    al["_ord"] = al.bank_txn_id.map(order).fillna(-1).to_numpy()
    al["_amt"] = pd.to_numeric(al.amount, errors="coerce").fillna(0.0)
    al["_d"] = np.where(al.exception.fillna("") == "RETURNED", al._amt, -al._amt)
    al = al.sort_values("_ord", kind="stable")
    # rows of the same line do not affect each other: cumulate per (item, line) blocks
    blk = al.groupby(["item_id", "_ord"], sort=False)._d.sum().reset_index().sort_values(["item_id", "_ord"])
    blk["_before"] = blk.groupby("item_id")._d.cumsum() - blk._d
    al = al.merge(blk[["item_id", "_ord", "_before"]], on=["item_id", "_ord"], how="left").set_axis(al.index)
    op0 = open_items.drop_duplicates("item_id").set_index("item_id").open_amount.astype(float)
    return (al.item_id.map(op0) + al._before).round(2).reindex(allocations.index)


def assign_exceptions(allocations: pd.DataFrame, bank_lines: pd.DataFrame, open_items: pd.DataFrame, **ctx) -> pd.Series:
    """Exception per matcher row, aligned with allocations.index (None = keep the matcher's exception).

    Per bank line: rebuild each item's open amount just before the line (replaying earlier allocations), run `explain`
    on (items, cash) and take its codes for rows whose matcher amount agrees with the rule's amount. Rows the rules
    cannot explain get the row-level rules: a cross-currency invoice row cleared in full -> FX_DIFFERENCE (credit
    notes stay empty); an invoice row applied for less than its open amount -> PARTIAL; applied in full -> empty.
    Matcher-only codes (SHORT_DISPUTE, DUPLICATE_PAYMENT, RETURNED, UNIDENTIFIED) are always kept. Items and
    amounts are never changed."""
    res = pd.Series([None] * len(allocations), index=allocations.index, dtype=object)
    if not len(allocations):
        return res
    masters = ctx.get("masters") or _load_masters(bank_lines)
    bcols = [c for c in ("bank_txn_id", "amount", "currency", "booking_date", "orig_amount", "orig_currency", "text")
             if c in bank_lines.columns]
    bank = {r["bank_txn_id"]: r for r in bank_lines[bcols].drop_duplicates("bank_txn_id").to_dict("records")}
    icols = [c for c in ("item_id", "item_type", "total_amount", "currency", "issue_date", "payment_terms", "party_id",
                         "entity", "payment_method") if c in open_items.columns]
    items = {r["item_id"]: r for r in open_items[icols].drop_duplicates("item_id").to_dict("records")}
    keep = {"SHORT_DISPUTE", "DUPLICATE_PAYMENT", "RETURNED", "UNIDENTIFIED"}
    op_all = running_open(allocations, bank_lines, open_items).to_numpy()
    txns = allocations.bank_txn_id.to_numpy()
    iids = allocations.item_id.to_numpy()
    excs = allocations.exception.fillna("").astype(str).to_numpy()
    amts = pd.to_numeric(allocations.amount, errors="coerce").fillna(0.0).to_numpy()
    out = [None] * len(allocations)
    groups: dict[str, list[int]] = {}
    for k, t in enumerate(txns):
        groups.setdefault(t, []).append(k)
    for txn, pos in groups.items():
        line = bank.get(txn)
        if line is None or any(iids[k] not in items for k in pos) or any(np.isnan(op_all[k]) for k in pos):
            continue
        its = []
        for k in pos:
            r = dict(items[iids[k]])
            r["open_amount"] = max(float(op_all[k]), 0.0)
            its.append(r)
        ex = None
        hard = [n for n, k in enumerate(pos) if excs[k] in keep]
        if not any(excs[pos[n]] != "SHORT_DISPUTE" for n in hard):
            # explain the rows the matcher left open; SHORT_DISPUTE rows consume their applied amount of the cash
            sub = [n for n in range(len(pos)) if n not in set(hard)]
            if sub and not any(its[n]["open_amount"] <= 0.005 for n in sub):
                ln, cash = line, abs(_f(line["amount"]))
                if hard:
                    d = sum((-1 if its[n]["item_type"] in CREDIT_TYPES else 1) * amts[pos[n]] for n in hard)
                    iccy = str(its[hard[0]]["currency"])
                    ln = dict(line)
                    if str(line.get("currency", "")) == iccy:
                        cash = cash - d
                    elif str(line.get("orig_currency", "")) == iccy and _f(line.get("orig_amount"), 0.0) != 0:
                        ln["orig_amount"] = abs(_f(line.get("orig_amount"))) - d
                    else:
                        cash = cash - to_ccy(d, iccy, str(line.get("currency", "")), str(line.get("booking_date", "")),
                                             masters["fx_rates"])
                if cash > 0:
                    try:
                        got = explain([its[n] for n in sub], cash, ln, masters)
                    except Exception:  # noqa: BLE001 - never break the merge
                        got = None
                    sub_amts = [amts[pos[n]] for n in sub]
                    if got is not None and not all(abs(a - x[1]) <= max(0.01, 0.005 * abs(x[1]))
                                                   for a, x in zip(sub_amts, got)):
                        # the rules found another explanation than the matcher's amounts: trust the amounts
                        try:
                            cc = explain_codes([its[n] for n in sub], sub_amts, cash, ln, masters)
                        except Exception:  # noqa: BLE001
                            cc = None
                        if cc is not None:
                            got = [(its[n]["item_id"], a, c) for n, a, c in zip(sub, sub_amts, cc)]
                    elif got is None:
                        try:
                            cc = explain_codes([its[n] for n in sub], sub_amts, cash, ln, masters)
                        except Exception:  # noqa: BLE001
                            cc = None
                        if cc is not None:
                            got = [(its[n]["item_id"], a, c) for n, a, c in zip(sub, sub_amts, cc)]
                    if got is not None:
                        ex = [None] * len(pos)
                        for n, x in zip(sub, got):
                            ex[n] = x
        bccy = str(line.get("currency", ""))
        for n, k in enumerate(pos):
            e0, a0 = excs[k], amts[k]
            if e0 in keep:
                continue
            if ex is not None and ex[n] is not None and abs(a0 - ex[n][1]) <= max(0.01, 0.005 * abs(ex[n][1])):
                # a PARTIAL set by a matcher (instalment / retention / advance from the PDF) is never downgraded
                if not (e0 == "PARTIAL" and ex[n][2] in FULL_CLEAR_CODES):
                    out[k] = ex[n][2]
                continue
            r = its[n]
            if r["item_type"] in CREDIT_TYPES:
                out[k] = ""
                continue
            op = r["open_amount"]
            cross = str(r["currency"]) != bccy
            if abs(a0 - op) <= max(0.01, 0.005 * op):
                if cross and e0 != "PARTIAL":
                    out[k] = "FX_DIFFERENCE"
            elif a0 < op and e0 in FULL_CLEAR_CODES:
                out[k] = "PARTIAL"
    return pd.Series(out, index=allocations.index, dtype=object)
