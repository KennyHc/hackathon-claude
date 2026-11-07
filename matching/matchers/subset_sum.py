"""Customer receipts without a usable reference: who paid (matching/parties.py) + which of the payer's open items the
amount settles.  Also hosts the shared engine used by `same_amount` (both are computed in ONE chronological pass so the
items consumed by one line are not offered to a later one).

Per line (incoming, not claimed by an earlier matcher), in booking-date order:
1. payer candidates = PartyResolver (IBAN / name / learned aliases / ORD. / EXPTE); then their subsidiaries / group.
2. pool = the candidate's open AR items in the line's entity & currency, issued on/before the booking date, minus what
   earlier matchers (ctx.allocations) and earlier lines of ours took, minus cancelled items (pdf_clauses).
3. solutions, first hit wins (kind = method suffix):
   - single       one invoice with exactly the amount (several -> same_amount: month named in the text -> the invoice
                  of that service month, issued at the start of the following month; else the oldest = FIFO)
   - discount     one invoice paid net of its early-payment discount (SHORT_DISCOUNT in time, PARTIAL late)
   - withholding  one invoice of a MX/BR/MA customer net of 1.25 / 1.5 / 10 % withheld tax (SHORT_WITHHOLDING)
   - cn_pair      one invoice net of its own credit note
   - fifo/subset  oldest-first prefix / unique exact subset (meet-in-the-middle, recent credit notes negative) whose
                  chance of being a coincidence (density of subset sums around the amount) is small
   - *_wht        same with every invoice withheld at one rate;  subset_disc: each invoice full or net of its discount
   - overpayment  one invoice + a round excess (1, 2, 5, 10, 50, 100 ...) -> OVERPAYMENT
   - partial      one invoice paid at a round fraction (25-70 %) -> PARTIAL ("RESTO"/"BALANCE" in text -> no exception)
   An allocation to an item that an earlier matcher settles on a LATER line gets PARTIAL (the item stays open).
4. claim only when exactly one candidate customer has a solution of the best tier (precision first).
"""
from __future__ import annotations

import math
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd

from matching.common import Ctx, alloc_frame
from matching.parties import PartyResolver, upper

NAME = "subset_sum"

MITM_MAX = 30          # max pool size for the exhaustive search
MITM_RECENT_DAYS = 400  # exhaustive search only over invoices issued in the last N days
DUE_STAGES = ((-90, 0), (-9999, 0), (-9999, 20))   # exhaustive pools: due-date windows around booking
SPURIOUS_MAX = 0.3      # max expected number of coincidental subsets hitting the amount
MULTI_ALT = True        # subset search with every invoice withheld / discounted
OVERPAY = False         # one invoice overpaid by a round amount (1, 2, 5, 10, 50, 100 ...) - short_payments does it
FX = True               # foreign-currency receipts: search on orig_amount / orig_currency -> FX_DIFFERENCE
OVERPAY_MAX = 0.35
PARTIAL_FRAC = True     # one invoice paid at a round fraction
MULTI_PART = True       # several invoices in full + one at a round fraction
PART_FRACTIONS = (0.4, 0.5, 0.6)
PART_MAX = 12           # max pool for that search (4 states per invoice)
FRACTIONS = (0.25, 0.3, 0.4, 0.5, 0.6, 0.7)
OVERPAY_EXCESS = {100, 200, 500, 1000, 2000, 5000, 10000, 20000}   # cents: 1, 2, 5, 10, 20, 50, 100, 200
TIER = {"discount": 1, "withholding": 1, "cn_pair": 1, "fifo_wht": 2, "subset_wht": 2, "fifo_disc": 2,
        "subset_disc": 2, "overpayment": 3, "partial": 3, "subset_part": 3}
WHT_COUNTRIES = {"MX", "BR", "MA"}
REMAINDER = re.compile(r"\b(?:RESTO|RESTANTE|REST|BALANCE|SOLDE|SALDO|REMAINDER|RESTBETRAG|SETTLEMENT)\b")
WHT_RATES = (0.0125, 0.015, 0.10)
DISCOUNT = True         # claim one invoice paid net of its early-payment discount
CN_RECENT_DAYS = 120    # open credit notes issued in the last N days take part (negative) in the exhaustive search
SPURIOUS_ALT = 0.05     # same, for the withheld / discounted valuations
MULTI_DISC = True       # subset search where each invoice is paid in full or net of its discount
DISC_MAX = 18           # max pool for that search (3 states per invoice)
TIE_MAX = 1             # exact subsets enumerated by mitm (only unique ones are used)
WINDOW = 20000          # +-cents window used to estimate the subset-sum density

# ---------------------------------------------------------------------------------------------------------------------
# month named in the text (service month)
MONTHS = {
    1: "JANUARY JAN ENERO ENE JANUAR JANVIER JANV GENNAIO GEN JANEIRO STYCZEN STY JANUARI",
    2: "FEBRUARY FEB FEBRERO FEBRUAR FEVRIER FEV FEVR FEBBRAIO FEVEREIRO LUTY LUT FEBRUARI",
    3: "MARCH MAR MARZO MAERZ MARZ MARS MARCO LUTEGO MAART",
    4: "APRIL APR ABRIL ABR AVRIL AVR APRILE KWIECIEN",   # not "KWI": ZA01 invoice prefix (FINAL)
    5: "MAY MAYO MAI MAGGIO MAG MAIO MAJ MEI",
    6: "JUNE JUN JUNIO JUNI JUIN GIUGNO GIU JUNHO CZERWIEC CZE",
    7: "JULY JUL JULIO JULI JUILLET JUIL LUGLIO LUG JULHO LIPIEC LIP",
    8: "AUGUST AUG AGOSTO AGO AOUT SIERPIEN SIE AUGUSTUS",
    9: "SEPTEMBER SEP SEPT SEPTIEMBRE SETIEMBRE SET SEPTEMBRE SETTEMBRE SETEMBRO WRZESIEN WRZ",
    10: "OCTOBER OCT OCTUBRE OKTOBER OKT OCTOBRE OTTOBRE OTT OUTUBRO OUT PAZDZIERNIK PAZ",
    11: "NOVEMBER NOV NOVIEMBRE NOVEMBRE NOVEMBRO LISTOPAD LIS",
    12: "DECEMBER DEC DICIEMBRE DIC DEZEMBER DEZ DECEMBRE DICEMBRE DEZEMBRO GRUDZIEN GRU",
}
MONTH_OF = {w: m for m, ws in MONTHS.items() for w in ws.split()}
SERVICE_KW = (r"SERVICE|SERVICES|SERVICIOS?|SERVIZIO|SERVIZI|SERVICO|SERVICOS|MANT|MANTENIMIENTO|MANUTENZIONE|"
              r"MANUTENCAO|CUOTA|QUOTA|WARTUNG|LEISTUNG|LEISTUNGEN|MONAT|ENTRETIEN|MAINTENANCE|MAINT|ABONNEMENT|"
              r"CANONE|MENSALIDADE|MENSUALIDAD|FEE|FEES|MONTHLY|MENSUEL|USLUGA|USLUGI|ONDERHOUD|ABO")
MONTH_RE = re.compile(r"\b(?:" + SERVICE_KW + r")\.?\s+(?:DE\s+|DEL\s+|FOR\s+|OF\s+|DI\s+|DU\s+)?([A-Z]{3,10})\.?"
                      r"(?:\s+(?:DE\s+)?('?\d{2,4}))?\b")
MONTH_YEAR_RE = re.compile(r"\b([A-Z]{3,10})\.?\s+(20\d{2})\b")


def service_month(text: str, booking: str) -> tuple[int, int] | None:
    """(year, month) of the service month named in the text, or None."""
    u = upper(text)
    for rx in (MONTH_RE, MONTH_YEAR_RE):
        for m in rx.finditer(u):
            mo = MONTH_OF.get(m.group(1))
            if not mo:
                continue
            y = m.group(2) if m.lastindex and m.lastindex >= 2 else None
            by, bm = int(booking[:4]), int(booking[5:7])
            if y:
                y = int(y.strip("'"))
                y = y + 2000 if y < 100 else y
            else:
                y = by if mo <= bm + 1 else by - 1
            return y, mo
    return None


def rnd(v: float) -> int:
    """Round half up (deductions are computed to the cent, half up - Python's round() is half-to-even)."""
    return int(math.floor(v + 0.5 + 1e-9))


def next_month(y: int, m: int) -> str:
    return f"{y + (m == 12):04d}-{(m % 12) + 1:02d}"


# ---------------------------------------------------------------------------------------------------------------------
def subset_sums(vals: np.ndarray) -> np.ndarray:
    s = np.zeros(1, dtype=np.int64)
    for v in vals:
        s = np.concatenate([s, s + v])
    return s


def mitm(vals: list[int], target: int, max_sols: int = 1) -> tuple[int, list[list[int]], float]:
    """Exact subset-sum by meet-in-the-middle. -> (number of solutions, the solutions' indexes when there are at most
    `max_sols`, expected number of coincidental solutions estimated from the density of subset sums within +-WINDOW of
    the target)."""
    n = len(vals)
    h = n // 2
    a, b = subset_sums(np.array(vals[:h], dtype=np.int64)), subset_sums(np.array(vals[h:], dtype=np.int64))
    order = np.argsort(b, kind="stable")
    bs = b[order]
    need = target - a
    lo, hi = np.searchsorted(bs, need, "left"), np.searchsorted(bs, need, "right")
    cnt = int((hi - lo).sum())
    wl, wh = np.searchsorted(bs, need - WINDOW, "left"), np.searchsorted(bs, need + WINDOW, "right")
    dens = float((wh - wl).sum()) / (2 * WINDOW + 1)
    sols = []
    if 0 < cnt <= max_sols:
        for i in np.nonzero(hi > lo)[0]:
            for jj in range(lo[i], hi[i]):
                j = int(order[jj])
                sols.append([k for k in range(h) if int(i) >> k & 1] + [h + k for k in range(n - h) if j >> k & 1])
    return cnt, sols, dens


def _opt_sums(opts: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """All sums choosing for each item 'absent' or one of its option values -> (sums, codes); code = mixed-radix
    choice per item (0 = absent, j+1 = option j)."""
    s = np.zeros(1, dtype=np.int64)
    c = np.zeros(1, dtype=np.int64)
    mult = 1
    for o in opts:
        s = np.concatenate([s] + [s + v for v in o])
        c = np.concatenate([c] + [c + (j + 1) * mult for j in range(len(o))])
        mult *= len(o) + 1
    return s, c


def mitm_opts(opts: list[list[int]], target: int) -> tuple[int, list[int] | None, float]:
    """Subset-sum where each item is absent or takes one of its alternative values. -> (number of solutions, the
    choice per item when unique (0 absent, j+1 option j), density of sums around the target)."""
    h = len(opts) // 2
    a, ac = _opt_sums(opts[:h])
    b, bc = _opt_sums(opts[h:])
    order = np.argsort(b, kind="stable")
    bs = b[order]
    need = target - a
    lo, hi = np.searchsorted(bs, need, "left"), np.searchsorted(bs, need, "right")
    cnt = int((hi - lo).sum())
    wl, wh = np.searchsorted(bs, need - WINDOW, "left"), np.searchsorted(bs, need + WINDOW, "right")
    dens = float((wh - wl).sum()) / (2 * WINDOW + 1)
    if cnt != 1:
        return cnt, None, dens
    i = int(np.nonzero(hi > lo)[0][0])
    codes = [int(ac[i]), int(bc[order[lo[i]]])]
    out = []
    for part, code in ((opts[:h], codes[0]), (opts[h:], codes[1])):
        for o in part:
            out.append(code % (len(o) + 1))
            code //= len(o) + 1
    return cnt, out, dens


# ---------------------------------------------------------------------------------------------------------------------
def cancelled_set(ctx: Ctx) -> set:
    """Items never settled by bank (cancelled / replaced invoices and their cancelling credit notes), cached on ctx."""
    s = getattr(ctx, "_cancelled_items", None)
    if s is None:
        try:
            from matching.pdf_clauses import cancelled_items
            s = set(cancelled_items(ctx))
        except Exception as e:  # pragma: no cover - the clause file may be missing
            print(f"[subset_sum] cancelled items unavailable ({e!r})")
            s = set()
        ctx._cancelled_items = s
    return s


class Engine:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.pr = PartyResolver.get(ctx)
        it = ctx.open_items[ctx.open_items.side == "AR"]
        it = it.sort_values(["issue_date", "document_number", "item_id"])
        # allocations of earlier matchers (the runner exposes them as ctx.allocations): consume those amounts
        prior = getattr(ctx, "allocations", None)
        taken: dict = {}
        if isinstance(prior, pd.DataFrame) and not prior.empty:
            p = prior[(prior.item_id.astype(str) != "") & (prior.exception != "RETURNED")]
            full = p.exception.isin(["SHORT_DISCOUNT", "SHORT_WITHHOLDING", "SHORT_BANK_CHARGES", "OVERPAYMENT",
                                     "FX_DIFFERENCE", "DUPLICATE_PAYMENT"])
            taken = p.assign(a=np.where(full, 1e12, p.amount.astype(float))).groupby("item_id").a.sum().to_dict()
            # when an earlier matcher settles (part of) an item on a LATER line, our line leaves it open -> PARTIAL
            dates = ctx.bank.set_index("bank_txn_id").booking_date
            pk = p.assign(k=p.bank_txn_id.map(dates).fillna("") + "|" + p.bank_txn_id)
            self.last_prior = pk.groupby("item_id").k.max().to_dict()
        else:
            self.last_prior = {}
        self.pool: dict[str, list] = {}
        cancelled = cancelled_set(ctx)
        for r in it.itertuples(index=False):
            left = r.open_amount - taken.get(r.item_id, 0.0)
            if left < 0.005 or r.item_id in cancelled:
                continue
            self.pool.setdefault(r.party_id, []).append(
                (r.item_id, r.issue_date, r.due_date or r.issue_date, int(round(left * 100)), r.item_type,
                 r.entity, r.currency, int(round(r.total_amount * 100)), r.payment_terms, r.document_number,
                 r.related_document))
        pt = ctx.masters["payment_terms"]
        self.terms = {c: (float(p), int(d)) for c, p, d in zip(pt.code, pt.early_payment_discount_pct, pt.discount_days)
                      if p}
        self.docs = {i: (a, n) for i, a, n in zip(it.item_id, it.document_alnum, it.document_number_norm)}
        self.used: set[str] = set()
        self.rem: dict = {}
        cust = ctx.masters["customers"]
        self.country = dict(zip(cust.customer_id, cust.country))
        self.diag: dict = {}
        self.conf = 0.8
        self.rows = {"subset_sum": [], "same_amount": []}

    # -- candidates ---------------------------------------------------------------------------------------------------
    def items(self, cid: str, t) -> list:
        out = []
        for x in self.pool.get(cid, ()):
            if x[0] in self.used or x[5] != t.entity or x[6] != t.currency or x[1] > t.booking_date:
                continue
            if x[0] in self.rem:     # partly paid by an earlier line of ours
                x = x[:3] + (self.rem[x[0]],) + x[4:]
            out.append(x)
        return out

    def solve(self, cid: str, t, cents: int):
        """-> (kind, [items], {item_id: (amount_cents, exception)}) or None."""
        r = self._solve(cid, t, cents)
        if r is None:
            return None
        return r if len(r) == 3 else (r[0], r[1], {})

    def _search(self, t, cid, cents, inv, cns, val, tag):
        """FIFO prefix, then exhaustive subset search, of invoices valued by `val` (cents, or None = not eligible)
        plus recent credit notes (negative). -> (kind, items) or None"""
        inv = [(x, val(x)) for x in inv]
        inv = [(x, v) for x, v in inv if v is not None and v > 0]
        acc = 0
        for k, (x, v) in enumerate(inv):
            acc += v
            if acc == cents and k >= 1:
                return "fifo" + tag, [x for x, _ in inv[:k + 1]]
            if acc > cents:
                break
        lim_iss = str(date.fromisoformat(t.booking_date) - timedelta(days=MITM_RECENT_DAYS))
        bd = date.fromisoformat(t.booking_date)
        prev = None
        for lo, hi in DUE_STAGES:
            lim_lo, lim_due = str(bd + timedelta(days=lo)), str(bd + timedelta(days=hi))
            cand = [(x, v) for x, v in inv if x[1] >= lim_iss and lim_lo <= x[2] <= lim_due and v <= cents][:MITM_MAX]
            if not cand or len(cand) + len(cns) < 2 or sum(v for _, v in cand) < cents or len(cand) == prev:
                continue
            prev = len(cand)
            cand = cand + [(c, -c[3]) for c in cns[:max(0, MITM_MAX + 4 - len(cand))]]
            cnt, sols, dens = mitm([v for _, v in cand], cents, max_sols=TIE_MAX)
            self.diag[(t.bank_txn_id, cid)] = (len(cand), cnt, dens)
            if cnt == 1 and dens <= (SPURIOUS_MAX if not tag else SPURIOUS_ALT) and len(sols[0]) >= 2 \
                    and any(cand[i][1] > 0 for i in sols[0]):
                self.conf = 0.85 if dens <= 0.01 else 0.75 if dens <= 0.1 else 0.6
                return "subset" + tag, [cand[i][0] for i in sols[0]]
            if cnt >= 1 and lo == DUE_STAGES[-1][0]:
                break   # a wider pool cannot make the solution unique
        return None

    def _solve(self, cid: str, t, cents: int):
        its = self.items(cid, t)
        if not its:
            return None
        inv = [x for x in its if x[4] == "ARI"]
        exact = [x for x in inv if x[3] == cents]
        if len(exact) == 1:
            return "single", exact
        if len(exact) > 1:
            refs = set(t.doc_tokens) | set(t.doc_refs)
            named = [x for x in exact if self.docs.get(x[0], ("", ""))[0] in refs or self.docs.get(x[0], ("", ""))[1] in refs]
            if len(named) == 1:
                return "single_ref", named
            sm = service_month(f"{t.text} {t.remittance_info}", t.booking_date)
            if sm:
                ym = next_month(*sm)
                hit = [x for x in exact if x[1][:7] == ym]
                if len(hit) == 1:
                    return "same_amount_month", hit
                return None
            return "same_amount_fifo", [exact[0]]
        # early-payment discount taken on one invoice (cash = total - pct x total): SHORT_DISCOUNT within terms
        # (+3 days grace), PARTIAL after the deadline (POLICIES §3)
        disc = []
        for x in inv:
            v = self.disc_value(x)
            if v is not None and abs(v - cents) <= 1:
                disc.append((x, self.disc_exc(x, t)))
        if disc:
            if len(disc) == 1 and DISCOUNT:
                x, exc = disc[0]
                return "discount", [x], {x[0]: (cents if exc == "PARTIAL" else x[3], exc)}
            return None
        # withholding tax at source (MX / BR / MA customers): cash = total x (1 - rate) on one invoice
        wht = self.country.get(cid, "") in WHT_COUNTRIES
        if wht:
            hits = [(x, r) for r in WHT_RATES for x in inv if x[3] == x[7] and x[7] - rnd(x[7] * r) == cents]
            if len(hits) == 1:
                x = hits[0][0]
                return "withholding", [x], {x[0]: (x[3], "SHORT_WITHHOLDING")}
            if hits:
                return None
        # one invoice net of its own credit note (credit_note.related_document = the invoice)
        by_doc = {x[9]: x for x in inv}
        pairs = [(by_doc[c[10]], c) for c in its if c[4] == "ARC" and c[10] in by_doc
                 and by_doc[c[10]][3] - c[3] == cents]
        if len(pairs) == 1:
            return "cn_pair", list(pairs[0])
        lim_cn = str(date.fromisoformat(t.booking_date) - timedelta(days=CN_RECENT_DAYS))
        cns = [x for x in its if x[4] == "ARC" and x[1] >= lim_cn][-6:] if CN_RECENT_DAYS else []
        r = self._search(t, cid, cents, inv, cns, lambda x: x[3], "")
        if r:
            return r
        # several invoices, all with tax withheld at the same rate
        if wht and MULTI_ALT:
            for rate in WHT_RATES:
                r = self._search(t, cid, cents, inv, cns,
                                 lambda x: x[7] - rnd(x[7] * rate) if x[3] == x[7] else None, "_wht")
                if r:
                    return r[0], r[1], {x[0]: (x[3], "SHORT_WITHHOLDING") for x in r[1] if x[4] == "ARI"}
        # several invoices, each paid in full or net of its early-payment discount (in time -> SHORT_DISCOUNT, late
        # -> PARTIAL with the cash applied)
        if MULTI_DISC and any(self.disc_value(x) is not None for x in inv):
            bd = date.fromisoformat(t.booking_date)
            lim_iss = str(bd - timedelta(days=MITM_RECENT_DAYS))
            for lo, hi in DUE_STAGES:
                lim_lo, lim_due = str(bd + timedelta(days=lo)), str(bd + timedelta(days=hi))
                cand = [x for x in inv if x[1] >= lim_iss and lim_lo <= x[2] <= lim_due and x[3] <= cents][:DISC_MAX]
                if len(cand) < 2:
                    continue
                opts = [[x[3]] + ([self.disc_value(x)] if self.disc_value(x) is not None else []) for x in cand]
                opts += [[-c[3]] for c in cns]
                allx = cand + cns
                cnt, ch, dens = mitm_opts(opts, cents)
                if cnt == 1 and dens <= SPURIOUS_ALT:
                    sel = [(allx[i], c) for i, c in enumerate(ch) if c]
                    if len(sel) >= 2 and any(c == 2 for _, c in sel):
                        exc = {}
                        for x, c in sel:
                            if c == 2:
                                e = self.disc_exc(x, t)
                                exc[x[0]] = (self.disc_value(x) if e == "PARTIAL" else x[3], e)
                        return "subset_disc", [x for x, _ in sel], exc
                if cnt >= 1:
                    break
        # several invoices in full + ONE paid at a round fraction (40 / 50 / 60 %) -> that one PARTIAL
        if MULTI_PART:
            bd = date.fromisoformat(t.booking_date)
            lim_iss = str(bd - timedelta(days=MITM_RECENT_DAYS))
            for lo, hi in DUE_STAGES:
                lim_lo, lim_due = str(bd + timedelta(days=lo)), str(bd + timedelta(days=hi))
                cand = [x for x in inv if x[1] >= lim_iss and lim_lo <= x[2] <= lim_due and x[3] <= cents][:PART_MAX]
                if len(cand) < 2:
                    continue
                opts = [[x[3]] + ([int(math.floor(f * x[7] + 1e-9)) for f in PART_FRACTIONS] if x[3] == x[7] else [])
                        for x in cand]
                cnt, ch, dens = mitm_opts(opts, cents)
                if cnt == 1 and dens <= SPURIOUS_ALT:
                    sel = [(cand[i], c) for i, c in enumerate(ch) if c]
                    parts = [(x, c) for x, c in sel if c >= 2]
                    if len(sel) >= 2 and len(parts) == 1:
                        x, c = parts[0]
                        return "subset_part", [y for y, _ in sel], {x[0]: (opts[cand.index(x)][c - 1], "PARTIAL")}
                if cnt >= 1:
                    break
        # one invoice overpaid by a round amount (the excess stays on the customer account)
        if OVERPAY:
            hits = [x for x in inv if (cents - x[3]) in OVERPAY_EXCESS]  # no size cap for round excesses
            if len(hits) == 1:
                x = hits[0]
                return "overpayment", [x], {x[0]: (x[3], "OVERPAYMENT")}
        # one invoice paid in part: a round fraction of its total / open amount
        if PARTIAL_FRAC:
            hits = {}
            for x in inv:
                for f in FRACTIONS:
                    if cents < x[3] and any(cents in (rnd(f * base), int(math.floor(f * base + 1e-9))) for base in (x[7], x[3])):
                        hits[x[0]] = x
            if len(hits) == 1:
                x = next(iter(hits.values()))
                # "RESTO" / "BALANCE": the rest of an earlier partial payment we did not see -> settles the item
                rest = REMAINDER.search(upper(f"{t.text} {t.remittance_info}"))
                return "partial", [x], {x[0]: (cents, "" if rest else "PARTIAL")}
        return None

    def disc_value(self, x):
        tm = self.terms.get(x[8])
        if not tm or x[3] != x[7] or x[4] != "ARI":
            return None
        return x[7] - rnd(x[7] * tm[0] / 100)

    def disc_exc(self, x, t):
        tm = self.terms[x[8]]
        late = (date.fromisoformat(t.booking_date) - date.fromisoformat(x[1])).days > tm[1] + 3
        return "PARTIAL" if late else "SHORT_DISCOUNT"

    # -- main loop ----------------------------------------------------------------------------------------------------
    def run(self):
        b = self.ctx.bank
        b = b[(b.amount > 0) & ~b.bank_txn_id.isin(self.ctx.claimed)].sort_values(["booking_date", "bank_txn_id"])
        for t in b.itertuples(index=False):
            fx = False
            oc = t.orig_currency if isinstance(t.orig_currency, str) else ""
            if FX and oc and oc != t.currency and t.orig_amount == t.orig_amount and t.orig_amount > 0:
                # the payer paid the invoices in their currency; the bank converted -> search on the original amount
                t = t._replace(currency=oc, amount=float(t.orig_amount))
                fx = True
            cents = int(round(t.amount * 100))
            cands = self.pr.resolve(t)
            if not cands:
                continue
            res = self.try_customers(t, cents, cands)
            if res is None:
                # a parent / group member paying for a subsidiary
                strong = [c for c, v in cands.items() if v[0] >= 3]
                ext = {}
                for c in strong:
                    for s in self.pr.expand_group(c):
                        ext.setdefault(s, (2, "group"))
                for c in cands:
                    ext.pop(c, None)
                if ext and len(ext) <= 20:
                    res = self.try_customers(t, cents, ext)
            if res is None:
                continue
            kind, its, excs = res
            method = "same_amount" if kind.startswith("same_amount") else "subset_sum"
            kind = kind + ("_fx" if fx else "")
            conf = self.conf if kind == "subset" else {"single": 0.9, "single_ref": 0.95, "fifo": 0.85, "discount": 0.9, "cn_pair": 0.9, "withholding": 0.85,
                                                       "same_amount_month": 0.9, "same_amount_fifo": 0.75}.get(kind, 0.7)
            key = f"{t.booking_date}|{t.bank_txn_id}"
            for x in its:
                amt, exc = excs.get(x[0], (x[3], ""))
                if exc == "" and x[4] == "ARI" and self.last_prior.get(x[0], "") > key:
                    exc = "PARTIAL"
                if fx and x[4] == "ARI" and exc in ("", "SHORT_DISCOUNT", "SHORT_WITHHOLDING", "SHORT_BANK_CHARGES",
                                                    "OVERPAYMENT"):
                    exc = "FX_DIFFERENCE"
                if exc == "PARTIAL" and amt < x[3]:
                    self.rem[x[0]] = x[3] - amt
                else:
                    self.used.add(x[0])
                self.rows[method].append((t.bank_txn_id, x[0], amt / 100, exc, conf, f"{method}:{kind}"))
        return self

    def try_customers(self, t, cents, cands: dict):
        sols = {}
        for c in cands:
            r = self.solve(c, t, cents)
            if r:
                sols[c] = r
        if not sols:
            return None
        if len(sols) > 1:   # an exact payment beats an explained difference
            tier = min(TIER.get(r[0], 0) for r in sols.values())
            sols = {c: r for c, r in sols.items() if TIER.get(r[0], 0) == tier}
        if len(sols) > 1:
            top = max(cands[c][0] for c in sols)
            sols = {c: r for c, r in sols.items() if cands[c][0] == top}
        if len(sols) > 1:
            # same items cannot belong to two customers; several customers with a solution -> ambiguous
            return None
        return next(iter(sols.values()))


def engine(ctx: Ctx) -> Engine:
    e = getattr(ctx, "_noref_engine", None)
    if e is None:
        e = Engine(ctx).run()
        ctx._noref_engine = e
    return e


def run(ctx: Ctx):
    rows = engine(ctx).rows[NAME]
    df = alloc_frame(rows)
    return df
