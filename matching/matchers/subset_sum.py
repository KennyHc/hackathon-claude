"""Customer receipts without a usable reference: who paid (matching/parties.py) + which of the payer's open items sum
to the amount.  Also hosts the shared engine used by `same_amount` (both are computed in ONE chronological pass so the
items consumed by one line are not offered to a later one).

Per line (incoming, not claimed by an earlier matcher), in booking-date order:
1. payer candidates = PartyResolver (IBAN / name / learned aliases / ORD. / EXPTE); then their subsidiaries / group.
2. pool = the candidate's open AR items in the line's entity & currency, issued on/before the booking date, not used.
3. solutions, in this order:
   - one invoice with exactly the amount: unique -> subset_sum; several with the same amount -> same_amount
     (month named in the text -> the invoice of that service month, issued at the start of the following month;
     else the oldest = FIFO);
   - oldest-first prefix of the open invoices summing to the amount (FIFO);
   - exact subset (meet-in-the-middle over the payer's recent invoices) that is unique and whose "chance" of being a
     coincidence (density of subset sums around the amount) is tiny.
4. claim only when exactly one candidate customer has a solution (precision first).
"""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd

from matching.common import Ctx, alloc_frame
from matching.parties import PartyResolver, upper

NAME = "subset_sum"

MITM_MAX = 30          # max pool size for the exhaustive search
MITM_RECENT_DAYS = 400  # exhaustive search only over invoices issued in the last N days
DUE_STAGES = (0, 20)    # exhaustive pools: invoices due by booking date + N days
SPURIOUS_MAX = 0.3      # max expected number of coincidental subsets hitting the amount
SPURIOUS_TIE = -1.0     # same, when several exact subsets exist and the oldest one is taken
CN_GUARD = ""
DISCOUNT = True         # claim one invoice paid net of its early-payment discount
CN_RECENT_DAYS = 120    # open credit notes issued in the last N days take part (negative) in the exhaustive search
TIE_MAX = 8             # max number of exact subsets to choose from
WINDOW = 20000          # +-cents window used to estimate the subset-sum density

# ---------------------------------------------------------------------------------------------------------------------
# month named in the text (service month)
MONTHS = {
    1: "JANUARY JAN ENERO ENE JANUAR JANVIER JANV GENNAIO GEN JANEIRO STYCZEN STY JANUARI",
    2: "FEBRUARY FEB FEBRERO FEBRUAR FEVRIER FEV FEVR FEBBRAIO FEVEREIRO LUTY LUT FEBRUARI",
    3: "MARCH MAR MARZO MAERZ MARZ MARS MARCO LUTEGO MAART",
    4: "APRIL APR ABRIL ABR AVRIL AVR APRILE KWIECIEN KWI",
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


# ---------------------------------------------------------------------------------------------------------------------
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
        self.pool: dict[str, list] = {}
        for r in it.itertuples(index=False):
            left = r.open_amount - taken.get(r.item_id, 0.0)
            if left < 0.005:
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
        self.diag: dict = {}
        self.conf = 0.8
        self.exc: dict = {}     # item_id -> (amount cents, exception) when not the plain open amount
        self.rows = {"subset_sum": [], "same_amount": []}

    # -- candidates ---------------------------------------------------------------------------------------------------
    def items(self, cid: str, t) -> list:
        return [x for x in self.pool.get(cid, ()) if x[0] not in self.used and x[5] == t.entity
                and x[6] == t.currency and x[1] <= t.booking_date]

    def solve(self, cid: str, t, cents: int):
        r = self._solve(cid, t, cents)
        if r is None or not CN_GUARD:
            return r
        # a credit note of the payer that cancels an open invoice of the same amount: the payment may well include the
        # pair (net zero) -> the found subset is not reliable
        sel = {x[0] for x in r[1]}
        its = self.items(cid, t)
        inv_amts = {x[3] for x in its if x[4] == "ARI" and x[0] not in sel}
        if any(x[4] == "ARC" and x[3] in inv_amts for x in its):
            return (r[0] + "+cn", r[1]) if CN_GUARD == "tag" else None
        return r

    def _solve(self, cid: str, t, cents: int):
        """-> (kind, [items]) or None. kind in single / same_amount / fifo / subset."""
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
            tm = self.terms.get(x[8])
            if tm and x[3] == x[7]:
                cash = x[7] - int(round(x[7] * tm[0] / 100))
                if abs(cash - cents) <= 1:
                    late = (date.fromisoformat(t.booking_date) - date.fromisoformat(x[1])).days > tm[1] + 3
                    disc.append((x, "PARTIAL" if late else "SHORT_DISCOUNT"))
        if disc:
            if len(disc) == 1 and DISCOUNT:
                x, exc = disc[0]
                self.exc[x[0]] = (cents if exc == "PARTIAL" else x[3], exc)
                return "discount", [x]
            return None
        # one invoice net of its own credit note (credit_note.related_document = the invoice)
        by_doc = {x[9]: x for x in inv}
        pairs = [(by_doc[c[10]], c) for c in its if c[4] == "ARC" and c[10] in by_doc
                 and by_doc[c[10]][3] - c[3] == cents]
        if len(pairs) == 1:
            return "cn_pair", list(pairs[0])
        # FIFO prefix of invoices
        acc = 0
        for k, x in enumerate(inv):
            acc += x[3]
            if acc == cents and k >= 1:
                return "fifo", inv[:k + 1]
            if acc > cents:
                break
        # exhaustive search over the recent invoices: first those already due at the booking date, then those due soon
        lim_iss = str(date.fromisoformat(t.booking_date) - timedelta(days=MITM_RECENT_DAYS))
        lim_cn = str(date.fromisoformat(t.booking_date) - timedelta(days=CN_RECENT_DAYS))
        cns = [x for x in its if x[4] == "ARC" and x[1] >= lim_cn][-6:] if CN_RECENT_DAYS else []
        prev = None
        for slack in DUE_STAGES:
            lim_due = str(date.fromisoformat(t.booking_date) + timedelta(days=slack))
            cand = [x for x in inv if x[1] >= lim_iss and x[2] <= lim_due and x[3] <= cents][:MITM_MAX]
            if not cand or len(cand) + len(cns) < 2 or sum(x[3] for x in cand) < cents or len(cand) == prev:
                continue
            prev = len(cand)
            cand = cand + cns[:max(0, MITM_MAX + 4 - len(cand))]
            cnt, sols, dens = mitm([-x[3] if x[4] == "ARC" else x[3] for x in cand], cents, max_sols=TIE_MAX)
            self.diag[(t.bank_txn_id, cid)] = (len(cand), cnt, dens)
            if cnt == 1 and dens <= SPURIOUS_MAX and len(sols[0]) >= 2 and any(cand[i][4] == "ARI" for i in sols[0]):
                self.conf = 0.85 if dens <= 0.01 else 0.75 if dens <= 0.1 else 0.6
                return "subset", [cand[i] for i in sols[0]]
            if 2 <= cnt <= TIE_MAX and dens <= SPURIOUS_TIE:
                # several exact subsets: the customer pays its OLDEST invoices -> the subset whose newest is oldest
                keys = [(max(sl), -min(sl), len(sl)) for sl in sols]
                best = min(keys)
                if keys.count(best) == 1:
                    sl = sols[keys.index(best)]
                    if len(sl) >= 2:
                        return "subset_tie", [cand[i] for i in sl]
            if cnt >= 1:
                break   # a wider pool cannot make the solution unique
        return None

    # -- main loop ----------------------------------------------------------------------------------------------------
    def run(self):
        b = self.ctx.bank
        b = b[(b.amount > 0) & ~b.bank_txn_id.isin(self.ctx.claimed)].sort_values(["booking_date", "bank_txn_id"])
        for t in b.itertuples(index=False):
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
            kind, its = res
            method = "same_amount" if kind.startswith("same_amount") else "subset_sum"
            conf = self.conf if kind == "subset" else {"single": 0.9, "single_ref": 0.95, "fifo": 0.85, "discount": 0.9, "cn_pair": 0.9,
                                                       "same_amount_month": 0.9, "same_amount_fifo": 0.75}.get(kind, 0.7)
            for x in its:
                self.used.add(x[0])
                amt, exc = self.exc.pop(x[0], (x[3], ""))
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
