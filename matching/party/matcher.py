"""K3 party matcher (stage 3): who paid → which of its open items sum to the cash.

match(phase, lines, items, claimed, **ctx) per docs/PLAN_LUIGI.md. Precision first: a line is claimed only when the
payer is identified and a subset of its open items explains the amount (exactly, or with an explained shortfall).
"""
from __future__ import annotations

import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import party as P  # noqa: E402
from integration.lib.masters import read_master  # noqa: E402

OUT_COLS = ["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"]
WHT_COUNTRIES = {"MX", "BR", "MA"}
WHT_RATES = (0.0125, 0.015, 0.10)
MAX_SUBSET = 26          # candidates for meet-in-the-middle
WINDOW = 30
ADVANCE = os.environ.get("PARTY_ADVANCE", "1") == "1"  # round 2 (val +0.05): PDF advance deduction → PARTIAL
LOW_TRUST = ("party_subset_cn", "party_window_cn", "party_discount_late")  # not for history-free / 2nd pass
DUP_ANY = os.environ.get("PARTY_DUP_ANY", "1") == "1"  # round 2: duplicates without a 'sure' payer
FX_ORIG = os.environ.get("PARTY_FX_ORIG", "1") == "1"  # round 2: match on orig_amount/orig_currency
VARIANT_WINDOW = os.environ.get("PARTY_VARIANT_WINDOW", "1") == "1"  # round 2: window for discount/wht
WINDOW_CN = os.environ.get("PARTY_WINDOW_CN", "1") == "1"  # round 2 (val +0.06)
WINDOW_MODE = os.environ.get("PARTY_WINDOW_MODE", "nearest")  # A/B on val: nearest 91.38 vs latest 91.28
UNCITED_PARTIAL = False  # 55 % exact on val
GROUP_FALLBACK = False   # measured 0 % exact on val (parents go through refs)
DUP_WITH_OPEN = True
ANYWHERE = True          # round 2: master names anywhere in the text (history-free payers)
OWN_HARVEST = True
HARVEST = True           # round 2: aliases from this phase's claimed lines + second pass
FACTORY = P.FACTORY

_MONTHS = {}
for i, names in enumerate([
        "JANUARY JAN ENERO ENE JANEIRO JANVIER JANV GENNAIO GEN JANUAR STYCZEN",
        "FEBRUARY FEB FEBRERO FEVEREIRO FEV FEVRIER FEVR FEBBRAIO FEBRUAR LUTY",
        "MARCH MAR MARZO MARCO MARS MARZ MARZEC",
        "APRIL APR ABRIL ABR AVRIL AVR APRILE KWIECIEN",
        "MAY MAYO MAIO MAI MAGGIO MAG MAJ",
        "JUNE JUN JUNIO JUNHO JUIN GIUGNO GIU JUNI CZERWIEC",
        "JULY JUL JULIO JULHO JUILLET JUIL LUGLIO LUG JULI LIPIEC",
        "AUGUST AUG AGOSTO AGO AOUT AUGUSTUS SIERPIEN",
        "SEPTEMBER SEP SEPT SEPTIEMBRE SETEMBRO SET SEPTEMBRE SETTEMBRE WRZESIEN",
        "OCTOBER OCT OCTUBRE OUTUBRO OUT OCTOBRE OTTOBRE OTT OKTOBER PAZDZIERNIK",
        "NOVEMBER NOV NOVIEMBRE NOVEMBRO NOVEMBRE LISTOPAD",
        "DECEMBER DEC DICIEMBRE DIC DEZEMBRO DEZ DECEMBRE DICEMBRE DEZEMBER GRUDZIEN"], start=1):
    for n in names.split():
        _MONTHS[n] = i
_MONTH_RE = re.compile(r"(?:SERVICE|SERVICES|SERVICIOS?|MANT\.?|MANTENIMIENTO|CUOTA MANTENIMIENTO|SERVICOS?|SERVIZIO|"
                       r"SERVIZI|LEISTUNG|PRESTATION|ENTRETIEN|ABONNEMENT|USLUGA|MAINTENANCE)\s+"
                       r"([A-Z]+)\.?(?:\s+(\d{2,4}))?")


def service_month(text: str, booking_date: str):
    """(year, month) of the service month named in the text, or None."""
    t = P.norm(text).replace(" | ", "")
    for m in _MONTH_RE.finditer(t):
        mon = _MONTHS.get(m.group(1))
        if not mon:
            continue
        y = m.group(2)
        by, bm = int(booking_date[:4]), int(booking_date[5:7])
        if y:
            y = int(y)
            y = y + 2000 if y < 100 else y
            if y < 2015 or y > 2035:
                y = None
        if not y:
            y = by if mon <= bm else by - 1
        return y, mon
    return None


def _subset_mitm(vals: np.ndarray, target: int, tol: int):
    """All index subsets of vals (int cents, len ≤ MAX_SUBSET) summing to target ± tol. Returns list of masks
    (python ints over positions), capped at 8."""
    n = len(vals)
    h = n // 2
    a, b = vals[:h], vals[h:]

    def sums(v):
        m = len(v)
        s = np.zeros(1 << m, dtype=np.int64)
        for i in range(m):
            s[1 << i: 1 << (i + 1)] = s[: 1 << i] + v[i]
        return s

    sa, sb = sums(a), sums(b)
    order = np.argsort(sb, kind="stable")
    sbs = sb[order]
    lo = np.searchsorted(sbs, target - tol - sa, side="left")
    hi = np.searchsorted(sbs, target + tol - sa, side="right")
    hit = np.nonzero(hi > lo)[0]
    out = []
    for ia in hit:
        for j in range(lo[ia], hi[ia]):
            out.append(int(ia) | (int(order[j]) << h))
            if len(out) >= 8:
                return out
    return out


def _variant_subset(IT, cp, v, target, d, tol=2):
    """Subset of the payer's positive items (values v) summing to target ± tol: all of them (≤ MAX_SUBSET, FIFO prefix
    first) or, behind VARIANT_WINDOW, the recent window as in the exact search."""
    sel = _best_subset(v[:200], target, tol)
    if sel is not None or not VARIANT_WINDOW or len(cp) <= 2:
        return sel
    bd = np.datetime64(d)
    age = (bd - IT.issue_dt[cp]).astype(np.int64)
    due = (IT.due_dt[cp] - bd).astype(np.int64)
    w = np.nonzero((age <= 400) & (due <= 16) & (due >= -300))[0]
    if len(w) > WINDOW:
        w = np.sort(w[np.argsort(due[w], kind="stable")[-WINDOW:]])
    if not (2 < len(w) < len(cp) or len(cp) > MAX_SUBSET):
        return None
    s2 = _best_subset(v[w], target, tol, prefix=False)
    return None if s2 is None else [int(w[i]) for i in s2]


def _best_subset(vals: np.ndarray, target: int, tol: int, prefix: bool = True):
    """Unique subset or the FIFO-preferred one. vals sorted oldest first. Returns list of positions or None."""
    n = len(vals)
    if n == 0:
        return None
    # FIFO prefix / all
    single = np.nonzero(np.abs(vals - target) <= tol)[0]
    if len(single):
        return [int(single[0])]
    if prefix:
        cs = np.cumsum(vals)
        k = np.nonzero(np.abs(cs - target) <= tol)[0]
        if len(k):
            return list(range(int(k[0]) + 1))
    if n > MAX_SUBSET:
        return None
    masks = _subset_mitm(vals, target, tol)
    masks = [m for m in masks if m]
    if not masks:
        return None
    if len(masks) == 1:
        m = masks[0]
        return [i for i in range(n) if m >> i & 1]
    # several: same multiset of values (equal-amount invoices) → oldest; else ambiguous
    sets = [tuple(sorted(vals[[i for i in range(n) if m >> i & 1]].tolist())) for m in masks]
    if len(set(sets)) == 1:
        # FIFO: lexicographically oldest positions
        best = min(masks, key=lambda m: sorted(i for i in range(n) if m >> i & 1))
        return [i for i in range(n) if best >> i & 1]
    return None


class _Items:
    adv: dict = {}

    def __init__(self, items: pd.DataFrame, phase: str):
        it = items.reset_index(drop=True)
        self.id = it.item_id.values
        self.typ = it.item_type.values
        self.party = it.party_id.values
        self.ccy = it.currency.values
        self.issue = it.issue_date.values
        self.total = np.round(it.total_amount.astype(float).values * 100).astype(np.int64)
        self.open = np.round(it.open_amount.astype(float).values * 100).astype(np.int64)
        self.sign = np.where(np.isin(self.typ, ["ARC", "APC"]), -1, 1)
        self.ar = np.isin(self.typ, ["ARI", "ARC"])
        # payment terms (item, else customer master)
        pt = read_master("payment_terms.csv", phase)
        disc = {}
        for c, pct, d in zip(pt.code, pt.early_payment_discount_pct, pt.discount_days):
            if pct and d:
                disc[c] = (float(pct) / 100.0, int(d))
        terms = it.payment_terms.values.copy() if "payment_terms" in it else np.array([""] * len(it), dtype=object)
        cu = read_master("customers.csv", phase)
        cterms = dict(zip(cu.customer_id, cu.payment_terms))
        self.country = dict(zip(cu.customer_id, cu.country))
        self.disc_pct = np.zeros(len(it))
        self.disc_days = np.zeros(len(it), dtype=np.int64)
        for i in range(len(it)):
            t = terms[i] or cterms.get(self.party[i], "")
            if t in disc:
                self.disc_pct[i], self.disc_days[i] = disc[t]
        self.issue_dt = pd.to_datetime(pd.Series(self.issue), errors="coerce").values.astype("datetime64[D]")
        self.due_dt = pd.to_datetime(it.due_date, errors="coerce").values.astype("datetime64[D]")
        self.idx = {i: k for k, i in enumerate(self.id)}
        self.docd = np.array([re.sub(r"\D", "", x).lstrip("0") for x in it.document_number.values], dtype=object)
        by = defaultdict(list)
        for k, (p, d) in enumerate(zip(self.party, self.issue)):
            by[p].append(k)
        self.by_party = {p: np.array(sorted(v, key=lambda k: (self.issue[k], self.id[k])), dtype=np.int64)
                         for p, v in by.items()}
        self.settled_on = {}  # item index -> booking_date it was fully settled (for duplicates)

    def apply(self, k: int, cents: int, exc: str, date: str):
        if exc == "RETURNED":
            self.open[k] += cents
        elif exc == "DUPLICATE_PAYMENT":
            pass
        else:
            self.open[k] = max(0, self.open[k] - cents)
            if self.open[k] <= 0:
                self.settled_on[k] = date


def _cutoffs(phase: str):
    if phase == "val":
        return "2023-07-01", "2025-01-01"
    if phase in ("sample",):
        return "2023-06-01", "2025-06-01"
    return "2023-07-01", "2025-07-01"


def match(phase: str, lines: pd.DataFrame, items: pd.DataFrame, claimed: pd.DataFrame, **ctx) -> pd.DataFrame:
    log = ctx.get("log", print)
    t0 = time.time()
    start, cutoff = _cutoffs(phase)
    loeo = ctx.get("loeo") or os.environ.get("PARTY_LOEO", "")
    aliases = P.learn_aliases(phase, start, cutoff, log=log, loeo=loeo)
    P.NO_PREFIX.clear()
    if loeo:
        P.NO_PREFIX.add(loeo)
    gm = P.group_map(phase)
    members = P.group_members(phase)
    R = P.Resolver(aliases, members)
    IT = _Items(items, phase)
    IT.adv = {}
    pdf = ctx.get("pdf_clauses")
    if ADVANCE and pdf is not None and len(pdf) and "advance_amount" in pdf:
        a = pdf[pdf.advance_amount.notna() & pdf.document_type.isin(["invoice"])]
        key = {(p_, d_): k for k, (p_, d_) in enumerate(zip(IT.party, items.reset_index(drop=True).document_number))}
        for p_, d_, adv in zip(a.party_id, a.document_number, a.advance_amount):
            k = key.get((p_, d_))
            if k is not None:
                IT.adv[k] = int(round(float(adv) * 100))
        log(f"party: {len(IT.adv):,} invoices with an advance to deduct (PDF)")
    # invoice party names of the phase's items (pre-rename names, new parties)
    ent_of = dict(zip(items.party_id, items.entity))
    if "party_name" in items:
        pn = items[["party_id", "party_name", "entity"]].drop_duplicates(["party_id", "party_name"])
        for pid, nm, ent in zip(pn.party_id, pn.party_name, pn.entity):
            for e2 in P._scopes(ent, pid.startswith("S")):
                for k in P.master_keys(nm):
                    R.add(e2 + "|" + k, pid, gm.get(pid, pid))
    log(f"party: resolver ready ({time.time() - t0:.0f}s)")

    L = lines.copy()
    L["amt_c"] = np.round(L.amount.astype(float) * 100).astype(np.int64)
    L = L.sort_values(["booking_date", "bank_txn_id"]).reset_index(drop=True)
    # claimed events in booking order
    cl = claimed.copy() if claimed is not None and len(claimed) else pd.DataFrame(columns=OUT_COLS + ["booking_date"])
    if len(cl):
        if "booking_date" not in cl:
            cl["booking_date"] = ""  # unknown → applied before every line
        cl["booking_date"] = cl.booking_date.fillna("")
        cl = cl[cl.item_id.fillna("") != ""]
        cl = cl.sort_values(["booking_date", "bank_txn_id"]).reset_index(drop=True)
    cl_dates = cl.booking_date.values if len(cl) else np.array([])
    cl_item = cl.item_id.values if len(cl) else np.array([])
    cl_amt = np.round(cl.amount.astype(float).values * 100).astype(np.int64) if len(cl) else np.array([])
    cl_exc = cl.exception.fillna("").values if len(cl) else np.array([])
    ci = 0

    # ---- round 2: history-free names (masters anywhere in text) + runtime harvest (flags)
    anywhere = _flag(ctx, "anywhere", ANYWHERE)
    harvest = _flag(ctx, "harvest", HARVEST)
    A = None
    if anywhere:
        cu = read_master("customers.csv", phase)
        su = read_master("suppliers.csv", phase)
        names = [(e, pid, nm) for pid, nm, e in zip(cu.customer_id, cu.name, cu.entity)]
        names += [(e, pid, nm) for pid, nm, e in zip(cu.customer_id, cu.trade_name, cu.entity) if nm]
        for pid, nm, e in zip(su.supplier_id, su.name, su.entity):
            for e2 in P._scopes(e, True):
                names.append((e2, pid, nm))
        if "party_name" in items:
            names += [(e2, pid, nm) for pid, nm, e in zip(pn.party_id, pn.party_name, pn.entity)
                      for e2 in P._scopes(e, pid.startswith("S"))]
        A = P.Anywhere(names)
    if harvest and len(cl):
        nh = _harvest(R, cl, items, phase, ctx)
        log(f"party: harvested {nh:,} alias keys from {cl.bank_txn_id.nunique():,} claimed lines")

    out = []
    dbg = {}
    stats = defaultdict(int)
    failed = []
    for r in L.itertuples(index=False):
        d = r.booking_date
        while ci < len(cl_dates) and cl_dates[ci] <= d:
            k = IT.idx.get(cl_item[ci])
            if k is not None:
                IT.apply(k, int(cl_amt[ci]), cl_exc[ci], cl_dates[ci])
            ci += 1
        if int(r.amt_c) == 0:
            continue
        if not _one(r, R, A, IT, gm, members, out, dbg, stats, harvest):
            failed.append(r)
    if harvest and failed:  # second pass with the aliases harvested from this phase's own claims
        n0 = stats["claimed"]
        for r in failed:
            _one(r, R, A, IT, gm, members, out, dbg, stats, harvest, second=True)
        log(f"party: second pass claimed {stats['claimed'] - n0:,} of {len(failed):,} lines")
    dbg = [(t,) + v for t, v in dbg.items()]
    log(f"party: {dict(stats)} ({time.time() - t0:.0f}s)")
    try:
        os.makedirs(P.CACHE, exist_ok=True)
        pd.DataFrame(dbg, columns=["bank_txn_id", "party_id", "group_id", "key", "status"]).to_parquet(
            os.path.join(P.CACHE, f"debug_{phase}.parquet"))
    except Exception as e:  # noqa: BLE001
        log(f"party: debug dump failed: {e}")
    return pd.DataFrame(out, columns=OUT_COLS)


def _flag(ctx, name, default):
    v = ctx.get("party_" + name)
    if v is None:
        v = os.environ.get("PARTY_" + name.upper())
        if v is not None:
            v = v not in ("", "0", "false", "False")
    return default if v is None else bool(v)


STRONG_HARVEST = ("party_single", "party_subset", "party_window", "party_discount", "party_wht", "party_month",
                  "party_fifo", "party_subset_cn")


def _harvest(R, cl, items, phase, ctx):
    """Alias keys of lines claimed earlier in this phase (by any stage) → the single party of their items."""
    pid = dict(zip(items.item_id, items.party_id))
    c = cl[["bank_txn_id", "item_id"]].copy()
    c["party_id"] = c.item_id.map(pid)
    lp = c.dropna().groupby("bank_txn_id").party_id.agg(["nunique", "first"])
    lp = lp[lp["nunique"] == 1]["first"]
    if not len(lp):
        return 0
    f = os.path.join(ctx.get("data_dir") or P.DATA, phase, "bank_transactions.csv.gz")
    b = pd.read_csv(f, dtype=str, keep_default_na=False, usecols=["bank_txn_id", "entity", "text",
                                                                   "counterparty_name", "counterparty_account"])
    b = b[b.bank_txn_id.isin(lp.index)]
    n = 0
    for t, e, txt, cpn, cpa in zip(b.bank_txn_id, b.entity, b.text, b.counterparty_name, b.counterparty_account):
        p = lp[t]
        for k, _ in P.keys_of(txt, cpn, cpa):
            R.add(e + "|" + k, p, purity=0.0)
            n += 1
    return n


def _one(r, R, A, IT, gm, members, out, dbg, stats, harvest, second=False):
    """Try one line; appends rows to out and applies them. True when claimed."""
    amt = int(r.amt_c)
    d = r.booking_date
    cands = R.resolve(r.entity, r.text, r.counterparty_name, r.counterparty_account)
    extra = []
    if A is not None:
        seen = {c_[0] for c_ in cands}
        for ib in P.ibans_in(r.text + " " + r.remittance_info):
            for p_, n_, pur_ in R.p.get(r.entity + "|I:" + ib, []):
                if p_ not in seen:
                    seen.add(p_)
                    extra.append((p_, 95, pur_, "I*:" + ib))
        for p_, ov in A.find(r.entity, r.text, r.counterparty_name, r.remittance_info):
            if p_ not in seen:
                seen.add(p_)
                extra.append((p_, ov, 0.0, "A:" + p_))
    if not cands and not extra:
        if not second:
            stats["no_party"] += 1
            dbg[r.bank_txn_id] = ("", "", "", "no_party")
        return False
    rows = None
    p0, rank0, pur0, key = (cands or extra)[0]
    for p, rank, pur, k in cands:  # strong fits on each candidate payer
        rows = _fit(IT, [p], r, amt, d, weak=False)
        if rows:
            key = k
            break
    if not rows and extra:  # history-free candidates: accepted only when exactly one of them fits
        fits = [(k, f_) for p, rank, pur, k in extra[:4] for f_ in [_fit(IT, [p], r, amt, d, weak=False)] if f_]
        fits = [f_ for f_ in fits if not f_[1][0][3].startswith(LOW_TRUST)]
        if len(fits) == 1:
            key = fits[0][0]
            rows = [(k_, c_, e_, m_ + "_any") for k_, c_, e_, m_ in fits[0][1]]
    gid = gm.get(p0, p0)
    if not rows and GROUP_FALLBACK and len(members.get(gid, [])) > 1:  # parent / group pays for its members
        rows = _fit(IT, members[gid], r, amt, d, weak=False)
        if rows:
            rows = [(k, c_, e, m + "_group") for k, c_, e, m in rows]
    if not rows and cands:
        sure = pur0 >= 0.9 and (len(cands) == 1 or cands[1][1] < rank0)
        rows = _weak_cited(IT, [c_[0] for c_ in cands], r, amt)
        if not rows and sure:
            rows = _fit(IT, [p0], r, amt, d, weak=True)
        if not rows and DUP_ANY:  # duplicate of a recently settled item: exactly one candidate payer has one
            ds = [x for x in (_duplicate(IT, [c_[0]], r, abs(amt), d, amt > 0) for c_ in cands[:4]) if x]
            if len(ds) == 1:
                rows = [(k_, c_, e_, m_ + "_any") for k_, c_, e_, m_ in ds[0]]
    if not rows:
        if not second:
            stats["no_fit"] += 1
            dbg[r.bank_txn_id] = (p0, gid, key, "no_fit")
        return False
    if second:
        if rows[0][3].startswith(LOW_TRUST + ("party_subset",)):  # measured 0–69 % exact in the 2nd pass
            return False
        rows = [(k_, c_, e_, m_ + "_2nd") for k_, c_, e_, m_ in rows]
    dbg[r.bank_txn_id] = (IT.party[rows[0][0]], gid, key, rows[0][3])
    stats["claimed"] += 1
    for k, cents, exc, meth in rows:
        out.append((r.bank_txn_id, IT.id[k], cents / 100.0, exc, 0.9, meth))
        IT.apply(k, cents, exc, d)
    if harvest and OWN_HARVEST and rows[0][3].startswith(STRONG_HARVEST):
        ps = {IT.party[k] for k, *_ in rows}
        if len(ps) == 1:
            p = ps.pop()
            for k, _ in P.keys_of(r.text, r.counterparty_name, r.counterparty_account):
                R.add(r.entity + "|" + k, p, purity=0.0)
    return True


def _fit(IT: _Items, parties, r, amt: int, d: str, weak: bool = False):
    """[(item_index, cents, exception, method)] or None."""
    receipt = amt > 0
    target = abs(amt)
    cand = []
    for p in parties:
        a = IT.by_party.get(p)
        if a is not None:
            cand.append(a)
    if not cand:
        return None
    c = np.concatenate(cand) if len(cand) > 1 else cand[0]
    side = IT.ar[c] if receipt else ~IT.ar[c]
    ccy = r.currency
    if FX_ORIG and r.orig_currency and r.orig_currency != r.currency and r.orig_amount:
        try:  # the bank shows the instructed amount in the invoice currency: match on that
            target = int(round(abs(float(r.orig_amount)) * 100))
            ccy = r.orig_currency
        except ValueError:
            pass
    m = side & (IT.open[c] > 0) & (IT.ccy[c] == ccy) & (IT.issue[c] <= d)
    c = c[m]
    if len(c) == 0:
        return _duplicate(IT, parties, r, target, d, receipt) if weak else None
    if weak:
        vals = IT.open[c] * IT.sign[c]
        if len(c) == 1 and vals[0] > 0:
            o = int(vals[0])
            if target < o and UNCITED_PARTIAL:
                return [(int(c[0]), target, "PARTIAL", "party_partial")]
            if o < target <= 1.1 * o:
                return [(int(c[0]), o, "OVERPAYMENT", "party_overpay")]
        return _duplicate(IT, parties, r, target, d, receipt) if DUP_WITH_OPEN else None
    if len(parties) > 1:
        c = c[np.lexsort((IT.id[c], IT.issue[c]))]
    vals = IT.open[c] * IT.sign[c]
    # 1. single exact item(s)
    eq = np.nonzero(vals == target)[0]
    if len(eq):
        pick = eq[0]
        if len(eq) > 1:
            sm = service_month(r.text + " " + r.remittance_info, d)
            if sm:
                y, mo = sm
                y2, mo2 = (y + 1, 1) if mo == 12 else (y, mo + 1)
                want = f"{y2:04d}-{mo2:02d}"
                hit = [e for e in eq if IT.issue[c[e]][:7] == want]
                if hit:
                    return [(int(c[hit[0]]), int(IT.open[c[hit[0]]]), "", "party_month")]
                return None
            return [(int(c[pick]), int(IT.open[c[pick]]), "", "party_fifo")]
        return [(int(c[pick]), int(IT.open[c[pick]]), "", "party_single")]
    # 2. exact subset (positives only first, then with credit notes)
    pos = vals > 0
    cp = c[pos]
    if len(cp) <= MAX_SUBSET or True:
        sel = _best_subset(vals[pos][:200], target, 0) if len(cp) else None
        if sel is not None:
            return [(int(cp[i]), int(IT.open[cp[i]]), "", "party_subset") for i in sel]
    bd = np.datetime64(d)
    if len(cp) > 2:  # recent window: not stale, due soon; the most recent WINDOW of them
        age = (bd - IT.issue_dt[cp]).astype(np.int64)
        due = (IT.due_dt[cp] - bd).astype(np.int64)
        w = np.nonzero((age <= 400) & (due <= 16) & (due >= -300))[0]
        if len(w) > WINDOW:  # keep the WINDOW latest by due date
            if WINDOW_MODE == "nearest":  # nearest to the booking date
                w = np.sort(w[np.argsort(np.abs(due[w]), kind="stable")[:WINDOW]])
            else:  # latest due
                w = np.sort(w[np.argsort(due[w], kind="stable")[-WINDOW:]])
        if 2 < len(w) < len(cp) or len(cp) > MAX_SUBSET:
            sel = _best_subset(vals[pos][w], target, 0, prefix=False)
            if sel is not None:
                return [(int(cp[w[i]]), int(IT.open[cp[w[i]]]), "", "party_window") for i in sel]
    if (~pos).any() and len(c) <= MAX_SUBSET:
        sel = _best_subset(vals, target, 0)
        if sel is not None:
            return [(int(c[i]), int(IT.open[c[i]]), "", "party_subset_cn") for i in sel]
    if WINDOW_CN and (~pos).any() and len(c) > MAX_SUBSET and len(cp):  # recent window + recent credit notes
        cn = c[~pos]
        cn = cn[(bd - IT.issue_dt[cn]).astype(np.int64) <= 400][-4:]
        age = (bd - IT.issue_dt[cp]).astype(np.int64)
        due = (IT.due_dt[cp] - bd).astype(np.int64)
        w = np.nonzero((age <= 400) & (due <= 16) & (due >= -300))[0]
        k = WINDOW - len(cn)
        if len(w) > k:
            w = np.sort(w[np.argsort(due[w], kind="stable")[-k:]])
        if len(cn) and len(w):
            idx = np.concatenate([cp[w], cn])
            v2 = IT.open[idx] * IT.sign[idx]
            sel = _best_subset(v2, target, 0, prefix=False)
            if sel is not None and any(IT.sign[idx[i]] < 0 for i in sel):
                return [(int(idx[i]), int(IT.open[idx[i]]), "", "party_window_cn") for i in sel]
    # 2b. invoice whose PDF says an advance must be deducted: cash = open − advance → PARTIAL (amount = cash)
    if IT.adv:
        for i, k in enumerate(c):
            a_ = IT.adv.get(int(k))
            if a_ and IT.open[k] == IT.total[k] and abs(int(IT.open[k]) - a_ - target) <= 1:
                return [(int(k), target, "PARTIAL", "party_advance")]
    # 3. early-payment discount within terms (+3 days grace): cash = Σ total·(1−pct) on eligible items
    bd = np.datetime64(d)
    dp = IT.disc_pct[c]
    if (dp > 0).any():
        within = (dp > 0) & ((bd - IT.issue_dt[c]).astype(np.int64) <= IT.disc_days[c] + 3) & \
                 (IT.open[c] == IT.total[c]) & (IT.sign[c] > 0)
        if within.any():
            dv = vals.copy()
            dv[within] = IT.total[c][within] - np.round(IT.total[c][within] * dp[within]).astype(np.int64)
            sel = _variant_subset(IT, cp, dv[pos], target, d) if pos.any() else None
            if sel is not None:
                return [(int(cp[i]), int(IT.open[cp[i]]), "SHORT_DISCOUNT" if within[pos][i] else "",
                         "party_discount") for i in sel]
        late = (dp > 0) & ~within & (IT.open[c] == IT.total[c]) & (IT.sign[c] > 0)
        if late.any():
            dv = vals.copy()
            dv[late] = IT.total[c][late] - np.round(IT.total[c][late] * dp[late]).astype(np.int64)
            sel = _variant_subset(IT, cp, dv[pos], target, d) if pos.any() else None
            if sel is not None:
                lp = late[pos]
                return [(int(cp[i]), int(dv[pos][i]) if lp[i] else int(IT.open[cp[i]]),
                         "PARTIAL" if lp[i] else "", "party_discount_late") for i in sel]
    # 4. withholding (MX/BR/MA customers): cash = Σ total·(1−r)
    if receipt and IT.country.get(IT.party[c[0]], "") in WHT_COUNTRIES:
        full = (IT.open[c] == IT.total[c]) & (IT.sign[c] > 0)
        for rate in WHT_RATES:
            dv = vals.copy()
            dv[full] = IT.total[c][full] - np.round(IT.total[c][full] * rate).astype(np.int64)
            sel = _variant_subset(IT, cp, dv[pos], target, d) if pos.any() else None
            if sel is not None:
                fp = full[pos]
                return [(int(cp[i]), int(IT.open[cp[i]]), "SHORT_WITHHOLDING" if fp[i] else "", "party_wht")
                        for i in sel]
    return None


def _weak_cited(IT, parties, r, amt):
    """Exactly one item of the candidate payers is cited: partial / overpayment / duplicate on it."""
    receipt = amt > 0
    target = abs(amt)
    cited = _cited(IT, parties, r, receipt)
    if len(cited) != 1:
        return None
    k = cited[0]
    o = int(IT.open[k])
    if o > 0 and target < o:
        return [(k, target, "PARTIAL", "party_cited_partial")]
    if o > 0 and o < target <= 1.5 * o:
        return [(k, o, "OVERPAYMENT", "party_cited_overpay")]
    if o == 0 and IT.total[k] == target and k in IT.settled_on:
        return [(k, target, "DUPLICATE_PAYMENT", "party_cited_duplicate")]
    return None


_RUN = re.compile(r"\d{4,}")


def _cited(IT, parties, r, receipt):
    """Item indices of the payer whose document number is cited in the text / remittance info."""
    runs = {x.lstrip("0") for x in _RUN.findall(r.text + " " + r.remittance_info + " " + r.ref1)}
    runs = {x for x in runs if len(x) >= 4}
    if not runs:
        return []
    out = []
    for p in parties:
        a = IT.by_party.get(p)
        if a is None:
            continue
        for k in a:
            if IT.ar[k] != receipt or IT.sign[k] < 0 or IT.ccy[k] != r.currency:
                continue
            dd = IT.docd[k]
            if dd and any(x == dd or (len(x) >= 5 and dd.endswith(x)) for x in runs):
                out.append(int(k))
    return out


def _duplicate(IT, parties, r, target, d, receipt):
    """Party has nothing open: an item of the party settled in the last 90 days with exactly this amount."""
    best = None
    for p in parties:
        a = IT.by_party.get(p)
        if a is None:
            continue
        for k in a:
            if IT.ar[k] != receipt or IT.ccy[k] != r.currency:
                continue
            s = IT.settled_on.get(int(k))
            if s and s <= d and IT.total[k] == target and (np.datetime64(d) - np.datetime64(s)).astype(int) <= 90:
                if best is None or s > IT.settled_on[best]:
                    best = int(k)
    if best is not None:
        return [(best, target, "DUPLICATE_PAYMENT", "party_duplicate")]
    return None
