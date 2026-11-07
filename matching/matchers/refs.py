"""refs matcher: customer transfers (AR side) identified by payer + document references + exact amounts.

    python -m matching.matchers.refs --phase val

Pipeline per bank line (credit, not claimed), lines processed in date order with a within-run item ledger:
  1. payer candidates (`matching.lib.party`): IBAN, customer/trade/invoice-printed name (truncations), learned aliases (train < cut),
     plus the owner of any of OUR document numbers quoted in the text (reference evidence).
  2. for each candidate: the customer's open items (issue <= booking, open > 0, same currency, ledger-adjusted) and the references
     found in the text (`refs_text.find_refs`: prefix/no-prefix, leading zeros, typos, truncation, N43 splits, customer PO).
  3. item selection (`refs_solve`), first rule that gives ONE canonical solution wins:
       ref_exact        subset of the quoted documents sums to the cash (ref_exact_amb: several, best evidenced)
       ref_short_*      quoted invoices paid short: discount (on time: full amounts; _late: PARTIAL at discounted cash),
                        withholding (POLICIES rates), charges (<= 2 % and <= 60): full open amounts, exception left to the engine
       ref_fill         quoted documents (strong) + a unique completion from the other open items
       partial          one quoted open invoice, cash clearly below it (PARTIAL, amount = cash)
       party_subset     unique subset of the payer's open items (credit notes netted, CN must come with its invoice); FIFO
                        among identical amounts; _month: text names the service month -> invoice issued the month after
       party_oldest     several subsets fit: the oldest-first one (low confidence)
     suffix _r2/_r3: risk tier = expected number of coincidental subset sums (#subsets of size <= k / cash in cents) >= 0.03 / 0.3
     suffix _grp: solved on the payer's group (parent pays for children); only ref_exact is accepted there.
  4. two candidate payers with different solutions -> the better evidenced one, else no claim.
Confidence = f(method, evidence, #open items): the orchestrator gates by `method`.  `claimed` lines are never touched.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from matching.lib import data, evalutil, party as partylib, refs_text as rt, refs_solve as rs

CUT = {"val": "2025-01-01", "sample": "2025-01-01"}
DEFAULT_CUT = "2025-07-01"  # live / final: learn on all of train
DROP = ("_r3", "ref_fill_oldest")  # measured far below the precision gate: never emitted
LEDGER_MIN_CONF = 0.6
PARTIAL_WORDS = ("A CUENTA", "ACUENTA", "ACCOUNT", "PARTIAL", "BALANCE", "ACOMPTE", "ACCONTO", "SALDO", "ANTICIPO", "ON ACC", "PARCIAL", "ABONO")


def _cents(x) -> int:
    return int(round(float(x) * 100))


class _Pool:
    """open items of one payer at one booking date (ledger adjusted)."""
    __slots__ = ("idx", "cents", "docs", "pos", "typ", "issue")


def _related_docs() -> dict:
    """credit note item_id -> related invoice number, from every credit_notes file available (inputs, not labels)."""
    out = {}
    for ph in ("train", "live", "final"):
        f = os.path.join(data.phase_dir(ph), "credit_notes.csv.gz")
        if os.path.exists(f):
            d = pd.read_csv(f, usecols=["item_id", "related_document"], dtype=str)
            out.update(dict(zip(d.item_id, d.related_document)))
    return out


def _load_items(phase: str):
    cols = ["item_id", "item_type", "side", "entity", "party_id", "document_number", "issue_date", "currency", "open_amount",
            "total_amount", "po_number", "related_document", "payment_terms"]
    cp = os.path.join(data.CACHE, phase, "items.parquet")
    if not os.path.exists(cp):
        data.items(phase)  # builds the cache
    it = pd.read_parquet(cp, columns=cols, filters=[("side", "==", "AR")])
    it = it[it.item_type.isin(["ARI", "ARC"])].copy()
    it = it[it.open_amount > 0.004]
    rd = _related_docs()
    miss = it.related_document.isna() & (it.item_type == "ARC")
    if miss.any():
        it.loc[miss, "related_document"] = it.loc[miss, "item_id"].map(rd)
    it["cents"] = np.round(it.open_amount.astype(float) * 100).astype("int64")
    it = it.sort_values(["issue_date", "item_id"]).reset_index(drop=True)
    # early-payment discount terms (invoice terms, else the customer's master terms)
    cu = data.customers().set_index("customer_id").payment_terms
    terms = it.payment_terms.fillna(it.party_id.map(cu))
    pt = data.payment_terms().set_index("code")
    it["disc_pct"] = terms.map(pt.early_payment_discount_pct).fillna(0.0).astype(float).values
    it["disc_days"] = terms.map(pt.discount_days).fillna(0.0).astype(float).values
    it["full_cents"] = np.round(it.total_amount.astype(float) * 100).astype("int64")
    return it


def match(phase: str, claimed: set | None = None, cut: str | None = None, verbose: bool = False, max_lines: int | None = None,
          prior: pd.DataFrame | None = None, p_items: pd.Series | None = None, p_min: float = 0.1,
          diag: dict | None = None) -> pd.DataFrame:
    """prior: allocations already made by higher-priority matchers (their items' amounts are consumed before we start);
    p_items: classification's P(line settles items) by bank_txn_id; lines below p_min are skipped."""
    t0 = time.time()
    claimed = set(claimed or ())
    if p_items is not None:
        claimed |= set(p_items.index[p_items.values < p_min])
    cut = cut or CUT.get(phase, DEFAULT_CUT)
    bank = data.bank(phase)
    b = bank[(bank.amount > 0) & ~bank.bank_txn_id.isin(claimed)].sort_values(["booking_date", "bank_txn_id"])
    if max_lines:
        b = b.head(max_lines)
    items = _load_items(phase)
    px = partylib.PartyIndex(phase, cut=cut)
    if verbose:
        print(f"[refs] index+learning {time.time() - t0:.0f}s; {len(b)} credit lines, {len(items)} open AR items")

    # ledger and lookups
    item_id = items.item_id.values
    cents0 = items.cents.values.copy()
    if prior is not None and len(prior):
        pos_of = pd.Series(np.arange(len(items)), index=items.item_id)
        pa = prior[prior.item_id.isin(pos_of.index)]
        pa = pa[pa.exception.fillna("") != "RETURNED"].groupby("item_id").amount.sum()
        ii = pos_of[pa.index].values
        cents0[ii] = np.maximum(0, cents0[ii] - np.round(pa.values * 100).astype("int64"))
    typ = items.item_type.values
    docs = items.document_number.astype(str).values
    pos = items.po_number.values
    cur = items.currency.values
    issue = items.issue_date.values
    ent = items.entity.values
    by_party: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(items.party_id.values):
        by_party[p].append(i)  # already sorted by issue date
    # entity-wide exact lookups of OUR numbers (document numbers are unique per entity)
    ref_idx: dict[tuple, list[int]] = defaultdict(list)
    for i in range(len(items)):
        k = rt.doc_keys(docs[i])
        if len(k["full"]) >= 8:
            ref_idx[(ent[i], k["full"])].append(i)
        if len(k["digits"]) >= 8 and k["digits"] != k["full"]:
            ref_idx[(ent[i], k["digits"])].append(i)

    rows = []
    stats = defaultdict(int)
    n_done = 0
    for r in b.itertuples(index=False):
        n_done += 1
        row = r._asdict()
        cash = _cents(r.amount_abs)
        bd = np.datetime64(r.booking_date)
        text = rt.bank_text(row)
        toks = rt.text_tokens(text)
        cands = px.candidates(row, r.entity)
        # reference evidence: owner of a quoted full number
        ref_owner: dict[str, int] = {}
        for t in toks:
            for key in (t, rt.digits(t)):
                for i in ref_idx.get((r.entity, key), ()):
                    if cents0[i] > 0:
                        ref_owner[items.party_id.values[i]] = ref_owner.get(items.party_id.values[i], 0) + 1
        known = {c.customer_id for c in cands}
        for p in ref_owner:
            if p not in known:
                cands.append(partylib.Cand(p, 0.6, "ref", "doc"))
        if not cands:
            stats["no_party"] += 1
            if diag is not None:
                diag[r.bank_txn_id] = ("no_party", [], [])
            continue
        cands = sorted(cands, key=lambda c: -c.score)[:30]
        results = []
        trace = [] if diag is not None else None
        for c in cands:
            res = _solve_party(c, px, by_party, items, cents0, docs, pos, cur, issue, typ, bd, r.currency, cash, toks, text, ent, r.entity, trace)
            if res is not None and not any(x in res["method"] for x in DROP):
                results.append((c, res))
        if diag is not None:
            diag[r.bank_txn_id] = ("pending", [c.customer_id for c in cands], trace)
        if not results:
            stats["no_solution"] += 1
            if diag is not None:
                diag[r.bank_txn_id] = ("no_solution", [c.customer_id for c in cands], trace)
            continue
        results.sort(key=lambda x: -(x[1]["conf"] * (0.5 + 0.5 * x[0].score)))
        best_c, best = results[0]
        if len(results) > 1:
            sig = lambda res: tuple(sorted(res["idx"]))
            others = [x for x in results[1:] if sig(x[1]) != sig(best)]
            if others:
                o = others[0]
                if o[1]["conf"] * (0.5 + 0.5 * o[0].score) > 0.85 * best["conf"] * (0.5 + 0.5 * best_c.score):
                    stats["ambiguous_payers"] += 1
                    if diag is not None:
                        diag[r.bank_txn_id] = ("ambiguous_payers", [c.customer_id for c in cands], trace)
                    continue
        stats["claimed"] += 1
        if diag is not None:
            diag[r.bank_txn_id] = ("claimed", [c.customer_id for c in cands], trace)
        stats["m_" + best["method"]] += 1
        for i, amt, exc in zip(best["idx"], best["amts"], best["excs"]):
            rows.append((r.bank_txn_id, item_id[i], amt / 100.0, exc, best["conf"], best["method"]))
            if best["conf"] >= LEDGER_MIN_CONF:  # weak claims do not consume items (they may be gated out downstream)
                cents0[i] = max(0, cents0[i] - amt)
        if verbose and n_done % 20000 == 0:
            print(f"[refs] {n_done}/{len(b)} lines, {time.time() - t0:.0f}s, {dict(stats)}")
    out = pd.DataFrame(rows, columns=["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"])
    if verbose:
        print("[refs] stats", dict(stats), f"{time.time() - t0:.0f}s")
    return out


def _pool_for(members, by_party, cents0, cur, issue, bd, currency, typ):
    idx = []
    for m in members:
        for i in by_party.get(m, ()):
            if issue[i] <= bd and cents0[i] > 0 and cur[i] == currency:
                idx.append(i)
    idx.sort(key=lambda i: (issue[i], i))
    return idx


def _signed(i, cents0, typ):
    return -cents0[i] if typ[i] == "ARC" else cents0[i]


def _canonical(sols, idx, cents0, typ, issue):
    """Reduce solutions (tuples of pool positions) to distinct amount-multisets; for each keep the FIFO-oldest instance."""
    best = {}
    for s in sols:
        sig = tuple(sorted((typ[idx[p]], cents0[idx[p]]) for p in s))
        key = sum(p for p in s)  # pool positions are in issue order -> smaller = older
        if sig not in best or key < best[sig][0]:
            best[sig] = (key, s)
    return [v[1] for v in best.values()]


def _solve_party(c, px, by_party, items, cents0, docs, pos, cur, issue, typ, bd, currency, cash, toks, text, ent, entity, trace=None):
    pid = c.customer_id
    tiers = [("own", [pid])]
    grp = px.group_members(pid)
    if len(grp) > 1:
        tiers.append(("group", grp))
    for tier, members in tiers:
        idx = _pool_for(members, by_party, cents0, cur, issue, bd, currency, typ)
        if not idx:
            continue
        res = _solve_pool(idx, cents0, docs, pos, typ, issue, cash, toks, text, c, tier, items, bd)
        if trace is not None:
            trace.append((pid, tier, len(idx), res["method"] if res else None))
        if res is not None and (tier == "own" or res["method"].startswith("ref_exact")):
            return res
    return None


def _cn_links(idx, typ, docs, items):
    """pool position of a credit note -> pool position of the invoice it rectifies (when that invoice is in the pool)."""
    rel = items.related_document.values
    dpos = {}
    for p, i in enumerate(idx):
        if typ[i] != "ARC":
            dpos[docs[i]] = p
    out = {}
    for p, i in enumerate(idx):
        if typ[i] == "ARC" and isinstance(rel[i], str) and rel[i] in dpos:
            out[p] = dpos[rel[i]]
    return out


def _month_fix(sel, idx, cents0, typ, issue, text, bd):
    """Same-amount invoices (monthly fees): when the text names the service month take the invoice issued the month after."""
    m = rt.service_month(text)
    if m is None:
        return sel, False
    tm = m % 12 + 1
    sel = list(sel)
    changed = False
    inpool = {p: idx[p] for p in range(len(idx))}
    for k, p in enumerate(sel):
        i = idx[p]
        if typ[i] != "ARI":
            continue
        same = [q for q in range(len(idx)) if q not in sel and typ[idx[q]] == "ARI" and cents0[idx[q]] == cents0[i]]
        if not same:
            continue
        cur_m = pd.Timestamp(issue[i]).month
        if cur_m == tm:
            continue
        cand = [q for q in same if pd.Timestamp(issue[idx[q]]).month == tm]
        if len(cand) >= 1:
            q = max(cand, key=lambda q: issue[idx[q]])  # the most recent such month (nearest to the payment)
            sel[k] = q
            changed = True
    return sel, changed


def _solve_pool(idx, cents0, docs, pos, typ, issue, cash, toks, text, c, tier, items, bd):
    n = len(idx)
    hits = rt.find_refs(toks, [docs[i] for i in idx])
    for k, lv in rt.find_refs(toks, [str(pos[i]) if isinstance(pos[i], str) else "" for i in idx], typo=False).items():
        hits[k] = max(hits.get(k, 0), min(lv, 3))
    sv = [_signed(i, cents0, typ) for i in idx]
    base = 1.0 if tier == "own" else 0.85
    links = _cn_links(idx, typ, docs, items) if any(typ[i] == "ARC" for i in idx) else {}

    def pack(sel, method, conf):
        sel = list(sel)
        return {"idx": [idx[p] for p in sel], "amts": [cents0[idx[p]] for p in sel], "excs": [None] * len(sel),
                "conf": round(conf * base, 3), "method": method + ("" if tier == "own" else "_grp")}

    def cn_ok(s):
        ss = set(s)
        return all((links[p] in ss) for p in s if p in links)

    def pick(sols, must=()):
        """canonical solutions -> (selection, n_alternatives). Credit-note rule first, then FIFO-oldest tie-break."""
        can = _canonical(sols, idx, cents0, typ, issue)
        if links and len(can) > 1:
            f = [s for s in can if cn_ok(s)]
            if f:
                can = f
        return can

    # --- 1. quoted documents alone
    H = sorted(hits)
    if H and len(H) <= rs.MAXN:
        sols = rs.subset_solutions([sv[p] for p in H], cash, limit=200)
        sols = [tuple(H[q] for q in s) for s in sols]
        can = pick(sols)
        if len(can) == 1:
            lv = max(hits[p] for p in can[0])
            return pack(can[0], "ref_exact", 0.99 if lv >= 3 else 0.9)
        if len(can) > 1:
            can.sort(key=lambda s: (-sum(hits[p] for p in s), len(s)))
            return pack(can[0], "ref_exact_amb", 0.6)
    # --- 1b. quoted invoices paid short (discount / withholding / bank charges): rows at FULL open amount (POLICIES s2)
    if H:
        r = _short([p for p in H if typ[idx[p]] == "ARI"], idx, cents0, issue, cash, items, bd)
        if r is not None:
            sel, amts, excs, how = r
            out = pack(sel, "ref_short_" + how, 0.9 if how.startswith("discount") else 0.6)
            out["amts"], out["excs"] = amts, excs
            return out
    # --- 1c. N:M — all quoted invoices paid, one of them by exactly half (the other half comes on another line)
    Hi = [p for p in H if typ[idx[p]] == "ARI"]
    if 2 <= len(Hi) <= 40:
        S = sum(cents0[idx[p]] for p in Hi)
        js = [p for p in Hi if abs((cash - (S - cents0[idx[p]])) - cents0[idx[p]] / 2) <= 1 and cash < S]
        if len(js) == 1:
            j = js[0]
            out = pack(Hi, "ref_half", 0.64)  # 95 % on val
            out["amts"] = [cents0[idx[p]] if p != j else cash - (S - cents0[idx[j]]) for p in Hi]
            out["excs"] = [None if p != j else "PARTIAL" for p in Hi]
            return out
    # --- 2. strong references + completion
    F = [p for p in H if hits[p] >= 3]
    if F:
        fs = sum(sv[p] for p in F)
        Fset = set(F)
        rest = _reduce([p for p in range(n) if p not in Fset], typ, idx)
        sols = rs.subset_solutions([sv[p] for p in rest], cash - fs, limit=200)
        sols = [tuple(F) + tuple(rest[q] for q in s) for s in sols]
        can = pick(sols)
        if len(can) == 1:
            rk = _risk(n - len(F), len(can[0]) - len(F), cash)
            return pack(can[0], "ref_fill" + _tier(rk), _conf(rk, 0.64))  # 96.6 % on val: below the 97 % gate
        if len(F) == 1 and typ[idx[F[0]]] == "ARI":
            op = cents0[idx[F[0]]]
            up = text.upper()
            if cash < op and (cash / op < 0.89 or any(w in up for w in PARTIAL_WORDS)) and cash / op > 0.02:
                r = pack([F[0]], "partial", 0.7)
                r["amts"] = [cash]
                r["excs"] = ["PARTIAL"]
                return r
        # several completions: no claim here (oldest-completion measured ~50 %); fall through to the prefix / discount rules
    # --- 2b. truncated / partial numbers listed in the text (camt 140-char cut, N43 38-char records): every token that is a
    #         PREFIX of some open document is a cover constraint; solve on the union of the matching documents only.
    ptoks = rt.prefix_tokens(text)
    if ptoks:
        M = {}
        for t in ptoks:
            m = [p for p in range(n) if rt.prefix_match(t, docs[idx[p]])]
            if m:
                M[t] = m
        C = sorted({p for m in M.values() for p in m})
        if M and 0 < len(C) <= rs.MAXN and len(C) < n:
            sols = rs.subset_solutions([sv[p] for p in C], cash, limit=200)
            sols = [tuple(C[q] for q in s_) for s_ in sols]
            sols = [s_ for s_ in sols if all(set(m) & set(s_) for m in M.values())]
            can = pick(sols)
            if len(can) == 1:
                rk = _risk(len(C), len(can[0]), cash)
                return pack(can[0], "ref_prefix" + _tier(rk), _conf(rk, 0.95))
    # --- 3. payer's items only
    rest = _reduce(list(range(n)), typ, idx)
    sols = rs.subset_solutions([sv[p] for p in rest], cash, limit=200)
    sols = [tuple(rest[q] for q in s) for s in sols]
    can = pick(sols)
    exact = None
    if len(can) == 1:
        sel, fixed = _month_fix(can[0], idx, cents0, typ, issue, text, None)
        rk = _risk(n, len(sel), cash)
        conf = _conf(rk, 0.97) if rk < 0.03 else 0.45  # unique but risky: measured ~73 % on val -> below the ledger/gate
        exact = pack(sel, "party_subset" + ("_month" if fixed else "") + _tier(rk), conf)
        if rk < 0.03:
            return exact
    elif len(can) > 1:
        can.sort(key=lambda s: (sum(s), len(s)))
        best, second = can[0], can[1]
        if sum(best) < sum(second):
            k = len(can)
            rk = _risk(n, len(best), cash) * k
            sel, fixed = _month_fix(best, idx, cents0, typ, issue, text, None)
            exact = pack(sel, "party_oldest" + ("_month" if fixed else "") + _tier(rk), _conf(rk, 0.85))
            if rk < 0.03:
                return exact
    # --- 3b. early-payment discount taken without a usable reference: invoices with discount terms, still fully open,
    #         counted at total - pct*total (POLICIES s3).  On time -> full amount (engine sets SHORT_DISCOUNT); late -> PARTIAL at
    #         the discounted cash.  Only a unique, low-risk solution that uses >= 1 discounted invoice is accepted.
    pct = items.disc_pct.values
    full = items.full_cents.values
    dv = list(sv)
    disc = {}
    for p, i in enumerate(idx):
        if typ[i] == "ARI" and pct[i] > 0 and cents0[i] == full[i]:
            dv[p] = full[i] - int(round(full[i] * pct[i] / 100))
            disc[p] = dv[p]
    if disc:
        sols = rs.subset_solutions([dv[p] for p in rest], cash, limit=200)
        sols = [tuple(rest[q] for q in s_) for s_ in sols]
        sols = [s_ for s_ in sols if any(p in disc for p in s_)]
        can = pick(sols)
        if len(can) == 1:
            sel = list(can[0])
            rk = _risk(n, len(sel), cash)
            if rk < 0.03:
                days = items.disc_days.values
                amts, excs, late = [], [], False
                for p in sel:
                    i = idx[p]
                    if p in disc and not (bd <= issue[i] + np.timedelta64(int(items.disc_days.values[i]) + 3, "D")):
                        amts.append(disc[p]); excs.append("PARTIAL"); late = True
                    else:
                        amts.append(cents0[i]); excs.append(None)
                out = pack(sel, "party_discount" + ("_late" if late else ""), 0.64 if late else 0.9)  # _late 93 % on val
                out["amts"], out["excs"] = amts, excs
                return out
    return exact


WHT_RATES = (0.0125, 0.015, 0.10)  # POLICIES s3 SHORT_WITHHOLDING


def _short(sel, idx, cents0, issue, cash, items, bd):
    """cash < open sum of the quoted invoices: explained shortfall?  returns (positions, amounts, exceptions, kind) or None.
    discount: cash == sum(total - pct*total) (cent rounding per invoice), on time -> full amounts; late -> PARTIAL at the cash per
    invoice. withholding: same with the POLICIES rates. charges: shortfall <= 2 % and <= 60 (line currency) -> full amounts."""
    if not sel or len(sel) > 40:
        return None
    ii = [idx[p] for p in sel]
    S = sum(cents0[i] for i in ii)
    if not (0 < S - cash <= 0.11 * S):
        return None
    full = items.full_cents.values
    if any(cents0[i] != full[i] for i in ii):
        return None  # only invoices still fully open
    tol = len(ii)
    pct = items.disc_pct.values
    if all(pct[i] > 0 for i in ii):
        D = sum(full[i] - int(round(full[i] * pct[i] / 100)) for i in ii)
        if abs(D - cash) <= tol:
            days = items.disc_days.values
            ontime = all(bd <= issue[i] + np.timedelta64(int(days[i]) + 3, "D") for i in ii)
            if ontime:
                return sel, [cents0[i] for i in ii], [None] * len(ii), "discount"
            return sel, [full[i] - int(round(full[i] * pct[i] / 100)) for i in ii], ["PARTIAL"] * len(ii), "discount_late"
    for r in WHT_RATES:
        D = sum(full[i] - int(round(full[i] * r)) for i in ii)
        if abs(D - cash) <= tol:
            return sel, [cents0[i] for i in ii], [None] * len(ii), "withholding"
    if S - cash <= min(6000, 0.02 * S):
        return sel, [cents0[i] for i in ii], [None] * len(ii), "charges"
    return None


def _risk(n: int, k: int, cash: int) -> float:
    """Expected number of SPURIOUS exact subset sums: #subsets of size <= k of n open items / #possible cent amounts (~cash).
    Structural (no fitted constant): a unique solution with risk << 1 is unlikely to be a coincidence."""
    from math import comb
    k = max(k, 1)
    return sum(comb(n, j) for j in range(1, min(k, n) + 1)) / max(cash, 100)


def _tier(rk: float) -> str:
    return "" if rk < 0.03 else "_r2" if rk < 0.3 else "_r3"


def _conf(rk: float, top: float) -> float:
    return top if rk < 0.03 else min(top, 0.6) if rk < 0.3 else 0.3


def _reduce(rest, typ, idx):
    """keep search tractable: credit notes first, then the oldest items up to MAXN."""
    if len(rest) <= rs.MAXN:
        return rest
    cn = [p for p in rest if typ[idx[p]] == "ARC"][:8]
    inv = [p for p in rest if typ[idx[p]] != "ARC"]
    keep = cn + inv[: rs.MAXN - len(cn)]
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="val")
    ap.add_argument("--max-lines", type=int)
    a = ap.parse_args()
    claimed, priors = set(), []
    for f in ("batches", "remittance"):
        fp = os.path.join("matching", "out", a.phase, f + ".parquet")
        if os.path.exists(fp):
            d = pd.read_parquet(fp, columns=["bank_txn_id", "item_id", "amount", "exception"])
            claimed |= set(d.bank_txn_id)
            priors.append(d)
    pi = None
    fp = os.path.join("classification", "out", a.phase, "line_labels.parquet")
    if os.path.exists(fp):
        pi = pd.read_parquet(fp, columns=["bank_txn_id", "p_items"]).set_index("bank_txn_id").p_items
    print(f"[refs] claimed by batches/remittance: {len(claimed)} lines; p_items available: {pi is not None}")
    out = match(a.phase, claimed=claimed, verbose=True, max_lines=a.max_lines, prior=pd.concat(priors) if priors else None, p_items=pi)
    os.makedirs(os.path.join("matching", "out", a.phase), exist_ok=True)
    out.to_parquet(os.path.join("matching", "out", a.phase, "refs.parquet"), index=False)
    try:
        evalutil.report(out, a.phase, name="refs")
    except Exception as e:
        print("no golden report:", e)
    if len(out):
        print(out.groupby("method").agg(lines=("bank_txn_id", "nunique"), pairs=("item_id", "size"), conf=("confidence", "mean")))


if __name__ == "__main__":
    main()
