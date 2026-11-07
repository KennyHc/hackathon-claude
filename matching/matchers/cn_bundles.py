"""cn_bundles: customer payments of several invoices net of credit notes that no earlier matcher could claim.

Runs LAST (only lines nobody claimed). Per incoming line, in booking order:
1. candidate payers = parties hit by the line's document tokens (refs.Solver.global_parties) + payers named in the
   text (refs.Solver.name_parties);
2. units = each open invoice of the payer together with its own open credit notes (credit_note.related_document =
   the invoice: POLICIES §2, a netted credit note is a row at its full amount); a credit note whose invoice is not
   open stands alone (negative);
3. fixed = units of items cited unambiguously (a complete number, or a cut number with exactly one candidate among
   the payer's open items); tokens with several candidates must each be covered by some chosen unit;
4. claim the whole line when the fixed units pay the cash exactly, or when exactly ONE set of the payer's other units
   completes it and the expected number of coincidental completions (subset-sum density, as in subset_sum.py) is small;
5. otherwise (PARTIAL_CLAIM) claim only the fixed units when they fit inside the cash: those pairs are right in ~all
   cases, and the line was going to be wrong anyway.
Items any earlier matcher allocated (on any line) are never offered.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict

from matching.common import Ctx, alloc_frame
from matching.matchers.refs import Solver
from matching.matchers.subset_sum import mitm
from matching.refs_index import tokens
from ingestion.text import norm_name

NAME = "cn_bundles"

SEARCH_MAX = int(os.environ.get("CNB_SEARCH_MAX", "26"))          # max units in the exhaustive completion search
SPURIOUS_MAX = float(os.environ.get("CNB_SPURIOUS", "0.05"))       # max expected coincidental completions (= subset_sum.SPURIOUS_ALT)
PARTIAL_CLAIM = os.environ.get("CNB_PARTIAL", "1") == "1"          # claim the fixed units alone as a fallback
SEARCH = os.environ.get("CNB_SEARCH", "1") == "1"
MAX_CANDS = int(os.environ.get("CNB_MAX_CANDS", "8"))   # a cited fragment with more candidates is ignored
MIN_DIGITS = int(os.environ.get("CNB_MIN_DIGITS", "6"))  # a cut / suffix fragment needs this many significant digits
WINDOW = os.environ.get("CNB_WINDOW", "1") == "1"         # pool too big: retry with units issued in the cited range
RESOLVER = os.environ.get("CNB_RESOLVER", "1") == "1"     # also payers from PartyResolver (IBAN / name / aliases)
INV_ANCHOR = os.environ.get("CNB_INV_ANCHOR", "0") == "1"  # >= 1 cited invoice; credit notes only by their full number
STAGES = os.environ.get("CNB_STAGES", "1") == "1"         # completion pools by due date around booking (as subset_sum)
DUE_STAGES = ((-90, 0), (-9999, 0), (-9999, 20))


class Bundles:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.sv = Solver(ctx)
        ix = self.sv.ix
        self.pos = {iid: k for k, iid in enumerate(ix.iid)}
        prior = getattr(ctx, "allocations", None)
        self.taken = set()
        if prior is not None and len(prior):
            self.taken = {self.pos[i] for i in prior.item_id.astype(str) if i in self.pos}
        self.used: set[int] = set()
        self.stats = defaultdict(int)
        self.debug = []   # (bank_txn_id, item k, kind, letters_ok, digits)
        due = dict(zip(ctx.items.item_id, ctx.items.due_date.fillna("").astype(str)))
        self.due = [due.get(i) or ix.issue[k] for k, i in enumerate(ix.iid)]
        self.pr = None
        if RESOLVER:
            from matching.parties import PartyResolver
            self.pr = PartyResolver.get(ctx)

    def units(self, P):
        """-> {invoice or lone credit note k: [k + its credit notes]}, {item k: unit key}"""
        ix = self.sv.ix
        inv_by_alnum = {}
        for k in P:
            if ix.sign[k] > 0 and ix.alnum[k]:
                inv_by_alnum.setdefault(ix.alnum[k], k)
        units = {k: [k] for k in P if ix.sign[k] > 0}
        for k in P:
            if ix.sign[k] < 0:
                inv = inv_by_alnum.get(ix.related[k]) if ix.related[k] else None
                if inv is not None and ix.cur[inv] == ix.cur[k]:
                    units[inv].append(k)
                else:
                    units[k] = [k]
        of = {m: u for u, ms in units.items() for m in ms}
        return units, of

    def uval(self, ms):
        ix = self.sv.ix
        return int(round(sum(ix.sign[m] * ix.open[m] for m in ms) * 100))

    def solve_party(self, line, party, toks):
        sv, ix = self.sv, self.sv.ix
        P = [k for k in sv.party_items("AR", party, line.entity, line.booking_date)
             if k not in self.taken and k not in self.used]
        if not P:
            return None
        units, of = self.units(P)
        fixed, constraints = set(), []
        why = {}
        for t in toks:
            kind, cands = sv.slot_cands(t, P)
            if not cands or kind == "F":
                continue
            nd = len(t.digits.lstrip("0"))
            if kind in ("S", "T") and nd < MIN_DIGITS:
                continue    # short fragments ("REF 12691") hit other years / series
            us = list(dict.fromkeys(of[k] for k in cands))
            lt = re.sub(r"[^A-Z]", "", ix.alnum[cands[0]])
            if INV_ANCHOR and len(cands) == 1 and ix.sign[cands[0]] < 0 and kind != "X":
                continue    # a credit note pinned only by a fragment
            if kind == "S" and t.letters and not lt.startswith(t.letters) and len(cands) == 1:
                continue    # same digits under another document prefix (PF proforma vs CRN credit note): not ours
            if len(us) == 1:
                fixed.add(us[0])
                why[us[0]] = (kind, (not t.letters) or lt.startswith(t.letters), nd)
            elif len(us) <= MAX_CANDS:
                constraints.append(set(us))
        if not fixed:
            return None
        if INV_ANCHOR and not any(ix.sign[u] > 0 for u in fixed):
            return None
        cur = {ix.cur[m] for u in fixed for m in units[u]}
        if len(cur) != 1:
            return None
        cur = cur.pop()
        cash, exact_cur = sv.to_item_cur(line, cur)
        if cash is None or not exact_cur:
            return None
        cents = int(round(cash * 100))
        fv = sum(self.uval(units[u]) for u in fixed)
        open_c = [c for c in constraints if not c & fixed]
        resid = cents - fv
        if abs(resid) <= 1 and not open_c:
            return "fixed", fixed, units, why
        if SEARCH and resid > 1:
            rest = [u for u in units if u not in fixed and ix.cur[u] == cur]
            vals = {u: self.uval(units[u]) for u in rest}
            rest = [u for u in rest if vals[u] != 0 and vals[u] <= resid]
            if STAGES:
                r = self.staged(line, rest, vals, resid, fixed, open_c)
                if r:
                    return "search", fixed | r, units, why
                rest = []
            elif len(rest) > SEARCH_MAX and WINDOW:
                lo = min(ix.issue[u] for u in fixed)
                hi = max(ix.issue[u] for u in fixed)
                rest = [u for u in rest if lo <= ix.issue[u] <= hi]
                self.stats["window"] += 1
            if 1 <= len(rest) <= SEARCH_MAX:
                cnt, sols, dens = mitm([vals[u] for u in rest], resid, max_sols=1)
                self.stats[f"search_cnt{min(cnt, 2)}"] += 1
                if cnt == 1 and dens <= SPURIOUS_MAX:
                    S = {rest[i] for i in sols[0]}
                    if all(c & (S | fixed) for c in open_c):
                        return "search", fixed | S, units, why
            elif not STAGES:
                self.stats["search_toobig" if rest else "search_empty"] += 1
        if PARTIAL_CLAIM and fv <= cents + 1:
            return "partial", fixed, units, why
        return None

    def staged(self, line, rest, vals, resid, fixed, open_c):
        """completion over the payer's units due in widening windows around the booking date (subset_sum.DUE_STAGES):
        the first stage with any exact completion decides; it must be the only one and pass the coincidence guard."""
        from datetime import date, timedelta
        ix = self.sv.ix
        bd = date.fromisoformat(line.booking_date)
        lone = [u for u in rest if ix.sign[u] < 0][-6:]
        prev = None
        for lo, hi in DUE_STAGES:
            a, b = str(bd + timedelta(days=lo)), str(bd + timedelta(days=hi))
            inv = [u for u in rest if ix.sign[u] > 0 and a <= self.due[u] <= b][-SEARCH_MAX:]
            cand = inv + lone
            if not inv or len(cand) == prev or sum(vals[u] for u in inv) < resid:
                continue
            prev = len(cand)
            cnt, sols, dens = mitm([vals[u] for u in cand], resid, max_sols=1)
            self.stats[f"stage_cnt{min(cnt, 2)}"] += 1
            if cnt == 1 and dens <= SPURIOUS_MAX:
                S = {cand[i] for i in sols[0]}
                if all(c & (S | fixed) for c in open_c):
                    return S
                return None
            if cnt >= 1:
                return None
        return None

    def solve(self, line):
        if not (line.amount > 0):
            return None
        sv = self.sv
        toks = tokens(line.text, line.remittance_info if line.remittance_info != line.text else "")
        if not toks:
            return None
        votes = sv.global_parties(line.entity, "AR", toks, line.booking_date, line.amount_abs)
        words = set(norm_name(" ".join([line.text or "", line.remittance_info or "", line.party_name_raw or ""])).split())
        for p in sv.name_parties(line.entity, "AR", words):
            votes[p] += 0.5
        res = []
        if self.pr is not None:
            for p, (st, _) in self.pr.resolve(line).items():
                votes[p] += 0.5 * st
        for party, _ in sorted(votes.items(), key=lambda x: (-x[1], x[0]))[:6]:   # ties: stable, not hash order
            r = self.solve_party(line, party, toks)
            if r:
                res.append(r)
        if not res:
            return None
        rank = {"fixed": 0, "search": 0, "partial": 1}
        best = min(rank[r[0]] for r in res)
        res = [r for r in res if rank[r[0]] == best]
        if len(res) != 1:
            self.stats["ambiguous_party"] += 1
            return None
        return res[0]

    def run(self):
        ix = self.sv.ix
        b = self.ctx.unclaimed()
        b = b[b.amount > 0].sort_values(["booking_date", "bank_txn_id"])
        rows = []
        for line in b.itertuples(index=False):
            r = self.solve(line)
            if not r:
                continue
            how, S, units, why = r
            for u, w in why.items():
                self.debug.append((line.bank_txn_id, ix.iid[u], how) + w)
            self.stats[how] += 1
            conf = {"fixed": 0.85, "search": 0.75, "partial": 0.5}[how]
            for u in S:
                for m in units[u]:
                    rows.append((line.bank_txn_id, ix.iid[m], round(float(ix.open[m]), 2), "", conf, f"{NAME}:{how}"))
                    self.used.add(m)
        print(f"[cn_bundles] {dict(self.stats)}")
        if os.environ.get("CNB_DEBUG"):
            import pandas as pd
            pd.DataFrame(self.debug, columns=["bank_txn_id", "item_id", "how", "kind", "let_ok", "ndig"]).to_parquet(
                os.environ["CNB_DEBUG"], index=False)
        return alloc_frame(rows)


def run(ctx: Ctx):
    return Bundles(ctx).run()
