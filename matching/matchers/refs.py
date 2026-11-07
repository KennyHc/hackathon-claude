"""refs (#10): our document numbers in the bank text, despite noise (POLICIES §5.5) -> the items they settle.

Per bank line (receipts -> AR items, debits -> AP items of the line's entity):
1. tokens of text + remittance_info (numbers re-joined across N43 ` | ` fields, amounts / IBANs / bank ids removed);
2. candidate parties from tokens that hit items exactly (alnum, digits, PO), by suffix (number without prefix),
   by prefix (truncated) or with one edit / transposition;
3. per candidate party, every token is matched against that party's open items only (exact > suffix > prefix > fuzzy)
   giving "slots"; choose one candidate per slot so that Σ open (invoices − credit notes) == cash ± 0.01;
4. otherwise complete with the party's uncited open items (credit notes first, small subset search);
5. otherwise explain the difference (POLICIES §3): FX_DIFFERENCE, SHORT_DISCOUNT, SHORT_WITHHOLDING,
   SHORT_BANK_CHARGES, OVERPAYMENT, PARTIAL.
Lines are processed in booking order and the open amounts consumed, so a later payment sees the remainder.
"""
from __future__ import annotations

import bisect
import itertools
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from matching.common import Ctx, alloc_frame  # noqa: E402
from ingestion.text import norm_name  # noqa: E402
from matching.refs_index import _WORD as WORD, RefIndex, _clean as clean_text, edit1, tokens  # noqa: E402

try:  # shared exception rules (classification/exceptions.py), local rules as fallback
    from classification.exceptions import explain as xexplain  # noqa: E402
except Exception:  # pragma: no cover
    xexplain = None
try:
    from classification.exceptions import overpayment_plausible  # noqa: E402
except Exception:  # pragma: no cover
    def overpayment_plausible(open_, cash):
        return 0 < cash - open_ <= OVERPAY_MAX * open_

NAME = "refs"
EPS = 0.011
STRONG = {"X": 3.0, "S": 2.0}
WEAK = {"T": 1.0, "F": 1.0}
FRACTIONS = [0.1, 0.2, 0.25, 0.3, 1 / 3, 0.4, 0.5, 0.6, 2 / 3, 0.7, 0.75, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99,
             1 / 6, 1 / 4, 1 / 2, 5 / 6]
NAME_MIN = 0.5
OVERPAY_MAX = 0.35  # excess cash on one invoice (val: p99 0.32)
FULL_CLEAR = {"", "SHORT_DISCOUNT", "SHORT_WITHHOLDING", "SHORT_BANK_CHARGES", "OVERPAYMENT"}
TRUNC_SPLIT = re.compile(r" \| |/[A-Z]{2,5}/| /REF\b|\\$")
DOC_PREFIXES = {"FV", "FA", "FT", "RE", "RF", "INV", "NF", "VTN", "AB", "GS", "NC", "AV", "CRN", "CM", "FAC", "FRA", "ND",
                "VCN", "FRAS", "FACT"}
RUN_ID = re.compile(r"PR\d{12}")
USE_SHARED = os.environ.get("REFS_SHARED") == "1"  # shared explain() only adds wrong lines on sample (fallback)
STRICT_EXC = {"", "SHORT_DISCOUNT", "SHORT_WITHHOLDING", "FX_DIFFERENCE"}
INSTAL = re.compile(r"(?:INSTALMENT|PLAZO|ECHEANCE|PRESTACAO|RATE|RATA|VENCIMIENTO|TEILZAHLUNG)\S*\s*(\d{1,2})\s*/\s*(\d{1,2})")
REMAINDER = re.compile(r"\b(?:SETTLEMENT|REGULARIZACION|REGULARISATION|REGOLARIZZAZIONE|BALANCE|RESTO|SOLDE|REST|"
                       r"RESTANTE|REMAINDER|RELEASE|RETENTION|GARANTIA|RETENCION|LIBERACION|RITENUTA|GARANTIE)\b")
WHT_COUNTRIES = {"MX", "BR", "MA"}
WHT_RATES = (0.0125, 0.015, 0.10)
SEPA = {"ES", "FR", "DE", "IT", "PT", "NL", "BE", "LU", "AT", "IE", "MT", "FI", "GR", "SK", "SI", "EE", "LV", "LT",
        "CY", "HR", "PL", "GB", "CH", "SE", "DK", "NO", "CZ", "HU", "RO", "BG", "IS", "LI", "MC", "SM"}


def unique_subset(vals, target, keys, max_states=400_000):
    """the only subset of vals (cents, signed) summing to target -> [keys] (None if none / several / too big)."""
    n = len(vals)
    if n == 0 or n > 80:
        return None
    pos_suffix = [0] * (n + 1)
    neg_suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        pos_suffix[i] = pos_suffix[i + 1] + max(vals[i], 0)
        neg_suffix[i] = neg_suffix[i + 1] + min(vals[i], 0)
    ways = {0: 1}
    parent = {0: None}
    for i, v in enumerate(vals):
        hi = target - neg_suffix[i + 1]
        lo = target - pos_suffix[i + 1]
        new = {}
        for sm, w in ways.items():
            ns = sm + v
            if ns < lo or ns > hi:
                continue
            new[ns] = new.get(ns, 0) + w
            if ns not in parent:
                parent[ns] = (sm, i)
        for ns, w in new.items():
            ways[ns] = min(2, ways.get(ns, 0) + w)
        # drop states that can no longer reach the target
        if len(ways) > max_states:
            return None
    if ways.get(target) != 1:
        return None
    out = []
    sm = target
    seen = set()
    while sm != 0 or (parent.get(sm) is not None and not out):
        pr = parent.get(sm)
        if pr is None or pr[1] in seen:
            return None
        seen.add(pr[1])
        out.append(keys[pr[1]])
        sm = pr[0]
        if sm == 0:
            break
    return out


class Solver:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        items = ctx.items[ctx.items.open_amount > 0.005]
        self.ix = RefIndex(items)
        m = ctx.masters
        pt = m["payment_terms"]
        self.terms = {r.code: (float(r.early_payment_discount_pct), int(r.discount_days))
                      for r in pt.itertuples() if pd.notna(r.early_payment_discount_pct)
                      and str(r.early_payment_discount_pct).strip() != ""}
        cu = m["customers"]
        self.country = dict(zip(cu.customer_id, cu.country))
        su = m.get("suppliers")
        if su is not None and "supplier_id" in su and "country" in su:
            self.country.update(dict(zip(su.supplier_id, su.country)))
        # party names: words + idf (for confirming the payer when cash != documents)
        it = items.drop_duplicates(["side", "party_id"])
        self.pname = {}
        self.pnorm = {}
        df = Counter()
        for sd, p, nm in zip(it.side, it.party_id, it.party_name_norm.fillna("")):
            ws = [w for w in nm.split() if len(w) >= 2]
            self.pname[(sd, p)] = ws
            self.pnorm[(sd, p)] = nm
            df.update(set(ws))
        self.word_parties = defaultdict(set)
        for sd, p, e, nm in zip(it.side, it.party_id, it.entity, it.party_name_norm.fillna("")):
            for w in nm.split():
                if len(w) >= 3:
                    self.word_parties[(e, sd, w)].add(p)
        n = max(len(self.pname), 1)
        self.idf = {w: float(np.log(1 + n / c)) for w, c in df.items()}
        bk = ctx.bank
        self.bank_by_id = dict(zip(bk.bank_txn_id, zip(bk.text.fillna(""), bk.remittance_info.fillna(""),
                                                        bk.party_name_raw.fillna(""))))
        self._words = {}
        self.receipts = defaultdict(list)
        for i, e, d, a in sorted(zip(bk.bank_txn_id, bk.entity, bk.booking_date, bk.amount), key=lambda x: (x[2], x[0])):
            self.receipts[(e, "AR" if a > 0 else "AP")].append((d, round(abs(a), 2), i))
        self.receipt_dates = {k: [x[0] for x in v] for k, v in self.receipts.items()}
        self.load_payment_runs()
        self.find_cancelled()
        fx = m["fx_rates"].copy()
        fx["date"] = fx.date.astype(str)
        self.fx = {c: (g.date.tolist(), g.units_per_eur.astype(float).tolist())
                   for c, g in fx.sort_values("date").groupby("currency")}

    def load_payment_runs(self):
        """AP debits covered by payment-run files belong to the payment_runs matcher; the files also tell which AP
        documents were already paid (in part) by an earlier run."""
        self.run_e2e, self.run_ids = set(), set()
        self.run_docs = defaultdict(list)
        d = self.ctx.data_path("payment_runs")
        if not os.path.isdir(d):
            return
        parts = []
        for f in sorted(os.listdir(d)):
            if f.endswith(".csv"):
                try:
                    parts.append(pd.read_csv(os.path.join(d, f), dtype=str,
                                             usecols=["run_id", "execution_date", "debtor_account_id", "end_to_end_id",
                                                      "document_number", "beneficiary"]))
                except Exception:
                    continue
        if not parts:
            return
        r = pd.concat(parts, ignore_index=True).fillna("")
        self.run_e2e = set(r.end_to_end_id) - {""}
        self.run_ids = set(r.run_id) - {""}
        ent = r.debtor_account_id.str.split("-").str[0]
        aln = r.document_number.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
        ben = r.beneficiary.map(norm_name)
        for e, a, b, dt in zip(ent, aln, ben, r.execution_date):
            self.run_docs[(e, a, b)].append(dt)

    def in_payment_run(self, line):
        if line.end_to_end_id and line.end_to_end_id in self.run_e2e:
            return True
        for v in (line.ref1, line.ref2):
            if v and v in self.run_ids:
                return True
        m = RUN_ID.search(line.text or "")
        return bool(m and m.group(0) in self.run_ids)

    # --- helpers -------------------------------------------------------------------------------------------------
    def rate(self, cur: str, date: str) -> float | None:
        if cur == "EUR":
            return 1.0
        v = self.fx.get(cur)
        if not v:
            return None
        i = bisect.bisect_right(v[0], date) - 1
        return v[1][max(i, 0)]

    def party_items(self, side, party, ent, bd):
        ix = self.ix
        return [k for k in ix.by_party.get((side, party), ()) if ix.open[k] > 0.005 and ix.entity[k] == ent
                and ix.issue[k] <= bd and k not in self.cancelled]

    def find_cancelled(self):
        """invoices cancelled by an 'Anulación' credit note (POLICIES §5.7): neither they nor that credit note are
        ever settled by bank; a payment citing them pays the replacement invoice issued around the credit note."""
        ix = self.ix
        self.cancelled = set()
        self.replaced = {}  # (side, party, alnum of cancelled invoice) -> credit note issue date
        pos0 = {iid: k for k, iid in enumerate(ix.iid)}
        try:  # PDF clauses: "anula y sustituye", credit notes cancelling an invoice in full (cached on ctx)
            ids = getattr(self.ctx, "cancelled_items", None)
            if ids is None:
                from matching.pdf_clauses import cancelled_items
                ids = cancelled_items(self.ctx)
                self.ctx.cancelled_items = ids
            self.cancelled |= {pos0[i] for i in ids if i in pos0}
        except Exception:
            pass
        it = self.ctx.items
        reason = it.reason.fillna("").str.upper()
        cn = it[(it.item_type.isin(["ARC", "APC"])) & reason.str.contains("ANULA|CANCEL")]
        pos = {iid: k for k, iid in enumerate(ix.iid)}
        for r in cn.itertuples(index=False):
            c = pos.get(r.item_id)
            rel = ix.related[c] if c is not None else ""
            if c is None or not rel:
                continue
            for k in ix.by_party.get((ix.side[c], ix.party[c]), ()):
                if ix.alnum[k] == rel and ix.sign[k] > 0 and abs(ix.total[k] - ix.total[c]) < 0.011:
                    self.cancelled |= {k, c}
                    self.replaced[(ix.side[c], ix.party[c], rel)] = ix.issue[c]
                    if ix.norm[k]:
                        self.replaced[(ix.side[c], ix.party[c], ix.norm[k])] = ix.issue[c]

    def slot_cands(self, t, P):
        """match one token against the party's items -> (kind, [items])"""
        ix = self.ix
        if P and self.replaced:
            side, party = ix.side[P[0]], ix.party[P[0]]
            dt = self.replaced.get((side, party, t.alnum)) or self.replaced.get((side, party, t.digits.lstrip("0")))
            if dt:
                lo = (pd.Timestamp(dt) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
                hi = (pd.Timestamp(dt) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
                rep = [k for k in P if ix.sign[k] > 0 and lo <= ix.issue[k] <= hi]
                if rep:
                    return "T", rep
        d0 = t.digits.lstrip("0")
        x = [k for k in P if ix.alnum[k] == t.alnum or (len(d0) >= 4 and ix.norm[k] == d0 and
                                                          (not t.letters or ix.alnum[k][:len(t.letters)] == t.letters))
             or (len(t.alnum) >= 5 and ix.po[k] == t.alnum)]
        if x:
            return "X", x
        td = t.digits
        if len(td) >= 4 and (not t.letters or len(t.letters) <= 3):
            s = [k for k in P if ix.norm[k].endswith(td) or (len(d0) >= 4 and ix.norm[k].endswith(d0))]
            if s:
                return "S", s
        if len(t.alnum) >= 4:
            if t.letters:
                p = [k for k in P if ix.alnum[k].startswith(t.alnum)]
            else:
                p = [k for k in P if len(td) >= 4 and td in ix.norm[k]]
            if p:
                return "T", p
        if len(d0) >= 5:
            f = [k for k in P if edit1(ix.norm[k], d0) or
                 (len(td) >= 5 and len(ix.norm[k]) > len(td) and edit1(ix.norm[k][-len(td):], td))]
            if f:
                return "F", f
        return None, []

    def global_parties(self, ent, side, toks, bd, cash):
        ix = self.ix
        votes = Counter()
        for t in toks:
            found = []
            hits = ix.exact(ent, side, t)
            if hits:
                found.append(("X", hits))
            else:
                h = ix.suffix(ent, side, t.digits) if len(t.digits.lstrip("0")) >= 5 else []
                if not h and len(t.digits.lstrip("0")) >= 5:
                    h = ix.suffix(ent, side, t.digits.lstrip("0"))
                if h:
                    found.append(("S", h))
                elif t.letters:
                    h = ix.prefix(ent, side, t.alnum)
                    if h:
                        found.append(("T", h))
            h = ix.fuzzy(ent, side, t)
            if h:
                found.append(("F", h))
            for kind, hits in found:
                hits = [k for k in hits if ix.open[k] > 0.005 and ix.issue[k] <= bd]
                if not hits:
                    continue
                parties = {ix.party[k] for k in hits}
                if len(parties) > 4:
                    # too ambiguous: keep only parties with an item of exactly the cash amount
                    parties = {ix.party[k] for k in hits if abs(ix.open[k] - cash) < EPS}
                    if len(parties) != 1:
                        continue
                w = (STRONG.get(kind) or WEAK.get(kind)) / len(parties)
                for p in parties:
                    votes[p] += w
        return votes

    def name_parties(self, ent, side, words, top=3):
        cand = set()
        for w in words:
            ps = self.word_parties.get((ent, side, w))
            if ps and len(ps) <= 300:
                cand |= ps
        sc = [(self.name_score(words, side, p), p) for p in cand]
        sc = sorted([x for x in sc if x[0] >= 0.6], reverse=True)[:top]
        return [p for _, p in sc]

    def name_score(self, words, side, party):
        """share (idf-weighted) of the party's name words found in the bank text (prefix = truncated word)."""
        nw = self.pname.get((side, party))
        if not nw:
            return 0.0
        tot = got = 0.0
        for w in nw:
            f = self.idf.get(w, 1.0)
            tot += f
            if w in words or any(len(x) >= 4 and w.startswith(x) for x in words):
                got += f
        return got / tot if tot else 0.0

    # --- amounts -------------------------------------------------------------------------------------------------
    def to_item_cur(self, line, cur):
        """cash in the item currency: exact when the bank gives the original amount, else (approx, None)."""
        if cur == line.currency:
            return line.amount_abs, True
        if isinstance(line.orig_currency, str) and line.orig_currency == cur and pd.notna(line.orig_amount):
            return abs(float(line.orig_amount)), True
        rb, ri = self.rate(line.currency, line.booking_date), self.rate(cur, line.booking_date)
        if not rb or not ri:
            return None, False
        return line.amount_abs / rb * ri, False

    def value(self, S):
        ix = self.ix
        return round(sum(ix.sign[k] * ix.open[k] for k in S), 2)

    # --- exceptions ----------------------------------------------------------------------------------------------
    def items_frame(self, S):
        ix = self.ix
        S = list(S)
        return pd.DataFrame({"item_id": ix.iid[S], "item_type": ix.typ[S], "total_amount": ix.total[S],
                             "open_amount": ix.open[S], "currency": ix.cur[S], "issue_date": ix.issue[S],
                             "payment_terms": ix.terms[S], "party_id": ix.party[S], "entity": ix.entity[S]}), S

    def explain(self, S, line, cash, exact_cur, party, strict):
        """S settles the line but cash != documents -> rows [(k, amount, exception)] or None.
        strict: only explanations that pin the cash to the cent (discount in terms, withholding, FX)."""
        ix = self.ix
        inv = [k for k in S if ix.sign[k] > 0]
        cns = [k for k in S if ix.sign[k] < 0]
        if not inv or cash is None:
            return None
        D = self.value(S)
        txt = f"{line.text or ''} {line.remittance_info or ''}".upper()
        # keyword cases on one invoice paid short (instalment k/n, remainder, dispute settlement, retention release)
        if not strict and len(inv) == 1 and exact_cur and cash < D - EPS and ix.cur[inv[0]] == line.currency:
            k = inv[0]
            paid = round(ix.open[k] - (D - cash), 2)
            if paid > 0.005:
                m = INSTAL.search(txt)
                if m:
                    a, b = int(m.group(1)), int(m.group(2))
                    exc = "PARTIAL" if a < b else ""
                    return [(k, paid, exc)] + [(c, ix.open[c], "") for c in cns]
                if REMAINDER.search(txt):
                    return [(k, paid, "")] + [(c, ix.open[c], "") for c in cns]
        rows = self.explain_local(S, line, cash, exact_cur, party)
        if rows is None and xexplain is not None and USE_SHARED and ix.cur[inv[0]] == line.currency:
            df, order = self.items_frame(S)
            try:
                res = xexplain(df, line.amount_abs, line, self.ctx.masters)
            except Exception:
                res = None
            if res:
                pos = {iid: k for iid, k in zip(ix.iid[order], order)}
                rows = [(pos[iid], amt, exc) for iid, amt, exc in res]
                if (sum(e == "PARTIAL" for _, _, e in rows) > 1
                        or any(e == "OVERPAYMENT" for _, _, e in rows) and not overpayment_plausible(D, cash)):
                    rows = None
        if not rows:
            return None
        excs = {e for _, _, e in rows}
        if strict and (not excs <= STRICT_EXC or any(a < ix.open[k] - 0.011 for k, a, _ in rows)):
            return None
        # a "partial" that is really the remainder of an earlier partial payment we did not see
        if "PARTIAL" in excs and not INSTAL.search(txt):
            out = []
            for k, amt, e in rows:
                if e == "PARTIAL" and self.prior_partial(line, party, ix.side[k], round(ix.open[k] - amt, 2), k):
                    e = ""
                out.append((k, amt, e))
            rows = out
        return rows

    def explain_local(self, S, line, cash, exact_cur, party):
        ix = self.ix
        inv = [k for k in S if ix.sign[k] > 0]
        D = self.value(S)
        if ix.cur[inv[0]] != line.currency:
            if not exact_cur:
                if 0.965 * D <= cash <= 1.003 * D:
                    return [(k, ix.open[k], "FX_DIFFERENCE" if ix.sign[k] > 0 else "") for k in S]
                return None
            if abs(cash - D) <= max(0.02, 0.0005 * D):
                return [(k, ix.open[k], "FX_DIFFERENCE" if ix.sign[k] > 0 else "") for k in S]
            rows = self.explain_same(S, line, cash, party)
            if not rows:
                return None
            return [(k, a, "FX_DIFFERENCE" if ix.sign[k] > 0 and e in FULL_CLEAR else e) for k, a, e in rows]
        return self.explain_same(S, line, cash, party)

    def explain_same(self, S, line, cash, party):
        ix = self.ix
        inv = [k for k in S if ix.sign[k] > 0]
        cns = [k for k in S if ix.sign[k] < 0]
        D = self.value(S)
        bd = line.booking_date
        if cash > D + EPS:
            if len(inv) == 1 and not cns and overpayment_plausible(D, cash):
                return [(inv[0], ix.open[inv[0]], "OVERPAYMENT")]
            return None
        short = round(D - cash, 2)
        # early-payment discount (single invoice: within terms + 3 days; several: the data accepts later ones)
        grace = 3 if len(inv) == 1 else 19
        disc, late = {}, {}
        for k in inv:
            tm = self.terms.get(ix.terms[k])
            if not tm or abs(ix.open[k] - ix.total[k]) > 0.005:
                continue
            dsc = round(tm[0] / 100.0 * ix.total[k], 2)
            lim = (pd.Timestamp(ix.issue[k]) + pd.Timedelta(days=tm[1] + grace)).strftime("%Y-%m-%d")
            (disc if bd <= lim else late)[k] = dsc
        if disc:
            ks = list(disc)[:12]
            for r in range(1, len(ks) + 1):
                hit = [c for c in itertools.combinations(ks, r) if abs(sum(disc[k] for k in c) - short) <= 0.011 * r]
                if len(hit) == 1:
                    c = set(hit[0])
                    return [(k, ix.open[k], "SHORT_DISCOUNT" if k in c else "") for k in S]
                if hit:
                    break
        if late:
            hit = [k for k in late if abs(late[k] - short) <= 0.011]
            if len(hit) == 1:
                k = hit[0]
                return [(j, round(ix.open[j] - short, 2) if j == k else ix.open[j], "PARTIAL" if j == k else "")
                        for j in S]
        # withholding at source
        if self.country.get(party) in WHT_COUNTRIES and ix.side[inv[0]] == "AR":
            for r in WHT_RATES:
                w = sum(round(r * ix.total[k], 2) for k in inv)
                if abs(w - short) <= 0.011 * len(inv) + 0.005:
                    return [(k, ix.open[k], "SHORT_WITHHOLDING" if ix.sign[k] > 0 else "") for k in S]
        # intermediary bank fee on a non-SEPA transfer
        if len(inv) == 1 and ix.side[inv[0]] == "AR" and self.country.get(party) not in SEPA:
            r = self.rate(line.currency, bd) or 1.0
            if short / r <= 60.0 and short <= 0.02 * D:
                return [(inv[0], ix.open[inv[0]], "SHORT_BANK_CHARGES")] + [(c, ix.open[c], "") for c in cns]
        # partial payment
        if len(inv) == 1:
            k = inv[0]
            paid = round(ix.open[k] - short, 2)
            if paid > 0.005:
                return [(k, paid, "PARTIAL")] + [(c, ix.open[c], "") for c in cns]
            return None
        hits = []
        for k in inv:
            paid = round(ix.open[k] - short, 2)
            if paid <= 0.005:
                continue
            fr = paid / ix.open[k]
            if any(abs(fr - f) * ix.open[k] <= 0.011 for f in FRACTIONS):
                hits.append((ix.issue[k], k, paid))
        if hits:
            _, k, paid = max(hits)
            # AR, two invoices, one short: in the data this is mostly the remainder of an earlier partial payment
            # (val: 72 % '' vs 28 % PARTIAL; with 3+ invoices it is always PARTIAL)
            code = "" if ix.side[k] == "AR" and len(inv) == 2 else "PARTIAL"
            return [(j, paid if j == k else ix.open[j], code if j == k else "") for j in S]
        return None

    def prior_partial(self, line, party, side, x, k):
        """an earlier receipt of this payer (name in its text) whose amount is x, or x + other open items of the party
        -> the item was already paid in part, this line pays the remainder."""
        if x <= 0.005:
            return False
        if side == "AP" and any(dt < line.booking_date for dt in self.run_docs.get((line.entity, self.ix.alnum[k], self.pnorm.get(("AP", party), "")), ())):
            return True
        lst = self.receipts.get((line.entity, side))
        if not lst:
            return False
        P = [j for j in self.ix.by_party.get((side, party), ()) if j != k and self.ix.open[j] > 0.005]
        v1 = [int(round(self.ix.sign[j] * self.ix.open[j] * 100)) for j in P[:150]]
        vals = set(v1)
        if len(v1) <= 80:
            vals |= {a + b for a, b in itertools.combinations(v1, 2)}
        lo = (pd.Timestamp(line.booking_date) - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
        dates = self.receipt_dates[(line.entity, side)]
        end = bisect.bisect_left(dates, line.booking_date)
        start = bisect.bisect_left(dates, lo)
        for j in range(end - 1, start - 1, -1):
            d, amt, i = lst[j]
            rest = round(amt - x, 2)
            if rest < -0.005:
                continue
            if abs(rest) < EPS or int(round(rest * 100)) in vals:
                if self.name_score(self.line_words(i), side, party) >= NAME_MIN:
                    return True
        return False

    def line_words(self, i):
        w = self._words.get(i)
        if w is None:
            L = self.bank_by_id[i]
            w = set(norm_name(" ".join([L[0] or "", L[1] or "", L[2] or ""])).split())
            self._words[i] = w
        return w

    def truncated(self, line, slots):
        """a field of the text ends with a reference fragment that resolved to nothing ('... FV', '... 250')."""
        used = {t.alnum for _, _, t in slots}
        # the structured remittance field is usually complete when the free text is cut: judge it alone if present
        srcs = (line.remittance_info,) if tokens(line.remittance_info or "") else (line.text, line.remittance_info)
        for txt in srcs:
            for chunk in TRUNC_SPLIT.split(clean_text(txt)):
                ws = WORD.findall(chunk)
                if not ws:
                    continue
                if re.fullmatch(r"\d{1,2}/\d{1,2}", ws[-1]):
                    continue
                w = re.sub(r"[^A-Z0-9]", "", ws[-1])
                if not w or w in used or (len(w) >= 4 and any(u.startswith(w) or (w.isdigit() and w in u) for u in used)):
                    continue
                dg = sum(c.isdigit() for c in w)
                if (dg == 0 and w in DOC_PREFIXES) or (w.isdigit() and len(w) <= 6) or (
                        dg and len(w) - dg >= 2 and w[:len(w) - dg] in DOC_PREFIXES and dg < 7):
                    return True
        return False

    # --- per party -----------------------------------------------------------------------------------------------
    def solve_party(self, line, side, party, toks, words):
        ix = self.ix
        name_ok = self.name_score(words, side, party) >= NAME_MIN
        P = self.party_items(side, party, line.entity, line.booking_date)
        if not P:
            return None
        slots = []
        for t in toks:
            kind, cands = self.slot_cands(t, P)
            if not cands:
                continue
            if len(cands) > 8:
                if kind in WEAK:
                    continue
                cands = cands[:8]
            slots.append((kind, tuple(cands), t))
        if not slots:
            return None
        # merge slots with identical candidate sets (same number cited twice / joined + parts)
        uniq = {}
        for kind, cands, t in slots:
            key = cands if len(cands) == 1 else (cands, t.alnum)  # two cut numbers may share their candidates
            if key not in uniq or (STRONG.get(kind, 0) > STRONG.get(uniq[key][0], 0)):
                uniq[key] = (kind, cands, t)
        slots = list(uniq.values())
        all_strong = all(kd == "X" for kd, _, _ in slots)
        cur = ix.cur[slots[0][1][0]]
        cash, exact_cur = self.to_item_cur(line, cur)
        if cash is None:
            return None

        fixed = set()
        choice = []
        for kind, cands, t in slots:
            if kind in STRONG and len(cands) == 1:
                fixed.add(cands[0])
            else:
                choice.append((kind, cands))
        choice = [(kind, c) for kind, c in choice if not set(c) & fixed]
        opts = [list(c) + [None] for _, c in choice]
        n = math.prod(len(o) for o in opts) if opts else 1
        if n > 4096:
            choice = [(kd, c) for kd, c in choice if kd in STRONG]
            opts = [list(c) + [None] for _, c in choice]
            n = math.prod(len(o) for o in opts) if opts else 1
            if n > 4096:
                return None
        base_sc = sum(STRONG[kd] for kd, c, t in slots if kd in STRONG and len(c) == 1)
        combos = []
        for pick in itertools.product(*opts) if opts else [()]:
            S = set(fixed)
            sc = base_sc
            for (kd, c), p in zip(choice, pick):
                if p is not None and p not in S:
                    S.add(p)
                    sc += STRONG.get(kd) or WEAK.get(kd)
            if not S or len({ix.cur[k] for k in S}) > 1:
                continue
            combos.append((sc, frozenset(S)))
        if not combos:
            return None
        combos = list({c[1]: c for c in combos}.values())
        combos.sort(key=lambda c: (-c[0], len(c[1])))

        def match(S):
            D = self.value(S)
            if exact_cur:
                return abs(D - cash) <= EPS
            # converted at our reference rate (no original amount on the line): only a named payer citing its
            # documents in full is trusted, the band is too wide otherwise
            return name_ok and not choice and 0.965 * D <= cash <= 1.003 * D

        exact = [c for c in combos if match(c[1])]
        if exact:
            best = exact[0]
            if any(c[0] == best[0] and c[1] != best[1] for c in exact):
                return None
            S = best[1]
            if exact_cur and ix.cur[next(iter(S))] == line.currency:
                return ("exact", best[0], [(k, ix.open[k], "") for k in S])
            rows = self.explain(S, line, cash, exact_cur, party, strict=True)
            return ("fx", best[0], rows) if rows else None

        weak_only = not any(kd in STRONG for kd, _, _ in slots)  # only cut / mistyped references
        top_sc = combos[0][0]
        tops = [c for c in combos if c[0] == top_sc]
        # 0) credit notes the payer netted without naming them (related to a cited invoice, or one CN == difference)
        if exact_cur and len(tops) == 1 and name_ok:
            S = set(tops[0][1])
            diff = round(cash - self.value(S), 2)
            if diff < -EPS:
                cn = [k for k in P if k not in S and ix.sign[k] < 0]
                rel = {ix.alnum[k] for k in S}
                cr = [k for k in cn if ix.related[k] in rel]
                one = [k for k in cn if abs(ix.open[k] + diff) < EPS]
                pick = None
                if cr and abs(sum(ix.open[k] for k in cr) + diff) < EPS:
                    pick = cr
                elif len(one) == 1:
                    pick = one
                if pick:
                    S2 = S | set(pick)
                    if len({ix.cur[k] for k in S2}) == 1:
                        fx = ix.cur[next(iter(S2))] != line.currency
                        return ("complete", top_sc, [(k, ix.open[k], "FX_DIFFERENCE" if fx and ix.sign[k] > 0 else "")
                                                     for k in S2])
        # 1) strict explanation (discount in terms, withholding, FX): pins the cash to the cent
        if (name_ok or all_strong) and not (weak_only and not name_ok) and (exact_cur or name_ok):
            got = [(c, self.explain(c[1], line, cash, exact_cur, party, strict=True)) for c in tops[:4]]
            got = [(c, r) for c, r in got if r]
            if len(got) == 1:
                return ("explain", top_sc, got[0][1])
            if len(got) > 1:
                return None
        if not name_ok or weak_only:
            return None
        # 2) completion with uncited open items of the party (credit notes first)
        if exact_cur and len(tops) == 1:
            S = set(tops[0][1])
            rest = [k for k in P if k not in S]
            diff = round(cash - self.value(S), 2)
            sol = self.complete(S, rest, diff, deep=self.truncated(line, slots))
            if sol is not None:
                S2 = S | set(sol)
                if len({ix.cur[k] for k in S2}) == 1:
                    fx = ix.cur[next(iter(S2))] != line.currency
                    return ("complete", top_sc, [(k, ix.open[k], "FX_DIFFERENCE" if fx and ix.sign[k] > 0 else "")
                                                 for k in S2])
            # 2b) credit notes netted AND a precise deduction (discount in terms / late discount / withholding)
            if len(tops) == 1:
                rows = self.cn_and_deduction(set(tops[0][1]), P, line, cash, exact_cur, party)
                if rows:
                    return ("complete", top_sc, rows)
        # 3) any explanation (overpayment, partial, bank charges) of the single best set -- not when the text is cut
        #    after an unresolved reference fragment (the uncited documents explain the difference)
        if len(tops) > 1 or self.truncated(line, slots):
            return None
        rows = self.explain(tops[0][1], line, cash, exact_cur, party, strict=False)
        if rows:
            return ("explain", top_sc, rows)
        return None

    def cn_and_deduction(self, S, P, line, cash, exact_cur, party):
        ix = self.ix
        cn = [k for k in P if k not in S and ix.sign[k] < 0]
        if not cn:
            return None
        rel = {ix.alnum[k] for k in S}
        groups = [[k for k in cn if ix.related[k] in rel]]
        if len(cn) <= 12:
            groups.append(cn)
        groups += [[k] for k in cn[:30]]
        seen, found = set(), []
        for gr in groups:
            key = frozenset(gr)
            if not gr or key in seen:
                continue
            seen.add(key)
            S2 = S | set(gr)
            if self.value(S2) <= cash + EPS:
                continue  # documents must exceed the cash: a deduction explains the rest
            rows = self.explain(S2, line, cash, exact_cur, party, strict=False)
            if rows and self.precise(rows):
                found.append(rows)
        return found[0] if len(found) == 1 else None

    def precise(self, rows):
        """deductions pinned to the cent: discount, withholding, FX, or a PARTIAL that is exactly a late discount."""
        ix = self.ix
        for k, amt, e in rows:
            if e in STRICT_EXC:
                continue
            if e != "PARTIAL":
                return False
            tm = self.terms.get(ix.terms[k])
            if not tm or abs(round(ix.open[k] - amt, 2) - round(tm[0] / 100.0 * ix.total[k], 2)) > 0.011:
                return False
        return True

    def complete(self, S, rest, diff, deep=False):
        """unique smallest set of the party's other items with signed value == diff."""
        ix = self.ix
        if abs(diff) < EPS or not rest:
            return None
        val = {k: round(ix.sign[k] * ix.open[k], 2) for k in rest}
        cn = [k for k in rest if ix.sign[k] < 0]
        rel = {ix.alnum[k] for k in S}
        cn_rel = [k for k in cn if ix.related[k] and ix.related[k] in rel]
        # 1) credit notes related to the cited invoices, 2) all credit notes
        for group in (cn_rel, cn):
            if group and len(group) <= 12:
                if abs(sum(val[k] for k in group) - diff) < EPS:
                    return list(group)
        # 3) small subset search over all uncited items
        rest = rest[:120]
        sols = []
        maxk = 3 if len(rest) > 40 else 4 if len(rest) > 18 else 5
        cents = {k: int(round(val[k] * 100)) for k in rest}
        target = int(round(diff * 100))
        by_val = defaultdict(list)
        for k in rest:
            by_val[cents[k]].append(k)
        for size in range(1, maxk + 1):
            if size == 1:
                for k in by_val.get(target, ()):
                    sols.append([k])
            else:
                for combo in itertools.combinations(rest, size - 1):
                    s = sum(cents[k] for k in combo)
                    for k in by_val.get(target - s, ()):
                        if k not in combo and all(rest.index(k) > rest.index(j) for j in combo):
                            sols.append(list(combo) + [k])
                    if len(sols) > 1:
                        break
            if sols:
                break
        if len(sols) == 1:
            return sols[0]
        if sols or not deep:
            return None
        # 4) cut text: any number of uncited items, accepted only if the subset is the only one
        return unique_subset([cents[k] for k in rest], target, rest)

    # --- per line ------------------------------------------------------------------------------------------------
    def solve(self, line):
        if not line.amount or abs(line.amount) < 0.005:
            return None
        side = "AR" if line.amount > 0 else "AP"
        if side == "AP" and self.in_payment_run(line):
            return None
        toks = tokens(line.text, line.remittance_info if line.remittance_info != line.text else "")
        if not toks:
            return None
        votes = self.global_parties(line.entity, side, toks, line.booking_date, line.amount_abs)
        words = set(norm_name(" ".join([line.text or "", line.remittance_info or "", line.party_name_raw or ""])).split())
        # payers named in the text: their items are searched too (number without prefix too ambiguous entity-wide)
        for p in self.name_parties(line.entity, side, words):
            votes[p] += 0.5
        if not votes:
            return None
        results = []
        for party, v in votes.most_common(6):
            r = self.solve_party(line, side, party, toks, words)
            if r and r[2]:
                results.append((r, party, v))
        if not results:
            return None
        rank = {"exact": 0, "fx": 1, "complete": 2, "explain": 3}
        results.sort(key=lambda x: (rank[x[0][0]], -x[0][1], -x[2]))
        best = results[0]
        if len(results) > 1:
            nxt = results[1]
            if rank[nxt[0][0]] == rank[best[0][0]] and nxt[0][1] == best[0][1]:
                return None
        return self.instalment(best[0], line)

    def instalment(self, res, line):
        """'INSTALMENT k/n' on a one-invoice line: every instalment but the last leaves the invoice open (PARTIAL)."""
        how, sc, rows = res
        ix = self.ix
        inv = [r for r in rows if ix.sign[r[0]] > 0]
        if len(inv) != 1:
            return res
        m = INSTAL.search(f"{line.text or ''} {line.remittance_info or ''}".upper())
        if not m:
            return res
        a, b = int(m.group(1)), int(m.group(2))
        if a < b:
            rows = [(k, amt, "PARTIAL" if ix.sign[k] > 0 else e) for k, amt, e in rows]
        else:
            fx = ix.cur[inv[0][0]] != line.currency
            rows = [(k, amt, ("FX_DIFFERENCE" if fx else "") if ix.sign[k] > 0 and e == "PARTIAL" else e)
                    for k, amt, e in rows]
        return (how, sc, rows)


CONF = {"exact": 0.95, "fx": 0.9, "complete": 0.85, "explain": 0.8}


def consume(ix, k, amt, exc, fx=False):
    """book an allocation on the running open amount of item k."""
    if exc == "RETURNED":
        ix.open[k] = round(ix.open[k] + amt, 2)
    elif exc in ("", "PARTIAL", "SHORT_DISPUTE", "DUPLICATE_PAYMENT") and not fx:
        ix.open[k] = max(0.0, round(ix.open[k] - amt, 2))
    else:
        ix.open[k] = 0.0


def run(ctx: Ctx):
    sv = Solver(ctx)
    ix = sv.ix
    pos = {iid: k for k, iid in enumerate(ix.iid)}
    bdate = dict(zip(ctx.bank.bank_txn_id, ctx.bank.booking_date))
    # allocations of the matchers that ran before (payment runs, batches, advices, PDF cases) consume open amounts
    prev = getattr(ctx, "allocations", None)
    events = []
    if prev is not None and len(prev):
        for t, iid, amt, exc in zip(prev.bank_txn_id, prev.item_id, prev.amount, prev.exception.fillna("")):
            k = pos.get(iid)
            if k is not None and t in bdate:
                events.append((bdate[t], k, float(amt), exc))
        events.sort(key=lambda e: e[0])
    ev = 0
    bank = ctx.unclaimed().sort_values(["booking_date", "bank_txn_id"])
    rows = []
    for line in bank.itertuples(index=False):
        while ev < len(events) and events[ev][0] <= line.booking_date:
            _, k, amt, exc = events[ev]
            consume(ix, k, amt, exc)
            ev += 1
        r = sv.solve(line)
        if not r:
            continue
        how, _, alloc = r
        if not alloc:
            continue
        for k, amt, exc in alloc:
            rows.append((line.bank_txn_id, ix.iid[k], round(float(amt), 2), exc, CONF[how], f"{NAME}:{how}"))
            consume(ix, k, amt, exc, fx=(how == "fx"))
    return alloc_frame(rows)
