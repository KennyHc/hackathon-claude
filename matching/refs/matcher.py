"""K2 refs matcher (stage 2 of the cascade): explicit document references in the bank text → open items.

Interface (docs/PLAN_LUIGI.md):  match(phase, lines, items, claimed, **ctx) -> DataFrame
    bank_txn_id, item_id, amount, exception, confidence, method

Precision first: a line is claimed only when the cited items (+ linked credit notes) explain the cash exactly
(same currency, ±0.02) or through a POLICIES §3 rule (discount within terms, withholding, FX, instalment, retention,
advance, single cited item paid partially).
"""
from __future__ import annotations

import bisect
import os
import re
import sys
from collections import defaultdict
from itertools import combinations, product

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from integration.lib import fx  # noqa: E402
from integration.lib.masters import masters_dir as _masters_dir  # noqa: E402

OUT_COLS = ["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"]
FACTORY = {"ES00": ("ES00", "ES01", "ES02", "PT01")}
WITHHOLDING_COUNTRIES = {"MX", "BR", "MA"}
WHT_RATES = (0.0125, 0.015, 0.10)
NON_ALNUM = re.compile(r"[^A-Z0-9]")
SPLIT = re.compile(r"[\s|,;*#\\:()\[\]{}<>=+'\"]+")
INSTAL_RE = re.compile(
    r"(?:PLAZO|RATA|INSTAL+MENT|VENCIMIENTO|VTO\.?|ECHEANCE|ÉCHÉANCE|PARCELA|PRESTA[CÇ][AÃ]O|TEILZAHLUNG|RATE|CUOTA|"
    r"TERMIJN|RATY?|INST\.?)\s*(?:N[ºO°.]?\s*)?(\d{1,2})\s*(?:/|DE|OF|SU|VON|SUR)\s*(\d{1,2})\b")
PARTIAL_RE = re.compile(r"A CUENTA|PAGO PARCIAL|PARTIAL|ACOMPTE|ON ACCOUNT|ACCONTO|ABSCHLAG|TEILZAHLUNG|PARCIAL")
RELEASE_RE = re.compile(r"RETENTION RELEASE|LIBERACI[OÓ]N|RETENCI[OÓ]N|RETENUE|EINBEHALT|RITENUTA|RETEN")
TOL = 0.02


def _norm(s: str) -> str:
    return NON_ALNUM.sub("", s.upper())


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


# ----------------------------------------------------------------------------------------------------------------
# tokens
def extract_tokens(line) -> list[tuple[str, str]]:
    """Ordered (raw_upper, norm) candidate-reference tokens of a bank line (≥4 alnum chars, ≥3 digits)."""
    t = str(line.get("text", ""))
    fields = [t, line.get("remittance_info", ""), line.get("ref2", ""), line.get("ref1", ""),
              line.get("end_to_end_id", "")]
    if " | " in t:
        fields.append(t.replace(" | ", ""))
    out, seen = [], set()
    for f in fields:
        if not f:
            continue
        for tok in SPLIT.split(str(f).upper()):
            if not tok:
                continue
            parts = [tok]
            if "/" in tok:
                parts += [p for p in tok.split("/") if p]
            for p in parts:
                n = _norm(p)
                if len(n) < 3 or n in seen:
                    continue
                nd = sum(c.isdigit() for c in n)
                if nd < 3:
                    continue
                seen.add(n)
                out.append((p, n))
    return out


# ----------------------------------------------------------------------------------------------------------------
class RefMatcher:
    def __init__(self, phase: str, items: pd.DataFrame, masters_dir: str, pdf: pd.DataFrame | None = None):
        self.phase = phase
        self.mdir = masters_dir
        it = items.reset_index(drop=True)
        self.n = len(it)
        self.iid = it.item_id.values
        self.itype = it.item_type.fillna("").values if "item_type" in it else it.item_id.str[:3].values
        self.ent = it.entity.values
        self.party = it.party_id.values
        self.doc = it.document_number.fillna("").values
        self.issue = it.issue_date.fillna("").values
        self.ccy = it.currency.values
        self.total = it.total_amount.astype(float).values
        self.open = it.open_amount.astype(float).values.copy()
        self.terms = it.payment_terms.fillna("").values if "payment_terms" in it else np.array([""] * self.n)
        self.reldoc = it.related_document.fillna("").values if "related_document" in it else np.array([""] * self.n)
        self.idx_of = {x: i for i, x in enumerate(self.iid)}
        self.normdoc = np.array([_norm(d) for d in self.doc], dtype=object)

        # exact index: (entity, norm) -> [idx]; digit-only for letter-prefixed docs
        self.ex = defaultdict(list)
        self.dg = defaultdict(list)
        self.po = defaultdict(list)
        for i in range(self.n):
            nd = self.normdoc[i]
            if not nd:
                continue
            self.ex[(self.ent[i], nd)].append(i)
            d = _digits(nd)
            if d != nd and len(d) >= 6:
                self.dg[(self.ent[i], d)].append(i)
        self.sfx = defaultdict(list)  # (entity, last-6 digits) for invoices
        for i in range(self.n):
            d = _digits(self.normdoc[i])
            if len(d) >= 7:
                self.sfx[(self.ent[i], d[-6:])].append(i)
        if "po_number" in it:
            for i, p in enumerate(it.po_number.fillna("").values):
                pn = _norm(p)
                if len(pn) >= 5 and sum(c.isdigit() for c in pn) >= 3:
                    self.po[(self.ent[i], pn)].append(i)
        # sorted keys per entity for prefix lookup (truncated refs)
        self.sorted_keys = {}
        tmp = defaultdict(list)
        for (e, k) in self.ex.keys():
            tmp[e].append(k)
        for e, ks in tmp.items():
            self.sorted_keys[e] = sorted(ks)
        # party -> items
        self.by_party = defaultdict(list)
        for i in range(self.n):
            self.by_party[(self.ent[i], self.party[i])].append(i)
        # credit notes linked to an invoice document: (entity, party, norm related doc) -> [cn idx]
        self.cn_by_rel = defaultdict(list)
        for i in range(self.n):
            if self.itype[i] in ("ARC", "APC") and self.reldoc[i]:
                self.cn_by_rel[(self.ent[i], _norm(self.reldoc[i]))].append(i)

        # cancelled invoices (an "Anulación de factura" credit note relates to them) and those credit notes
        self.blocked = set()
        if "cn_reason" in it:
            reason = it.cn_reason.fillna("").values
            for i in range(self.n):
                if self.itype[i] in ("ARC", "APC") and reason[i].upper().startswith("ANULACI") and self.reldoc[i]:
                    self.blocked.add(i)
                    for j in self.ex.get((self.ent[i], _norm(self.reldoc[i])), []):
                        if self.party[j] == self.party[i]:
                            self.blocked.add(j)
        # payment terms with discount
        self.disc = {}
        pt = pd.read_csv(os.path.join(masters_dir, "payment_terms.csv"), dtype=str, keep_default_na=False)
        for _, r in pt.iterrows():
            if r.early_payment_discount_pct and r.discount_days:
                self.disc[r.code] = (float(r.early_payment_discount_pct) / 100, int(r.discount_days))
        cust = pd.read_csv(os.path.join(masters_dir, "customers.csv"), dtype=str, keep_default_na=False,
                           usecols=["customer_id", "country"])
        self.ccountry = dict(zip(cust.customer_id, cust.country))

        # PDF clauses (optional)
        self.inst_n, self.ret_pct, self.adv_amt, self.replaced_by = {}, {}, {}, {}
        if pdf is not None and len(pdf):
            self._load_pdf(pdf)

    # ------------------------------------------------------------------
    def _load_pdf(self, pdf):
        key = {(e, _norm(d)): i for i, (e, d) in enumerate(zip(self.ent, self.doc)) if d}
        for r in pdf.itertuples(index=False):
            i = key.get((r.entity, _norm(str(r.document_number))))
            if i is None:
                continue
            n = getattr(r, "n_instalments", None)
            if n is not None and pd.notna(n) and 2 <= int(n) <= 24:
                self.inst_n[i] = int(n)
            p = getattr(r, "retention_pct", None)
            if p is not None and pd.notna(p) and 0 < float(p) < 50:
                self.ret_pct[i] = float(p) / 100
            a = getattr(r, "advance_amount", None)
            if a is not None and pd.notna(a) and float(a) > 0:
                self.adv_amt[i] = float(a)
            rep = getattr(r, "replaces_document", None)
            if rep is not None and isinstance(rep, str) and rep:
                j = key.get((r.entity, _norm(rep)))
                if j is not None:
                    self.replaced_by[j] = i

    def fill(self, line, cand, p, cash, valid):
        """Cash > cited items (truncated list): find the remainder among the party's other open items of the same
        currency: one or two items, or the oldest-first prefix. Same-currency lines only."""
        if not cand or line["currency"] != self.ccy[cand[0]]:
            return None
        sign = {i: (-1 if self.itype[i] in ("ARC", "APC") else 1) for i in cand}
        gross = sum(sign[i] * self.open[i] for i in cand)
        rem = round(cash - gross, 2)
        if rem <= 0.01:
            return None
        inv_t = "ARI" if self.itype[cand[0]] in ("ARI", "ARC") else "API"
        pool = [k for k in self.by_party.get(p, []) if k not in cand and valid(k) and self.itype[k] == inv_t
                and self.ccy[k] == line["currency"]]
        if not pool or len(pool) > 150:
            return None
        cents = {}
        for k in pool:
            cents.setdefault(int(round(self.open[k] * 100)), []).append(k)
        r = int(round(rem * 100))
        sols = []
        if r in cents and len(cents[r]) == 1:
            sols.append([cents[r][0]])
        if len(pool) <= 80:
            for a in pool:
                ca = int(round(self.open[a] * 100))
                for b in cents.get(r - ca, []):
                    if b > a:
                        sols.append([a, b])
        # oldest-first prefix
        po = sorted(pool, key=lambda k: (self.issue[k], self.doc[k]))
        acc = 0
        for j, k in enumerate(po):
            acc += int(round(self.open[k] * 100))
            if abs(acc - r) <= 1:
                sols.append(po[: j + 1])
                break
            if acc > r:
                break
        uniq = {tuple(sorted(x)) for x in sols}
        if len(uniq) != 1:
            return None
        return cand + list(uniq.pop())

    @staticmethod
    def variants(n):
        out = set()
        for j in range(len(n) - 1):
            if n[j] != n[j + 1]:
                out.add(n[:j] + n[j + 1] + n[j] + n[j + 2:])
        for j, ch in enumerate(n):
            if ch.isdigit():
                for d in "0123456789":
                    if d != ch:
                        out.add(n[:j] + d + n[j + 1:])
        return out

    def replacement_guess(self, i):
        """Cancelled invoice without PDF: open invoices of the same party issued on/after the cancelling credit note
        with the same total (the replacement usually re-bills the same amount)."""
        return [k for k in self.by_party.get((self.ent[i], self.party[i]), [])
                if k != i and self.itype[k] == self.itype[i] and k not in self.blocked
                and self.issue[k] >= self.issue[i] and abs(self.total[k] - self.total[i]) <= 0.01][:3]

    # ------------------------------------------------------------------
    def lookup(self, ents, n):
        """Exact (norm doc), then digit-only, then PO. Returns (idx list, how)."""
        for e in ents:
            r = self.ex.get((e, n))
            if r:
                return list(r), "exact"
        if n.isdigit():
            out = []
            for e in ents:
                out += self.dg.get((e, n), [])
            if out:
                return out, "digits"
        out = []
        for e in ents:
            out += self.po.get((e, n), [])
        if out:
            return out, "po"
        return [], ""

    def prefix_lookup(self, ents, n, cap=30):
        out = []
        for e in ents:
            ks = self.sorted_keys.get(e)
            if not ks:
                continue
            j = bisect.bisect_left(ks, n)
            while j < len(ks) and ks[j].startswith(n) and len(out) <= cap:
                if ks[j] != n:
                    out += self.ex[(e, ks[j])]
                j += 1
        return out if len(out) <= cap else []

    # ------------------------------------------------------------------
    def candidate_slots(self, line, ents):
        """List of slots: (how, [idx...]) in text order."""
        toks = extract_tokens(line)
        slots, prev_full = [], None
        used_keys = set()
        for raw, n in toks:
            idx, how = self.lookup(ents, n)
            if not idx and prev_full and n.isdigit() and 3 <= len(n) < len(prev_full):
                comp = prev_full[: len(prev_full) - len(n)] + n
                idx, how = self.lookup(ents, comp)
                how = "suffix" if idx else ""
            if not idx and len(n) >= 7 and sum(c.isdigit() for c in n) >= 5:
                idx = self.prefix_lookup(ents, n)
                how = "prefix" if idx else ""
            if not idx and n.isdigit() and 6 <= len(n) <= 8:
                for e in ents:
                    idx += self.sfx.get((e, n[-6:]), [])
                idx = [i for i in idx if _digits(self.normdoc[i]).endswith(n)]
                how = "esuffix" if 0 < len(idx) <= 40 else ""
                if not how:
                    idx = []
            if idx:
                key = tuple(sorted(idx))
                if key in used_keys:
                    continue
                used_keys.add(key)
                slots.append((how, idx, n))
                if how in ("exact", "digits"):
                    prev_full = self.normdoc[idx[0]]
            else:
                slots.append(("none", [], n))
        return slots

    # ------------------------------------------------------------------
    def explain(self, line, cand, cash, lccy, bdate, text, single_slot, strict=True):
        """cand: list of item idx (distinct). Returns (rows[(idx, amount, exc)], method) or None."""
        ccys = {self.ccy[i] for i in cand}
        if len(ccys) != 1:
            return None
        iccy = ccys.pop()
        inv = [i for i in cand if self.itype[i] in ("ARI", "API")]
        cns = [i for i in cand if self.itype[i] in ("ARC", "APC")]
        o = {i: self.open[i] for i in cand}
        if any(o[i] <= 0.005 for i in cand):
            return None
        gross = sum(o[i] for i in inv) - sum(o[i] for i in cns)
        if not inv:  # credit notes only (supplier refund / paid-out)
            gross = sum(o[i] for i in cns)
        if gross <= 0:
            return None
        full = [(i, o[i], "") for i in cand]

        # FX line
        if lccy != iccy:
            oc, oa = line.get("orig_currency", ""), line.get("orig_amount", "")
            if oc == iccy and oa:
                c = abs(float(oa))
                if abs(c - gross) <= TOL:
                    return [(i, o[i], "FX_DIFFERENCE" if i in inv else "") for i in cand], "fx_orig"
                return None
            try:
                c = fx.convert(cash, lccy, iccy, bdate, self.mdir)
            except Exception:
                return None
            if False and abs(c - gross) <= 0.03 * gross:
                return [(i, o[i], "FX_DIFFERENCE" if i in inv else "") for i in cand], "fx_ref"
            return None

        c = cash
        if abs(c - gross) <= TOL:
            return full, "full"
        short = gross - c
        if short > 0 and inv:
            # discount within terms (on total), possibly only some invoices eligible
            elig, late = [], []
            for i in inv:
                d = self.disc.get(self.terms[i])
                if d:
                    lim = (pd.Timestamp(self.issue[i]) + pd.Timedelta(days=d[1] + 3)).strftime("%Y-%m-%d")
                    (elig if bdate <= lim else late).append(i)
            if elig or late:
                dsum = sum(round(self.disc[self.terms[i]][0] * self.total[i], 2) for i in elig)
                if elig and abs(short - dsum) <= TOL * max(1, len(elig)):
                    return [(i, o[i], "SHORT_DISCOUNT" if i in elig else "") for i in cand], "discount"
                lsum = sum(round(self.disc[self.terms[i]][0] * self.total[i], 2) for i in elig + late)
                if late and abs(short - lsum) <= TOL * max(1, len(elig + late)) and len(inv) == 1:
                    i = inv[0]
                    return [(i, round(c + sum(o[k] for k in cns), 2), "PARTIAL")] + \
                        [(k, o[k], "") for k in cns], "late_discount"
            # withholding (customers in MX/BR/MA)
            if self.itype[inv[0]] == "ARI" and self.ccountry.get(self.party[inv[0]], "") in WITHHOLDING_COUNTRIES:
                for r in WHT_RATES:
                    w = sum(round(r * self.total[i], 2) for i in inv)
                    if abs(short - w) <= TOL * len(inv):
                        return [(i, o[i], "SHORT_WITHHOLDING" if i in inv else "") for i in cand], "withholding"
            if strict:
                return None
            if len(inv) == 1 and not cns:
                i = inv[0]
                t = self.total[i]
                # instalments: text k/n or PDF
                m = INSTAL_RE.search(text)
                ns = []
                if m and 2 <= int(m.group(2)) <= 24:
                    ns.append(int(m.group(2)))
                if i in self.inst_n:
                    ns.append(self.inst_n[i])
                for nn in ns:
                    if abs(c - round(t / nn, 2)) <= 0.02 * nn or (m and abs(c - t / nn) <= max(0.05, 0.001 * c)):
                        return [(i, c, "PARTIAL")], "instalment"
                # retention (PDF pct, else 5/10 % exact)
                rs = [self.ret_pct[i]] if i in self.ret_pct else [0.05, 0.10]
                for r in rs:
                    if abs(c - round(t * (1 - r), 2)) <= TOL and abs(o[i] - t) <= TOL:
                        return [(i, c, "PARTIAL")], "retention"
                # advance (PDF)
                if i in self.adv_amt and abs(c - (t - self.adv_amt[i])) <= TOL:
                    return [(i, c, "PARTIAL")], "advance"
                # single cited item, less cash, keyword or strong single reference
                if single_slot and 0 < c < o[i] and self.itype[i] == "ARI":
                    return [(i, c, "PARTIAL")], "partial_single"
            # n_to_m: all full but one partial
            if len(inv) >= 2 and self.itype[inv[0]] == "ARI":
                opts = []
                for j in inv:
                    rem = c - (gross - o[j])
                    if TOL < rem < o[j] - TOL:
                        ratio = rem / o[j]
                        roundness = abs(ratio * 10 - round(ratio * 10))
                        opts.append((roundness, j, rem))
                if opts:
                    opts.sort()
                    if len(opts) == 1 or (opts[0][0] < 0.002 and opts[1][0] > 0.002):
                        _, j, rem = opts[0]
                        return [(i, (round(rem, 2) if i == j else o[i]), ("PARTIAL" if i == j else ""))
                                for i in cand], "partial_one"
        if short < 0 and len(inv) == 1 and not cns:
            # retention release / last instalment paid against remaining open handled by "full"; release when ledger
            # does not know the first payment: cash = total * r and text says release
            i = inv[0]
            if RELEASE_RE.search(text):
                pass
        if short > 0 and len(inv) == 1 and not cns and RELEASE_RE.search(text):
            i = inv[0]
            t = self.total[i]
            for r in ([self.ret_pct[i]] if i in self.ret_pct else [0.05, 0.10]):
                if abs(c - round(t * r, 2)) <= TOL:
                    return [(i, c, "")], "retention_release"
        return None

    # ------------------------------------------------------------------
    def match_line(self, line):
        amt = float(line["amount"])
        if amt == 0:
            return None
        receipt = amt > 0
        cash = abs(amt)
        ents = FACTORY.get(line["entity"], (line["entity"],))
        bdate = line["booking_date"]
        text = " ".join(str(line.get(k, "")) for k in ("text", "remittance_info", "ref1", "ref2")).upper()
        slots = self.candidate_slots(line, ents)
        if not any(s[1] for s in slots):
            return None
        ok_types = ("ARI", "ARC", "APC") if receipt else ("API", "APC", "ARC")

        def valid(i):
            return self.itype[i] in ok_types and self.open[i] > 0.005 and self.issue[i] <= bdate \
                and i not in self.blocked

        # replaced invoices: a cited cancelled invoice → its replacement
        rslots = []
        for how, idx, n in slots:
            new = []
            for i in idx:
                if i in self.replaced_by:
                    new.append(self.replaced_by[i])
                elif i in self.blocked and self.itype[i] in ("ARI", "API"):
                    new += self.replacement_guess(i)
                else:
                    new.append(i)
            new = [i for i in dict.fromkeys(new) if valid(i)]
            if receipt:
                # supplier refund: a receipt citing an AP invoice settles its APC (related_document)
                ext = []
                for i in new:
                    if self.itype[i] == "API":
                        ext += [k for k in self.cn_by_rel.get((self.ent[i], self.normdoc[i]), []) if valid(k)]
                new = [i for i in new if self.itype[i] != "API"] + ext
            else:
                ext = []
                for i in new:
                    if self.itype[i] == "ARI":
                        pass
                new = [i for i in new if self.itype[i] != "ARI"]
            if new:
                rslots.append((how, new, n))
        if not rslots:
            return None

        # parties: rank by number of slots covered (strong slots weigh more)
        cover = defaultdict(float)
        for how, idx, n in rslots:
            w = 1.0 if how in ("exact", "suffix", "digits") else 0.5
            for p in {(self.ent[i], self.party[i]) for i in idx}:
                cover[p] += w
        parties = sorted(cover, key=lambda p: -cover[p])[:4]
        best_cov = cover[parties[0]]
        n_strong = sum(1 for s in rslots if s[0] in ("exact", "suffix", "digits"))
        weak_ok = []
        for p in parties:
            if cover[p] < best_cov - 1e-9 and len(parties) > 1 and cover[p] < 1:
                continue
            ps = [[i for i in idx if (self.ent[i], self.party[i]) == p] for how, idx, n in rslots]
            hows = [rslots[k][0] for k in range(len(rslots)) if ps[k]]
            ps = [x for x in ps if x]
            if not ps:
                continue
            # party-suffix: unresolved digit tokens → this party's open items ending with them
            got = {i for x in ps for i in x}
            pitems = [k for k in self.by_party.get(p, []) if valid(k) and k not in got]
            unres = [n for how, idx, n in slots if how == "none"] + \
                [n for how, idx, n in rslots if how not in ("exact",) and
                 not any((self.ent[i], self.party[i]) == p for i in idx)]
            for n in unres:
                if n.isdigit() and len(n) >= 5 and pitems:
                    hit = [k for k in pitems if self.normdoc[k].endswith(n)]
                    if len(hit) == 1:
                        ps.append(hit)
                        hows.append("psuffix")
                        got.add(hit[0])
                        pitems.remove(hit[0])
            combos = 1
            for x in ps:
                combos *= len(x)
            if combos > 64:
                ps = [x[:1] if len(x) > 1 else x for x in ps]
            for choice in product(*ps):
                cand = list(dict.fromkeys(choice))
                variants = [(cand, "")]
                linked = []
                for i in cand:
                    if self.itype[i] in ("ARI", "API"):
                        linked += [k for k in self.cn_by_rel.get((self.ent[i], self.normdoc[i]), [])
                                   if valid(k) and k not in cand and (self.ent[k], self.party[k]) == p]
                linked = list(dict.fromkeys(linked))
                if linked:
                    variants.append((cand + linked, "+cn"))
                inv_c = [i for i in cand if self.itype[i] in ("ARI", "API")]
                if inv_c:
                    pcn = [k for k in self.by_party.get(p, []) if self.itype[k] in ("ARC", "APC") and valid(k)
                           and k not in cand and k not in linked]
                    if 0 < len(pcn) <= 6:
                        for rsz in range(1, len(pcn) + 1):
                            for sub in combinations(pcn, rsz):
                                variants.append((cand + linked + list(sub), "+pcn"))
                if 3 <= len(cand) <= 12:
                    for k in range(len(cand)):
                        variants.append((cand[:k] + cand[k + 1:], "-1"))
                for vc, tag in variants:
                    r = self.explain(line, vc, cash, line["currency"], bdate, text, False, strict=True)
                    if r:
                        return (r[0], r[1] + tag), p
                fill = self.fill(line, cand + linked, p, cash, valid)
                if fill:
                    r = self.explain(line, fill, cash, line["currency"], bdate, text, False, strict=True)
                    if r and r[1] == "full":
                        return (r[0], "fill"), p
                uniq = len(ps) == 1 and hows[0] in ("exact", "digits", "suffix") and len(rslots) == 1 \
                    and len(rslots[0][1]) == 1 and len(self.normdoc[cand[0]]) >= 6
                weak_ok.append((variants[:2] if linked else variants[:1], uniq, p))
        # fuzzy: 1 transposition / 1 substitution of a cited token → an item that explains the cash exactly
        hits = {}
        for how, idx, n in slots:
            if len(n) < 6 or sum(c.isdigit() for c in n) < 4:
                continue
            for v in self.variants(n):
                for e in ents:
                    for i in self.ex.get((e, v), []):
                        if valid(i) and self.itype[i] in ("ARI", "API", "APC", "ARC") and i not in hits:
                            r = self.explain(line, [i], cash, line["currency"], bdate, text, False, strict=True)
                            if r:
                                hits[i] = r
        if len(hits) == 1:
            i, r = hits.popitem()
            return (r[0], "fuzzy"), (self.ent[i], self.party[i])
        # same party, a different open item with exactly the cash amount (customer quoted the wrong invoice)
        for p in parties[:2]:
            if cover[p] < best_cov - 1e-9:
                continue
            same = [k for k in self.by_party.get(p, []) if valid(k) and self.itype[k] in ("ARI", "API")
                    and self.ccy[k] == line["currency"] and abs(self.open[k] - cash) <= 0.005]
            if len(same) == 1:
                return ([(same[0], self.open[same[0]], "")], "party_amt"), p
        for vs, uniq, p in weak_ok:
            for vc, tag in vs:
                r = self.explain(line, vc, cash, line["currency"], bdate, text, uniq and not tag, strict=False)
                if r:
                    return (r[0], r[1] + tag), p
        return None


CONF = {"full": 0.97, "fx_orig": 0.95, "fx_ref": 0.85, "discount": 0.93, "withholding": 0.9, "instalment": 0.85,
        "retention": 0.8, "advance": 0.8, "partial_single": 0.7, "partial_one": 0.6, "late_discount": 0.75,
        "retention_release": 0.8}


def match(phase: str, lines: pd.DataFrame, items: pd.DataFrame, claimed: pd.DataFrame, **ctx) -> pd.DataFrame:
    mdir = ctx.get("masters_dir") or _masters_dir(phase)
    pdf = ctx.get("pdf_clauses")
    m = RefMatcher(phase, items, mdir, pdf)
    # claimed allocations of earlier stages, applied in booking order
    ev = []
    if claimed is not None and len(claimed):
        cl = claimed.copy()
        if "booking_date" not in cl:
            cl["booking_date"] = ""
        cl = cl.sort_values(["booking_date", "bank_txn_id"])
        for r in cl.itertuples(index=False):
            ev.append((str(r.booking_date), str(r.bank_txn_id), r.item_id, float(r.amount), str(r.exception or "")))
    ln = lines.sort_values(["booking_date", "bank_txn_id"]).to_dict("records")
    rows, k = [], 0
    for line in ln:
        key = (line["booking_date"], line["bank_txn_id"])
        while k < len(ev) and (ev[k][0], ev[k][1]) < key:
            _, _, it, a, e = ev[k]
            i = m.idx_of.get(it)
            if i is not None:
                if e == "RETURNED":
                    m.open[i] += a
                elif e != "DUPLICATE_PAYMENT":
                    m.open[i] = max(0.0, m.open[i] - a)
            k += 1
        try:
            res = m.match_line(line)
        except Exception:
            res = None
        if not res:
            continue
        (alloc, how), p = res
        for i, a, e in alloc:
            rows.append((line["bank_txn_id"], m.iid[i], round(float(a), 2), e, CONF.get(how.split("+")[0].split("-")[0], 0.6), how))
            m.open[i] = max(0.0, m.open[i] - a)
    return pd.DataFrame(rows, columns=OUT_COLS)
