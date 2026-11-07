"""Who paid?  Resolve a customer receipt (bank line) to candidate customers of the line's entity.

Signals
- IBAN of the payer: `counterparty_account` (camt) or an IBAN inside the text ("ORDENANTE ES14...") -> customers.iban,
  plus IBANs learned from the TRAIN golden.
- Name: the payer name is cut out of the text with per-format patterns ("ACH CREDIT <name> CCD", "/ORDP/<name>/REMI/",
  "nnnnnnnn-<name> |", "TRANSFERENCIA DE <name> | CONCEPTO", ...), plus camt `counterparty_name` and German `ABWA+<name>`
  (ultimate debtor = the real customer). Names are compared "squashed" (A-Z0-9 only): a known name equal to the start of
  the segment (the text has extra words after the name) or the segment equal to the start of a known name (truncated,
  >= 8 chars). The longest match wins. Fallback: a known full name anywhere in the text (SPEI "CONCEPTO ... <name>").
- Known names: master name + trade name, the names printed on the invoices (names before a rename), and aliases learned
  from the TRAIN golden (bank-side spellings, old names). Leakage: for sample/val we only learn from lines booked before
  the window start; for live/final from all of TRAIN.
- Third parties: "CONFIRMING ... ORD. <customer>", "EXPTE nnnn <customer, truncated>" (collection agency; the visible
  payer name is then ignored), a parent paying for its subsidiaries (customers.parent_customer_id / group_id) ->
  `expand_group`.

API
    pr = PartyResolver.get(ctx)
    pr.resolve(line) -> {customer_id: (strength, source)}   strength 3 = IBAN / name, 2 = learned alias, 1 = truncated
"""
from __future__ import annotations

import bisect
import collections
import os
import pickle
import re
import sys
import time
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from ingestion import lib  # noqa: E402
from ingestion.text import LEGAL, norm_name  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "matching", "out", "_cache")
CUTOFF = {"sample": "2025-06-01", "val": "2025-01-01", "hold": "2024-01-01"}       # learn aliases only from TRAIN lines booked before this
ALIAS_VERSION = 4

IBAN_RE = re.compile(r"\b([A-Z]{2}\d{2}[A-Z0-9]{11,30})\b")
_SQ = re.compile(r"[^A-Z0-9]")
_WS = re.compile(r"[^A-Z0-9]+")
LEGAL_X = set(LEGAL) | {"S", "R", "L", "C", "V", "A", "B", "N", "U", "K", "E"}

# Payer-name segments, applied to the upper-cased, accent-free text.  Group 1 = segment that starts with the name.
NAME_PATTERNS = [
    r"ABWA\+(.+?)(?: [A-Z]{4}\+|$)",
    r"\bORD\. (.+?)(?: FRA\b| FR \||$)",
    r"POR CTA DE (.+)$",
    r"EXPTE \d+ (.+?)(?: \d{3,}|$)",
    r"ACH CREDIT (.+?) CCD",
    r"DIRECT CREDIT (.+)$",
    r"/ORG/(.+?)(?: /RFB/|/|$)",
    r"/ORDP/(.+?)(?:/REMI/|/|$)",
    r"/NAME/(.+?)(?:/REMI/|/|$)",
    r"/DE (.+?)(?: /MOTIF|$)",
    r"BONIFICO DA (.+?)(?: CAUSALE|$)",
    r"PRZELEW PRZYCH\. (.+?)(?: TYTUL|$)",
    r"TT IN (.+)$",
    r"^FP (.+)$",
    r"PIX RECEBIDO (.+)$",
    r"LIQUIDACAO BOLETO \S+ (.+)$",
    r"TRANSFERENCIAS (.+?)(?: \| |$)",
    r"TRANSFERENCIAS .+? \| (.+?)$",
    r"\b\d{8}-(.+?)(?: \| |$)",
    r"TRANSFERENCIA DE (.+?)(?: \| CONCEPTO|CONCEPTO|$)",
    r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})? -(.+?)(?: \| |$)",
]
REMIT_PATTERNS = [
    r"CONCEPTO:? (.+)$",
    r"SVWZ\+(.+?)(?: [A-Z]{4}\+|$)",
    r"/REMI/(.+?)(?:/EREF/|$)",
    r"TYTUL: (.+)$",
    r"CAUSALE (.+)$",
    r"/MOTIF (.+?)(?: /REF|$)",
]
NAME_PATTERNS = [re.compile(p) for p in NAME_PATTERNS]
REMIT_PATTERNS = [re.compile(p) for p in REMIT_PATTERNS]
AGENCY = re.compile(r"\bEXPTE\b")
CONFIRMING = re.compile(r"CONFIRMING|\bORD\. ")
THIRD_PARTY = re.compile(r"\bEXPTE\b|CONFIRMING|\bORD\. |POR CTA DE|ABWA\+")


def upper(s: str) -> str:
    s = (s or "").upper().replace("ß", "SS")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def squash(s: str) -> str:
    return _SQ.sub("", s)


def tokens(seg: str) -> list[str]:
    return [t for t in _WS.split(seg) if t]


def name_keys(name: str) -> set[str]:
    """Squashed forms of a known name: full (with legal form), normalised (legal forms dropped), and the core (tokens
    before the first legal-form token: 'Piscinas Nuevo Valle S. de R.L. de C.V.' -> 'PISCINASNUEVOVALLE')."""
    u = upper(name)
    toks = tokens(u.replace("&", " "))
    out = {squash(u), squash(norm_name(name)), squash(u.replace("&", "AND"))}
    core = []
    for t in toks:
        if t in LEGAL_X and core:
            break
        core.append(t)
    out.add("".join(core))
    out.add("".join(t for t in toks if t not in LEGAL_X))
    return {k for k in out if len(k) >= 5}


def line_texts(line) -> list[str]:
    """[text, remittance_info] (upper-cased, accent-free; the remittance only when not already inside the text)."""
    t = line.text or ""
    ri = getattr(line, "remittance_info", "") or ""
    return [upper(t)] + ([upper(ri)] if ri and ri not in t else [])


def segments(u: str, counterparty_name: str = "", remit: bool = True) -> list[str]:
    """Candidate payer-name segments (upper-cased, accent-free), most specific first."""
    segs = []
    for p in NAME_PATTERNS:
        for m in p.finditer(u):
            s = m.group(1).strip(" |")
            if s:
                segs.append(s)
    if counterparty_name:
        segs.append(upper(counterparty_name))
    if remit:
        for p in REMIT_PATTERNS:
            for m in p.finditer(u):
                s = m.group(1).strip(" |")
                if s:
                    segs.append(s)
    return segs


def _learn_aliases(cutoff: str) -> dict:
    g = lib.golden_train()
    g = g[g.item_id.str.startswith("AR")]
    bank = lib.bank_of("train")
    bank = bank[(bank.amount > 0) & (bank.booking_date < cutoff)]
    items = lib.items_of("train").drop_duplicates("item_id")
    party = dict(zip(items.item_id, items.party_id))
    g = g[g.bank_txn_id.isin(set(bank.bank_txn_id))]
    g = g.assign(party=g.item_id.map(party)).dropna(subset=["party"])
    per_line = g.groupby("bank_txn_id").party.agg(lambda s: s.iloc[0] if s.nunique() == 1 else None).dropna()
    bank = bank[bank.bank_txn_id.isin(per_line.index)]
    names: dict = collections.defaultdict(collections.Counter)
    ibans: dict = collections.defaultdict(collections.Counter)
    for tid, ent, txt, ri, cpn, cpa in zip(bank.bank_txn_id, bank.entity, bank.text, bank.remittance_info,
                                           bank.counterparty_name, bank.counterparty_account):
        cid = per_line[tid]
        srcs = [upper(txt)] + ([upper(ri)] if ri and ri not in txt else [])
        u = " ".join(srcs)
        for ib in set(([cpa.replace(" ", "").upper()] if cpa else []) + IBAN_RE.findall(u)):
            ibans[(ent, ib)][cid] += 1
        third = bool(THIRD_PARTY.search(u))
        if third:  # only the segments naming the customer (ORD. / EXPTE / ABWA / POR CTA DE)
            segs = [m.group(1).strip(" |") for x in srcs for p in NAME_PATTERNS[:4] for m in p.finditer(x)]
        else:
            segs = [sg for x in srcs for sg in segments(x, remit=False)] + ([upper(cpn)] if cpn else [])
        seen = set()
        for seg in segs[:3]:
            toks = tokens(seg)
            acc = ""
            for t in toks[:8]:
                if any(ch.isdigit() for ch in t):
                    break
                acc += t
                if len(acc) >= 6 and acc not in seen:
                    seen.add(acc)
                    names[(ent, acc)][cid] += 1
    out_names = {}
    for key, c in names.items():
        tot = sum(c.values())
        if len(c) <= 4 and tot >= 2:
            out_names[key] = {cid for cid, n in c.items() if n >= 2 or len(c) == 1}
    out_ibans = {}
    for (ent, ib), c in ibans.items():
        if len(c) <= 3:
            out_ibans[(ent, ib)] = set(c)
    return {"names": out_names, "ibans": out_ibans}


class PartyResolver:
    _inst: dict = {}

    @classmethod
    def get(cls, ctx) -> "PartyResolver":
        k = (ctx.phase, id(ctx))
        if k not in cls._inst:
            cls._inst[k] = cls(ctx)
        return cls._inst[k]

    def __init__(self, ctx):
        t0 = time.time()
        self.ctx = ctx
        cust = ctx.masters["customers"]
        self.cust = cust.set_index("customer_id", drop=False)
        self.entity_of = dict(zip(cust.customer_id, cust.entity))
        # ---- known names: entity -> squashed name -> {customer_id}
        self.keys: dict = collections.defaultdict(lambda: collections.defaultdict(set))
        self.iban: dict = collections.defaultdict(set)
        for r in cust.itertuples(index=False):
            for nm in (r.name, r.trade_name):
                if nm:
                    for k in name_keys(nm):
                        self.keys[r.entity][k].add(r.customer_id)
            if r.iban:
                self.iban[(r.entity, r.iban.replace(" ", "").upper())].add(r.customer_id)
        ar = ctx.items[ctx.items.side == "AR"].drop_duplicates(["party_id", "party_name"])
        for pid, ent, nm in zip(ar.party_id, ar.entity, ar.party_name):
            if nm:
                for k in name_keys(nm):
                    self.keys[ent][k].add(pid)
        # ---- learned aliases (TRAIN golden)
        al = self._aliases()
        self.alias: dict = collections.defaultdict(dict)
        for (ent, k), cids in al["names"].items():
            self.alias[ent][k] = cids
        self.liban = al["ibans"]
        ents = set(self.keys) | set(self.alias)
        self.sorted_keys = {e: sorted(set(self.keys.get(e, ())) | set(self.alias.get(e, ()))) for e in ents}
        self.key_lens = {e: sorted({len(k) for k in self.sorted_keys[e]}, reverse=True) for e in ents}
        # ---- groups
        self.children = collections.defaultdict(set)
        self.group = collections.defaultdict(set)
        for r in cust.itertuples(index=False):
            if r.parent_customer_id:
                self.children[r.parent_customer_id].add(r.customer_id)
            if r.group_id:
                self.group[r.group_id].add(r.customer_id)
        self.group_of = dict(zip(cust.customer_id, cust.group_id))
        self.parent_of = dict(zip(cust.customer_id, cust.parent_customer_id))
        print(f"[parties] resolver ready ({time.time() - t0:.1f}s, {len(al['names']):,} name aliases, "
              f"{len(al['ibans']):,} learned IBANs)")

    def _aliases(self) -> dict:
        cutoff = CUTOFF.get(self.ctx.phase, "9999-99-99")
        os.makedirs(CACHE_DIR, exist_ok=True)
        cp = os.path.join(CACHE_DIR, f"party_aliases_v{ALIAS_VERSION}_{cutoff}.pkl")
        if os.path.exists(cp):
            with open(cp, "rb") as f:
                return pickle.load(f)
        t0 = time.time()
        res = _learn_aliases(cutoff)
        with open(cp, "wb") as f:
            pickle.dump(res, f)
        print(f"[parties] learned aliases from TRAIN < {cutoff} in {time.time() - t0:.0f}s")
        return res

    # ------------------------------------------------------------------------------------------------------------------
    def _lookup(self, ent: str, s: str, alias: bool = True) -> set:
        out = set()
        d = self.keys.get(ent)
        if d and s in d:
            out |= d[s]
        if alias:
            a = self.alias.get(ent)
            if a and s in a:
                out |= a[s]
        return out

    def name_hits(self, ent: str, seg: str) -> tuple[set, int]:
        """-> (customers, matched length) for one segment: (a) the longest master name equal to a token-prefix of the
        segment, (a') the same for learned aliases, (b) a token-prefix of the segment (>= 8 chars, longer than (a)) that
        known names start with (truncated name)."""
        toks = tokens(seg)
        if not toks or ent not in self.sorted_keys:
            return set(), 0
        bounds, acc = [], ""
        for t in toks[:12]:
            acc += t
            bounds.append(acc)
        best, best_len = set(), 0
        d, a = self.keys.get(ent, {}), self.alias.get(ent, {})
        for p in reversed(bounds):                     # (a) master names
            if len(p) < 5:
                break
            if p in d:
                best, best_len = set(d[p]), len(p)
                break
        for p in reversed(bounds):                     # (a') learned aliases
            if len(p) < 6:
                break
            if p in a:
                best |= a[p]
                best_len = max(best_len, len(p))
                break
        sk = self.sorted_keys[ent]
        for p in reversed(bounds):                     # (b) truncated
            if len(p) < 8 or len(p) <= best_len:
                break
            i = bisect.bisect_left(sk, p)
            hit = set()
            while i < len(sk) and sk[i].startswith(p):
                if sk[i] != p:
                    hit |= self._lookup(ent, sk[i])
                i += 1
                if len(hit) > 12:
                    break
            if hit:
                if len(hit) <= 12:
                    best, best_len = hit, len(p)
                break
        return best, best_len

    def resolve(self, line) -> dict:
        """line: namedtuple with entity, text, remittance_info, counterparty_name, counterparty_account.
        -> {customer_id: (strength, source)} restricted to the line's entity."""
        ent = line.entity
        out: dict = {}

        def add(cids, strength, src):
            for c in cids:
                if self.entity_of.get(c, ent) != ent:
                    continue
                if c not in out or out[c][0] < strength:
                    out[c] = (strength, src)

        srcs = line_texts(line)
        u = " ".join(srcs)
        cpa = (line.counterparty_account or "").replace(" ", "").upper()
        third = bool(THIRD_PARTY.search(u))
        agency = bool(AGENCY.search(u))
        for ib in set(([cpa] if cpa else []) + IBAN_RE.findall(u)):
            if agency and ib == cpa:
                continue
            add(self.iban.get((ent, ib), ()), 3, "iban")
            add(self.liban.get((ent, ib), ()), 2, "iban_learned")
        if agency or (CONFIRMING.search(u) and not line.counterparty_name):
            # the visible payer is an agency / confirming bank: only the customer named after EXPTE / ORD.
            segs = [m.group(1).strip(" |") for x in srcs for p in NAME_PATTERNS[:4] for m in p.finditer(x)]
            remit = []
        else:
            segs = [sg for x in srcs for sg in segments(x, remit=False)]
            if line.counterparty_name and not third:
                segs.append(upper(line.counterparty_name))
            remit = [m.group(1).strip(" |") for x in srcs for p in REMIT_PATTERNS for m in p.finditer(x)]
        found = False
        for seg in segs:
            hit, L = self.name_hits(ent, seg)
            if hit:
                add(hit, 3, "name")
                found = True
        if not found and not agency:
            for seg in remit:
                hit, L = self.name_hits(ent, seg)
                if hit:
                    add(hit, 2, "remit")
            # a known full name (>= 8 chars) starting at any word of the text
            d, a = self.keys.get(ent, {}), self.alias.get(ent, {})
            lens = self.key_lens.get(ent, ())
            for x in srcs:
                toks = tokens(x)
                for i in range(len(toks)):
                    s = "".join(toks[i:i + 10])
                    got_d = got_a = False
                    for L in lens:
                        if L < 8 or L > len(s):
                            continue
                        k = s[:L]
                        if not got_d and k in d:
                            add(d[k], 2, "scan")
                            got_d = True
                        if not got_a and k in a:
                            add(a[k], 2, "scan")
                            got_a = True
                        if got_d and got_a:
                            break
        return out

    def expand_group(self, cid: str) -> list[str]:
        """Customers a payer may pay for: its subsidiaries (parent_customer_id) and the members of its group."""
        out = set(self.children.get(cid, ()))
        g = self.group_of.get(cid, "")
        if g:
            out |= self.group[g]
        out.discard(cid)
        return sorted(out)
