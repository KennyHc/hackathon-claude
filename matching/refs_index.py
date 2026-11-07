"""Reference index for the `refs` matcher: noisy document numbers in bank texts -> open items (POLICIES §5.5).

Bank texts carry our document numbers with noise: no prefix / leading zeros (`RE 108031` -> RE250108031), a shared
prefix with short follow-ups (`FA250211336 209808 212043`), typos and transposed digits, truncation (16/35/38 chars,
N43 38-char fields joined with ` | ` so a number may be cut in two), or the customer's PO instead of our number.

`tokens(text)`  -> reference-like tokens of a bank text (incl. numbers re-joined across ` | `)
`RefIndex(items)`: per (entity, side) exact indexes on alnum / digits / PO, plus sorted arrays for suffix and prefix
lookups (truncation, number without prefix), and per-party item lists for party-restricted fuzzy matching.
"""
from __future__ import annotations

import bisect
import re
from collections import defaultdict

import numpy as np
import pandas as pd

_WORD = re.compile(r"[A-Z0-9]+(?:[\-/\.][A-Z0-9]+)*")
_AMOUNT = re.compile(r"^\d{1,9}[\.,]\d{2}$")
# structured noise that must not be read as references
_NOISE = [
    re.compile(r"/OCMT/[A-Z]{3}[\d,\.]+"), re.compile(r"/EXCH/[\d,\.]+"), re.compile(r"\*PI\*[\d\.,]+"),
    re.compile(r"PAYMENT ID:\s*\d+"), re.compile(r"\bCLAVE \d+"), re.compile(r"\bORDENANTE [A-Z]{2}\d{2}[A-Z0-9]+"),
    re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{14,30}\b"),  # IBAN
    re.compile(r"\bEREF\+\S+"), re.compile(r"/EREF/[^/]*"), re.compile(r"\bMREF\+\S+"),
    re.compile(r"\bID:\s*\d+"),
]
# end of a field: N43 joins, mt940 tags, end of text -> the token before may be truncated
_FIELD_SPLIT = re.compile(r" \| |/[A-Z]{2,5}/|\\$")


def _clean(s: str) -> str:
    s = (s or "").upper()
    for p in _NOISE:
        s = p.sub(" ", s)
    return s


class Tok:
    __slots__ = ("raw", "alnum", "digits", "letters", "trunc", "joined")

    def __init__(self, raw: str, trunc: bool, joined: bool = False):
        self.raw = raw
        self.alnum = re.sub(r"[^A-Z0-9]", "", raw)
        self.digits = re.sub(r"\D", "", raw)
        self.letters = re.sub(r"[^A-Z]", "", self.alnum)
        self.trunc = trunc      # last token of a field: may be cut
        self.joined = joined    # re-joined across a ` | ` boundary

    def __repr__(self):
        return f"Tok({self.raw!r}{', T' if self.trunc else ''}{', J' if self.joined else ''})"


def tokens(*texts: str) -> list[Tok]:
    out: list[Tok] = []
    seen = set()
    for text in texts:
        if not text:
            continue
        s = _clean(text)
        fields = _FIELD_SPLIT.split(s)
        raw_fields = s.split(" | ")
        for f in fields:
            ws = _WORD.findall(f)
            ws = [w for w in ws if not _AMOUNT.match(w)]
            for j, w in enumerate(ws):
                t = Tok(w, trunc=(j == len(ws) - 1))
                if sum(c.isdigit() for c in t.alnum) >= 3 and t.alnum not in seen:
                    seen.add(t.alnum)
                    out.append(t)
        # numbers cut by the N43 38-char fields
        for a, b in zip(raw_fields, raw_fields[1:]):
            if a and b and a[-1].isalnum() and b[0].isalnum():
                wa, wb = _WORD.findall(a), _WORD.findall(b)
                if wa and wb:
                    t = Tok(wa[-1] + wb[0], trunc=False, joined=True)
                    if sum(c.isdigit() for c in t.alnum) >= 3 and t.alnum not in seen:
                        seen.add(t.alnum)
                        out.append(t)
    return out


def edit1(a: str, b: str) -> bool:
    """Damerau distance <= 1 (one substitution, insertion, deletion or adjacent transposition), a != b."""
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        if len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]:
            return True
        return False
    if la > lb:
        a, b = b, a
    # b is a with one char inserted
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


class RefIndex:
    def __init__(self, items: pd.DataFrame):
        it = items.reset_index(drop=True)
        self.df = it
        self.n = len(it)
        self.iid = it.item_id.to_numpy()
        self.side = it.side.to_numpy()
        self.entity = it.entity.to_numpy()
        self.party = it.party_id.fillna("").to_numpy()
        self.typ = it.item_type.to_numpy()
        self.sign = np.where(np.isin(self.typ, ["ARC", "APC"]), -1.0, 1.0)
        self.open = it.open_amount.astype(float).to_numpy().copy()
        self.total = it.total_amount.astype(float).to_numpy()
        self.cur = it.currency.to_numpy()
        self.issue = it.issue_date.astype(str).to_numpy()
        self.alnum = it.document_alnum.fillna("").to_numpy()
        self.norm = it.document_number_norm.fillna("").to_numpy()
        self.po = it.po_norm.fillna("").to_numpy()
        self.terms = it.payment_terms.fillna("").to_numpy()
        rel = it.related_document.fillna("").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
        self.related = rel.to_numpy()

        self.by_alnum: dict = defaultdict(list)
        self.by_norm: dict = defaultdict(list)
        self.by_po: dict = defaultdict(list)
        self.by_party: dict = defaultdict(list)
        rev: dict = defaultdict(list)
        pre: dict = defaultdict(list)
        for k in range(self.n):
            key = (self.entity[k], self.side[k])
            if self.alnum[k]:
                self.by_alnum[key + (self.alnum[k],)].append(k)
                pre[key].append((self.alnum[k], k))
            if self.norm[k]:
                self.by_norm[key + (self.norm[k],)].append(k)
                rev[key].append((self.norm[k][::-1], k))
            if self.po[k] and len(self.po[k]) >= 5:
                self.by_po[key + (self.po[k],)].append(k)
            self.by_party[(self.side[k], self.party[k])].append(k)
        self.rev = {}
        for key, v in rev.items():
            v.sort()
            self.rev[key] = ([a for a, _ in v], [b for _, b in v])
        self.pre = {}
        for key, v in pre.items():
            v.sort()
            self.pre[key] = ([a for a, _ in v], [b for _, b in v])
        for key in self.by_party:
            self.by_party[key].sort(key=lambda k: (self.issue[k], self.iid[k]))

    # --- global lookups (entity + side) -----------------------------------------------------------------------------
    def exact(self, ent: str, side: str, t: Tok) -> list[int]:
        key = (ent, side)
        out = list(self.by_alnum.get(key + (t.alnum,), ()))
        d = t.digits.lstrip("0")
        if len(d) >= 5 and (not t.letters or not out):
            out += self.by_norm.get(key + (d,), ())
        if len(t.alnum) >= 5:
            out += self.by_po.get(key + (t.alnum,), ())
        return list(dict.fromkeys(out))

    def suffix(self, ent: str, side: str, digits: str, cap: int = 60) -> list[int]:
        """items whose digits end with `digits` (number without prefix / series short form)."""
        arr = self.rev.get((ent, side))
        if not arr or len(digits) < 4:
            return []
        r = digits[::-1]
        keys, ids = arr
        lo = bisect.bisect_left(keys, r)
        hi = bisect.bisect_left(keys, r + "\x7f")
        if hi - lo > cap:
            return []
        return [ids[i] for i in range(lo, hi) if keys[i] != r or True]

    def prefix(self, ent: str, side: str, alnum: str, cap: int = 5000) -> list[int]:
        """items whose alnum starts with `alnum` (truncated reference)."""
        arr = self.pre.get((ent, side))
        if not arr or len(alnum) < 5:
            return []
        keys, ids = arr
        lo = bisect.bisect_left(keys, alnum)
        hi = bisect.bisect_left(keys, alnum + "\x7f")
        if hi - lo > cap:
            return []
        return [ids[i] for i in range(lo, hi)]

    def fuzzy(self, ent: str, side: str, t: Tok) -> list[int]:
        """one edit / transposition on the digits (full number) -> items."""
        d = t.digits.lstrip("0")
        if len(d) < 7:
            return []
        key = (ent, side)
        out = []
        seen = set()
        cands = set()
        # transpositions + substitutions + deletions + insertions
        for i in range(len(d) - 1):
            cands.add(d[:i] + d[i + 1] + d[i] + d[i + 2:])
        for i in range(len(d)):
            for c in "0123456789":
                cands.add(d[:i] + c + d[i + 1:])
            cands.add(d[:i] + d[i + 1:])
        for i in range(len(d) + 1):
            for c in "0123456789":
                cands.add(d[:i] + c + d[i:])
        cands.discard(d)
        for v in cands:
            for k in self.by_norm.get(key + (v.lstrip("0"),), ()):
                if k not in seen:
                    seen.add(k)
                    out.append(k)
        return out
