"""Text / reference normalisation helpers shared by the AR matchers (owner: refs matcher).

Public API
----------
norm(s)                      uppercase, no accents, non-alphanumerics -> single space
compact(s)                   norm(s) without any space (used for prefix/substring name matching)
bank_text(row)               one searchable string made of text + ref1 + ref2 + remittance_info + end_to_end_id (+ counterparty name)
doc_keys(doc)                the forms in which customers quote one of OUR document numbers
text_tokens(s)               candidate reference tokens of a bank text (N43 ' | ' record joins handled)
find_refs(tokens, items)     fuzzy lookup of the tokens among a small list of items -> {item_index: level}
edit1(a, b)                  True when a and b differ by <= 1 insertion/deletion/substitution/adjacent transposition

Reference levels (higher = stronger evidence), see `find_refs`:
  4  full printed number incl. prefix (or a truncation of it of >= 9 chars)
  3  digits of the number (prefix dropped), >= 7 digits
  2  last 6 digits of the number (customers often quote a 6-digit tail)
  1  one typo / transposition in the full or digit form
All functions are pure and never raise on odd input (None, NaN, numbers).
"""
from __future__ import annotations

import re
import unicodedata

_NONALNUM = re.compile(r"[^A-Z0-9]+")
_DIGITS = re.compile(r"\d+")


def _s(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if x != x:
            return ""
        return str(int(x)) if x == int(x) else str(x)
    return str(x)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm(s) -> str:
    """Uppercase, accents removed, every run of non-alphanumerics -> one space."""
    s = strip_accents(_s(s)).upper().replace("&", " AND ")
    return _NONALNUM.sub(" ", s).strip()


def compact(s) -> str:
    return norm(s).replace(" ", "")


# ---------------------------------------------------------------- legal suffixes / names
LEGAL = {
    "SA", "SL", "SLU", "SAU", "SLL", "SCOOP", "COOP", "GMBH", "AG", "KG", "EK", "EV", "UG", "MBH", "CO", "KGAA", "EURL", "SARL", "SAS", "SASU", "SNC",
    "SRL", "SPA", "SCRL", "LTD", "LLC", "INC", "CORP", "PTY", "LIMITED", "LLP", "PLC", "BV", "NV", "SP", "ZOO", "OO", "SPZOO", "SKA", "DE", "CV", "RL",
    "SAPI", "LTDA", "EIRELI", "ME", "EPP", "EI", "SC", "SCA", "SAE", "DOO", "THE", "AND", "Y", "E", "UND", "ET", "DI", "DA", "DO", "DOS", "DAS", "OF", "CIE", "CIA",
    "COMPANY", "INCORPORATED", "CORPORATION", "GROUP", "GRP", "HOLDING", "HOLDINGS",
}
# "GROUP" etc. are dropped from the *key* only; the full compact name is kept as a second key.


def name_tokens(name) -> list[str]:
    return norm(name).split()


def name_key_tokens(name) -> tuple:
    """Name tokens without trailing legal-form tokens ('POOL AND SPA HOWARD PTY LTD' -> ('POOL','AND','SPA','HOWARD'))."""
    t = name_tokens(name)
    while len(t) > 1 and t[-1] in LEGAL:
        t.pop()
    return tuple(t)


def name_keys(name) -> set:
    """Matching keys of a party name: sets of token tuples (legal-form-free and full)."""
    out = set()
    for t in (name_key_tokens(name), tuple(name_tokens(name))):
        if t and sum(map(len, t)) >= 5:
            out.add(t)
    return out


# ---------------------------------------------------------------- bank text
def bank_text(r) -> str:
    """Searchable text of a bank row (mapping-like). Keeps field order; N43 ' | ' joins preserved."""
    parts = []
    for c in ("text", "remittance_info", "ref1", "ref2", "end_to_end_id"):
        v = _s(r.get(c) if hasattr(r, "get") else r[c])
        if v and v not in ("NOTPROVIDED", "NONREF", "None", "nan"):
            parts.append(v)
    return " ; ".join(parts)


# ---------------------------------------------------------------- reference tokens
_TOK = re.compile(r"[A-Za-z]*\d[A-Za-z0-9]*(?:[-/.][A-Za-z0-9]+)*")


def text_tokens(s) -> list[str]:
    """Candidate reference tokens (alphanumeric, upper, separators removed) of a bank text.

    Besides the plain tokens it adds, for N43 texts, the tokens obtained when the ' | ' record boundary is removed (a number
    can be cut in two) and consecutive digit runs joined by a hyphen/slash ('FT25-01-27109' -> 'FT250127109')."""
    s = strip_accents(_s(s)).upper()
    out: list[str] = []
    seen = set()
    for variant in (s, s.replace(" | ", "")):
        for m in _TOK.finditer(variant):
            t = re.sub(r"[^A-Z0-9]", "", m.group(0))
            if len(t) >= 4 and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def digits(s) -> str:
    return "".join(_DIGITS.findall(_s(s)))


def doc_keys(doc) -> dict:
    """Forms of an item's document number a customer may quote.

    returns {'full': 'INV210100108', 'digits': '210100108', 'seq': '100108' (digits of the last hyphen-group, zeros stripped),
             'tail6': '100108'}  (empty strings when not derivable)"""
    d = norm(_s(doc))
    full = d.replace(" ", "")
    dig = digits(full)
    parts = d.split()
    last = digits(parts[-1]) if parts else ""
    seq = last.lstrip("0")
    tail6 = dig[-6:] if len(dig) >= 6 else ""
    return {"full": full, "digits": dig, "seq": seq, "tail6": tail6}


def edit1(a: str, b: str) -> bool:
    """Damerau distance <= 1 (insert, delete, substitute, swap adjacent)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    if la > lb:
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def find_refs(tokens: list[str], docs: list, typo: bool = True) -> dict:
    """Which of `docs` (document numbers, list index = item index) are quoted by `tokens`? -> {index: level} (best level per item).

    Matching is meant for the SMALL list of a payer's open items; typo matching (level 1) is only attempted for tokens of
    >= 7 chars that contain >= 6 digits and for numbers with >= 7 digits."""
    hits: dict[int, int] = {}
    if not tokens:
        return hits
    tset = set(tokens)
    tdig = {t: digits(t) for t in tokens}
    for i, doc in enumerate(docs):
        k = doc_keys(doc)
        full, dg, seq, t6 = k["full"], k["digits"], k["seq"], k["tail6"]
        lvl = 0
        if full and len(full) >= 6:
            if full in tset:
                lvl = 4
            else:
                for t in tokens:
                    # truncated quote of the full number (cut at 16/35/38) or number glued to extra text
                    if len(t) >= 9 and len(t) < len(full) and full.startswith(t):
                        lvl = max(lvl, 4)
                    elif len(full) >= 9 and full in t:
                        lvl = max(lvl, 4)
        if lvl < 4 and len(dg) >= 7:
            for t in tokens:
                td = tdig[t]
                if td == dg or (len(dg) >= 8 and dg in td and len(td) <= len(dg) + 3):
                    lvl = max(lvl, 3)
                    break
        if lvl < 3 and seq and len(seq) >= 5:
            for t in tokens:
                td = tdig[t]
                if td and (td == seq or td.lstrip("0") == seq or (len(td) >= 6 and td.endswith(seq) and len(td) <= len(seq) + 2)):
                    lvl = max(lvl, 2)
                    break
        if lvl < 2 and t6:
            for t in tokens:
                td = tdig[t]
                if td and (td.endswith(t6) and len(td) <= 8) and len(t) <= 10:
                    lvl = max(lvl, 2)
                    break
        if typo and lvl < 2 and len(dg) >= 7:
            for t in tokens:
                td = tdig[t]
                if len(td) >= 6 and (edit1(td, dg) or (len(seq) >= 6 and edit1(td.lstrip("0"), seq))):
                    lvl = max(lvl, 1)
                    break
        if lvl:
            hits[i] = lvl
    return hits


# ---------------------------------------------------------------- service month ("MANT. MAYO 25", "SERVICE MAY 2025")
_MONTHS = {
    1: "ENERO ENE JANUARY JAN JANUAR JANVIER JANV GENNAIO GEN JANEIRO JANUARI STYCZEN",
    2: "FEBRERO FEB FEBRUARY FEBRUAR FEVRIER FEV FEBBRAIO FEVEREIRO FEBRUARI LUTY",
    3: "MARZO MAR MARCH MARZ MRZ MARS MARCO MAART MARZEC",
    4: "ABRIL ABR APRIL APR AVRIL AVR APRILE KWIECIEN",
    5: "MAYO MAY MAI MAGGIO MAG MAIO MEI MAJ",
    6: "JUNIO JUN JUNE JUNI JUIN GIUGNO GIU JUNHO CZERWIEC",
    7: "JULIO JUL JULY JULI JUILLET JUIL LUGLIO LUG JULHO LIPIEC",
    8: "AGOSTO AGO AUGUST AUG AOUT GOSTO AUGUSTUS SIERPIEN",
    9: "SEPTIEMBRE SEP SEPT SEPTEMBER SEPTEMBRE SETTEMBRE SETEMBRO WRZESIEN",
    10: "OCTUBRE OCT OCTOBER OKTOBER OCTOBRE OTTOBRE OUTUBRO OKT PAZDZIERNIK",
    11: "NOVIEMBRE NOV NOVEMBER NOVEMBRE NOVEMBRO LISTOPAD",
    12: "DICIEMBRE DIC DECEMBER DEC DEZEMBER DECEMBRE DICEMBRE DEZEMBRO DEZ GRUDZIEN",
}
_MONTH_OF = {w: m for m, ws in _MONTHS.items() for w in ws.split()}


def service_month(s):
    """Month number (1..12) named in a bank text ('SERVICIOS MAYO', 'MANT. MAY 25'), else None. Whole words only; ambiguous -> None."""
    found = {_MONTH_OF[t] for t in norm(s).split() if t in _MONTH_OF}
    return found.pop() if len(found) == 1 else None


# ---------------------------------------------------------------- truncated numbers ("RE2401184", "FV24-03 032", "0214")
_PTOK = re.compile(r"[A-Za-z]*\d[A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")


def prefix_tokens(s) -> list[str]:
    """Short alphanumeric tokens with >= 3 digits (or letters+digits >= 4 chars) that may be a CUT document number."""
    s = strip_accents(_s(s)).upper()
    out, seen = [], set()
    for m in _PTOK.finditer(s):
        t = re.sub(r"[^A-Z0-9]", "", m.group(0))
        nd = sum(ch.isdigit() for ch in t)
        if nd >= 3 and len(t) <= 16 and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def prefix_match(t: str, doc) -> bool:
    """Is token `t` the beginning of document `doc` as customers print it: the full number, its digits, the sequence after the
    last hyphen, or its last-6-digit tail (e.g. 'RE2401184' / '0322619' / '032' / '1262' for FA240126xxx)."""
    k = doc_keys(doc)
    if not k["full"]:
        return False
    if t[0].isalpha():
        return len(t) >= 4 and k["full"].startswith(t)
    if len(t) >= 3 and (k["digits"].startswith(t) and len(t) >= 5):
        return True
    parts = norm(_s(doc)).split()
    last = digits(parts[-1]) if parts else ""
    if len(parts) > 1 and last.startswith(t) and len(t) >= 3:
        return True
    return len(t) >= 4 and k["tail6"].startswith(t)
