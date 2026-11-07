"""Normalisers for party names and document references (importable by every piece)."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

LEGAL = {
    "SL", "SLU", "SA", "SAU", "SCOOP", "SCCL", "SLL", "CB", "GMBH", "CO", "KG", "EK", "AG", "OHG", "UG", "SARL", "SAS",
    "SASU", "EURL", "SNC", "SRL", "SRLS", "SPA", "SAPA", "SNC", "LTD", "LIMITED", "PLC", "LLP", "PTY", "LLC", "INC", "CORP",
    "LDA", "UNIPESSOAL", "BV", "NV", "VOF", "SP", "ZOO", "SPZOO", "SK", "DE", "CV", "SAPI", "LTDA", "ME", "EIRELI", "SARLAU",
    "SCS", "CIA", "Y", "AND", "UND", "ET", "E", "THE", "DA", "DO", "DOS", "DEL", "LA", "LE", "LES", "EL", "ZHONGGUO",
    "COLTD", "GMBHCOKG", "SOCIEDAD", "LIMITADA", "ANONIMA",
}
_SPLIT = re.compile(r"[^A-Z0-9]+")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_name(s: str) -> str:
    """'Riegos Oasis S.L.' -> 'RIEGOS OASIS'; 'Pooltechnik Weißenau GmbH & Co. KG' -> 'POOLTECHNIK WEISSENAU'."""
    if not s:
        return ""
    s = strip_accents(s.upper().replace("ß", "SS").replace("&", " ")).replace(".", "")
    toks = [t for t in _SPLIT.split(s) if t and t not in LEGAL]
    return " ".join(toks)


def norm_name_series(s: pd.Series) -> pd.Series:
    cache: dict[str, str] = {}
    return s.map(lambda x: cache.setdefault(x, norm_name(x)) if x not in cache else cache[x])


_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/\.]*[A-Za-z0-9]|[0-9]")


def doc_tokens(text: str) -> list[str]:
    """Reference-like tokens (≥ 3 digits), uppercase alphanumeric without separators: 'FV24-0223' -> 'FV240223'."""
    out = []
    for t in _TOK.findall(text or ""):
        u = re.sub(r"[^A-Z0-9]", "", t.upper())
        if sum(ch.isdigit() for ch in u) >= 3:
            out.append(u)
    return out


def digits_norm(tok: str) -> str:
    """Digits only, no leading zeros: 'RE250108031' -> '250108031', 'INV0005403' -> '5403'."""
    d = re.sub(r"\D", "", tok).lstrip("0")
    return d


def doc_number_norm_series(s: pd.Series) -> pd.Series:
    return s.str.replace(r"\D", "", regex=True).str.lstrip("0")


def alnum_series(s: pd.Series) -> pd.Series:
    return s.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
