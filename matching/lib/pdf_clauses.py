"""PDF clauses -> CONTRACTS doc_clauses rows, cached as matching/out/cache/<phase>/pdf_clauses.parquet.

    from matching.lib import pdf_clauses
    cl = pdf_clauses.clauses("val")      # one row per clause
    pdf_clauses.parse_text(text)         # -> list[dict] for one PDF's clause text (pure, never raises)

Language-agnostic parsing: a clause is recognised by a keyword family (es/en/fr/de/it/pt/nl/pl) AND its number pattern;
amounts are parsed whatever the separators ('5.871,50', '10 304,42', '5,511.15'). Unparsed keyword hits are counted
(`parse_stats`) so a new layout shows up as a lower parse rate instead of wrong numbers.

clause_type: INSTALMENTS (n_instalments, amounts in raw 'amounts' list), RETENTION (pct, amount = retained, payable),
ADVANCE_DEDUCT (amount = advance, ref_document = proforma, payable = net), REPLACES (ref_document = cancelled invoice),
CANCELS (credit note cancelling ref_document), SERVICE_MONTH (YYYY-MM), NEW_IBAN (iban = new, old_iban, from_date),
PROFORMA (advance request: amount = order total).
"""
from __future__ import annotations

import os
import re
from collections import Counter

import pandas as pd

from . import data

_NUM = r"\d{1,3}(?:[ .,  ]\d{3})*(?:[.,]\d{2})"
NUM = re.compile(_NUM)
_CUR = r"(?:EUR|USD|GBP|CHF|PLN|MXN|BRL|CNY|AUD|MAD|ZAR|CAD|€|\$|£)"
_DOC = r"([A-Z0-9][A-Z0-9/-]*\d[A-Z0-9/-]*[A-Z0-9])"
_DATE = r"(\d{2}[/.-]\d{2}[/.-]\d{4}|\d{4}-\d{2}-\d{2})"
_IBAN = re.compile(r"\b([A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}\s?[A-Z0-9]{1,4})\b")

# keyword families (lowercase, accents kept and stripped variants)
K_INST = r"(vencimientos|instal?ments?|[ée]ch[ée]ances|raten|rate|presta[cç][oõ]es|parcelas|termijnen|rat)"
K_RET = r"(retenci[oó]n|retention|retenue|einbehalt|sicherheitseinbehalt|garantieeinbehalt|ritenuta|reten[cç][aã]o|garantie|inhouding)"
K_ADV = r"(anticipo|advance|acompte|anzahlung|adiantamento|avance|vooruitbetaling|zaliczk)"
K_REPL = r"(anula y sustituye|cancels and replaces|annule et remplace|storniert und ersetzt|ersetzt die rechnung|annulla e sostituisce|anula e substitui|vervangt|annule et remplace)"
K_CANC = r"(anulaci[oó]n total|cancell?ing of invoice|annulation (?:totale )?de la facture|stornierung der rechnung|storno|annullamento|anula[cç][aã]o|creditering)"
K_MONTH = r"(correspondiente a|for the month of|pour le mois d[e']|f[uü]r den monat|relativ[ao] al mese di|referente ao m[eê]s de|over de maand|za miesi)"
K_IBAN = r"(cuenta|account|compte|konto|conto|conta|rekening|rachunek)"
K_PROF = r"(proforma|pro-forma|pro forma)"

MONTHS = {
    "enero": 1, "january": 1, "janvier": 1, "januar": 1, "gennaio": 1, "janeiro": 1, "januari": 1, "styczen": 1,
    "febrero": 2, "february": 2, "fevrier": 2, "février": 2, "februar": 2, "febbraio": 2, "fevereiro": 2, "februari": 2,
    "marzo": 3, "march": 3, "mars": 3, "märz": 3, "marz": 3, "março": 3, "marco": 3, "maart": 3,
    "abril": 4, "april": 4, "avril": 4, "aprile": 4,
    "mayo": 5, "may": 5, "mai": 5, "maggio": 5, "maio": 5, "mei": 5,
    "junio": 6, "june": 6, "juin": 6, "juni": 6, "giugno": 6, "junho": 6,
    "julio": 7, "july": 7, "juillet": 7, "juli": 7, "luglio": 7, "julho": 7,
    "agosto": 8, "august": 8, "août": 8, "aout": 8, "augustus": 8,
    "septiembre": 9, "september": 9, "septembre": 9, "settembre": 9, "setembro": 9,
    "octubre": 10, "october": 10, "octobre": 10, "oktober": 10, "ottobre": 10, "outubro": 10,
    "noviembre": 11, "november": 11, "novembre": 11, "novembro": 11,
    "diciembre": 12, "december": 12, "décembre": 12, "decembre": 12, "dezember": 12, "dicembre": 12, "dezembro": 12,
}

COLS = ["pdf_file", "document_type", "document_number", "entity", "party_id", "clause_type", "n_instalments", "pct", "amount",
        "ref_document", "service_month", "iban", "raw_text", "amounts", "payable", "old_iban", "from_date"]


def num(s: str) -> float | None:
    """'5.871,50' / '10 304,42' / '5,511.15' / '193,55' -> float. Last separator followed by 2 digits is the decimal mark."""
    try:
        s = s.strip().replace(" ", " ").replace(" ", " ")
        m = re.match(r"^(.*?)[.,](\d{2})$", s)
        if not m:
            return float(re.sub(r"[^\d]", "", s))
        return float(re.sub(r"[^\d]", "", m.group(1)) + "." + m.group(2))
    except Exception:
        return None


def _date(s: str | None):
    if not s:
        return None
    try:
        if re.match(r"\d{4}-", s):
            return s[:10]
        d, m, y = re.split(r"[/.-]", s)
        return f"{y}-{m}-{d}"
    except Exception:
        return None


def _sentences(text: str) -> list[str]:
    # clause lines were joined with ' || ' and may wrap; re-join and split on sentence-like boundaries that start a new clause
    return [text.replace(" || ", " ")]


def parse_text(text: str) -> tuple[list[dict], Counter]:
    """-> (clauses, stats). stats counts keyword hits ('hit:X') and successful parses ('ok:X') per clause type."""
    st: Counter = Counter()
    out: list[dict] = []
    if not text:
        return out, st
    t = text.replace(" || ", " ")
    low = t.lower()

    # INSTALMENTS: "<n> <kw> ... (<a> / <b> / ... CUR)"
    if re.search(K_INST, low) and "/" in t:
        st["hit:INSTALMENTS"] += 1
        m = re.search(r"(\d{1,2})\s+" + K_INST + r"[^()]{0,80}\(((?:\s*" + _NUM + r"\s*/)+\s*" + _NUM + r")\s*" + _CUR + r"?\s*\)", t, re.I)
        if m:
            n = int(m.group(1))
            am = [num(x) for x in m.group(3).split("/")]
            if len(am) == n and all(a is not None and a > 0 for a in am):
                out.append({"clause_type": "INSTALMENTS", "n_instalments": n, "amounts": am, "amount": round(sum(am), 2)})
                st["ok:INSTALMENTS"] += 1

    # RETENTION: "<kw> ... <pct> % (<amt> CUR) ... <date> ... : <payable> CUR"
    if re.search(K_RET, low) and "%" in t:
        st["hit:RETENTION"] += 1
        m = re.search(K_RET + r"[^%]{0,40}?(\d{1,2}(?:[.,]\d+)?)\s*%\s*\(\s*(" + _NUM + r")\s*" + _CUR + r"?\s*\)(.{0,160})", t, re.I)
        if m:
            pct = float(m.group(2).replace(",", ".")) / 100
            ret = num(m.group(3))
            rest = m.group(4)
            d = re.search(_DATE, rest)
            pays = NUM.findall(rest[d.end():] if d else rest)
            payable = num(pays[0]) if pays else None
            if ret and 0 < pct < 0.5:
                out.append({"clause_type": "RETENTION", "pct": pct, "amount": ret, "payable": payable,
                            "from_date": _date(d.group(1)) if d else None})
                st["ok:RETENTION"] += 1

    # ADVANCE_DEDUCT (invoice): "<kw> ... <proforma no> ... <advance> CUR ... <net> CUR"
    if re.search(K_ADV, low) and re.search(K_PROF, low) and not re.search(r"(no v[aá]lida|not a tax invoice|n'est pas une facture|keine rechnung|non valida|n[aã]o v[aá]lida)", low):
        st["hit:ADVANCE_DEDUCT"] += 1
        m = re.search(K_PROF + r"\s*(?:n[º°o.]*\s*)?(PF[A-Z0-9/-]*\d)\b(.{0,160})", t, re.I)
        if m:
            nums = [num(x) for x in NUM.findall(m.group(3))]
            nums = [x for x in nums if x]
            if len(nums) >= 2:
                out.append({"clause_type": "ADVANCE_DEDUCT", "ref_document": m.group(2), "amount": nums[0], "payable": nums[1]})
                st["ok:ADVANCE_DEDUCT"] += 1

    # PROFORMA (the proforma itself)
    if re.search(r"(no v[aá]lida como factura|not a tax invoice|n'est pas une facture|keine rechnung|non valida come fattura|n[aã]o v[aá]lida como fatura)", low):
        st["hit:PROFORMA"] += 1
        nums = [num(x) for x in NUM.findall(t)]
        nums = [x for x in nums if x]
        out.append({"clause_type": "PROFORMA", "amount": nums[-1] if nums else None})
        st["ok:PROFORMA"] += 1

    # REPLACES: "<kw> <invoice no>"
    if re.search(K_REPL, low):
        st["hit:REPLACES"] += 1
        m = re.search(K_REPL + r"\s+(?:\w+\s+){0,3}?(?:n[º°o.]*\s*)?" + _DOC, t, re.I)
        if m:
            out.append({"clause_type": "REPLACES", "ref_document": m.group(2)})
            st["ok:REPLACES"] += 1

    # CANCELS (credit note)
    if re.search(K_CANC, low):
        st["hit:CANCELS"] += 1
        m = re.search(K_CANC + r"\s+(?:\w+\s+){0,3}?(?:n[º°o.]*\s*)?" + _DOC, t, re.I)
        if m:
            out.append({"clause_type": "CANCELS", "ref_document": m.group(2)})
            st["ok:CANCELS"] += 1

    # SERVICE_MONTH
    if re.search(K_MONTH, low):
        st["hit:SERVICE_MONTH"] += 1
        m = re.search(K_MONTH + r"\s+([a-zäéèûçã]+)\s+(?:de\s+|del\s+|of\s+)?(\d{4})", low)
        if m and m.group(2) in MONTHS:
            out.append({"clause_type": "SERVICE_MONTH", "service_month": f"{m.group(3)}-{MONTHS[m.group(2)]:02d}"})
            st["ok:SERVICE_MONTH"] += 1

    # NEW_IBAN (bank-account notice): first IBAN = new, second = old; first date after the notice date = from
    ibs = [x.replace(" ", "") for x in _IBAN.findall(t)]
    if ibs and re.search(K_IBAN, low) and not t.upper().startswith(("FORMA", "CONDITIONS", "ZAHLUNG", "PAYMENT")):
        st["hit:NEW_IBAN"] += 1
        ds = re.findall(_DATE, t)
        out.append({"clause_type": "NEW_IBAN", "iban": ibs[0], "old_iban": ibs[1] if len(ibs) > 1 else None,
                    "from_date": _date(ds[-1]) if ds else None})
        st["ok:NEW_IBAN"] += 1
    return out, st


def clauses(phase: str, verbose: bool = True) -> pd.DataFrame:
    p = os.path.join(data.CACHE, phase, "pdf_clauses.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    from . import pdf_text
    tx = pdf_text.build(phase, verbose=verbose)
    rows, st, fails = [], Counter(), 0
    for r in tx.itertuples(index=False):
        if r.document_type == "bank_account_notice":
            text = r.text or ""
        else:
            text = r.text or ""
        try:
            cl, s = parse_text(text)
        except Exception:
            fails += 1
            continue
        st.update(s)
        for c in cl:
            if r.document_type == "bank_account_notice" and c["clause_type"] != "NEW_IBAN":
                continue
            if r.document_type != "bank_account_notice" and c["clause_type"] == "NEW_IBAN":
                continue
            row = {k: None for k in COLS}
            row.update(pdf_file=r.file, document_type=r.document_type, document_number=r.document_number, entity=r.entity,
                       party_id=r.party_id, raw_text=text[:600])
            row.update(c)
            rows.append(row)
    df = pd.DataFrame(rows, columns=COLS)
    df["amounts"] = df.amounts.apply(lambda a: list(a) if isinstance(a, list) else None)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_parquet(p, index=False)
    if verbose:
        print(f"  pdf_clauses {phase}: {len(tx)} pdfs, {len(df)} clauses, {fails} parse crashes")
        for k in sorted({k.split(':')[1] for k in st}):
            h, o = st.get("hit:" + k, 0), st.get("ok:" + k, 0)
            print(f"    {k:15s} keyword hits {h:6d} parsed {o:6d} ({o / max(h, 1):.1%})")
    return df


def parse_stats(phase: str) -> pd.DataFrame:
    from . import pdf_text
    tx = pdf_text.build(phase, verbose=False)
    st = Counter()
    for t in tx.text:
        st.update(parse_text(t or "")[1])
    return pd.Series(st).sort_index()
