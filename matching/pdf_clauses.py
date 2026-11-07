"""Extract payment clauses from the document PDFs -> matching/out/<phase>/doc_clauses.parquet

    python matching/pdf_clauses.py --phase sample      # (sample/val read train PDFs; live reads train + live)

One row per clause found:
    pdf_file, folder, document_type, document_number, entity, party_id, item_type, item_id,
    clause_type, n_instalments, pct, amount, amounts, instalment_days, net_amount, ref_document, ref_item_id,
    service_month (YYYY-MM), clause_date (YYYY-MM-DD), iban, old_iban, counterparty, raw_text

clause_type
    INSTALMENTS     invoice paid in N instalments; amounts = the instalment amounts (amount = first one),
                    instalment_days = days from the invoice date of each instalment
    RETENTION       pct % retained as guarantee; amount = retention, net_amount = payable at due date,
                    clause_date = final acceptance (release) date
    ADVANCE_DEDUCT  advance received against proforma ref_document; amount = advance, net_amount = net due
    REPLACES        this invoice cancels and replaces invoice ref_document (ref_item_id = the cancelled item)
    CANCELS         credit note that cancels invoice ref_document in full (extra type: neither is settled by bank)
    SERVICE_MONTH   monthly maintenance fee for service_month
    RAPPEL_OFFSET   rappel credit note offset in the next payment (pct, amount = rebate base, ref_document = period)
    RAPPEL_PAYOUT   rappel credit note paid out by transfer (our outgoing transfer settles the ARC)
    CN_REFUND       supplier credit note refunded by the supplier (an incoming transfer settles the APC)
    CN_OFFSET       supplier credit note deducted from our next payment
    FACTORING       supplier invoice assigned to a factor (counterparty)
    NEW_IBAN        supplier invoice (iban = new supplier IBAN) or customer notice (iban = new paying account,
                    old_iban = previous); clause_date = effective date

Clauses exist only in ES and EN wording in this data set (surveyed all 131k TRAIN PDFs).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from datetime import date

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ingestion import lib  # noqa: E402

COLS = ["pdf_file", "folder", "document_type", "document_number", "entity", "party_id", "item_type", "item_id",
        "clause_type", "n_instalments", "pct", "amount", "amounts", "instalment_days", "net_amount", "ref_document", "ref_item_id",
        "service_month", "clause_date", "iban", "old_iban", "counterparty", "raw_text"]

DOC_ITEM_TYPE = {"invoice": "ARI", "credit_note": "ARC", "supplier_invoice": "API", "supplier_credit_note": "APC"}

AMT = r"\d+(?:[.,\s]\d{3})*[.,]\d{2}"
CUR = r"[A-Z]{3}"
IBAN = r"[A-Z]{2}\d{2}[A-Z0-9]{8,30}"
DATE = r"\d{1,2}/\d{1,2}/\d{4}"
DOC = r"[A-Z0-9][A-Za-z0-9\-/.]*?[A-Z0-9]"

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
     "december"])}
MONTHS.update({m: i + 1 for i, m in enumerate(
    ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre",
     "diciembre"])})
MONTHS["setiembre"] = 9

P = {
    "INST_EN": re.compile(r"Payment terms:\s*(\d+)\s+instalments at\s+([\d/ ]+?)\s*days\s*\(([^)]*)\)"),
    "INST_ES": re.compile(r"Forma de pago:\s*(\d+)\s+vencimientos a\s+([\d/ ]+?)\s*d[ií]as\s*\(([^)]*)\)"),
    "RET_EN": re.compile(rf"Retention\s+([\d.,]+)\s*%\s*\(\s*({AMT})\s*({CUR})\s*\)\s*withheld until final acceptance"
                         rf"\s*\(\s*({DATE})\s*\)\.?(?:\s*Amount payable at due date:\s*({AMT})\s*{CUR})?"),
    "RET_ES": re.compile(rf"Retenci[oó]n en garant[ií]a del\s+([\d.,]+)\s*%\s*\(\s*({AMT})\s*({CUR})\s*\),?\s*"
                         rf"liberable a la recepci[oó]n definitiva\s*\(\s*({DATE})\s*\)\.?"
                         rf"(?:\s*Importe a pagar al vencimiento:\s*({AMT})\s*{CUR})?"),
    "ADV_EN": re.compile(rf"Less advance received against proforma\s+({DOC}):\s*({AMT})\s*({CUR})\.?"
                         rf"(?:\s*Net amount due\s*({AMT})\s*{CUR})?"),
    "ADV_ES": re.compile(rf"A deducir anticipo recibido seg[uú]n proforma\s+({DOC})\s+de\s+({AMT})\s*({CUR})\.?"
                         rf"(?:\s*Importe neto a pagar\s*({AMT})\s*{CUR})?"),
    "REPL_EN": re.compile(rf"This invoice cancels and replaces invoice\s+({DOC})\s+dated\s+({DATE})"),
    "REPL_ES": re.compile(rf"Esta factura anula y sustituye a la factura\s+({DOC})\s+de\s+({DATE})"),
    "CANC_EN": re.compile(rf"CANCELLING OF INVOICE\s+N[º°o]?\.?\s*({DOC})(?=[\s.]|$)"),
    "CANC_ES": re.compile(rf"Abono por anulaci[oó]n total de la factura\s+({DOC})(?=[\s.]|$)"),
    "SM_EN": re.compile(r"Maintenance and service fee for the month of\s+([A-Za-z]+)\s+(\d{4})"),
    "SM_ES": re.compile(r"Cuota de mantenimiento y servicio t[eé]cnico correspondiente a\s+([a-záéíóú]+)\s+(\d{4})",
                        re.I),
    "RAP_EN": re.compile(rf"Volume rebate\s+Q(\d)\s+(\d{{4}})\s*\(\s*([\d.,]+)\s*%\s*on\s+({AMT})\s*({CUR})\s*\):\s*"
                         r"(to be offset against your next payment|will be paid to your bank account)"),
    "RAP_ES": re.compile(rf"Rappel\s+[TQ](\d)\s+(\d{{4}})\s+por volumen\s*\(\s*([\d.,]+)\s*%\s*s/\s*({AMT})\s*({CUR})\s*\):\s*"
                         r"(se compensar[aá] en la pr[oó]xima liquidaci[oó]n|se abonar[aá] por transferencia a su cuenta)"),
    "CNR": re.compile(r"(The amount of this credit note will be refunded by bank transfer"
                      r"|El importe de este abono les ser[aá] reembolsado mediante transferencia bancaria)"),
    "CNO": re.compile(r"(This credit note is to be deducted from your next payment"
                      r"|Este abono se compensar[aá] en sus pr[oó]ximos pagos)"),
    "FACT_EN": re.compile(r"The receivable represented by this invoice has been assigned to\s+(.+?)\.\s*"
                          r"Payment is only discharging"),
    "FACT_ES": re.compile(r"El cr[eé]dito representado por esta factura ha sido cedido a\s+(.+?)\.\s*"
                          r"El pago solo ser[aá] liberatorio"),
    "IBAN_EN": re.compile(rf"from\s+({DATE})\s+our bank details change\.\s*New IBAN\s+({IBAN})"),
    "IBAN_ES": re.compile(rf"a partir del\s+({DATE})\s+nuestros datos bancarios cambian\.\s*Nuevo IBAN\s+({IBAN})"),
    "NOT_EN": re.compile(rf"from\s+({DATE})\s+our payments will be made from account\s+({IBAN})\s*"
                         rf"\(previously\s+({IBAN})\)"),
    "NOT_ES": re.compile(rf"a partir del\s+({DATE})\s+(?:los recibos domiciliados deber[aá]n cargarse en nuestra nueva "
                         rf"cuenta|nuestros pagos a su favor se realizar[aá]n desde la cuenta)\s+({IBAN}),?\s*"
                         rf"en sustituci[oó]n de la anterior\s+({IBAN})"),
}

# keyword presence checks (to measure parser coverage: a keyword without a parsed clause = a miss)
KEYWORDS = {
    "INSTALMENTS": re.compile(r"instalments at|vencimientos a"),
    "RETENTION": re.compile(r"withheld until final acceptance|Retenci[oó]n en garant"),
    "ADVANCE_DEDUCT": re.compile(r"Less advance received|A deducir anticipo"),
    "REPLACES": re.compile(r"cancels and replaces|anula y sustituye"),
    "CANCELS": re.compile(r"CANCELLING OF INVOICE|anulaci[oó]n total de la factura"),
    "SERVICE_MONTH": re.compile(r"for the month of|servicio t[eé]cnico correspondiente"),
    "RAPPEL": re.compile(r"Volume rebate|Rappel"),
    "CN_SUPPLIER": re.compile(r"credit note will be refunded|deducted from your next payment|abono les ser|abono se compensar"),
    "FACTORING": re.compile(r"has been assigned to|ha sido cedido a"),
    "NEW_IBAN": re.compile(r"bank details change|datos bancarios cambian|Change of bank account|Cambio de cuenta bancaria"),
}


def parse_amount(s: str | None) -> float | None:
    if not s:
        return None
    s = re.sub(r"\s", "", s)
    m = re.match(r"^(.*)[.,](\d{2})$", s)
    if not m:
        try:
            return float(re.sub(r"[^\d]", "", s))
        except ValueError:
            return None
    return float(re.sub(r"[^\d]", "", m.group(1)) + "." + m.group(2))


def parse_pct(s: str) -> float | None:
    try:
        return float(s.replace(",", "."))
    except (ValueError, AttributeError):
        return None


def parse_date(s: str | None, us: bool) -> str:
    """dd/mm/yyyy (mm/dd/yyyy for US documents) -> YYYY-MM-DD; falls back to the other order if invalid."""
    if not s:
        return ""
    try:
        a, b, y = (int(x) for x in s.split("/"))
    except ValueError:
        return ""
    for d, m in [(a, b), (b, a)]:  # clause dates are dd/mm even on US documents
        try:
            return date(y, m, d).isoformat()
        except ValueError:
            continue
    return ""


def _clean_doc(s: str) -> str:
    return s.rstrip(".,;:")


def extract(text: str, entity: str) -> list[dict]:
    """All clauses of one PDF text."""
    s = re.sub(r"\s+", " ", text)
    us = entity.startswith("US")
    out: list[dict] = []

    def add(ct, m, **kw):
        out.append({"clause_type": ct, "raw_text": m.group(0)[:400], **kw})

    for k in ("INST_EN", "INST_ES"):
        for m in P[k].finditer(s):
            amts = [parse_amount(x) for x in re.findall(AMT, m.group(3))]
            amts = [a for a in amts if a is not None]
            days = [int(x) for x in re.findall(r"\d+", m.group(2))]
            add("INSTALMENTS", m, n_instalments=int(m.group(1)), amounts=amts, amount=amts[0] if amts else None,
                instalment_days=days)
    for k in ("RET_EN", "RET_ES"):
        for m in P[k].finditer(s):
            add("RETENTION", m, pct=parse_pct(m.group(1)), amount=parse_amount(m.group(2)),
                net_amount=parse_amount(m.group(5)), clause_date=parse_date(m.group(4), us))
    for k in ("ADV_EN", "ADV_ES"):
        for m in P[k].finditer(s):
            add("ADVANCE_DEDUCT", m, ref_document=_clean_doc(m.group(1)), amount=parse_amount(m.group(2)),
                net_amount=parse_amount(m.group(4)))
    for k in ("REPL_EN", "REPL_ES"):
        for m in P[k].finditer(s):
            add("REPLACES", m, ref_document=_clean_doc(m.group(1)), clause_date=parse_date(m.group(2), us))
    for k in ("CANC_EN", "CANC_ES"):
        for m in P[k].finditer(s):
            add("CANCELS", m, ref_document=_clean_doc(m.group(1)))
    for k in ("SM_EN", "SM_ES"):
        for m in P[k].finditer(s):
            mo = MONTHS.get(m.group(1).lower())
            if mo:
                add("SERVICE_MONTH", m, service_month=f"{int(m.group(2)):04d}-{mo:02d}")
    for k in ("RAP_EN", "RAP_ES"):
        for m in P[k].finditer(s):
            payout = bool(re.search(r"paid to your bank|transferencia", m.group(6)))
            add("RAPPEL_PAYOUT" if payout else "RAPPEL_OFFSET", m, pct=parse_pct(m.group(3)),
                amount=parse_amount(m.group(4)), ref_document=f"{m.group(2)}Q{m.group(1)}")
    for m in P["CNR"].finditer(s):
        add("CN_REFUND", m)
    for m in P["CNO"].finditer(s):
        add("CN_OFFSET", m)
    for k in ("FACT_EN", "FACT_ES"):
        for m in P[k].finditer(s):
            add("FACTORING", m, counterparty=m.group(1).strip())
    for k in ("IBAN_EN", "IBAN_ES"):
        for m in P[k].finditer(s):
            add("NEW_IBAN", m, iban=m.group(2), clause_date=parse_date(m.group(1), us))
    for k in ("NOT_EN", "NOT_ES"):
        for m in P[k].finditer(s):
            add("NEW_IBAN", m, iban=m.group(2), old_iban=m.group(3), clause_date=parse_date(m.group(1), us))
    # SERVICE_MONTH appears in the description table and sometimes again in the footer: keep one per month
    seen, dedup = set(), []
    for c in out:
        k = (c["clause_type"], c.get("service_month"), c.get("ref_document"), c.get("iban"), c.get("amount"))
        if c["clause_type"] == "SERVICE_MONTH" and k in seen:
            continue
        seen.add(k)
        dedup.append(c)
    return dedup


def load_texts(folder: str, files: set[str] | None = None) -> dict[str, str]:
    p = os.path.join(lib.DATA, "cache", f"pdftext_{folder}.jsonl.gz")
    if not os.path.exists(p):
        print(f"[pdf_clauses] no text cache {p}")
        return {}
    d = {}
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if files is None or r["file"] in files:
                d[r["file"]] = r["text"]
    return d


def build(phase: str, items: pd.DataFrame | None = None, verbose: bool = True) -> pd.DataFrame:
    folders = lib.PHASES[phase][3] if not lib.PHASES[phase][4] else ["train"]
    rows, miss = [], {}
    for folder in folders:
        ip = os.path.join(lib.DATA, folder, "pdfs", "index.csv")
        if not os.path.exists(ip):
            continue
        ix = lib.read_csv(ip)
        texts = load_texts(folder)
        for r in ix.itertuples(index=False):
            t = texts.get(r.file)
            if t is None:
                miss["<no text>"] = miss.get("<no text>", 0) + 1
                continue
            cl = extract(t, r.entity)
            got = {c["clause_type"] for c in cl}
            for k, rx in KEYWORDS.items():
                if rx.search(t):
                    want = {"RAPPEL": {"RAPPEL_OFFSET", "RAPPEL_PAYOUT"}, "CN_SUPPLIER": {"CN_REFUND", "CN_OFFSET"}}.get(k, {k})
                    if not (got & want):
                        miss[k] = miss.get(k, 0) + 1
            for c in cl:
                rows.append({"pdf_file": r.file, "folder": folder, "document_type": r.document_type,
                             "document_number": r.document_number, "entity": r.entity, "party_id": r.party_id,
                             "item_type": DOC_ITEM_TYPE.get(r.document_type, ""), **c})
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    for c in ("ref_document", "service_month", "clause_date", "iban", "old_iban", "counterparty", "item_id",
              "ref_item_id"):
        df[c] = df[c].fillna("").astype(str)
    df["amounts"] = df.amounts.map(lambda v: list(v) if isinstance(v, list) else [])
    df["instalment_days"] = df.instalment_days.map(lambda v: list(v) if isinstance(v, list) else [])
    df["n_instalments"] = pd.to_numeric(df.n_instalments, errors="coerce").fillna(0).astype(int)
    for c in ("pct", "amount", "net_amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(["pdf_file", "clause_type", "ref_document", "service_month", "iban", "amount"])
    if items is not None and len(df):
        df = resolve_items(df, items)
    if verbose:
        print(f"[pdf_clauses] {len(df):,} clauses from {df.pdf_file.nunique():,} PDFs; keyword-without-clause: {miss}")
        print(df.clause_type.value_counts().to_string())
    return df[COLS].reset_index(drop=True)


def resolve_items(df: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """item_id via (item_type, entity, document_number[, party_id]); ref_item_id for REPLACES / CANCELS."""
    it = items[["item_id", "item_type", "entity", "party_id", "document_number"]]
    k3 = {(r.item_type, r.entity, r.document_number, r.party_id): r.item_id for r in it.itertuples()}
    g2 = it.groupby(["item_type", "entity", "document_number"]).item_id.agg(list)
    k2 = {k: v[0] for k, v in g2.items() if len(v) == 1}
    df = df.copy()
    df["item_id"] = [k3.get((t, e, d, p)) or k2.get((t, e, d), "")
                     for t, e, d, p in zip(df.item_type, df.entity, df.document_number, df.party_id)]
    ref = df.clause_type.isin(["REPLACES", "CANCELS"])
    df.loc[ref, "ref_item_id"] = [k3.get(("ARI", e, d, p)) or k2.get(("ARI", e, d), "")
                                  for e, d, p in zip(df.entity[ref], df.ref_document[ref], df.party_id[ref])]
    return df


def load_or_build(ctx) -> pd.DataFrame:
    """Clauses for ctx.phase (cached in matching/out/<phase>/doc_clauses.parquet, rebuilt when missing)."""
    p = os.path.join(lib.out_dir("matching", ctx.phase), "doc_clauses.parquet")
    if os.path.exists(p) and os.path.getmtime(p) >= os.path.getmtime(os.path.abspath(__file__)):
        return pd.read_parquet(p)
    df = build(ctx.phase, ctx.items, verbose=False)
    df.to_parquet(p, index=False)
    return df


def cancelled_items(ctx) -> set[str]:
    """item_ids never settled by bank (POLICIES §5.7): invoices cancelled by "anula y sustituye" / "cancels and
    replaces", and credit notes that cancel an invoice in full (plus that invoice). Matchers should not allocate to
    them; a payment citing a cancelled invoice settles its replacement (pdf_cases)."""
    cl = load_or_build(ctx)
    out = set(cl.ref_item_id[(cl.clause_type == "REPLACES") & (cl.ref_item_id != "")])
    canc = cl[cl.clause_type == "CANCELS"]
    out |= set(canc.item_id[canc.item_id != ""]) | set(canc.ref_item_id[canc.ref_item_id != ""])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="sample", choices=list(lib.PHASES))
    a = ap.parse_args()
    t = time.time()
    from matching.common import Ctx
    ctx = Ctx(a.phase)
    df = build(a.phase, ctx.items)
    od = lib.out_dir("matching", a.phase)
    df.to_parquet(os.path.join(od, "doc_clauses.parquet"), index=False)
    res = df[df.item_id != ""]
    print(f"[pdf_clauses] resolved to an item of the phase: {len(res):,} clauses "
          f"({res.groupby('clause_type').size().to_dict()})")
    print(f"[pdf_clauses] -> {od}/doc_clauses.parquet ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
