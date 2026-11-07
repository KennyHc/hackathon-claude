"""Parse remittance advices (PDF text / CSV / XLSX / e-mail TXT) into one row per advice line.

    python matching/remittance_parse.py --phase sample      # -> matching/out/<phase>/remittance_lines.parquet

Layouts seen (TRAIN 2021-06..2025-06, LIVE 2025-07..12):
- CSV  (`,` or `;`): `payment_ref,<ref>,payment_date,<d>,currency,<ccy>` / `document,doc_date,gross,discount_or_deduction,
  reason,paid,your_order` / rows / `TOTAL,,,,,<total>,`
- PDF  (text cached in data/cache/pdftext_<folder>_remittance_advices.jsonl.gz): English "REMITTANCE ADVICE"
  (Payment ref./Date/Payee/Amount) or Spanish "AVISO DE PAGO" (Ref./Fecha/Beneficiario/Importe); rows
  `<doc> <dd/mm/yyyy> <gross> [<deduction> <REASON>] <paid>`; numbers US (1,234.56) or European (1.234,56).
- TXT  (e-mail): `Subject: Payment advice <ref>` / `Aviso de pago <ref>`, "payment of <amt> <ccy> on <mm/dd/yyyy>" or
  "transferencia por importe de <amt> <ccy> con fecha <dd/mm/yyyy>", rows `  <doc> <dd/mm/yy> <gross> [<ded> <REASON>]`.
- XLSX old: `<payer>` / `Payment advice,<ref>,<date>` / `Document,Date,Gross,Deduction,Reason,Paid` / rows / TOTAL.
- XLSX new (from Aug/Sep 2025): `Voucher,Invoice,PO,Inv. date,Amount,Deduction,Net paid`, where Invoice is only the
  last 6 digits of our number and there is no reason column (the TOTAL sits in the Deduction column).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from datetime import date, datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ingestion import lib  # noqa: E402

COLS = ["file", "entity", "payer_name", "received_date", "format", "layout", "payment_ref", "payment_date",
        "currency", "stated_total", "line_no", "doc", "doc_date", "gross", "deduction", "reason", "paid", "po",
        "voucher"]

NUM = r"-?\d[\d.,]*"


def num(s) -> float | None:
    """'1,234.56' / '1.234,56' / '-1 234,56' / 1234.56 -> float"""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().replace(" ", "").replace(" ", "")
    if not s:
        return None
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    s = s.strip("-()")
    m = re.search(r"[.,](\d{1,2})$", s)
    if m:
        intpart, dec = s[: m.start()], m.group(1)
    else:
        intpart, dec = s, "0"
    intpart = re.sub(r"[.,]", "", intpart)
    try:
        v = float(f"{intpart}.{dec}")
    except ValueError:
        return None
    return -v if neg else v


def pdate(s, order="dmy") -> str | None:
    if s is None or s == "":
        return None
    if isinstance(s, (datetime, date)):
        return s.strftime("%Y-%m-%d")
    s = str(s).strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return m.group(0)
    m = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})$", s)
    if not m:
        return None
    a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100:
        y += 2000
    d, mo = (a, b) if order == "dmy" else (b, a)
    if mo > 12 and d <= 12:
        d, mo = mo, d
    try:
        return date(y, mo, d).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _row(doc, doc_date, gross, ded, reason, paid, po="", voucher=""):
    gross = num(gross)
    ded = num(ded) or 0.0
    paid = num(paid)
    if paid is None and gross is not None:
        paid = round(gross + ded, 2)
    if gross is None and paid is not None:
        gross = round(paid - ded, 2)
    return dict(doc=str(doc).strip(), doc_date=doc_date, gross=gross, deduction=ded,
                reason=(reason or "").strip().upper(), paid=paid, po=str(po or "").strip(),
                voucher=str(voucher or "").strip())


# ---------------------------------------------------------------------------------------------------------------------
def parse_csv(path: str) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        text = fh.read()
    lines = [l for l in text.splitlines() if l.strip()]
    sep = ";" if lines and lines[0].count(";") > lines[0].count(",") else ","
    head = {}
    rows = []
    cols = None
    for l in lines:
        f = [x.strip().strip('"') for x in l.split(sep)]
        if f[0].lower() == "payment_ref":
            kv = dict(zip(f[0::2], f[1::2]))
            head = dict(payment_ref=kv.get("payment_ref", ""), payment_date=pdate(kv.get("payment_date")),
                        currency=kv.get("currency", ""))
            continue
        if f[0].lower() == "document":
            cols = [c.lower() for c in f]
            continue
        if f[0].upper() == "TOTAL":
            vals = [num(x) for x in f[1:] if x]
            head["stated_total"] = vals[-1] if vals else None
            continue
        if cols is None:
            continue
        r = dict(zip(cols, f))
        rows.append(_row(r.get("document"), pdate(r.get("doc_date")), r.get("gross"),
                         r.get("discount_or_deduction"), r.get("reason"), r.get("paid"), r.get("your_order", "")))
    head["layout"] = "csv"
    return head, rows


ROW_RE = re.compile(r"^(?P<doc>[A-Z]{0,4}[\d][\w\-/]*)\s+(?P<date>\d{1,2}/\d{1,2}/\d{2,4})\s+(?P<gross>" + NUM +
                    r")(?:(?:\s+|(?=-))(?P<ded>" + NUM + r")\s+(?P<reason>[A-Za-z][A-Za-z .'/&-]*?))?(?:\s+(?P<paid>" + NUM +
                    r"))?\s*$")


def parse_pdf_text(text: str) -> tuple[dict, list[dict]]:
    head = {"layout": "pdf"}
    rows = []
    for raw in text.splitlines():
        l = raw.strip()
        if not l:
            continue
        m = re.match(r"^(Payment ref\.?|Ref\.?)\s{2,}(\S.*)$", l)
        if m:
            head["payment_ref"] = m.group(2).strip()
            continue
        m = re.match(r"^(Date|Fecha)\s{2,}(\S+)$", l)
        if m:
            head["payment_date"] = pdate(m.group(2))
            continue
        m = re.match(r"^(Amount|Importe)\s{2,}(" + NUM + r")\s*([A-Z]{3})?$", l)
        if m:
            head["stated_total"] = num(m.group(2))
            head["currency"] = m.group(3) or ""
            continue
        if l in ("AVISO DE PAGO",):
            head["layout"] = "pdf_es"
        m = ROW_RE.match(l)
        if m:
            g = m.groupdict()
            if g["paid"] is None:          # no paid column -> gross is paid
                g["paid"] = g["gross"]
            rows.append(_row(g["doc"], pdate(g["date"]), g["gross"], g["ded"], g["reason"], g["paid"]))
    return head, rows


def parse_txt(path: str) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    head = {"layout": "txt"}
    rows = []
    for raw in text.splitlines():
        l = raw.strip()
        if not l:
            continue
        m = re.match(r"^Subject:\s*(?:Payment advice|Aviso de pago|Remittance advice|Avis de paiement)\s+(\S+)", l,
                     re.I)
        if m:
            head["payment_ref"] = m.group(1)
            continue
        m = re.search(r"payment of (" + NUM + r") ([A-Z]{3}) on (\S+?)\s", l)
        if m:
            head.update(stated_total=num(m.group(1)), currency=m.group(2), payment_date=pdate(m.group(3), "mdy"))
            continue
        m = re.search(r"importe de (" + NUM + r") ([A-Z]{3}) con fecha (\S+?)\s", l)
        if m:
            head.update(stated_total=num(m.group(1)), currency=m.group(2), payment_date=pdate(m.group(3), "dmy"),
                        layout="txt_es")
            continue
        if raw.startswith("  "):
            m = ROW_RE.match(l)
            if m:
                g = m.groupdict()
                gross = num(g["gross"])
                ded = num(g["ded"]) or 0.0
                if g["paid"] is not None and g["ded"] is None:   # "<doc> <date> <gross> <paid>"
                    rows.append(_row(g["doc"], pdate(g["date"]), g["gross"], None, None, g["paid"]))
                else:
                    rows.append(_row(g["doc"], pdate(g["date"]), gross, ded, g["reason"],
                                     g["paid"] if g["paid"] is not None else round(gross + ded, 2)))
    return head, rows


def parse_xlsx(path: str) -> tuple[dict, list[dict]]:
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    head: dict = {"layout": "xlsx"}
    rows = []
    for ws in wb.worksheets:
        cols = None
        for r in ws.iter_rows(values_only=True):
            if r is None or all(v is None for v in r):
                continue
            f0 = str(r[0]).strip() if r[0] is not None else ""
            if f0.lower() in ("payment advice", "payment ref", "payment_ref", "aviso de pago"):
                head["payment_ref"] = str(r[1]).strip() if r[1] is not None else ""
                head["payment_date"] = pdate(r[2]) if len(r) > 2 else None
                continue
            low = [str(v).strip().lower() if v is not None else "" for v in r]
            if "document" in low or "voucher" in low or "documento" in low:
                cols = low
                head["layout"] = "xlsx_voucher" if "voucher" in low else "xlsx"
                continue
            if f0.upper() == "TOTAL":
                vals = [num(v) for v in r[1:] if v is not None and num(v) is not None]
                head["stated_total"] = vals[-1] if vals else None
                continue
            if cols is None:
                if not head.get("payer"):
                    head["payer"] = f0
                continue
            d = dict(zip(cols, r))
            if "voucher" in cols:
                rows.append(_row(d.get("invoice"), pdate(d.get("inv. date")), d.get("amount"), d.get("deduction"),
                                 d.get("reason"), d.get("net paid"), d.get("po") or "", d.get("voucher")))
            else:
                rows.append(_row(d.get("document") or d.get("documento"), pdate(d.get("date") or d.get("fecha")),
                                 d.get("gross"), d.get("deduction"), d.get("reason"), d.get("paid"),
                                 d.get("your_order") or d.get("po") or ""))
    wb.close()
    return head, rows


# ---------------------------------------------------------------------------------------------------------------------
def _pdf_texts(folder: str) -> dict[str, str]:
    from ingestion import pdftext
    p = pdftext.path(folder, "remittance_advices")
    out = {}
    if not os.path.exists(p) and os.path.isdir(os.path.join(lib.DATA, folder, "remittance_advices")):
        print(f"[remittance] WARNING no PDF text cache {p}: PDF advices of {folder} are skipped (run ingestion/pdftext.py {folder})")
    if os.path.exists(p):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                out[r["file"]] = r.get("text") or ""
    return out


def _pdf_extract(path: str) -> str:
    """Fallback when the cache has no text for a PDF."""
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(path) as pdf:
            return "\n".join(pg.extract_text(layout=True) or "" for pg in pdf.pages)
    except Exception:
        pass
    try:
        import subprocess
        return subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""


def parse_folder(folder: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    base = os.path.join(lib.DATA, folder, "remittance_advices")
    idx = pd.read_csv(os.path.join(base, "index.csv"), dtype=str).fillna("")
    if start:
        idx = idx[idx.received_date >= start]
    if end:
        idx = idx[idx.received_date <= end]
    pdf = _pdf_texts(folder) if (idx.format == "pdf").any() else {}
    out = []
    for a in idx.itertuples(index=False):
        path = os.path.join(base, a.file)
        try:
            if a.format == "csv":
                head, rows = parse_csv(path)
            elif a.format == "txt":
                head, rows = parse_txt(path)
            elif a.format == "xlsx":
                head, rows = parse_xlsx(path)
            elif a.format == "pdf":
                text = pdf.get(a.file)
                if text is None:
                    text = _pdf_extract(path)
                head, rows = parse_pdf_text(text)
            else:
                continue
        except Exception as e:  # noqa: BLE001
            print(f"[remittance_parse] {a.file}: {e}")
            head, rows = {"layout": "error"}, []
        base_row = dict(file=a.file, entity=a.entity, payer_name=a.payer_name, received_date=a.received_date,
                        format=a.format, layout=head.get("layout", ""), payment_ref=str(head.get("payment_ref", "")),
                        payment_date=head.get("payment_date"), currency=head.get("currency", ""),
                        stated_total=head.get("stated_total"))
        if not rows:
            out.append({**base_row, "line_no": -1, "doc": "", "doc_date": None, "gross": None, "deduction": None,
                        "reason": "", "paid": None, "po": "", "voucher": ""})
        for i, r in enumerate(rows):
            out.append({**base_row, "line_no": i, **r})
    df = pd.DataFrame(out, columns=COLS)
    for c in ("stated_total", "gross", "deduction", "paid"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load(phase: str, refresh: bool = False) -> pd.DataFrame:
    """Parsed advice lines for a phase (cached in matching/out/<phase>/remittance_lines.parquet)."""
    od = lib.out_dir("matching", phase)
    p = os.path.join(od, "remittance_lines.parquet")
    if os.path.exists(p) and not refresh:
        return pd.read_parquet(p)
    folder, start, end, _, _ = lib.PHASES[phase]
    s = e = None
    if start:
        s = (pd.Timestamp(start) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    if end:
        e = (pd.Timestamp(end) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    # Advices arrive a few days after the cash, so the advices of the last payments of a folder sit in the next
    # folder (train ends 2025-06-30, live starts 2025-07-01), and vice versa for the first payments.
    chain = ["train", "live", "final"]
    parts = [parse_folder(folder, s, e)]
    i = chain.index(folder) if folder in chain else -1
    if i >= 0 and i + 1 < len(chain) and os.path.exists(os.path.join(lib.DATA, chain[i + 1], "remittance_advices")):
        last = e or parts[0].received_date.max()
        lo = parts[0].received_date.max() if len(parts[0]) else None
        nxt = parse_folder(chain[i + 1], None, (pd.Timestamp(last) + pd.Timedelta(days=0)).strftime("%Y-%m-%d")
                           if e else (pd.Timestamp(lo) + pd.Timedelta(days=30)).strftime("%Y-%m-%d"))
        parts.append(nxt)
    if i > 0 and os.path.exists(os.path.join(lib.DATA, chain[i - 1], "remittance_advices")):
        first = parts[0].received_date.min()
        parts.append(parse_folder(chain[i - 1], (pd.Timestamp(first) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")))
    df = pd.concat([x for x in parts if len(x)], ignore_index=True).drop_duplicates(["file", "line_no"])
    df.to_parquet(p, index=False)
    return df


def report(df: pd.DataFrame) -> None:
    adv = df.groupby("file").agg(fmt=("format", "first"), layout=("layout", "first"), n=("line_no", "max"),
                                 ref=("payment_ref", "first"), tot=("stated_total", "first"),
                                 s=("paid", "sum"))
    adv["has_lines"] = adv.n >= 0
    adv["has_ref"] = adv.ref.str.len() > 0
    adv["tot_ok"] = adv.tot.isna() | ((adv.tot - adv.s).abs() < 0.015)
    adv["ok"] = adv.has_lines & adv.has_ref & adv.tot_ok
    print(adv.groupby(["fmt", "layout"])[["has_lines", "has_ref", "tot_ok", "ok"]].mean().round(4))
    print(f"advices {len(adv):,}, parsed ok {adv.ok.mean():.2%}, lines {len(df):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="sample", choices=list(lib.PHASES))
    a = ap.parse_args()
    d = load(a.phase, refresh=True)
    report(d)
