"""Remittance-advice parser (csv / xlsx / txt e-mail / pdf) -> CONTRACTS `remittance_lines` table.

Content driven (no fixed positions, no filename logic): the layout is recognised from column headers or from the shape of a
line, so the September-2025 layout change (xlsx: Voucher | Invoice | PO | Inv. date | Amount | Deduction | Net paid, no reason
column) is handled by the same code as the old layout (Document | Date | Gross | Deduction | Reason | Paid).

    from matching.lib import remit_parse
    df = remit_parse.lines(phase)                 # cached: matching/out/cache/<phase>/remittance_lines.parquet
    df = remit_parse.lines(phase, date_from='2023-01-01', date_to='2023-03-31', cache=False)   # slice of an index

Columns: ra_file, entity, payer_name, received_date, format, payment_ref, doc_ref_raw, doc_ref_norm, amount_paid, deduction,
deduction_reason, line_total  (+ extras: currency, voucher, po, doc_date, gross, header_total, total_ok, ra_id).
`amount_paid` is the cash for that document (negative for credit notes), `gross` the document amount shown, `deduction` <= 0 as shown.
`line_total` = total of the whole advice (header amount, else TOTAL row, else sum of paid). `total_ok` = sum(paid) matches it.
"""
from __future__ import annotations

import csv
import io
import os
import re
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "matching", "out", "cache")

_NUM = r"-?\d[\d.,]*"
_NUM_RE = re.compile(r"^[-+]?\d[\d.,' ]*$")
_DATE_RE = re.compile(r"^\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}$")
_CCY = r"(?:EUR|USD|GBP|CHF|AUD|CAD|PLN|MXN|BRL|CNY|ZAR|MAD)"


def num(s):
    """Parse '3,333.73' / '2.525,06' / '-1 234,5' / float -> float, None if not a number."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return None if s != s else round(float(s), 2)
    t = str(s).strip().replace(" ", " ")
    if not t or not _NUM_RE.match(t):
        return None
    t = t.replace(" ", "").replace("'", "")
    neg = t.startswith("-")
    t = t.lstrip("+-")
    seps = [i for i, ch in enumerate(t) if ch in ".,"]
    if seps:
        last = seps[-1]
        dec_len = len(t) - last - 1
        if dec_len in (1, 2) or (len(seps) == 1 and dec_len != 3):
            ip, fp = t[:last], t[last + 1:]
            ip = re.sub(r"[.,]", "", ip)
            t = ip + "." + fp
        else:  # thousand separators only
            t = re.sub(r"[.,]", "", t)
    try:
        v = float(t)
    except ValueError:
        return None
    return round(-v if neg else v, 2)


def norm_doc(s) -> str:
    """Normalised document key: upper alphanumerics only (prefix kept)."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper()) if s is not None else ""


def digits_key(s) -> str:
    """Digits only, leading zeros stripped (prefix-less / zero-stripped key)."""
    d = re.sub(r"\D", "", str(s)) if s is not None else ""
    return d.lstrip("0")


def doc_ref_norm(s) -> str:
    """Contract doc_ref_norm: no letter prefix, no leading zeros (digits of the number part)."""
    return digits_key(s)


# ---------------------------------------------------------------------------------------------------------------- headers
_ROLE_PATTERNS = [
    ("voucher", r"^(voucher|vchr|beleg|vale|bon|comprobante)"),
    ("doc", r"^(document|documento|invoice|factura|doc\b|dokument|rechnung|facture|fattura|fatura|nf\b|inv\b|inv\.\s*(no|num|number)|ref\.?\s*doc)"),
    ("po", r"^(po\b|your_order|your order|order|pedido|bestellung|commande|ordine|p\.o)"),
    ("date", r"^(doc_date|date|inv\.?\s*date|fecha|datum|data\b|invoice date)"),
    ("ded", r"^(discount_or_deduction|deduction|deducción|deduccion|dto|ded|abzug|déduction|deduzione|desconto|discount)"),
    ("reason", r"^(reason|motivo|grund|motif|motivazione|razão|razao)"),
    ("paid", r"^(paid|net paid|net_paid|pagado|bezahlt|payé|paye|pagato|pago|net\b|neto)"),
    ("gross", r"^(gross|amount|bruto|betrag|brut|lordo|importe|valor)"),
]


def _role(cell) -> str | None:
    t = str(cell or "").strip().lower()
    if not t:
        return None
    for role, pat in _ROLE_PATTERNS:
        if re.match(pat, t):
            return role
    return None


def _header_map(cells):
    """cells -> {col index: role} if this row looks like the header of the line table, else None."""
    roles = {}
    for i, c in enumerate(cells):
        r = _role(c)
        if r and r not in roles.values():
            roles[i] = r
        elif r == "doc" and "doc" in roles.values():
            pass
    vals = set(roles.values())
    if ("doc" in vals or "voucher" in vals) and ("paid" in vals or "gross" in vals) and len(vals) >= 3:
        return roles
    return None


# ---------------------------------------------------------------------------------------------------------------- tables
def _table_lines(rows):
    """rows: list of list of cells (csv or xlsx). Returns (header dict, line dicts)."""
    meta = {"payment_ref": None, "currency": None, "total": None, "payment_date": None}
    lines = []
    hm = None
    for cells in rows:
        cells = [(c.strip() if isinstance(c, str) else c) for c in cells]
        nz = [c for c in cells if c not in (None, "")]
        if not nz:
            continue
        if hm is None:
            h = _header_map(cells)
            if h:
                hm = h
                continue
            # metadata rows: key,value pairs (csv) or label,ref,date (xlsx)
            low = [str(c).strip().lower() if c is not None else "" for c in cells]
            for i, k in enumerate(low):
                if k in ("payment_ref", "payment ref", "payment ref.", "ref", "ref.", "payment advice", "aviso de pago", "referencia") and meta["payment_ref"] is None:
                    for v in cells[i + 1:]:
                        if v not in (None, ""):
                            meta["payment_ref"] = str(v).strip()
                            break
                if k == "currency" and i + 1 < len(cells):
                    meta["currency"] = str(cells[i + 1]).strip() or None
            continue
        first = str(nz[0]).strip().upper()
        if first in ("TOTAL", "TOTALE", "SUMME", "SUMA", "TOTALS", "GESAMT"):
            vals = [num(c) for c in nz[1:]]
            vals = [v for v in vals if v is not None]
            if vals:
                meta["total"] = vals[-1]
            continue
        rec = {}
        for i, role in hm.items():
            rec[role] = cells[i] if i < len(cells) else None
        doc = rec.get("doc") if rec.get("doc") not in (None, "") else rec.get("voucher")
        if doc in (None, ""):
            continue
        gross, ded, paid = num(rec.get("gross")), num(rec.get("ded")), num(rec.get("paid"))
        if paid is None and gross is None:
            continue
        if gross is None:
            gross = paid if ded is None else round(paid - ded, 2)
        if paid is None:
            paid = round(gross + (ded or 0.0), 2)
        if ded is None:
            ded = round(paid - gross, 2) if abs(paid - gross) > 0.004 else 0.0
        lines.append({"doc": str(doc).strip(), "voucher": str(rec["voucher"]).strip() if rec.get("voucher") not in (None, "") and rec.get("doc") not in (None, "") else None,
                      "po": str(rec["po"]).strip() if rec.get("po") not in (None, "") else None,
                      "doc_date": rec.get("date"), "gross": gross, "deduction": ded,
                      "reason": (str(rec["reason"]).strip() if rec.get("reason") not in (None, "") else None), "paid": paid})
    return meta, lines


def parse_csv(path):
    raw = open(path, "rb").read().decode("utf-8-sig", errors="replace")
    first = raw.splitlines()[0] if raw else ""
    delim = ";" if first.count(";") > first.count(",") else ","
    rows = list(csv.reader(io.StringIO(raw), delimiter=delim))
    meta, lines = _table_lines(rows)
    # metadata row 'payment_ref,X,payment_date,Y,currency,Z' (key/value pairs on one row)
    for r in rows[:4]:
        low = [str(c).strip().lower() for c in r]
        if "payment_ref" in low:
            for k, name in (("payment_ref", "payment_ref"), ("payment_date", "payment_date"), ("currency", "currency")):
                if k in low and low.index(k) + 1 < len(r):
                    meta[name] = r[low.index(k) + 1].strip()
    return meta, lines


def parse_xlsx(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()
    meta, lines = _table_lines(rows)
    for r in rows[:4]:
        vals = [str(c).strip() for c in r if c not in (None, "")]
        if len(vals) >= 2 and vals[0].lower() in ("payment advice", "aviso de pago", "payment ref", "payment ref.", "ref", "ref."):
            meta["payment_ref"] = vals[1]
            if len(vals) > 2:
                meta["payment_date"] = vals[2]
    return meta, lines


# ---------------------------------------------------------------------------------------------------------------- text lines
_LINE_RE = re.compile(r"\S")


def _text_line(line: str):
    """One document line of a txt e-mail or a pdf (layout mode). Returns dict or None."""
    s = re.sub(r"([.,]\d\d)(-\d)", r"\1  \2", line.rstrip())   # pdf: wide gross glued to the deduction cell
    if not s or not s[0].isspace() and not re.match(r"^[A-Za-z0-9]", s):
        return None
    toks = [t for t in re.split(r"\s{2,}", s.strip()) if t]
    if len(toks) < 3:
        return None
    di = next((i for i, t in enumerate(toks) if _DATE_RE.match(t)), None)
    if di is None or di == 0 or di > 3:
        return None
    ids, rest = toks[:di], toks[di + 1:]
    if not rest or any(" " in t for t in ids) and len(ids) == 1 and False:
        return None
    # ids: doc | voucher doc po | doc po
    ids = [t for t in ids]
    voucher = po = None
    if len(ids) == 1:
        doc = ids[0]
    elif len(ids) == 2:
        doc, po = ids
    else:
        voucher, doc, po = ids[0], ids[1], ids[2]
    if " " in doc:
        return None
    nums, ded, reason = [], None, None
    for t in rest:
        v = num(t)
        if v is not None:
            nums.append(v)
            continue
        m = re.match(r"^(-?\d[\d.,]*)\s+(.+)$", t)
        if m and num(m.group(1)) is not None:
            ded, reason = num(m.group(1)), m.group(2).strip()
            nums.append(("D", ded))
        else:
            return None
    plain = [n for n in nums if not isinstance(n, tuple)]
    has_ded_cell = any(isinstance(n, tuple) for n in nums)
    if has_ded_cell:
        if len(plain) == 1:               # txt: gross, "ded reason"
            gross = plain[0]
            paid = round(gross + ded, 2)
        elif len(plain) == 2:             # pdf: gross, "ded reason", paid
            gross, paid = plain
        else:
            return None
    else:
        if len(plain) == 1:
            gross = paid = plain[0]; ded = 0.0
        elif len(plain) == 2:
            gross, paid = plain; ded = round(paid - gross, 2) if abs(paid - gross) > 0.004 else 0.0
        elif len(plain) == 3:
            gross, ded, paid = plain
        else:
            return None
    return {"doc": doc, "voucher": voucher, "po": po, "doc_date": toks[di], "gross": gross, "deduction": ded or 0.0,
            "reason": reason, "paid": paid}


_REF_PATTERNS = [
    r"(?:payment ref\.?|ref\.?|referencia|référence|zahlungsreferenz|riferimento|referência)\s*[:\s]\s*([A-Z0-9][A-Z0-9/\-]{5,})",
]


def _text_meta(txt: str):
    meta = {"payment_ref": None, "currency": None, "total": None, "payment_date": None}
    m = re.search(r"^Subject:\s*(.+)$", txt, re.M)
    if m:
        toks = m.group(1).strip().split()
        for t in reversed(toks):
            if re.search(r"\d", t) and len(t) >= 6:
                meta["payment_ref"] = t
                break
    if meta["payment_ref"] is None:
        for line in txt.splitlines()[:25]:
            for pat in _REF_PATTERNS:
                mm = re.search(pat, line, re.I)
                if mm:
                    meta["payment_ref"] = mm.group(1)
                    break
            if meta["payment_ref"]:
                break
    # total + currency: first "<amount> <CCY>" in the header zone
    for mm in re.finditer(r"(" + _NUM + r")\s*(" + _CCY + r")\b", txt):
        v = num(mm.group(1))
        if v is not None:
            meta["total"], meta["currency"] = v, mm.group(2)
            break
    return meta


def parse_txt(path):
    txt = open(path, "rb").read().decode("utf-8-sig", errors="replace")
    meta = _text_meta(txt)
    lines = []
    for ln in txt.splitlines():
        if ln.startswith((" ", "\t")):
            r = _text_line(ln)
            if r:
                lines.append(r)
    return meta, lines


def parse_pdf(path):
    import pypdf
    rd = pypdf.PdfReader(path)
    pages = []
    for p in rd.pages:
        try:
            pages.append(p.extract_text(extraction_mode="layout"))
        except Exception:
            pages.append(p.extract_text() or "")
    txt = "\n".join(pages)
    meta = {"payment_ref": None, "currency": None, "total": None, "payment_date": None}
    flat = [re.split(r"\s{2,}", l.strip()) for l in txt.splitlines() if l.strip()]
    lines = []
    for ln in txt.splitlines():
        r = _text_line(ln)
        if r:
            lines.append(r)
    # header block: "label   value" pairs on layout lines
    for toks in flat:
        for i, t in enumerate(toks[:-1]):
            tl = t.strip().lower()
            v = toks[i + 1].strip()
            if meta["payment_ref"] is None and tl in ("payment ref.", "payment ref", "ref.", "ref", "referencia", "réf.", "ref.:"):
                meta["payment_ref"] = v
            mm = re.match(r"^(" + _NUM + r")\s*(" + _CCY + r")$", v)
            if mm and meta["total"] is None and tl in ("amount", "importe", "montant", "betrag", "importo", "valor", "total"):
                meta["total"], meta["currency"] = num(mm.group(1)), mm.group(2)
    if meta["total"] is None:
        m = re.search(r"(" + _NUM + r")\s*(" + _CCY + r")\b", txt)
        if m:
            meta["total"], meta["currency"] = num(m.group(1)), m.group(2)
    if meta["payment_ref"] is None:
        m = re.search(r"(?:Payment ref\.?|Ref\.?|Referencia)\s+([A-Z0-9][A-Z0-9/\-]{5,})", txt)
        if m:
            meta["payment_ref"] = m.group(1)
    return meta, lines


PARSERS = {"csv": parse_csv, "xlsx": parse_xlsx, "txt": parse_txt, "pdf": parse_pdf}


def parse_file(path: str, fmt: str | None = None):
    fmt = (fmt or os.path.splitext(path)[1][1:]).lower()
    return PARSERS[fmt](path)


# ---------------------------------------------------------------------------------------------------------------- table build
COLS = ["ra_id", "ra_file", "entity", "payer_name", "received_date", "format", "payment_ref", "doc_ref_raw", "doc_ref_norm",
        "amount_paid", "deduction", "deduction_reason", "line_total", "currency", "voucher", "po", "doc_date", "gross",
        "header_total", "total_ok"]


def _index(phase):
    d = pd.read_csv(os.path.join(_ra_dir(phase), "index.csv"), dtype=str)
    d["received_date"] = pd.to_datetime(d.received_date)
    return d


def _ra_dir(phase):
    return os.path.join(DATA, "train-lite", "train", "remittance_advices") if phase == "train_lite" else os.path.join(DATA, phase, "remittance_advices")


def lines(phase: str, date_from=None, date_to=None, cache: bool = True, stats: dict | None = None, quiet=False) -> pd.DataFrame:
    """Parse all advices of the phase (or those received in [date_from, date_to]). Per-file try/except; `stats` (dict) gets
    per-format counters {fmt: {files, ok, failed, empty, total_ok}}."""
    cpath = os.path.join(CACHE, phase, "remittance_lines.parquet")
    if cache and date_from is None and date_to is None and os.path.exists(cpath):
        return pd.read_parquet(cpath)
    idx = _index(phase)
    if date_from is not None:
        idx = idx[idx.received_date >= pd.Timestamp(date_from)]
    if date_to is not None:
        idx = idx[idx.received_date <= pd.Timestamp(date_to)]
    base = _ra_dir(phase)
    st = stats if stats is not None else {}
    out = []
    for n, r in enumerate(idx.itertuples(index=False)):
        s = st.setdefault(r.format, {"files": 0, "ok": 0, "failed": 0, "empty": 0, "total_ok": 0, "no_ref": 0})
        s["files"] += 1
        try:
            meta, ls = parse_file(os.path.join(base, r.file), r.format)
        except Exception as e:  # noqa: BLE001
            s["failed"] += 1
            if not quiet and s["failed"] <= 3:
                print(f"  parse failed {r.file}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if not ls:
            s["empty"] += 1
            continue
        s["ok"] += 1
        if not meta.get("payment_ref"):
            s["no_ref"] += 1
        sp = round(sum(l["paid"] for l in ls), 2)
        tot = meta.get("total")
        ok = tot is not None and abs(sp - tot) <= max(0.05, 0.0005 * abs(tot))
        s["total_ok"] += int(ok)
        for l in ls:
            out.append((r.file, r.file, r.entity, r.payer_name, r.received_date, r.format, meta.get("payment_ref"), l["doc"],
                        doc_ref_norm(l["doc"]), l["paid"], l["deduction"], l["reason"], tot if tot is not None else sp,
                        meta.get("currency"), l["voucher"], l["po"], str(l["doc_date"])[:10] if l["doc_date"] is not None else None,
                        l["gross"], tot, bool(ok)))
    df = pd.DataFrame(out, columns=COLS)
    for c in ("entity", "format", "currency"):
        df[c] = df[c].astype("category")
    if cache and date_from is None and date_to is None:
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        df.to_parquet(cpath, index=False)
    return df


def print_stats(st: dict):
    print("format  files   ok  failed empty  no_ref  total_ok(sum==header)")
    for f, s in sorted(st.items()):
        print(f"{f:6} {s['files']:6} {s['ok']:5} {s['failed']:5} {s['empty']:5} {s['no_ref']:6}  {s['total_ok']:5} ({s['total_ok'] / max(1, s['ok']):.1%})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="sample")
    ap.add_argument("--from", dest="dfrom"); ap.add_argument("--to", dest="dto")
    a = ap.parse_args()
    st = {}
    df = lines(a.phase, a.dfrom, a.dto, cache=not (a.dfrom or a.dto), stats=st)
    print(len(df), "lines in", df.ra_file.nunique(), "advices")
    print_stats(st)
