"""Remittance-advice parser (csv / xlsx / txt / pdf) → heads (one per advice) + docs (one per document line).
Cached to matching/out/<phase>/remit_{heads,docs}.parquet."""
import csv
import io
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from matching.batch.common import DATA, PHASE_START, ROOT, parse_amount

DATE_RE = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})\s*$|^\s*(\d{4})-(\d{2})-(\d{2})\s*$")
ANY_DATE = re.compile(r"(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})|(\d{4})-(\d{2})-(\d{2})")
AMT_CCY = re.compile(r"(-?[\d][\d.,' ]*\d)\s*([A-Z]{3})\b|\b([A-Z]{3})\s*(-?[\d][\d.,' ]*\d)")
NUM_RE = re.compile(r"^\(?-?[\d][\d.,' ]*\)?$")
TXT_DOC = re.compile(r"^\s+(\S+)\s+(\d{1,2}[/.]\d{1,2}[/.]\d{2,4}|\d{4}-\d{2}-\d{2})\s+(-?[\d.,]+)"
                     r"(?:\s+(-?[\d.,]+))?(?:\s+(.*?))?\s*$")
HEAD_COLS = ["file", "entity", "payer_name", "received_date", "format", "payment_ref", "payment_date", "currency",
             "total"]
DOC_COLS = ["file", "document", "doc_date", "gross", "deduction", "reason", "paid", "your_order"]


def _iso(s, us=False):
    if s is None:
        return ""
    if hasattr(s, "strftime"):
        return s.strftime("%Y-%m-%d")
    m = ANY_DATE.search(str(s))
    if not m:
        return ""
    if m.group(4):
        return f"{m.group(4)}-{m.group(5)}-{m.group(6)}"
    a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
    y = int(y) + 2000 if len(y) == 2 else int(y)
    d, mo = (b, a) if us or a > 12 and False else (a, b)
    if mo > 12:
        d, mo = mo, d
    try:
        return f"{y:04d}-{mo:02d}-{d:02d}"
    except ValueError:
        return ""


def _amt(x):
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    return parse_amount(str(x).replace("'", "").replace(" ", ""))


def _doc_row(doc, date, gross, ded, reason, paid, order=""):
    g = _amt(gross)
    d = _amt(ded)
    d = 0.0 if d != d else abs(d)
    p = _amt(paid)
    if p != p:
        p = g - d if g == g else float("nan")
    if g != g:
        g = p + d
    return {"document": str(doc).strip(), "doc_date": _iso(date), "gross": g, "deduction": d,
            "reason": re.sub(r"^\(?-?[\d.,]+\)?\s*", "", (reason or "").strip() if isinstance(reason, str)
                             or reason is None else str(reason)),
            "paid": p, "your_order": "" if order is None else str(order).strip()}


def parse_csv(text):
    sep = ";" if text.split("\n", 1)[0].count(";") > text.split("\n", 1)[0].count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=sep))
    head = {"payment_ref": "", "payment_date": "", "currency": "", "total": float("nan")}
    docs, cols = [], None
    for r in rows:
        if not r or not any(c.strip() for c in r):
            continue
        k0 = r[0].strip().lower()
        if cols is None and k0 in ("payment_ref", "ref", "referencia", "reference"):
            kv = {r[i].strip().lower(): r[i + 1].strip() for i in range(0, len(r) - 1, 2)}
            head["payment_ref"] = kv.get("payment_ref", kv.get(k0, ""))
            head["payment_date"] = _iso(kv.get("payment_date", ""))
            head["currency"] = kv.get("currency", "")
            continue
        if cols is None and k0 in ("document", "documento", "doc", "invoice", "factura"):
            cols = [c.strip().lower() for c in r]
            continue
        if cols is not None:
            if k0.startswith("total"):
                pv = [c for c in r[1:] if c.strip()]
                if pv:
                    head["total"] = _amt(pv[-1])
                continue
            d = dict(zip(cols, r))
            docs.append(_doc_row(d.get(cols[0]), d.get("doc_date", d.get("date")), d.get("gross"),
                                 d.get("discount_or_deduction", d.get("deduction")), d.get("reason"),
                                 d.get("paid"), d.get("your_order")))
    return head, docs


def parse_xlsx(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    head = {"payment_ref": "", "payment_date": "", "currency": "", "total": float("nan")}
    docs = []
    for ws in wb.worksheets:
        cols = None
        for row in ws.iter_rows(values_only=True):
            vals = list(row)
            if not any(v is not None and str(v).strip() for v in vals):
                continue
            s0 = str(vals[0]).strip().lower() if vals[0] is not None else ""
            if cols is None:
                low = [str(v).strip().lower() if v is not None else "" for v in vals]
                if s0 in ("document", "documento", "doc", "invoice", "factura", "rechnung", "facture", "fattura",
                          "invoice no", "invoice number", "our invoice", "voucher") or ("gross" in low or "paid" in low
                                                                         or "net paid" in low):
                    cols = low
                    continue
                rest = [v for v in vals[1:] if v is not None and str(v).strip()]
                if rest and not head["payment_ref"] and ("advice" in s0 or "ref" in s0 or "pago" in s0
                                                         or "payment" in s0 or "aviso" in s0):
                    head["payment_ref"] = str(rest[0]).strip()
                    if len(rest) > 1:
                        head["payment_date"] = _iso(rest[1])
                    for v in rest[2:]:
                        if isinstance(v, str) and re.fullmatch(r"[A-Z]{3}", v.strip()):
                            head["currency"] = v.strip()
                continue
            if s0.startswith("total"):
                nums = [v for v in vals[1:] if v is not None and str(v).strip()]
                if nums:
                    head["total"] = _amt(nums[-1])
                continue
            d = dict(zip(cols, vals))
            gk = next((c for c in cols if c in ("gross", "bruto", "amount", "importe", "brutto")), None)
            pk = next((c for c in cols if c in ("paid", "pagado", "net", "bezahlt", "payé", "pagato")
                       or "paid" in c), None)
            dk = next((c for c in cols if "deduc" in c or "discount" in c or "dto" in c), None)
            rk = next((c for c in cols if "reason" in c or "motivo" in c), None)
            ok = next((c for c in cols if "order" in c or "pedido" in c or "po" == c), None)
            datek = next((c for c in cols if "date" in c or "fecha" in c), None)
            dockey = "invoice" if "invoice" in cols and cols[0] != "invoice" else cols[0]
            docs.append(_doc_row(d.get(dockey), d.get(datek), d.get(gk), d.get(dk), d.get(rk), d.get(pk), d.get(ok)))
    wb.close()
    return head, docs


def _head_from_text(lines, us):
    head = {"payment_ref": "", "payment_date": "", "currency": "", "total": float("nan")}
    for ln in lines:
        if ln.lower().startswith("subject:"):
            toks = ln.split()
            head["payment_ref"] = toks[-1] if len(toks) > 1 else ""
            break
    body = "\n".join(l for l in lines if not re.match(r"^(From|To|Date|Subject):", l))
    m = AMT_CCY.search(body)
    if m:
        a, c = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
        head["total"], head["currency"] = _amt(a), c
        after = body[m.end():]
        dm = ANY_DATE.search(after)
        if dm:
            head["payment_date"] = _iso(dm.group(0), us=us)
    return head


def parse_txt(text):
    lines = text.splitlines()
    us = " on " in text and "payment of" in text
    head = _head_from_text(lines, us)
    docs = []
    for ln in lines:
        m = TXT_DOC.match(ln)
        if not m:
            continue
        doc, date, gross, ded, reason = m.groups()
        if ANY_DATE.fullmatch(doc or ""):
            continue
        g = _amt(gross)
        d = _amt(ded) if ded else 0.0
        docs.append(_doc_row(doc, date, g, d, reason or "", g - abs(d) if g >= 0 else g))
    return head, docs


def parse_pdf_text(text):
    L = [l.strip() for l in text.splitlines() if l.strip()]
    head = {"payment_ref": "", "payment_date": "", "currency": "", "total": float("nan")}
    amt_idx = None
    for i, l in enumerate(L[:30]):
        if not head["payment_ref"] and re.search(r"\bref|réf|rif|referenz|nr\.|n[º°o]\b", l, re.I) and i + 1 < len(L):
            head["payment_ref"] = L[i + 1].strip()
        m = AMT_CCY.fullmatch(l)
        if m and amt_idx is None:
            a, c = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
            head["total"], head["currency"] = _amt(a), c
            amt_idx = i
        if not head["payment_date"] and DATE_RE.match(l):
            head["payment_date"] = _iso(l)
    start = (amt_idx or 0) + 1
    starts = [i for i in range(start, len(L) - 1) if DATE_RE.match(L[i + 1]) and not DATE_RE.match(L[i])]
    docs = []
    for k, i in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(L)
        vals = L[i + 2:end]
        nums, texts = [], []
        for v in vals:
            if NUM_RE.match(v.replace(" ", "")):
                nums.append(_amt(v))
            elif len(v) < 40 and not v.upper().startswith("TOTAL"):
                texts.append(v)
            else:
                break
        if not nums:
            continue
        g, p = nums[0], nums[-1]
        d = nums[1] if len(nums) >= 3 else (abs(g - p) if len(nums) == 2 else 0.0)
        docs.append(_doc_row(L[i], L[i + 1], g, d, " ".join(texts), p))
    return head, docs


def parse_file(args):
    path, fmt = args
    try:
        if fmt == "csv":
            raw = open(path, "rb").read()
            try:
                txt = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                txt = raw.decode("latin-1")
            return parse_csv(txt)
        if fmt == "xlsx":
            return parse_xlsx(path)
        if fmt == "txt":
            raw = open(path, "rb").read()
            try:
                txt = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                txt = raw.decode("latin-1")
            return parse_txt(txt)
        if fmt == "pdf":
            import pypdf
            r = pypdf.PdfReader(path)
            return parse_pdf_text("\n".join(p.extract_text() or "" for p in r.pages))
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}, []
    return {"error": "fmt"}, []


def _index(phase, since_days):
    """(abs_path, row) for the phase's advices plus earlier-phase ones received within since_days before start."""
    out = []
    srcs = [phase] + {"final": ["live", "train"]}.get(phase, ["train"] if phase not in ("train",) else [])
    start = PHASE_START.get(phase, "0000")
    lo = (pd.Timestamp(start) - pd.Timedelta(days=int(since_days))).strftime("%Y-%m-%d") if start != "0000" else ""
    seen = set()
    for k, src in enumerate(srcs):
        f = os.path.join(DATA, src, "remittance_advices", "index.csv")
        if not os.path.exists(f):
            continue
        ix = pd.read_csv(f, dtype=str, keep_default_na=False)
        if k > 0:
            ix = ix[(ix.received_date >= lo) & (ix.received_date < start)]
        for r in ix.itertuples(index=False):
            if r.file in seen:
                continue
            seen.add(r.file)
            out.append((os.path.join(DATA, src, "remittance_advices", r.file), r))
    return out


def parse_remittances(phase, since_days=45, cache=True, workers=8):
    od = os.path.join(ROOT, "matching", "out", phase)
    hp, dp = os.path.join(od, "k1_remit_heads.parquet"), os.path.join(od, "k1_remit_docs.parquet")
    ixf = os.path.join(DATA, phase, "remittance_advices", "index.csv")
    if (cache and os.path.exists(hp) and os.path.exists(dp)
            and (not os.path.exists(ixf) or os.path.getmtime(hp) > os.path.getmtime(ixf))):
        H = pd.read_parquet(hp)
        if {"sum_paid", "total", "payment_ref"} <= set(H.columns):
            return H, pd.read_parquet(dp)
    t0 = time.time()
    ix = _index(phase, since_days)
    import multiprocessing as mp
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(parse_file, [(p, r.format) for p, r in ix], chunksize=64))
    heads, docs = [], []
    for (p, r), (h, ds) in zip(ix, res):
        heads.append({"file": r.file, "entity": r.entity, "payer_name": r.payer_name,
                      "received_date": r.received_date, "format": r.format, **{k: h.get(k, "") for k in
                                                                               ("payment_ref", "payment_date",
                                                                                "currency")},
                      "total": h.get("total", float("nan")), "error": h.get("error", "")})
        for d in ds:
            docs.append({"file": r.file, **d})
    H = pd.DataFrame(heads)
    D = pd.DataFrame(docs, columns=DOC_COLS)
    H["total"] = pd.to_numeric(H.total, errors="coerce")
    sums = D.groupby("file").paid.sum()
    H["sum_paid"] = H.file.map(sums).fillna(0.0)
    H.loc[H.total.isna(), "total"] = H.sum_paid
    os.makedirs(od, exist_ok=True)
    H.to_parquet(hp, index=False)
    D.to_parquet(dp, index=False)
    print(f"  [remit_io] parsed {len(H):,} advices / {len(D):,} doc lines in {time.time() - t0:.1f}s", flush=True)
    return H, D


if __name__ == "__main__":
    import sys
    ph = sys.argv[1] if len(sys.argv) > 1 else "val"
    H, D = parse_remittances(ph, cache=False)
    H["ok"] = (H.total - H.sum_paid).abs() <= 0.02
    print(H.groupby("format").agg(n=("file", "size"), ok=("ok", "mean"), ref=("payment_ref", lambda s: (s != "").mean()),
                                  date=("payment_date", lambda s: (s != "").mean()),
                                  ccy=("currency", lambda s: (s != "").mean()),
                                  err=("error", lambda s: (s != "").sum())))
    print(H[~H.ok].groupby("format").head(2)[["file", "payment_ref", "payment_date", "currency", "total", "sum_paid"]])
