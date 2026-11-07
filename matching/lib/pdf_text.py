"""Raw text of the PDFs a phase can use, cached as matching/out/cache/<phase>/pdf_text.parquet.

Which PDFs: the phase's own pdfs/index.csv (customer invoices, credit notes, proformas, bank-account notices) plus, for
documents issued BEFORE the phase (open_items_at_start), the same document's PDF from data/train/pdfs when the phase index
does not have it. PDFs are documents, not labels, so reading old ones is not leakage. Supplier documents are skipped (the
AR clauses are what matters here).

Only the clause part is kept: the lines after the last `IBAN` line (the free-text clauses), the payment-terms line and, for
credit notes, the reference line. Per-file try/except, failures counted.
"""
from __future__ import annotations

import os
import re
import sys
import time

import pandas as pd

from . import data

TYPES = ("invoice", "credit_note", "proforma", "bank_account_notice")
FOOTER = re.compile(r"sint[eé]tico|synthetic test document", re.I)
TERMS = re.compile(r"^(Forma de pago|Conditions|Zahlungsbedingungen|Payment terms|Condizioni|Condi[cç][oõ]es|Termos|"
                   r"Warunki|Betalingsvoorwaarden|Modalit)", re.I)


def _index(phase: str) -> pd.DataFrame:
    p = os.path.join(data.phase_dir(phase), "pdfs", "index.csv")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["file", "document_type", "document_number", "entity", "party_id", "document_date", "root"])
    i = pd.read_csv(p, dtype=str)
    i = i[i.document_type.isin(TYPES)].copy()
    i["root"] = os.path.join(data.phase_dir(phase), "pdfs")
    return i


def file_list(phase: str) -> pd.DataFrame:
    idx = _index(phase)
    if phase not in ("train", "train_lite"):
        tr = _index("train")
        if len(tr):
            it = data.items(phase)
            it = it[it.side.eq("AR")]
            have = set(zip(idx.entity, idx.document_number))
            want = set(zip(it.entity, it.document_number)) - have
            tr = tr[[k in want for k in zip(tr.entity, tr.document_number)]]
            idx = pd.concat([idx, tr], ignore_index=True)
    return idx.drop_duplicates(["file"]).reset_index(drop=True)


def extract(path: str) -> str:
    import pypdf
    txt = "\n".join((p.extract_text() or "") for p in pypdf.PdfReader(path).pages)
    lines = [l.strip() for l in txt.split("\n")]
    k = max([j for j, l in enumerate(lines) if l.upper().startswith("IBAN")] or [-1])
    keep = [l for l in lines if TERMS.match(l)]
    for j, l in enumerate(lines):  # credit notes: "Ref." then the invoice number on the next line
        if l.rstrip(".:").lower() in ("ref", "réf", "rif", "referencia", "referenz", "reference") and j + 1 < len(lines):
            keep.append("REF " + lines[j + 1])
    tail = lines[k + 1:] if k >= 0 else lines[-8:]
    keep += [l for l in tail if l and not FOOTER.search(l)]
    return re.sub(r"\s+", " ", " || ".join(keep)).strip()


def build(phase: str, verbose: bool = True) -> pd.DataFrame:
    p = os.path.join(data.CACHE, phase, "pdf_text.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    fl = file_list(phase)
    out, fails, t0 = [], 0, time.time()
    for n, (root, f) in enumerate(zip(fl.root, fl.file)):
        try:
            out.append(extract(os.path.join(root, f)))
        except Exception:
            out.append(None); fails += 1
        if verbose and n % 5000 == 0:
            print(f"  pdf_text {phase}: {n}/{len(fl)} ({time.time() - t0:.0f}s, {fails} failures)", file=sys.stderr)
    fl["text"] = out
    fl = fl.drop(columns=["root"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fl.to_parquet(p, index=False)
    if verbose:
        print(f"  pdf_text {phase}: {len(fl)} files, {fails} extraction failures, {time.time() - t0:.0f}s", file=sys.stderr)
    return fl
