# Brief k1-remit: remittance-advice parser (one module, one function)

Create `matching/batch/remit_parse.py` with:

```python
def parse_remittances(phase: str, data_dir: str = "data", since_days: int = 45) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (heads, docs).
    heads: one row per advice file: file, entity, payer_name, received_date, format, payment_ref, payment_date
           (ISO YYYY-MM-DD), currency, total (float, the cash total of the advice; sum of `paid` if no total line)
    docs:  one row per document line: file, document (string exactly as printed, stripped), doc_date (ISO or ''),
           gross (float), deduction (float, >= 0, absolute value of discount_or_deduction; 0 when empty),
           reason (str, '' when none), paid (float), your_order (str, '' when none)"""
```

Sources: `data/<phase>/remittance_advices/index.csv` (file, entity, payer_name, received_date, format) and the
files it lists (`data/<phase>/remittance_advices/<file>`). ALSO include advices of `data/train/remittance_advices/`
whose received_date is within `since_days` days BEFORE the first received_date of the phase (advices can arrive before
the cash; for phase `val` train contains them, for `live` too). Dedupe by `file`. Phases to support: sample, val,
train, live, final (final may not exist yet: skip missing folders silently).

Formats (look at many examples of each, in data/val, data/train AND data/live — one big customer changes layout in
September 2025, which only shows in data/live; find it and support both layouts):
- csv: header line `payment_ref,<ref>,payment_date,<date>,currency,<ccy>`, then a table
  `document,doc_date,gross,discount_or_deduction,reason,paid,your_order`, then `TOTAL,...`. Separator may be `,` or
  `;`. Decimal may be `.` or `,` (European `1.234,56`).
- xlsx: open with openpyxl (read_only=True, data_only=True); find the same kind of header/table.
- txt: e-mail text; `Subject:` carries the payment reference (e.g. `Aviso de pago 241231885236`, `Payment advice
  REM/2021/0798`), the body gives total, currency and date ("transferencia por importe de 42.483,40 EUR con fecha
  31/12/2024", "payment of 143,072.41 AUD on 07/01/2021"), then one line per document
  (`FV24-0323522  26/09/24  2.525,06`), possibly with deductions/reasons. Inspect many files and languages
  (ES/EN/FR/DE/IT/PT/PL…). Dates can be dd/mm/yy, dd/mm/yyyy, mm/dd/yyyy (US), ISO.
- pdf: use `pypdf.PdfReader(path).pages[i].extract_text()`; parse the text like txt. PDFs are the slow part: keep it
  fast (only open each file once; if ~10k pdfs take > 3 min, report the timing).

Amount parsing: write one robust `parse_amount(s)` handling `1,234.56`, `1.234,56`, `1234.56`, `1234,56`, `-1.111,25`,
`(1,111.25)`, currency symbols/codes around it.

Validation (write a small script `matching/batch/eval_remit.py` that prints):
- files parsed / failed per format, for val and live (live has no golden; just parse counts and a few samples);
- share of heads with payment_ref, total, currency non-empty, and |sum(paid) - total| <= 0.01;
- for val: the golden is `data/val/golden_allocations.csv.gz` (bank_txn_id,item_id,amount,…,segment). Items:
  `from integration.lib.ledger import load_items; items = load_items("val")` (columns item_id, entity,
  document_number, …). Report the share of parsed `docs.document` that equal some item `document_number` exactly in the
  same entity, and the share after normalisation (strip, upper, remove spaces) — the documents may also be the
  customer's own voucher numbers or our number without prefix; report examples of documents that do not match.
- runtime per phase.

Cache: `parse_remittances` should save results to `matching/out/<phase>/remit_heads.parquet` and
`remit_docs.parquet` and reuse them if present and newer than index.csv (pass `cache=False` to force).

Report to `matching/batch/briefs/k1-remit.REPORT.md`. Do not edit any other file.
