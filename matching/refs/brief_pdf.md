# Brief k2-pdf: PDF clause extractor (matching/refs/pdf_clauses.py)

Goal: one script that parses the document PDFs and writes a cached table of the free-text clauses that matter for
matching (docs/POLICIES.md §5.7-5.9 and §3). Read docs/POLICIES.md and docs/DATA_DICTIONARY.md (pdfs section) first.

## Inputs
- `data/<phase>/pdfs/index.csv` (file, document_type, document_number, entity, party_id, document_date); files are
  `data/<phase>/pdfs/<file>`. document_type ∈ invoice, credit_note, supplier_invoice, supplier_credit_note, proforma,
  bank_account_notice.
- Text: `pypdf.PdfReader(path)`, join `page.extract_text()` of all pages. Each PDF is a short synthetic doc
  (header, lines, TOTAL, then "Terms:/Forma de pago:" and 0-2 free-text clause lines, then a disclaimer line
  "Documento sintético de prueba ...").

## Output
`matching/refs/pdf_clauses.py` with:
- `def extract(phase: str, extra_phases: list[str] = None) -> pd.DataFrame` and a CLI
  `.venv/bin/python matching/refs/pdf_clauses.py --phase {sample,val,train,live,final} [--extra train,live]`
  writing `matching/out/<phase>/pdf_clauses.parquet` (create the folder). Cached: if the parquet exists and `--force`
  is not given, `load(phase)` just reads it.
- `extra_phases`: also parse PDFs of those phases' folders (same output table, column `src_phase`), but ONLY rows of
  their index whose (entity, document_number) is in `data/<phase>/open_items_at_start.csv.gz` (columns entity,
  document_number) when that file exists (old open invoices whose PDF lives in an earlier phase).
- Use `multiprocessing.Pool` (8 procs) — val has ~17.8k PDFs; it must finish in < 3 min for one phase.
- Columns (all str except numbers float / int, NaN when absent):
  `file, document_type, document_number, entity, party_id, document_date, src_phase,
   clause_text` (all free-text clause lines joined with " || ": everything between the Terms/Forma de pago line and
   the disclaimer, plus any other line that is not header/table/total — be inclusive),
  `terms_text` (the Terms / Forma de pago / Conditions line),
  `n_instalments` (int: "3 vencimientos", "in 3 instalments", "3 rate", "3 échéances", "3 Raten", etc.),
  `retention_pct` (e.g. 10.0), `retention_amount`, `retention_release_date` (ISO), `amount_payable` (the "Amount payable
  at due date" figure, if any),
  `advance_amount` (advance / anticipo / proforma amount to deduct), `advance_proforma` (proforma number cited, e.g.
  PF21-0400008),
  `replaces_document` (the document number X in "anula y sustituye a la factura X" / "cancels and replaces invoice X" /
  any language), `cancels_document` (credit notes: "CANCELLING OF INVOICE Nº X" / "anulación de la factura X"),
  `ref_document` (the "Ref." header field of credit notes, e.g. INV0004248),
  `service_month` (ISO YYYY-MM when the PDF names a service month, e.g. "Servicio de mantenimiento mayo 2025"),
  `cn_settlement` ∈ {"offset","paid_out",""} (rappel / supplier credit note: "to be offset against your next payment",
  "deducted from your next payment", "will be paid to your bank account", "reembolsado mediante transferencia" …),
  `rappel` (bool: volume rebate text), `factoring` (bool: assignment to a factor), `new_iban` (IBAN in a bank-account
  change notice), `customer_po` (Your PO / Su pedido), `total` (TOTAL figure as float; parse both 1,234.56 and
  1.234,56 formats).

## How
1. FIRST do a census: parse ~600 PDFs sampled across all entities and document types of `data/val/pdfs`, print the
   distinct clause lines (numbers replaced by #) with counts, so you see every phrasing/language (ES, EN, FR, DE, IT,
   PT, PL…). Write regexes for ALL variants you see. Keep the census output in your report.
2. Implement, run on `--phase sample` and `--phase val --extra train` (time both), print counts of non-null per column.
3. Sanity: for invoices with retention, `retention_amount ≈ retention_pct/100 * total`; for instalments n between 2 and 12.

## Report
Write `matching/refs/REPORT_pdf.md`: what you built, how to run, run times, non-null counts per column on val, the
census of clause phrasings (top 40), known gaps. Then stop.
