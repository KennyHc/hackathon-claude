# Report: PDF Clause Extractor (`matching/refs/pdf_clauses.py`)

## 1. What Was Built

We implemented a high-throughput, multi-core PDF clause and metadata extraction engine in `matching/refs/pdf_clauses.py` that parses document PDFs (invoices, supplier invoices, credit notes, supplier credit notes, proformas, bank account notices) and creates structured tables for reconciliation matching per **POLICIES.md §5.7–5.9 and §3**.

### Core Architecture & Features
- **Parallel Processing**: Employs `multiprocessing.Pool(8)` with chunked work dispatch (`chunksize=100`), achieving ~1,350 PDFs/second on macOS Apple Silicon.
- **Robust Regex & Layout Parsers**: Multi-lingual clause parsing across 17 entities and all languages in the synthetic corpus (English, Spanish, French, German, Italian, Portuguese, Polish, Dutch, etc.).
- **Dual Amount Parsing**: Handles both European (`1.234,56`) and Anglo-Saxon (`1,234.56`) number formats, including thin spaces and currency suffixes.
- **Smart Line Reconstruction**: Automatically bridges multi-line wrapped sentences (such as factoring notices, retention clauses, and proforma advance mentions).
- **Caching Mechanism**:
  - `load(phase, extra_phases=None, force=False)`: Instantly loads `matching/out/<phase>/pdf_clauses.parquet` if it exists.
  - `--force`: Triggers full re-extraction.
  - `--extra train,live`: Parses predecessor phases for old open items listed in `open_items_at_start.csv.gz`.

---

## 2. Output Schema

The output Parquet table contains 26 columns matching the specification:

| Column | Type | Description |
|---|---|---|
| `file` | str | Relative PDF path from index |
| `document_type` | str | Document type (`invoice`, `credit_note`, etc.) |
| `document_number` | str | Document identifier |
| `entity` | str | Legal entity identifier |
| `party_id` | str | Customer / Supplier ID |
| `document_date` | str | Document issue date (ISO `YYYY-MM-DD`) |
| `src_phase` | str | Source phase (`sample`, `val`, `train`, etc.) |
| `clause_text` | str | Joined free-text clause lines (`line1 \|\| line2`) |
| `terms_text` | str | Terms / payment conditions header line |
| `n_instalments` | Int64 | Number of instalments (2–12) |
| `retention_pct` | float64 | Guarantee retention percentage |
| `retention_amount` | float64 | Retained guarantee amount |
| `retention_release_date` | str | ISO date (`YYYY-MM-DD`) for retention release |
| `amount_payable` | float64 | Amount payable at initial due date |
| `advance_amount` | float64 | Proforma advance amount deducted |
| `advance_proforma` | str | Proforma invoice identifier cited |
| `replaces_document` | str | Cancelled document replaced by this invoice |
| `cancels_document` | str | Original invoice cancelled by credit note |
| `ref_document` | str | Credit note reference header field |
| `service_month` | str | Service month (ISO `YYYY-MM`) |
| `cn_settlement` | str | Settlement mode: `offset`, `paid_out`, or `""` |
| `rappel` | bool | Whether the clause specifies a volume rebate |
| `factoring` | bool | Whether the invoice has been assigned to a factor |
| `new_iban` | str | New bank account IBAN from change notices |
| `customer_po` | str | Customer purchase order / reference number |
| `total` | float64 | Total invoice / document amount |

---

## 3. How to Run

### Command Line Interface (CLI)
```bash
# Run on sample (writes matching/out/sample/pdf_clauses.parquet)
.venv/bin/python matching/refs/pdf_clauses.py --phase sample

# Run on validation dataset with historical train open items
.venv/bin/python matching/refs/pdf_clauses.py --phase val --extra train

# Force re-extraction (bypassing cache)
.venv/bin/python matching/refs/pdf_clauses.py --phase val --force

# Print clause census
.venv/bin/python matching/refs/pdf_clauses.py --census
```

### Python API
```python
from matching.refs.pdf_clauses import extract, load

# Fast load from cached parquet
df = load('val')

# Extract fresh DataFrame
df = extract(phase='val', extra_phases=['train'])
```

---

## 4. Measured Performance & Sanity Checks

### Run Times
| Phase | Document Count | Extraction Time | Throughput | Parquet Size |
|---|---|---|---|---|
| `sample` | 2,298 PDFs | 1.74 s | 1,321.9 PDFs/s | ~280 KB |
| `val` (alone) | 17,794 PDFs | 15.84 s | 1,123.5 PDFs/s | ~2.1 MB |
| `val` + `--extra train` | 39,138 PDFs | 28.48 s | 1,374.0 PDFs/s | ~4.7 MB |
| Cached load (`load('val')`) | 39,138 rows | 0.08 s | Instantaneous | — |

*Target limit was < 3 minutes for val; implementation finishes in under 29 seconds for 39,138 PDFs.*

### Sanity Check Verification
1. **Retention Integrity**: For all invoices with retention clauses, checked that:
   $$\text{retention\_amount} \approx \frac{\text{retention\_pct}}{100} \times \text{total}$$
   - Evaluated on 3,243 rows: **Max absolute difference = 0.0095** (< 0.01 cent / rounding error).
   - **Status: PASSED**.
2. **Instalment Bounds**: Checked that $n \in [2, 12]$:
   - Evaluated on 2,896 instalment rows: **$\min = 2, \max = 6$**.
   - **Status: PASSED**.

---

## 5. Non-Null Column Counts

### Validation Set (17,794 PDFs alone)
| Column | Non-Null Count | Percentage |
|---|---|---|
| `file` | 17,794 | 100.0% |
| `document_type` | 17,794 | 100.0% |
| `document_number` | 17,794 | 100.0% |
| `entity` | 17,794 | 100.0% |
| `party_id` | 17,794 | 100.0% |
| `document_date` | 17,794 | 100.0% |
| `src_phase` | 17,794 | 100.0% |
| `clause_text` | 12,716 | 71.5% |
| `terms_text` | 17,412 | 97.9% *(all non-notices/proformas)* |
| `n_instalments` | 1,579 | 8.9% |
| `retention_pct` | 849 | 4.8% |
| `retention_amount` | 849 | 4.8% |
| `retention_release_date` | 849 | 4.8% |
| `amount_payable` | 849 | 4.8% |
| `advance_amount` | 278 | 1.6% |
| `advance_proforma` | 278 | 1.6% |
| `replaces_document` | 1,779 | 10.0% |
| `cancels_document` | 319 | 1.8% |
| `ref_document` | 1,881 | 10.6% |
| `service_month` | 453 | 2.5% |
| `cn_settlement` | 2,709 | 15.2% *(offset: 1,786; paid_out: 923)* |
| `rappel` | 1,330 | 7.5% True |
| `factoring` | 1,542 | 8.7% True |
| `new_iban` | 98 | 0.6% *(100% of bank account notices)* |
| `customer_po` | 8,465 | 47.6% |
| `total` | 17,696 | 99.4% *(100% of commercial docs)* |

### Validation Set with `--extra train` (39,138 PDFs total)
| Column | Non-Null Count | Percentage |
|---|---|---|
| `total documents` | 39,138 | 100.0% |
| `clause_text` | 30,901 | 79.0% |
| `terms_text` | 38,756 | 99.0% |
| `n_instalments` | 2,896 | 7.4% |
| `retention_amount` | 3,243 | 8.3% |
| `advance_amount` | 2,084 | 5.3% |
| `replaces_document` | 2,418 | 6.2% |
| `cancels_document` | 319 | 0.8% |
| `ref_document` | 2,400 | 6.1% |
| `service_month` | 565 | 1.4% |
| `cn_settlement` | 11,407 | 29.1% *(offset: 7,906; paid_out: 3,501)* |
| `rappel` | 9,637 | 24.6% True |
| `factoring` | 3,256 | 8.3% True |
| `new_iban` | 98 | 0.3% |
| `customer_po` | 16,392 | 41.9% |
| `total` | 39,040 | 99.7% |

---

## 6. Census of Clause Phrasings (Top 40 from Sample N=531)

Numbers replaced with `#`:

| Rank | Frequency | Phrasing / Pattern |
|---|---|---|
| 1 | 45 | `This credit note is to be deducted from your next payment.` |
| 2 | 41 | `Change of bank account` |
| 3 | 41 | `Kind regards,` |
| 4 | 36 | `IBAN: ES#  BIC: PENAESM#XXX` |
| 5 | 35 | `The amount of this credit note will be refunded by bank transfer.` |
| 6 | 21 | `PROFORMA INVOICE — not a tax invoice. Advance payment for order; the final invoice will deduct it. Order total #,#.# EUR.` |
| 7 | 16 | `PROFORMA INVOICE — not a tax invoice. Advance payment for order; the final invoice will deduct it.` |
| 8 | 16 | `Este abono se compensará en sus próximos pagos.` |
| 9 | 14 | `IBAN: ES#` |
| 10 | 12 | `IBAN: CH#  BIC: RHEIDEFFXXX` |
| 11 | 12 | `IBAN: DE#  BIC: RHEIDEFFXXX` |
| 12 | 12 | `IBAN: IT#A#  BIC: VALDITMMXXX` |
| 13 | 12 | `IBAN: MA#  BIC: LORVFRPPXXX` |
| 14 | 12 | `IBAN: NL#ZZTJ#  BIC: RHEIDEFFXXX` |
| 15 | 12 | `IBAN: PL#  BIC: RHEIDEFFXXX` |
| 16 | 8 | `Volume rebate Q# # (#.#% on #.#,# EUR): to be offset against your next payment.` |
| 17 | 8 | `Cambio de cuenta bancaria` |
| 18 | 8 | `Atentamente,` |
| 19 | 8 | `FACTURA PROFORMA — no válida como factura. Pago anticipado del pedido por transferencia; la factura definitiva se emitirá a` |
| 20 | 8 | `This invoice cancels and replaces invoice INV#-# dated #/#/#. Do not pay both.` |
| 21 | 8 | `IBAN: GB#ZZWJ#  BIC: NORTGB#LXXX` |
| 22 | 7 | `Please note that from #/#/# our payments will be made from account` |
| 23 | 7 | `Rappel T# # por volumen (#.# % s/ #.#,# EUR): se compensará en la próxima liquidación.` |
| 24 | 6 | `IBAN: DE#` |
| 25 | 6 | `El importe de este abono les será reembolsado mediante transferencia bancaria.` |
| 26 | 6 | `entidad.` |
| 27 | 6 | `IBAN: FR#  BIC: LORVFRPPXXX` |
| 28 | 6 | `IBAN: DE#  BIC: LORVFRPPXXX` |
| 29 | 6 | `IBAN: FR#` |
| 30 | 6 | `IBAN: IT#A#` |
| 31 | 6 | `IBAN: MA#` |
| 32 | 6 | `Volume rebate Q# # (#.#% on #,#.# EUR): to be offset against your next payment.` |
| 33 | 6 | `IBAN: NL#ZZTJ#` |
| 34 | 6 | `IBAN: PL#` |
| 35 | 6 | `PROFORMA INVOICE — not a tax invoice. Advance payment for order; the final invoice will deduct it. Order total #,#.# USD.` |
| 36 | 5 | `CANCELLING OF INVOICE Nº INV#-#` |
| 37 | 5 | `The receivable represented by this invoice has been assigned to Ribagorza Confirming. Payment is only discharging if made to it.` |
| 38 | 5 | `Volume rebate Q# # (#.#% on #,#.# USD): to be offset against your next payment.` |
| 39 | 5 | `PROFORMA INVOICE — not a tax invoice. Advance payment for order; the final invoice will deduct it. Order total #,#.# CNY.` |
| 40 | 5 | `Order total #,#.# EUR.` |

---

## 7. Known Gaps & Edge Cases

1. **Multi-page Invoices**: While synthetic invoices in the current phases are 1 page, `extract_one_pdf` is fully multi-page aware (`\n.join(p.extract_text() for p in reader.pages)`).
2. **Proforma Clauses**: Proforma invoices do not feature payment terms lines; the extraction logic specifically accounts for this by anchoring to the `TOTAL` block.
3. **Drift in LIVE/FINAL**:
   - South Africa entity `ZA01` (introduced 2026-01-01 in ZAR currency) with numbering pattern `KWI...` will be processed without code change as regexes are currency-agnostic and match any document identifier.
   - Bank statement text shifts (e.g. Banco del Moncayo Norma 43 to camt.053) do not affect PDF layouts.
