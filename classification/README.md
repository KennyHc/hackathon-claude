# classification (piece C)

**Mission (two parts):**

1. **Lines that settle no item:** category (`data/masters/categories.csv`) + `UNIDENTIFIED` when it's a receipt with no
   open item, plus `p_items` = probability that the line settles items at all, so integration can pick B vs C per line.
2. **Exception rules library** `classification/exceptions.py`: the POLICIES §3 logic (SHORT_DISCOUNT window and
   rounding, SHORT_BANK_CHARGES ≤ 60 EUR & ≤ 2 %, SHORT_WITHHOLDING rates, FX_DIFFERENCE precedence, PARTIAL…), applied by
   integration to matching's allocation rows. Exceptions are 20 % of the score, and are mostly rule-based.

**Reads:** `ingestion/out/<phase>/*` (until v1 lands: `data/<phase>/` directly).
**Writes:** `classification/out/<phase>/line_labels.parquet` + the importable `classification/exceptions.py`
(`docs/CONTRACTS.md` §C).
**Entry point:** `python classification/run.py --phase sample`

TRAIN golden rows with empty `item_id` are your labels for part 1 (bank_code, text, sign, counterparty are strong
features); golden item rows with their `exception` are your test set for part 2.
