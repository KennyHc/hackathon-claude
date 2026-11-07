"""South-African amount formats in remittance advices (FINAL adds ZA01 / ZAR).  .venv/bin/python matching/test_za_amounts.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matching import remittance_parse as R  # noqa: E402

NUM_CASES = {"1 234,56": 1234.56, "1 234,56": 1234.56, "1 234,56": 1234.56, "1 234,56": 1234.56,
             "R 5,100.69": 5100.69, "R5,100.69": 5100.69, "ZAR 5 100,69": 5100.69, "5 100,69 ZAR": 5100.69,
             "-R 1 234,56": -1234.56, "R -1 234,56": -1234.56, "(R 250,00)": -250.0, "12 345 678,90": 12345678.9,
             "1,234": 1234.0, "1.234,56": 1234.56, "-45.10": -45.1, "(12,50)": -12.5, "": None, "abc": None}

for s, want in NUM_CASES.items():
    assert R.num(s) == want, (s, R.num(s), want)

g = R.ROW_RE.match(R._norm_row("KWI000123 03/05/2026 R 5 100,69 -R 102,00 SHORT DELIVERY R 4 998,69")).groupdict()
assert (g["doc"], g["gross"], g["ded"], g["reason"], g["paid"]) == \
    ("KWI000123", "5100,69", "-102,00", "SHORT DELIVERY", "4998,69"), g
g = R.ROW_RE.match(R._norm_row("R2025-0045 03/05/2026 1 234,56 1 234,56")).groupdict()
assert g["doc"] == "R2025-0045" and g["gross"] == "1234,56", g          # a document number starting with R is kept
assert "12500,00 ZAR" in R._norm_amounts("payment of R 12 500,00 ZAR on 05/03/2026 ")
g = R.ROW_RE.match(R._norm_row("KWI-26-0308068 2026-03-05 R 5 100,69 R 5 100,69")).groupdict()   # ISO date row (Kenny)
assert (g["doc"], g["gross"], g["paid"]) == ("KWI-26-0308068", "5100,69", "5100,69"), g
assert R._norm_row("A2025-01-0001 05/03/2026 1 234,56").startswith("A2025-01-0001 05/03/2026 1234,56")
print("ok: ZA amount formats")
