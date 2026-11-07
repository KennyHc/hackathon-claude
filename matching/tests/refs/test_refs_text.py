"""Unit tests for matching.lib.refs_text / refs_solve.   python -m pytest matching/tests/refs -q   (or run the file directly)"""
import sys
sys.path.insert(0, ".")
from matching.lib import refs_text as rt, refs_solve as rs


def test_norm_and_name_key():
    assert rt.norm("Climatización Muñoz, S.L.") == "CLIMATIZACION MUNOZ S L"
    assert rt.name_key_tokens("Pool & Spa Howard Pty Ltd") == ("POOL", "AND", "SPA", "HOWARD")
    assert rt.name_key_tokens("SL") == ("SL",)


def test_doc_keys_and_find_refs():
    docs = ["INV21-0100108", "FV25-0102742", "1142301382", "FT25-01-27109", "RE250102511"]
    # printed number, no prefix, leading zeros, tail6, truncation, typo
    assert rt.find_refs(rt.text_tokens("PAGO INV21-0100108"), docs).get(0) == 4
    assert rt.find_refs(rt.text_tokens("FRA. FV25-010274"), docs).get(1) == 4           # truncated at 16
    assert rt.find_refs(rt.text_tokens("FRA. FV25-01027 | 42"), docs).get(1) is not None  # split across N43 records
    assert rt.find_refs(rt.text_tokens("REF 100108"), docs).get(0) == 2                 # tail without zeros
    assert rt.find_refs(rt.text_tokens("FACT 1142301382"), docs).get(2) == 4
    assert rt.find_refs(rt.text_tokens("FATT. FT25-01-27109"), docs).get(3) == 4
    assert rt.find_refs(rt.text_tokens("RECHNUNG RE250102151"), docs).get(4) == 1       # transposed digits
    assert 0 not in rt.find_refs(rt.text_tokens("PAYMENT ID:0745132965"), docs)


def test_edit1():
    assert rt.edit1("12345", "12354") and rt.edit1("12345", "1234") and rt.edit1("12345", "12945")
    assert not rt.edit1("12345", "12984")


def test_service_month():
    assert rt.service_month("MANT. MAYO 25") == 5 and rt.service_month("SERVICE MAY 2025") == 5
    assert rt.service_month("SEPA CREDIT MARKETING") is None


def test_subset_solutions():
    assert rs.subset_solutions([100, 250, -50, 300], 300)  # 250-50+100 and 300
    assert (1, 2) in rs.subset_solutions([5, 7, 9], 16)
    assert rs.subset_solutions([5, 7], 1) == []


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
    print("ok")
