"""Small unit tests for the `batches` matcher helpers (no data files needed).
Run: PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest matching/tests/batches -q   (or python matching/tests/batches/test_batches.py)"""
import sys

sys.path.insert(0, ".")
import pandas as pd

from matching.lib import batches_core as bc
from matching.matchers import batches as B


def _items():
    return pd.DataFrame({
        "item_id": ["I1", "I2", "I3"], "item_type": "ARI", "side": "AR", "entity": "ES01", "party_id": ["C1", "C2", "C1"],
        "party_name": ["Aguas Norte SL", "Piscinas Sur SL", "Aguas Norte SL"],
        "document_number": ["FV23-0000123", "FV24-0000123", "FV24-0000999"],
        "issue_date": pd.to_datetime(["2023-02-01", "2024-02-01", "2024-03-01"]), "currency": "EUR",
        "total_amount": [100.0, 100.0, 250.0], "open_amount": [100.0, 100.0, 250.0]})


def test_exact_and_suffix():
    A = bc.prep_items(_items())
    R = pd.DataFrame({"rid": [0, 1, 2], "entity": "ES01", "ccy": "EUR", "ref": ["FV24-0000123", "RECIBO 0000999", "RECIBO 0000123"],
                      "amount": [100.0, 250.0, 100.0], "name": [None, None, "Piscinas Sur SL"],
                      "adate": pd.to_datetime(["2024-05-01"] * 3)})
    r = bc.resolve(R, A).set_index("rid")
    assert r.loc[0, "item_id"] == "I2" and r.loc[1, "item_id"] == "I3"
    assert r.loc[2, "item_id"] == "I2"  # same digits in two years: customer name breaks the tie


def test_future_invoice_not_used():
    A = bc.prep_items(_items())
    R = pd.DataFrame({"rid": [0], "entity": "ES01", "ccy": "EUR", "ref": ["FV24-0000123"], "amount": [100.0], "name": [None],
                      "adate": pd.to_datetime(["2023-12-01"])})
    assert bc.resolve(R, A).empty


def test_name_fallback():
    A = bc.prep_items(_items())
    R = pd.DataFrame({"rid": [0], "entity": "ES01", "ccy": "EUR", "ref": [None], "amount": [250.0], "name": ["Aguas Norte S.L."],
                      "adate": pd.to_datetime(["2024-05-01"])})
    assert bc.resolve_by_name(R, A).item_id.tolist() == ["I3"]


def test_link_batches_requires_unique_or_id():
    bank = pd.DataFrame({"bank_txn_id": ["B1", "B2"], "account_id": "ES01-X", "amount": 50.0,
                         "booking_date": pd.to_datetime(["2025-01-02", "2025-01-02"]), "text": ["REMESA 111", "OTHER"]})
    T = pd.DataFrame({"batch_id": ["111"], "account_id": "ES01-X", "tot": 50.0, "date": pd.to_datetime(["2025-01-02"])})
    L = B._link_batches(bank, T, "batch_id")
    assert L.bank_txn_id.tolist() == ["B1"]  # two same-amount lines: the one quoting the batch id wins
    T["batch_id"] = "999"
    assert B._link_batches(bank, T, "batch_id").empty  # ambiguous and no id in text -> no claim


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
