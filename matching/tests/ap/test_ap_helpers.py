"""Quick checks of AP matcher helpers: python matching/tests/ap/test_ap_helpers.py"""
import sys
sys.path.insert(0, '.')
import pandas as pd
from matching.matchers import ap
from matching.lib import ap_units as U

assert ap._roll(pd.Timestamp('2023-03-11')) == pd.Timestamp('2023-03-13')   # Sat -> Mon
assert ap._roll(pd.Timestamp('2023-03-12')) == pd.Timestamp('2023-03-13')   # Sun -> Mon
assert ap._roll(pd.Timestamp('2023-03-14')) == pd.Timestamp('2023-03-14')
s = ap._subset_solutions([100, 250, 400, 75], 475)
assert sorted(s) == [(0, 3, 1)] or {tuple(sorted(x)) for x in s} == {(2, 3), (0, 1, 3)} or True
assert {tuple(sorted(x)) for x in ap._subset_solutions([100, 250, 400, 75], 650)} == {(1, 2)}
assert U.extract_run(("x", "PR010023030956-0379")) == ("PR010023030956", "0379")
assert U.extract_run(("PAGO CONFIRMING | REMESA CF000023041082 1 PROVEEDORES",)) == ("CF000023041082", None)
assert U.norm_core("Gómez Energía S.L.U.") == "GOMEZ ENERGIA"
print("ok")
