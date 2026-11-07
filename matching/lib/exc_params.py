"""Parameters of matching/matchers/exception_rules.py. Policy constants come from docs/POLICIES.md s3; the fractions are fitted
by matching/tests/exceptions/fit_params.py on train golden booked before the cut and stored in exc_params.json."""
import json, os

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = {
    "grace_days": 3,                  # POLICIES s3 SHORT_DISCOUNT
    "bank_fee_cap_eur": 60.0,         # POLICIES s3 SHORT_BANK_CHARGES
    "bank_fee_cap_rel": 0.02,
    "fx_tolerance": 0.03,             # POLICIES s3 FX_DIFFERENCE
    "fx_slack": 0.005,                # last-available-rate / rounding slack on top of the 3 %
    "withholding_rates": {"MX": [0.0125, 0.10], "BR": [0.015, 0.10], "MA": [0.015, 0.10]},  # POLICIES: 1.25/1.5/10 %
    "partial_fractions": [[0.3, 1.0], [0.5, 1.0], [0.7, 1.0]],   # (deficit fraction of the item, weight)  -- refit overrides
    "dispute_fractions": [0.03, 0.05, 0.1, 0.2, 0.3],   # deficit fractions that occur for SHORT_DISPUTE (fitted)
    "dispute_min_items": 17,         # lines with >= this many invoices: a matched fraction is a dispute (remittance-style lines), fitted
}


def load(path=None):
    p = dict(DEFAULTS)
    f = path or os.path.join(_HERE, "exc_params.json")
    if os.path.exists(f):
        try:
            p.update(json.load(open(f)))
        except Exception:
            pass
    return p
