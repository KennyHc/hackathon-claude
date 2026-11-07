"""Exception rules library (POLICIES §3). v0: keep the matcher's exception. Replaced by issue #22/#23.

    assign_exceptions(allocations, bank_lines, open_items, **ctx) -> pd.Series aligned with allocations.index
    (None = keep the matcher's exception)
"""
from __future__ import annotations

import pandas as pd


def assign_exceptions(allocations: pd.DataFrame, bank_lines: pd.DataFrame, open_items: pd.DataFrame, **ctx) -> pd.Series:
    return pd.Series([None] * len(allocations), index=allocations.index, dtype=object)
