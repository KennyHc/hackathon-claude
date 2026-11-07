"""Same amount, several open invoices of the payer (fixed monthly fees): the month named in the text ("SERVICE MAY 2025",
"MANT. MAYO 25", "SERVICIOS MAYO", ...) -> the invoice of that service month (issued at the start of the following
month); otherwise the OLDEST open one (FIFO).  POLICIES §5.6.

The decision is made inside the chronological pass of matchers/subset_sum.py (shared item consumption); this module
returns that pass's same-amount lines.
"""
from __future__ import annotations

from matching.common import Ctx, alloc_frame
from matching.matchers.subset_sum import engine

NAME = "same_amount"


def run(ctx: Ctx):
    df = alloc_frame(engine(ctx).rows[NAME])
    return df
