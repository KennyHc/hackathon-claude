"""FX rates and currency conversion utilities for Grupo Valtermia.

Provides reference rates (ECB TARGET days and derived MAD basket) and
currency conversions via EUR, supporting both fast scalar lookups and
high-performance vectorized operations via merge_asof.

How to run tests:
    .venv/bin/python integration/lib/test_lib.py
"""

import os
from typing import Optional, Union

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MASTERS = os.path.join(ROOT, "data", "masters")

# Global caches keyed by masters_dir
_rates_cache: dict[str, pd.DataFrame] = {}
_by_ccy_cache: dict[str, dict[str, pd.DataFrame]] = {}
_daily_lookup_cache: dict[str, dict[str, dict[str, float]]] = {}


def load_rates(masters_dir: str = MASTERS) -> pd.DataFrame:
    """Load and cache exchange rates from masters directory.

    Reads fx_rates.csv, parses units_per_eur as float, indexes by date
    for fast as-of lookup, and builds lookup indices.

    Args:
        masters_dir: Directory containing fx_rates.csv.

    Returns:
        pd.DataFrame containing date, currency, units_per_eur, source.
    """
    if masters_dir in _rates_cache:
        return _rates_cache[masters_dir]

    csv_path = os.path.join(masters_dir, "fx_rates.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(masters_dir, "fx_rates.csv.gz")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df["units_per_eur"] = df["units_per_eur"].astype(float)
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt").reset_index(drop=True)

    # Index by DatetimeIndex (unnamed to avoid ambiguity during merge_asof)
    df.index = pd.DatetimeIndex(df["date_dt"])
    df.index.name = None

    _rates_cache[masters_dir] = df

    # Build per-currency sorted DataFrames for merge_asof
    by_ccy: dict[str, pd.DataFrame] = {}
    for c, grp in df.groupby("currency"):
        by_ccy[c] = (
            grp[["date_dt", "units_per_eur"]]
            .rename(columns={"date_dt": "date"})
            .sort_values("date")
            .reset_index(drop=True)
        )
    _by_ccy_cache[masters_dir] = by_ccy

    # Build dense daily lookup table for O(1) scalar lookups (2021 through 2026)
    piv = df.pivot(index="date", columns="currency", values="units_per_eur").astype(float)
    full_calendar = pd.date_range(piv.index.min(), "2026-12-31").strftime("%Y-%m-%d")
    piv_daily = piv.reindex(full_calendar).ffill().bfill()
    _daily_lookup_cache[masters_dir] = piv_daily.to_dict()

    return df


def rate(ccy: str, date: str, masters_dir: str = MASTERS) -> float:
    """Get units of currency per 1 EUR on date.

    EUR is defined as 1.0. For weekends, bank holidays, or missing dates,
    the last available earlier rate is used.

    Args:
        ccy: Currency code (e.g., 'USD', 'GBP', 'EUR').
        date: ISO date string ('YYYY-MM-DD') or timestamp.
        masters_dir: Masters directory path.

    Returns:
        Exchange rate in units of ccy per 1 EUR.
    """
    ccy_clean = str(ccy).strip().upper()
    if ccy_clean == "EUR":
        return 1.0

    date_str = str(date)[:10]

    if masters_dir not in _daily_lookup_cache:
        load_rates(masters_dir)

    lookup = _daily_lookup_cache[masters_dir]
    if ccy_clean not in lookup:
        raise ValueError(f"Unknown currency: {ccy}")

    c_dict = lookup[ccy_clean]
    if date_str in c_dict:
        return c_dict[date_str]

    # Fallback for dates outside precomputed calendar range
    by_ccy = _by_ccy_cache[masters_dir][ccy_clean]
    dt = pd.to_datetime(date_str)
    if dt < by_ccy["date"].iloc[0]:
        return float(by_ccy["units_per_eur"].iloc[0])

    idx = by_ccy["date"].searchsorted(dt, side="right") - 1
    return float(by_ccy["units_per_eur"].iloc[idx])


def to_eur(amount: float, ccy: str, date: str, masters_dir: str = MASTERS) -> float:
    """Convert amount in given currency to EUR at the rate on date."""
    ccy_clean = str(ccy).strip().upper()
    if ccy_clean == "EUR":
        return float(amount)
    r = rate(ccy_clean, date, masters_dir)
    return float(amount) / r


def convert(
    amount: float,
    from_ccy: str,
    to_ccy: str,
    date: str,
    masters_dir: str = MASTERS,
) -> float:
    """Convert amount from one currency to another via EUR at the rate on date."""
    fc = str(from_ccy).strip().upper()
    tc = str(to_ccy).strip().upper()
    if fc == tc:
        return float(amount)
    if fc == "EUR":
        return float(amount) * rate(tc, date, masters_dir)
    if tc == "EUR":
        return to_eur(amount, fc, date, masters_dir)
    return to_eur(amount, fc, date, masters_dir) * rate(tc, date, masters_dir)


def rel_diff(
    bank_amount: float,
    bank_ccy: str,
    item_amount: float,
    item_ccy: str,
    date: str,
    masters_dir: str = MASTERS,
) -> float:
    """Calculate relative difference between bank amount and item amount.

    Formula: (convert(bank_amount -> item_ccy) - item_amount) / item_amount.
    Negative indicates less cash than item amount.
    """
    bank_in_item_ccy = convert(float(bank_amount), bank_ccy, item_ccy, date, masters_dir)
    item_amt = float(item_amount)
    if item_amt == 0.0:
        return 0.0 if bank_in_item_ccy == 0.0 else float("inf")
    return (bank_in_item_ccy - item_amt) / item_amt


def is_fx(bank_ccy: str, item_ccy: str) -> bool:
    """Check if bank currency and item currency differ."""
    return str(bank_ccy).strip().upper() != str(item_ccy).strip().upper()


def rate_v(
    ccy: Union[pd.Series, str],
    date: Union[pd.Series, str],
    masters_dir: str = MASTERS,
) -> pd.Series:
    """Vectorized rate lookup returning Series of units per 1 EUR on date.

    Uses per-currency merge_asof with backward direction for high performance.
    Handles ~1M rows in under a second.
    """
    if masters_dir not in _by_ccy_cache:
        load_rates(masters_dir)
    by_ccy = _by_ccy_cache[masters_dir]

    if isinstance(ccy, str) and isinstance(date, pd.Series):
        ccy = pd.Series(ccy, index=date.index)
    elif isinstance(date, str) and isinstance(ccy, pd.Series):
        date = pd.Series(date, index=ccy.index)
    elif not isinstance(ccy, pd.Series) and not isinstance(date, pd.Series):
        return pd.Series([rate(str(ccy), str(date), masters_dir)])

    ccy_s = ccy.astype(str).str.strip().str.upper()
    res = pd.Series(index=ccy_s.index, dtype=float)

    eur_mask = (ccy_s == "EUR")
    res.loc[eur_mask] = 1.0

    non_eur = ~eur_mask
    if not non_eur.any():
        return res

    date_sub = date[non_eur]
    if not pd.api.types.is_datetime64_any_dtype(date_sub):
        date_sub = pd.to_datetime(date_sub)
    ccy_sub = ccy_s[non_eur]

    for c in ccy_sub.unique():
        if c not in by_ccy:
            raise ValueError(f"Unknown currency: {c}")
        m = (ccy_sub == c)
        sub_indices = ccy_sub.index[m]
        sub = pd.DataFrame({"_idx": sub_indices, "date": date_sub[m]}).sort_values("date")
        merged = pd.merge_asof(sub, by_ccy[c], on="date", direction="backward")
        # bfill in case any date is strictly before the first available rate
        units = merged["units_per_eur"].bfill()
        res.loc[merged["_idx"]] = units.values

    return res


def convert_v(
    amount: Union[pd.Series, float, int],
    from_ccy: Union[pd.Series, str],
    to_ccy: Union[pd.Series, str],
    date: Union[pd.Series, str],
    masters_dir: str = MASTERS,
) -> pd.Series:
    """Vectorized currency conversion via EUR.

    Converts amounts from from_ccy to to_ccy as of date.
    Fast-paths identical currencies without rate lookup.
    """
    if isinstance(amount, pd.Series):
        ref_idx = amount.index
    elif isinstance(from_ccy, pd.Series):
        ref_idx = from_ccy.index
    elif isinstance(to_ccy, pd.Series):
        ref_idx = to_ccy.index
    elif isinstance(date, pd.Series):
        ref_idx = date.index
    else:
        return pd.Series([convert(float(amount), str(from_ccy), str(to_ccy), str(date), masters_dir)])

    amt_s = amount.astype(float) if isinstance(amount, pd.Series) else pd.Series(amount, index=ref_idx, dtype=float)
    fc_s = (
        from_ccy.astype(str).str.strip().str.upper()
        if isinstance(from_ccy, pd.Series)
        else pd.Series(from_ccy, index=ref_idx).astype(str).str.strip().str.upper()
    )
    tc_s = (
        to_ccy.astype(str).str.strip().str.upper()
        if isinstance(to_ccy, pd.Series)
        else pd.Series(to_ccy, index=ref_idx).astype(str).str.strip().str.upper()
    )
    dt_s = date if isinstance(date, pd.Series) else pd.Series(date, index=ref_idx)

    res = amt_s.copy()
    diff_mask = (fc_s != tc_s)
    if not diff_mask.any():
        return res

    r_from = rate_v(fc_s[diff_mask], dt_s[diff_mask], masters_dir)
    r_to = rate_v(tc_s[diff_mask], dt_s[diff_mask], masters_dir)
    res.loc[diff_mask] = amt_s[diff_mask] * (r_to / r_from)

    return res


def to_eur_v(
    amount: Union[pd.Series, float, int],
    ccy: Union[pd.Series, str],
    date: Union[pd.Series, str],
    masters_dir: str = MASTERS,
) -> pd.Series:
    """Vectorized conversion to EUR helper."""
    return convert_v(amount, ccy, "EUR", date, masters_dir)


def rel_diff_v(
    bank_amount: Union[pd.Series, float, int],
    bank_ccy: Union[pd.Series, str],
    item_amount: Union[pd.Series, float, int],
    item_ccy: Union[pd.Series, str],
    date: Union[pd.Series, str],
    masters_dir: str = MASTERS,
) -> pd.Series:
    """Vectorized relative difference helper."""
    conv_bank = convert_v(bank_amount, bank_ccy, item_ccy, date, masters_dir)
    item_amt = item_amount if isinstance(item_amount, pd.Series) else pd.Series(item_amount, index=conv_bank.index, dtype=float)
    return (conv_bank - item_amt) / item_amt.replace(0.0, np.nan)
