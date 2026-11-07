"""Master files for a phase: FINAL ships updated masters in data/final/masters_final/ (new subsidiary ZA01, ZAR,
renamed customers, new IBANs, 2026 FX rates). Use this everywhere instead of hard-coding data/masters.

    from integration.lib.masters import masters_path, read_master
    customers = read_master("customers.csv", phase)      # final → masters_final version if present, else data/masters
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")


def masters_path(name: str, phase: str = "", data_dir: str = DATA) -> str:
    """Path of master file `name` for `phase`: data/<phase>/masters_final/<name> if it exists, else data/masters/<name>."""
    if phase:
        f = os.path.join(data_dir, phase, "masters_final", name)
        if os.path.exists(f):
            return f
    return os.path.join(data_dir, "masters", name)


def masters_dir(phase: str = "", data_dir: str = DATA) -> str:
    d = os.path.join(data_dir, phase, "masters_final")
    return d if phase and os.path.isdir(d) else os.path.join(data_dir, "masters")


def read_master(name: str, phase: str = "", data_dir: str = DATA) -> pd.DataFrame:
    return pd.read_csv(masters_path(name, phase, data_dir), dtype=str, keep_default_na=False)
