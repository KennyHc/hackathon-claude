"""Master files for a phase: FINAL ships updated masters in data/final/masters_final/ (new subsidiary ZA01, ZAR,
renamed customers, new IBANs, 2026 FX rates). Use this everywhere instead of hard-coding data/masters.

    from integration.lib.masters import masters_path, read_master, masters_dir
    customers = read_master("customers.csv", phase)      # data/masters ∪ data/<phase>/masters_final (final wins per key)
    d = masters_dir(phase)                               # a directory with every master file, merged the same way

We do not know whether masters_final holds full files or only the changes, so every file is the union of the base
file and the masters_final one, deduplicated by its key with the masters_final row winning (fx_rates: date+currency).
Files only in masters_final are taken as they are. Without masters_final everything is plain data/masters.
The merged directory is materialised once under integration/out/masters_merged/<phase>/ (rebuilt when a source
file is newer).
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("BANKREC_DATA", os.path.join(ROOT, "data"))
MERGED = os.path.join(ROOT, "integration", "out", "masters_merged")

KEYS = {"customers.csv": ["customer_id"], "suppliers.csv": ["supplier_id"], "bank_accounts.csv": ["account_id"],
        "entities.csv": ["entity"], "fx_rates.csv": ["date", "currency"], "payment_terms.csv": ["code"],
        "categories.csv": ["category"]}


def _final_dir(phase: str, data_dir: str) -> str:
    return os.path.join(data_dir, phase, "masters_final") if phase else ""


def _merge(name: str, base_dir: str, final_dir: str) -> pd.DataFrame:
    parts = []
    for d in (base_dir, final_dir):
        f = os.path.join(d, name)
        if d and os.path.exists(f):
            parts.append(pd.read_csv(f, dtype=str, keep_default_na=False))
    if not parts:
        raise FileNotFoundError(name)
    if len(parts) == 1:
        return parts[0]
    base, fin = parts
    df = pd.concat([base, fin], ignore_index=True, sort=False).fillna("")
    key = [k for k in KEYS.get(name, [df.columns[0]]) if k in df.columns] or [df.columns[0]]
    df = df.drop_duplicates(key, keep="last")
    cols = list(base.columns) + [c for c in fin.columns if c not in base.columns]
    return df[cols]


def masters_dir(phase: str = "", data_dir: str = DATA) -> str:
    """Directory with every master file for `phase` (data/masters when there is no masters_final)."""
    base = os.path.join(data_dir, "masters")
    fd = _final_dir(phase, data_dir)
    if not (fd and os.path.isdir(fd)):
        return base
    out = os.path.join(MERGED, phase)
    os.makedirs(out, exist_ok=True)
    names = sorted({n for d in (base, fd) for n in os.listdir(d) if n.endswith((".csv", ".csv.gz"))})
    for n in names:
        dst = os.path.join(out, n)
        srcs = [os.path.join(d, n) for d in (base, fd) if os.path.exists(os.path.join(d, n))]
        if os.path.exists(dst) and os.path.getmtime(dst) >= max(os.path.getmtime(s) for s in srcs):
            continue
        tmp = dst + f".tmp{os.getpid()}"
        _merge(n, base, fd).to_csv(tmp, index=False, compression="gzip" if n.endswith(".gz") else None)
        os.replace(tmp, dst)
    return out


def masters_path(name: str, phase: str = "", data_dir: str = DATA) -> str:
    """Path of master file `name` for `phase` (merged with masters_final when the phase has one)."""
    return os.path.join(masters_dir(phase, data_dir), name)


def read_master(name: str, phase: str = "", data_dir: str = DATA) -> pd.DataFrame:
    return pd.read_csv(masters_path(name, phase, data_dir), dtype=str, keep_default_na=False)
