"""Matching entry point (piece B): python matching/run.py --phase val [--min-conf 0.0] [--no-exceptions]

Runs the matchers in priority order (most certain evidence first). A bank line claimed by an earlier matcher is never given to a
later one. Writes matching/out/<phase>/allocations.parquet with the CONTRACTS section B columns and, when the phase has a golden
(sample/train/val), prints the per-segment report.

Priority = how deterministic the evidence is: batch files > payment-run ids > remittance advices > explicit references.
A matcher that is missing or crashes is skipped (and counted): the pipeline must always produce a valid file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import os
import sys
import time
import traceback

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from matching.lib import data, evalutil  # noqa: E402

# matcher modules under matching/matchers/, in priority order (a missing module is skipped)
ORDER = ["batches", "ap", "remittance", "pdf_cases", "refs", "shortpay", "duplicates"]
COLS = ["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"]
# Per-matcher confidence gate. Structural matchers (batches, ap, remittance) are kept whole: their low-confidence methods
# still measured >= 98% precise. refs' weak tiers (risky subset sums, withholding/charges guesses) sit below 0.65.
GATES = {"refs": 0.65}
# Methods kept despite the gate: 93-97% pair precise on val, and an unclaimed line is wrong anyway (it becomes
# UNIDENTIFIED / no match), so keeping them gains ~1,300 correct lines for ~80 wrong ones on val.
GATE_EXEMPT = {"ref_fill", "ref_half", "party_discount_late"}
CACHE_VERSION = "1"  # bump to invalidate every cached matcher output


def _p_items(phase: str):
    p = os.path.join(ROOT, "classification", "out", phase, "line_labels.parquet")
    if not os.path.exists(p):
        return None
    lab = pd.read_parquet(p, columns=["bank_txn_id", "p_items"])
    return lab.set_index("bank_txn_id").p_items


def _save_cache(df: pd.DataFrame, cpath: str, kpath: str, key: str) -> None:
    """Write output then key (key last: a half-written cache never matches). Mixed-type object columns go as strings."""
    try:
        d = df.reset_index(drop=True).copy()
        for c in ("bank_txn_id", "item_id", "exception", "method"):
            d[c] = d[c].astype("string")
        d["amount"] = pd.to_numeric(d.amount, errors="coerce").astype(float)
        d["confidence"] = pd.to_numeric(d.confidence, errors="coerce").astype(float)
        if os.path.exists(kpath):
            os.remove(kpath)
        d.to_parquet(cpath, index=False)
        with open(kpath, "w") as fh:
            fh.write(key)
    except Exception:
        pass  # caching is an optimisation only


def _tree_stat(path: str) -> str:
    """Names, sizes and mtimes of every file under path (cheap: no reads)."""
    h = hashlib.sha1()
    for dp, dns, fns in os.walk(path):
        dns.sort()
        for f in sorted(fns):
            try:
                st = os.stat(os.path.join(dp, f))
            except OSError:
                continue
            h.update(f"{dp}/{f}:{st.st_size}:{st.st_mtime_ns};".encode())
    return h.hexdigest()


def _src_hash(*paths: str) -> str:
    h = hashlib.sha1()
    for p in paths:
        files = [p] if os.path.isfile(p) else sorted(
            os.path.join(dp, f) for dp, _, fns in os.walk(p) for f in fns if f.endswith(".py"))
        for f in files:
            with open(f, "rb") as fh:
                h.update(f.encode() + b"\0" + fh.read())
    return h.hexdigest()


def _base_key(phase: str, min_conf: float) -> str:
    """What every matcher's output depends on besides its own code: shared libs, data, classifier labels, gates."""
    lab = os.path.join(ROOT, "classification", "out", phase, "line_labels.parquet")
    parts = [
        CACHE_VERSION, phase, repr(min_conf), repr(sorted(GATES.items())), repr(sorted(GATE_EXEMPT)),
        # matchers importing other matchers (shortpay -> remittance) are covered by the chain: imports point upstream only
        _src_hash(os.path.join(ROOT, "matching", "lib"), os.path.join(ROOT, "integration", "lib")),
        _tree_stat(os.path.join(ROOT, "data", phase)),
        _tree_stat(os.path.join(ROOT, "data", "train")),
        str(os.stat(lab).st_mtime_ns) if os.path.exists(lab) else "no-labels",
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def run(phase: str, min_conf: float = 0.0, only: list[str] | None = None, exceptions: bool = False, quiet: bool = False,
        cache: bool = True) -> pd.DataFrame:
    """cache: reuse a matcher's previous output when its code, every earlier matcher in the chain, the shared libs, the data
    and the classifier labels are all unchanged (matching/out/<phase>/cache/). Any change re-runs that matcher and all later."""
    t0 = time.time()
    claimed: set[str] = set()
    parts, stats = [], []
    cdir = os.path.join(ROOT, "matching", "out", phase, "cache")
    os.makedirs(cdir, exist_ok=True)
    chain = _base_key(phase, min_conf)
    for name in (only or ORDER):
        src = os.path.join(ROOT, "matching", "matchers", f"{name}.py")
        chain = hashlib.sha1((chain + name + (_src_hash(src) if os.path.exists(src) else "missing")).encode()).hexdigest()
        cpath, kpath = os.path.join(cdir, f"{name}.parquet"), os.path.join(cdir, f"{name}.key")
        if cache and os.path.exists(cpath) and os.path.exists(kpath):
            with open(kpath) as fh:
                if fh.read().strip() == chain:
                    df = pd.read_parquet(cpath)
                    for c in ("bank_txn_id", "item_id", "exception", "method"):  # back to object/None, as a fresh run gives
                        df[c] = df[c].astype(object).where(df[c].notna(), None)
                    claimed |= set(df.bank_txn_id)
                    if len(df):
                        parts.append(df[COLS])
                    stats.append((name, "cached", df.bank_txn_id.nunique(), len(df), "0s"))
                    continue
        try:
            mod = importlib.import_module(f"matching.matchers.{name}")
        except ImportError as e:
            stats.append((name, "missing", 0, 0, f"{e}"))
            continue
        try:
            t1 = time.time()
            params = inspect.signature(mod.match).parameters
            kw = {}
            if "prior" in params:      # items already settled by earlier matchers, so a later one does not reuse them
                kw["prior"] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
            if "p_items" in params:
                kw["p_items"] = _p_items(phase)
            df = mod.match(phase, set(claimed), **kw)
            if df is None or df.empty:
                _save_cache(pd.DataFrame(columns=COLS), cpath, kpath, chain)
                stats.append((name, "ok", 0, 0, f"{time.time() - t1:.0f}s"))
                continue
            for c in COLS:
                if c not in df:
                    df[c] = None
            df = df[~df.bank_txn_id.isin(claimed)]
            gate = max(min_conf, GATES.get(name, 0.0))
            if gate > 0:
                df = df[(df.confidence.fillna(1.0) >= gate) | df.method.isin(GATE_EXEMPT)]
            claimed |= set(df.bank_txn_id)
            parts.append(df[COLS])
            _save_cache(df[COLS], cpath, kpath, chain)
            stats.append((name, "ok", df.bank_txn_id.nunique(), len(df), f"{time.time() - t1:.0f}s"))
        except Exception:  # never let one matcher kill the run
            stats.append((name, "CRASHED", 0, 0, traceback.format_exc().strip().splitlines()[-1]))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
    if exceptions and len(out):
        try:
            from matching.matchers import exception_rules
            out = exception_rules.assign(out, phase)[COLS]
        except ImportError:
            pass
        except Exception:
            stats.append(("exception_rules", "CRASHED", 0, 0, traceback.format_exc().strip().splitlines()[-1]))
    out["amount"] = out.amount.astype(float).round(2)
    out = out.drop_duplicates(["bank_txn_id", "item_id"])
    p = os.path.join(ROOT, "matching", "out", phase)
    os.makedirs(p, exist_ok=True)
    out.to_parquet(os.path.join(p, "allocations.parquet"), index=False)
    if not quiet:
        print(f"[{time.time() - t0:.0f}s] allocations: {out.bank_txn_id.nunique()} lines / {len(out)} rows -> {p}")
        for s in stats:
            print("  %-16s %-8s lines=%-7s rows=%-8s %s" % s)
        if phase in ("sample", "train", "train_lite", "val") and len(out):
            evalutil.report(out, phase, name="matching (all)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="val")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--exceptions", action="store_true", help="also apply matching.matchers.exception_rules (integration already applies classification/exceptions.py)")
    ap.add_argument("--no-cache", action="store_true", help="re-run every matcher (default: reuse unchanged ones, see run())")
    a = ap.parse_args()
    run(a.phase, a.min_conf, a.only, a.exceptions, cache=not a.no_cache)
