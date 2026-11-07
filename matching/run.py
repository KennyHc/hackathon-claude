#!/usr/bin/env python3
"""B. Matching cascade: one chronological pass of the stages over the phase's bank lines and items.

    .venv/bin/python matching/run.py --phase val                  # all stages present: batch → refs → party
    .venv/bin/python matching/run.py --phase val --stages refs     # only some
    .venv/bin/python matching/run.py --phase val --fallback-files  # a stage that is missing/crashes → its own
                                                                   #   matching/out/P/alloc_<stage>.parquet if present

Each stage is `matching/<stage>/matcher.py: match(phase, lines, items, claimed, **ctx) -> DataFrame` (docs/PLAN_LUIGI.md).
A missing module, an import error, an exception or a timeout in a stage is logged and the stage is skipped.
The cascade enforces, after each stage:
  1. a line is claimed by at most one stage (rows for lines claimed earlier, or not in the phase, are dropped);
  2. rows on unknown items are dropped (items of train/live/final closed before the phase are kept as given:
     returns, duplicates, re-presentations); duplicate (line, item) rows are summed;
  3. an item is never allocated beyond its open balance: rows are walked in booking-date order against the balance
     left by the opening balance and every row kept so far (RETURNED re-opens, DUPLICATE_PAYMENT has no effect);
     an amount above the balance is trimmed to it; a row on an item with no balance left is dropped
     (--overclaim dup: kept as DUPLICATE_PAYMENT when its amount ≈ the item total instead). A line that loses all
     its rows is released for the next stages.
Writes matching/out/P/allocations.parquet (CONTRACTS §B), cascade_<stage>.parquet (rows kept per stage) and
summary.csv (per-stage counts and timing).
"""
from __future__ import annotations

import argparse
import importlib
import os
import signal
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from matching.common import (ALLOC_COLS, BUILTIN_STAGES, DATA, NO_BALANCE, REOPENS, Balances, all_item_ids, load_items,  # noqa: E402
                             load_line_labels, load_lines, masters_dir, normalize_stage_output, out_dir, tol)

STAGES = ["batch", "refs", "party", "k4_sbc"]
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


class StageTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise StageTimeout()


def run_stage(stage, phase, lines_left, items, claimed, ctx, timeout, fallback_files, pkg="matching"):
    """Returns (DataFrame or None, status)."""
    t = time.time()
    if stage in BUILTIN_STAGES:  # residual stages owned by the cascade (matching/common.py)
        try:
            return BUILTIN_STAGES[stage](phase, lines_left.copy(), items.copy(), claimed.copy(), **ctx), \
                f"ok {time.time() - t:.1f}s"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return None, f"error: {type(e).__name__}: {e}"
    try:
        mod = importlib.import_module(f"{pkg}.{stage}.matcher")
    except ModuleNotFoundError as e:
        status = f"missing ({e.name})"
        mod = None
    except Exception as e:  # noqa: BLE001
        status = f"import error: {type(e).__name__}: {e}"
        traceback.print_exc()
        mod = None
    if mod is not None and not hasattr(mod, "match"):
        status, mod = "no match()", None
    if mod is not None:
        old = signal.signal(signal.SIGALRM, _alarm) if timeout else None
        if timeout:
            signal.alarm(int(timeout))
        try:
            df = mod.match(phase, lines_left.copy(), items.copy(), claimed.copy(), **ctx)
            return df, f"ok {time.time() - t:.1f}s"
        except StageTimeout:
            status = f"timeout after {timeout}s"
        except Exception as e:  # noqa: BLE001
            status = f"error: {type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            if timeout:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
    if fallback_files:
        f = os.path.join(out_dir(phase), f"alloc_{stage}.parquet")
        if os.path.exists(f):
            return pd.read_parquet(f), status + f"; using {os.path.relpath(f, ROOT)}"
    return None, status


def enforce(new, stage, lines_left_ids, booking, items_total, bal, overclaim="drop", universe=frozenset(),
            conflict="rows"):
    """Apply cascade rules 1-3 to one stage's normalized output. Mutates `bal`. Returns (kept rows, counts)."""
    c = {"rows_in": len(new), "lines_in": new.bank_txn_id.nunique()}
    m = new.bank_txn_id.isin(lines_left_ids)
    c["drop_claimed_line"] = int((~m).sum())
    new = new[m]
    known = new.item_id.isin(bal.open.keys())
    m = known | new.item_id.isin(universe)  # items closed before the phase (returns, duplicates, re-presentations)
    c["drop_unknown_item"] = int((~m).sum())
    c["closed_item_rows"] = int((m & ~known).sum())
    new = new[m]
    n = len(new)
    if n:
        new = (new.groupby(["bank_txn_id", "item_id"], sort=False, as_index=False)
               .agg(amount=("amount", "sum"), exception=("exception", "first"), confidence=("confidence", "first"),
                    method=("method", "first")))
    c["merged_dup_rows"] = n - len(new)
    new = new.assign(booking_date=new.bank_txn_id.map(booking)).sort_values(["booking_date", "bank_txn_id"],
                                                                            kind="stable")
    keep = np.ones(len(new), dtype=bool)
    amounts = new.amount.values.astype(float).copy()
    excs = new.exception.astype(object).values.copy()
    trimmed = dropped = to_dup = 0
    for i, (it, amt, ex) in enumerate(zip(new.item_id.values, amounts, excs)):
        if ex in REOPENS:
            bal.give(it, amt)
            continue
        if ex in NO_BALANCE or it not in bal.open:  # closed before the phase: balance unknown, kept as given
            continue
        avail = bal.get(it)
        if avail <= 0.01:
            if overclaim == "dup" and abs(amt - items_total.get(it, -1e9)) <= tol(amt):
                excs[i] = "DUPLICATE_PAYMENT"
                to_dup += 1
            else:
                keep[i] = False
                dropped += 1
            continue
        if amt > avail + 0.005:
            amounts[i] = round(avail, 2)
            trimmed += 1
        bal.take(it, amounts[i])
    new = new.assign(amount=amounts, exception=excs)
    if conflict == "line":  # a line with any dropped row loses all its rows (balances given back)
        bad = set(new.bank_txn_id.values[~keep])
        for it, amt, ex, tid, k in zip(new.item_id.values, amounts, excs, new.bank_txn_id.values, keep):
            if tid in bad and k and it in bal.open and ex not in NO_BALANCE:
                bal.take(it, amt) if ex in REOPENS else bal.give(it, amt)
        keep &= ~new.bank_txn_id.isin(bad).values
        c["drop_conflict_lines"] = len(bad)
    new = new[keep]
    c.update(trimmed=trimmed, drop_overalloc=dropped, to_duplicate=to_dup, rows_kept=len(new),
             lines_kept=new.bank_txn_id.nunique())
    new = new.assign(stage=stage)
    return new, c


def load_pdf_clauses(phase):
    """PDF clauses (instalments, retentions, replacements…) for the refs stage: matching/out/P/pdf_clauses.parquet,
    built with matching.refs.pdf_clauses when missing (as matching/refs/run.py does). None when unavailable."""
    pp = os.path.join(out_dir(phase), "pdf_clauses.parquet")
    try:
        if not os.path.exists(pp) and os.path.exists(os.path.join(DATA, phase, "pdfs", "index.csv")):
            t = time.time()
            from matching.refs import pdf_clauses
            extra = {"val": ["train"], "live": ["train"], "final": ["train", "live"]}.get(phase)
            cwd = os.getcwd()
            os.chdir(ROOT)
            try:
                pdf_clauses.load(phase, extra_phases=extra)
            finally:
                os.chdir(cwd)
            log(f"pdf clauses built in {time.time() - t:.0f}s")
        if os.path.exists(pp):
            return pd.read_parquet(pp)
    except Exception as e:  # noqa: BLE001  (PDFs are a bonus: never block the cascade)
        traceback.print_exc()
        log(f"pdf clauses unavailable: {e}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--stages", default=",".join(STAGES), help="comma list, in cascade order")
    ap.add_argument("--pkg", default="matching", help="package holding the stage folders (tests use a temp one)")
    ap.add_argument("--out", default=None, help="output dir (default matching/out/<phase>)")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per stage (0 = none)")
    ap.add_argument("--fallback-files", action="store_true",
                    help="stage missing/failing → read its own matching/out/P/alloc_<stage>.parquet")
    ap.add_argument("--overclaim", choices=["drop", "dup"], default="drop",
                    help="row on an item with no balance left: drop it, or keep it as DUPLICATE_PAYMENT when its "
                         "amount ≈ the item total")
    ap.add_argument("--conflict", choices=["rows", "line"], default="rows",
                    help="rows on items with no balance left: drop just those rows, or the whole line")
    a = ap.parse_args()
    od = a.out or out_dir(a.phase)
    os.makedirs(od, exist_ok=True)

    lines = load_lines(a.phase)
    items = load_items(a.phase)
    labels = load_line_labels(a.phase)
    log(f"{a.phase}: {len(lines):,} lines, {len(items):,} items, line_labels "
        f"{'none' if labels is None else f'{len(labels):,}'} (data {os.path.relpath(DATA, ROOT)})")
    pdf = load_pdf_clauses(a.phase)
    log(f"pdf clauses: {'none' if pdf is None else f'{len(pdf):,}'}")
    ctx = {"line_labels": labels, "masters_dir": masters_dir(a.phase), "data_dir": DATA, "pdf_clauses": pdf}
    booking = dict(zip(lines.bank_txn_id, lines.booking_date))
    items_total = dict(zip(items.item_id, pd.to_numeric(items.total_amount, errors="coerce")))
    line_cols = [c for c in lines.columns if c != "amount_f"]
    universe = all_item_ids(a.phase) | frozenset(items.item_id)
    bal = Balances(items)

    claimed = pd.DataFrame({c: pd.Series(dtype=object) for c in ALLOC_COLS + ["booking_date", "stage"]})
    claimed["amount"] = claimed["amount"].astype(float)
    summary = []
    for stage in [s.strip() for s in a.stages.split(",") if s.strip()]:
        taken = set(claimed.bank_txn_id)
        left = lines[~lines.bank_txn_id.isin(taken)]
        log(f"stage {stage}: {len(left):,} lines left")
        raw, status = run_stage(stage, a.phase, left[line_cols], items, claimed, ctx, a.timeout,
                                a.fallback_files, a.pkg)
        log(f"stage {stage}: {status}")
        row = {"stage": stage, "status": status, "lines_offered": len(left)}
        if raw is not None:
            try:
                new = normalize_stage_output(raw, stage)
                kept, cnt = enforce(new, stage, set(left.bank_txn_id), booking, items_total, bal, a.overclaim,
                                     universe, a.conflict)
                row.update(cnt)
                log(f"stage {stage}: " + ", ".join(f"{k} {v:,}" for k, v in cnt.items()))
                kept.to_parquet(os.path.join(od, f"cascade_{stage}.parquet"), index=False)
                claimed = pd.concat([claimed, kept[claimed.columns]], ignore_index=True) if len(claimed) else \
                    kept[claimed.columns].reset_index(drop=True)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                row["status"] = status + f"; rejected: {type(e).__name__}: {e}"
                log(f"stage {stage}: output rejected ({e})")
        summary.append(row)

    out = claimed[ALLOC_COLS].copy()
    out["amount"] = out["amount"].astype(float).round(2)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").astype(float)
    out["exception"] = out["exception"].astype(object).where(out["exception"].notna(), None)
    out.to_parquet(os.path.join(od, "allocations.parquet"), index=False)
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(od, "summary.csv"), index=False)
    log(f"wrote {len(out):,} rows on {out.bank_txn_id.nunique():,} lines "
        f"({out.bank_txn_id.nunique() / max(len(lines), 1):.1%} of the phase) → {os.path.relpath(od, ROOT)}/allocations.parquet")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
