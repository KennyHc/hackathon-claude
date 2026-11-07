"""Drift alarm: compare a phase's pipeline output with LIVE's before submitting (no labels needed).

    python integration/drift.py --phase final            # vs live
    python integration/drift.py --phase final --ref val

For each view it prints the share of bank lines per group in the reference and in the phase, and flags a group whose
share moved more than --tol (relative, default 30 %) or that appears / disappears:
  - matcher family that claimed the line (from integration/out/<phase>/methods.parquet)
  - entity: share of its lines claimed by a matcher (a new company with ~0 % claimed = matchers don't understand it)
  - currency: share of lines claimed
  - exception code share among submission rows, category share among no-item rows
Exit code 1 if anything is flagged (the runbook prints it; submitting is still the user's call).
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ingestion import lib  # noqa: E402


def _load(phase: str):
    od = lib.out_dir("integration", phase)
    bank = pd.read_parquet(os.path.join(ROOT, "ingestion", "out", phase, "bank_lines.parquet"))
    meth = pd.read_parquet(os.path.join(od, "methods.parquet")).set_index("bank_txn_id").method
    sub = pd.read_csv(os.path.join(od, "submission.csv.gz"), dtype=str, keep_default_na=False)
    fam = meth.str.split(":").str[0].str.replace(r"^(cards_cheques|dd_batches|remittance|duplicates)_.*", r"\1", regex=True)
    bank = bank.assign(family=bank.bank_txn_id.map(fam).fillna("?"))
    bank["claimed"] = bank.family != "cls"
    return bank, sub


def _compare(name: str, ref: pd.Series, cur: pd.Series, tol: float, min_share: float = 0.002) -> list[str]:
    t = pd.DataFrame({"ref": ref, "cur": cur}).fillna(0.0)
    t = t[(t.ref >= min_share) | (t.cur >= min_share)]
    t["rel"] = (t.cur - t.ref) / t.ref.where(t.ref > 0)
    flags = []
    print(f"\n--- {name} ---")
    for k, r in t.sort_values("ref", ascending=False).iterrows():
        bad = r.ref == 0 or r.cur == 0 or abs(r.rel) > tol
        mark = "  <-- CHECK" if bad else ""
        rel = ("new" if k not in ref.index else "was 0") if r.ref == 0 else ("gone" if r.cur == 0 else f"{r.rel:+.0%}")
        print(f"{str(k):32s} ref {r.ref:7.2%}   now {r.cur:7.2%}   {rel:>6s}{mark}")
        if bad:
            flags.append(f"{name}: {k} ({rel})")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="final", choices=list(lib.PHASES))
    ap.add_argument("--ref", default="live", choices=list(lib.PHASES))
    ap.add_argument("--tol", type=float, default=0.30)
    a = ap.parse_args()
    rb, rs = _load(a.ref)
    cb, cs = _load(a.phase)
    print(f"[drift] {a.phase}: {len(cb):,} lines vs {a.ref}: {len(rb):,} lines")
    flags = []
    flags += _compare("matcher share of lines", rb.family.value_counts(normalize=True), cb.family.value_counts(normalize=True), a.tol)
    flags += _compare("claimed share by entity", rb.groupby("entity").claimed.mean(), cb.groupby("entity").claimed.mean(), a.tol, 0.0)
    flags += _compare("claimed share by currency", rb.groupby("currency").claimed.mean(), cb.groupby("currency").claimed.mean(), a.tol, 0.0)
    ex = lambda s: s.exception.replace("", "(none)").value_counts(normalize=True)
    flags += _compare("exception share of rows", ex(rs), ex(cs), a.tol, 0.0005)
    cat = lambda s: s[s.item_id == ""].category.value_counts(normalize=True)
    flags += _compare("category share of no-item rows", cat(rs), cat(cs), a.tol)
    # entities that are new OR nearly absent from the reference (ZA01 has 2 LIVE lines, so it is not "new")
    for e, n in cb.entity.value_counts().items():
        nr = int((rb.entity == e).sum())
        if nr < 100:
            print(f"[drift] small/new entity {e}: {n:,} lines now ({nr} in {a.ref}), {cb[cb.entity == e].claimed.mean():.1%} claimed")
    rl = os.path.join(ROOT, "matching", "out", a.phase, "remittance_lines.parquet")
    if os.path.exists(rl):   # advice parse coverage per format (a new layout parses to advices without lines)
        r = pd.read_parquet(rl, columns=["file", "format", "line_no"]).groupby("file").agg(f=("format", "first"), n=("line_no", "max"))
        cov = (r.n >= 0).groupby(r.f).mean()
        print("\n--- remittance advices with parsed lines, by format ---\n" + cov.map("{:.1%}".format).to_string())
        flags += [f"remittance: {k} advices only {v:.0%} parsed" for k, v in cov.items() if v < 0.9]
    new_ent = sorted(set(cb.entity) - set(rb.entity))
    if new_ent:
        for e in new_ent:
            sh = cb[cb.entity == e].claimed.mean()
            print(f"\n[drift] new entity {e}: {int((cb.entity == e).sum()):,} lines, {sh:.1%} claimed by matchers")
    print(f"\n[drift] {len(flags)} flag(s)" + ("".join(f"\n  - {f}" for f in flags) if flags else ": none"))
    sys.exit(1 if flags else 0)


if __name__ == "__main__":
    main()
