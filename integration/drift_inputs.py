"""Input-side drift alarm (complements integration/drift.py, which only looks at the pipeline OUTPUT). No labels needed.

    .venv/bin/python integration/drift_inputs.py --phase final            # vs live
    .venv/bin/python integration/drift_inputs.py --phase live --ref val   # self-test

Compares the phase with a reference phase on what can fail SILENTLY before matching:
  1. bank statements  : lines per (bank, format); per entity the share of lines with empty text / counterparty / payer
                        name / document tokens (a new bank or format parsed badly -> matchers see nothing)
  2. remittance advices: per layout the advices with zero parsed rows or a parse error, rows per advice, rows without a
                        document or paid amount, advices whose rows add up to their stated total
  3. PDF text         : share of PDFs (documents, advices) whose extracted text is empty
  4. invoice numbers  : per entity the main document-number shape, and the share of receipts whose bank text cites a
                        document number of that entity (exact token hit) -> new numbering the refs matcher can't see
  5. FX coverage      : per currency used in the phase, the last FX date vs the last booking date, and missing days
  6. classifier       : per entity p_items spread (share < 0.1, share in the grey zone 0.1-0.97) and UNIDENTIFIED share
  7. money claimed    : per entity the share of |amount_eur| on lines that got item rows
A row is FLAGGED when a share moves by more than --abs (default 0.05) AND more than --rel (default 30 %) relative, or
when a group is new / gone. Exit code 1 if anything is flagged; submitting stays the human's call.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ingestion import lib  # noqa: E402

FLAGS: list[str] = []


def _p(*a):
    print(*a, flush=True)


def _rd(path):
    return pd.read_parquet(path) if os.path.exists(path) else None


def _compare(title: str, ref: pd.DataFrame, cur: pd.DataFrame, key: list[str], cols: list[str], a: argparse.Namespace,
             new_ok: bool = False):
    """ref/cur: one row per group (key) with share columns (0..1). Prints a table and records flags."""
    _p(f"\n--- {title}")
    if ref is None or cur is None or ref.empty and cur.empty:
        _p("  (no data)")
        return
    m = ref.merge(cur, on=key, how="outer", suffixes=("_ref", "_cur"), indicator=True)
    rows = []
    for _, r in m.iterrows():
        g = " / ".join(str(r[k]) for k in key)
        if r["_merge"] == "right_only":
            note = "NEW" + ("" if new_ok else " (FLAG)")
            if not new_ok:
                FLAGS.append(f"{title}: new group {g}")
        elif r["_merge"] == "left_only":
            note = "GONE (FLAG)"
            FLAGS.append(f"{title}: group gone {g}")
        else:
            note = ""
        parts = []
        for c in cols:
            x, y = r.get(f"{c}_ref"), r.get(f"{c}_cur")
            parts.append(f"{c} {'' if pd.isna(x) else f'{x:.3f}'}→{'' if pd.isna(y) else f'{y:.3f}'}")
            if r["_merge"] == "both" and pd.notna(x) and pd.notna(y):
                d = abs(y - x)
                if d > a.abs and d > a.rel * max(abs(x), 1e-9):
                    note += f" FLAG {c}"
                    FLAGS.append(f"{title}: {g} {c} {x:.3f} -> {y:.3f}")
            elif r["_merge"] == "right_only" and pd.notna(y) and c in ("empty_text", "empty_doc", "zero_rows", "err", "pdf_empty") \
                    and y > 0.2:
                note += f" FLAG {c}"
                FLAGS.append(f"{title}: new group {g} has {c} = {y:.3f}")
        rows.append(f"  {g:28} " + " | ".join(parts) + (f"   <-- {note.strip()}" if note.strip() else ""))
    for r in rows:
        _p(r)


# ------------------------------------------------------------------------------------------------------------ views
def bank_view(bank: pd.DataFrame):
    b = bank.copy()
    empty = lambda s: s.fillna("").astype(str).str.strip().eq("")  # noqa: E731
    b["empty_text"] = empty(b.text) & empty(b.remittance_info)
    b["empty_cpty"] = empty(b.counterparty_name)
    b["empty_payer"] = empty(b.party_name_norm)
    b["empty_doc"] = b.doc_tokens.map(lambda x: x is None or len(x) == 0)
    fmt = b.groupby(["bank", "statement_format"]).size().div(len(b)).rename("share").reset_index()
    ent = b.groupby("entity")[["empty_text", "empty_cpty", "empty_payer", "empty_doc"]].mean().reset_index()
    return fmt, ent


def advice_view(rl: pd.DataFrame | None):
    if rl is None or rl.empty:
        return None
    rl = rl.copy()
    adv = rl.groupby("file").agg(layout=("layout", "first"), fmt=("format", "first"), n=("line_no", "size"),
                                 zero=("line_no", lambda s: bool((s == -1).all())),
                                 nodoc=("doc", lambda s: float(s.fillna("").astype(str).eq("").mean())),
                                 nopaid=("paid", lambda s: float(s.isna().mean())),
                                 paid=("paid", "sum"), total=("stated_total", "first"))
    adv["err"] = adv.layout.eq("error")
    has_t = adv.total.notna() & ~adv.zero
    adv["recon"] = np.where(has_t, (adv.paid - adv.total).abs() <= 0.01 * adv.total.abs().clip(lower=1), np.nan)
    adv["key"] = adv.fmt + ":" + adv.layout.replace("", "?")
    g = adv.groupby("key").agg(advices=("n", "size"), zero_rows=("zero", "mean"), err=("err", "mean"),
                               rows_per_adv=("n", "mean"), nodoc=("nodoc", "mean"), nopaid=("nopaid", "mean"),
                               recon=("recon", "mean")).reset_index()
    g["share"] = g.advices / g.advices.sum()
    g["rows_per_adv"] = g.rows_per_adv / 100.0          # scaled so the 0.05 abs rule means 5 rows
    return g[["key", "share", "zero_rows", "err", "nodoc", "nopaid", "recon", "rows_per_adv"]]


def pdf_view(folder: str):
    out = []
    for kind in ("", "_remittance_advices"):
        p = os.path.join(ROOT, "ingestion", "out", "_cache", f"pdftext_{folder}{kind}.jsonl.gz")
        if not os.path.exists(p):
            p2 = os.path.join(lib.DATA, "cache", f"pdftext_{folder}{kind}.jsonl.gz")
            p = p2 if os.path.exists(p2) else p
        if not os.path.exists(p):
            out.append({"kind": kind.strip("_") or "pdfs", "pdf_empty": np.nan, "n": 0})
            continue
        n = e = 0
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                n += 1
                e += len((json.loads(line).get("text") or "").strip()) < 20
        out.append({"kind": kind.strip("_") or "pdfs", "pdf_empty": e / max(n, 1), "n": n})
    return pd.DataFrame(out)


def numbering_view(bank: pd.DataFrame, items: pd.DataFrame):
    it = items[items.side == "AR"] if "side" in items else items
    shape = it.document_number.fillna("").astype(str).str.replace(r"\d", "#", regex=True).str.replace(r"#+", "#",
                                                                                                         regex=True)
    top = shape.groupby(it.entity).agg(lambda s: s.value_counts().index[0] if len(s) else "").rename("shape")
    docs = it.groupby("entity").document_alnum.agg(lambda s: set(s.dropna().astype(str)))
    rec = bank[bank.amount > 0]
    hit = []
    for ent, toks in zip(rec.entity, rec.doc_tokens):
        ds = docs.get(ent, set())
        hit.append(bool(ds) and toks is not None and any(str(t) in ds for t in toks))
    rec = rec.assign(hit=hit)
    h = rec.groupby("entity").hit.mean().rename("ref_hit")
    return pd.concat([top, h], axis=1).reset_index().rename(columns={"index": "entity"})


def fx_view(bank: pd.DataFrame, items: pd.DataFrame, folder: str):
    m = lib.masters(folder)
    fx = m["fx_rates"]
    fx = fx.assign(date=pd.to_datetime(fx.date))
    last_book = pd.to_datetime(bank.booking_date).max()
    first_book = pd.to_datetime(bank.booking_date).min()
    curs = sorted((set(bank.currency.dropna()) | set(items.currency.dropna())) - {"EUR"})
    rows = []
    bdays = pd.bdate_range(first_book, last_book)
    for c in curs:
        f = fx[fx.currency == c]
        last = f.date.max() if len(f) else pd.NaT
        have = set(f.date)
        miss = sum(d not in have for d in bdays) / max(len(bdays), 1)
        rows.append({"currency": c, "fx_last": "" if pd.isna(last) else last.strftime("%Y-%m-%d"),
                     "behind_days": np.nan if pd.isna(last) else max((last_book - last).days, 0),
                     "missing_bdays": miss})
    return pd.DataFrame(rows), last_book


def cls_view(bank: pd.DataFrame, labels: pd.DataFrame | None):
    if labels is None:
        return None
    l = labels.merge(bank[["bank_txn_id", "entity", "amount"]], on="bank_txn_id")
    rec = l.amount > 0
    g = l.assign(low=l.p_items < 0.1, grey=l.p_items.between(0.1, 0.97), unid=rec & (l.category == "UNIDENTIFIED"))
    return g.groupby("entity")[["low", "grey", "unid"]].mean().reset_index()


def money_view(bank: pd.DataFrame, sub: pd.DataFrame | None):
    if sub is None:
        return None
    claimed = set(sub.loc[sub.item_id.fillna("").astype(str).ne(""), "bank_txn_id"])
    b = bank.assign(w=bank.amount_eur.abs().fillna(bank.amount.abs()), c=bank.bank_txn_id.isin(claimed))
    g = b.groupby("entity").apply(lambda x: x.w[x.c].sum() / max(x.w.sum(), 1e-9), include_groups=False)
    return g.rename("money_claimed").reset_index()


# ------------------------------------------------------------------------------------------------------------- main
def load_phase(phase: str):
    folder = lib.PHASES[phase][0]
    bank = pd.read_parquet(os.path.join(ROOT, "ingestion", "out", phase, "bank_lines.parquet"))
    items = pd.read_parquet(os.path.join(ROOT, "ingestion", "out", phase, "open_items.parquet"))
    rl = _rd(os.path.join(ROOT, "matching", "out", phase, "remittance_lines.parquet"))
    labels = _rd(os.path.join(ROOT, "classification", "out", phase, "line_labels.parquet"))
    sp = os.path.join(ROOT, "integration", "out", phase, "submission.csv.gz")
    sub = pd.read_csv(sp, dtype=str, usecols=["bank_txn_id", "item_id"]) if os.path.exists(sp) else None
    return folder, bank, items, rl, labels, sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="final", choices=list(lib.PHASES))
    ap.add_argument("--ref", default="live", choices=list(lib.PHASES))
    ap.add_argument("--abs", type=float, default=0.05)
    ap.add_argument("--rel", type=float, default=0.30)
    a = ap.parse_args()
    R, C = load_phase(a.ref), load_phase(a.phase)
    _p(f"=== input drift: {a.phase} vs {a.ref}  (flag = |Δ| > {a.abs} and > {a.rel:.0%} relative, or new/gone group)")
    _p(f"bank lines {len(R[1]):,} -> {len(C[1]):,} · items {len(R[2]):,} -> {len(C[2]):,}")

    (fr, er), (fc, ec) = bank_view(R[1]), bank_view(C[1])
    _compare("1a bank statements: share of lines per bank / format", fr, fc, ["bank", "statement_format"], ["share"],
             a, new_ok=True)
    _compare("1b bank lines with empty fields, per entity", er, ec, ["entity"],
             ["empty_text", "empty_cpty", "empty_payer", "empty_doc"], a, new_ok=True)
    _compare("2 remittance advices per format:layout (rows_per_adv shown /100)", advice_view(R[3]), advice_view(C[3]),
             ["key"], ["share", "zero_rows", "err", "nodoc", "nopaid", "recon", "rows_per_adv"], a)
    _compare("3 PDF text extraction (share empty)", pdf_view(R[0]), pdf_view(C[0]), ["kind"], ["pdf_empty"], a)
    nr, nc = numbering_view(R[1], R[2]), numbering_view(C[1], C[2])
    _p("\n--- 4 invoice numbering: main AR document-number shape per entity")
    for _, r in nc.merge(nr, on="entity", how="left", suffixes=("", "_ref")).iterrows():
        tag = "" if r.get("shape_ref") == r["shape"] else f"   <-- shape changed (ref {r.get('shape_ref')})"
        _p(f"  {r['entity']:6} {r['shape']}{tag}")
    _compare("4 receipts citing a document number of their entity (exact token hit)", nr[["entity", "ref_hit"]],
             nc[["entity", "ref_hit"]], ["entity"], ["ref_hit"], a, new_ok=True)
    fxc, last_book = fx_view(C[1], C[2], C[0])
    _p(f"\n--- 5 FX coverage for {a.phase} (last booking date {last_book.date()})")
    for _, r in fxc.iterrows():
        bad = (pd.isna(r.behind_days) or r.behind_days > 4 or r.missing_bdays > 0.1)
        _p(f"  {r.currency:4} last rate {r.fx_last or 'NONE':10} behind {r.behind_days} d, business days missing "
           f"{r.missing_bdays:.1%}" + ("   <-- FLAG" if bad else ""))
        if bad:
            FLAGS.append(f"5 FX: {r.currency} last {r.fx_last or 'none'}, missing {r.missing_bdays:.1%}")
    _compare("6 classifier per entity: p_items<0.1 / grey zone 0.1-0.97 / UNIDENTIFIED receipts", cls_view(R[1], R[4]),
             cls_view(C[1], C[4]), ["entity"], ["low", "grey", "unid"], a, new_ok=True)
    _compare("7 money claimed (share of |amount_eur| on lines with item rows), per entity", money_view(R[1], R[5]),
             money_view(C[1], C[5]), ["entity"], ["money_claimed"], a, new_ok=True)
    nm = money_view(C[1], C[5])
    if nm is not None:
        newe = set(nm.entity) - set(R[1].entity)
        for e in sorted(newe):
            v = float(nm.loc[nm.entity == e, "money_claimed"].iloc[0])
            if v < 0.5:
                FLAGS.append(f"7 money: new entity {e} has only {v:.1%} of its money claimed")
    _p(f"\n=== {len(FLAGS)} flag(s)")
    for f in FLAGS:
        _p("  - " + f)
    sys.exit(1 if FLAGS else 0)


if __name__ == "__main__":
    main()
