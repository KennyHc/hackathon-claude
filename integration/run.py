#!/usr/bin/env python3
"""D. Merge matching (B) + classification (C) into the submission, validate, and score it when the phase has labels.

    .venv/bin/python integration/run.py --phase val      # → integration/out/val/submission.csv.gz + score
    .venv/bin/python integration/run.py --phase live     # → integration/out/live/submission.csv.gz (then submit.py)

Inputs (each optional, so the pipeline always produces a valid file):
  matching/out/<phase>/allocations.parquet       B: bank_txn_id,item_id,amount,exception,confidence,method
  classification/out/<phase>/line_labels.parquet C: bank_txn_id,p_items,category,exception,confidence,method
  classification/exceptions.py                   C: assign_exceptions(allocations, bank_lines, open_items, **ctx)
Merge rule v0 (docs/CONTRACTS.md §D): a line gets B's item rows if B has it and (C's p_items ≥ --p-items or C has no
category for it); otherwise C's single category row; lines nobody covers get a default (receipt → UNIDENTIFIED,
payment → BANK_FEES, like the starter). Exceptions on item rows: assign_exceptions() where not null, else B's.
"""
import argparse
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from integration.scoring import SUB_COLS, has_golden, phase_dir, txns_path  # noqa: E402

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


def load_optional(path, what):
    if os.path.exists(path):
        df = pd.read_parquet(path)
        log(f"{what}: {len(df):,} rows from {os.path.relpath(path, ROOT)}")
        return df
    log(f"{what}: not found ({os.path.relpath(path, ROOT)}), skipped")
    return None


def exceptions_from_c(alloc, lines, phase, fix_amounts=True):
    try:
        from classification.exceptions import assign_exceptions
        from integration.lib.ledger import load_items
        from integration.lib.masters import masters_dir
    except ImportError as e:
        log(f"exceptions: classification/exceptions.py or integration/lib not available ({e}); keeping B's")
        return None
    items = load_items(phase)
    if fix_amounts:
        try:
            from classification.exceptions import normalize_amounts
            new = normalize_amounts(alloc, lines, items, masters_dir=masters_dir(phase))
            ch = (new - pd.to_numeric(alloc.amount, errors="coerce")).abs() > 0.005
            log(f"amounts: normalize_amounts changed {int(ch.sum()):,} rows")
            alloc["amount"] = new.values
        except ImportError:
            pass
    exc = assign_exceptions(alloc, lines, items, masters_dir=masters_dir(phase))
    log(f"exceptions: assign_exceptions gave {exc.notna().sum():,} codes")
    return exc


def unid_evidence(lines, phase):
    """bank_txn_id → True when there is evidence the line settles items. K3's matching/party/unid.py when present,
    else a local stub: an invoice-like reference (≥ 5-digit document number of an item of that entity) or a payer
    (IBAN or normalised name) known among that entity's customers."""
    from integration.lib.ledger import load_items
    items = load_items(phase)
    try:
        pdir = os.path.join(ROOT, "matching", "party")
        if pdir not in sys.path:
            sys.path.insert(0, pdir)  # K3's modules import each other by bare name (import matcher, party)
        from matching.party.unid import unid_evidence as k3
        r = k3(lines, items, phase)
        if isinstance(r, pd.DataFrame):  # bank_txn_id, has_ref_like, payer_known, …
            ev = r.has_ref_like.fillna(False).astype(bool) | r.payer_known.fillna(False).astype(bool)
            out = dict(zip(r.bank_txn_id, ev))
            log(f"unid evidence: K3 unid_evidence on {len(r):,} lines, {int(ev.sum()):,} with evidence")
            return out
        return dict(r) if isinstance(r, dict) else dict(zip(lines.bank_txn_id, pd.Series(r).astype(bool).values))
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        log(f"unid evidence: K3 unid_evidence failed ({e}); using the local stub")
    import re
    from integration.lib.masters import read_master
    from matching.common import norm_name
    cu = read_master("customers.csv", phase)
    out = {}
    for entity, L in lines.groupby("entity"):
        it = items[items.entity == entity]
        docs = {re.sub(r"\D", "", d).lstrip("0") for d in it.document_number if len(re.sub(r"\D", "", d)) >= 5}
        c = cu[cu.entity == entity]
        ibans = set(c.iban) - {""}
        names = {n for n in (norm_name(x) for x in list(c.name) + list(c.get("trade_name", []))) if len(n) >= 5}
        for r in L.itertuples(index=False):
            txt = f"{r.text} {r.remittance_info} {r.ref1} {r.ref2}"
            toks = {t.lstrip("0") for t in re.findall(r"\d{5,}", txt)}
            # also digit runs inside alphanumeric refs (KWI25-000123 → 25000123)
            toks |= {re.sub(r"\D", "", t).lstrip("0") for t in re.findall(r"[A-Z]{2,4}\d{2}-?\d{4,}", txt.upper())}
            nm = norm_name(f"{r.counterparty_name} {r.text}")
            out[r.bank_txn_id] = bool(toks & docs) or (r.counterparty_account in ibans) or \
                any(n in nm for n in names)
    return out


def train_entities():
    from integration.scoring import txns_path
    return set(pd.read_csv(txns_path("train"), dtype=str, keep_default_na=False, usecols=["entity"]).entity)


# matchers that know the exception AND the amount better than the ledger-based engine (val, 14:40: engine
# overrides on these methods were wrong in 32/32 rows)
TRUST_METHODS = {"dd_batch", "dd_batch_part", "card_settle", "late_discount", "retention_release",
                 "party_discount_late", "k4_sbc"}
KEEP_EMPTY_METHODS = {"instalment_last", "settlement"}


def drop_wild_overpay(items, lines, phase, ratio):
    from integration.lib.ledger import load_items
    it = load_items(phase).set_index("item_id")
    n = items.groupby("bank_txn_id").item_id.transform("size")
    cash = items.bank_txn_id.map(dict(zip(lines.bank_txn_id, lines.amount.astype(float).abs())))
    lc = items.bank_txn_id.map(dict(zip(lines.bank_txn_id, lines.currency)))
    ic = items.item_id.map(it.currency)
    amt = items.amount.astype(float)
    bad = (n == 1) & items.item_id.str[:3].isin(["ARI", "API"]) & (lc == ic) & (cash > amt * (1 + ratio)) \
        & ~items.exception.isin(["DUPLICATE_PAYMENT", "RETURNED"])
    log(f"overpay guard: dropping {int(bad.sum()):,} single-invoice lines with cash > amount·{1 + ratio:g}")
    return items[~bad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--p-items", type=float, default=0.5, help="B wins a line when C's p_items ≥ this")
    ap.add_argument("--unid-p", type=float, default=0.9,
                    help="a receipt nobody covers is UNIDENTIFIED when C's p_items < this (tune on val once B lands)")
    ap.add_argument("--no-c-exceptions", action="store_true", help="use only B's exceptions")
    ap.add_argument("--no-amount-fix", action="store_true", help="skip classification.exceptions.normalize_amounts")
    ap.add_argument("--b-exc-fallback", type=int, default=0,
                    help="1: where assign_exceptions says no exception, keep B's guess; 0: C decides alone")
    ap.add_argument("--overpay-max", type=float, default=0.35,
                    help="> 0: drop B's line when it is one same-currency invoice and cash > amount·(1+this) "
                         "(such 'overpayments' are mostly wrong items)")
    ap.add_argument("--trust-methods", default="default",
                    help="'default' (TRUST_METHODS) or a comma list: keep the matcher's exception and amount there")
    ap.add_argument("--unid-evidence", choices=["off", "new", "all"], default="new",
                    help="leftover receipts: UNIDENTIFIED only without evidence of items (new = unseen entities)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--b", default=None, help="override B's allocations.parquet path (testing)")
    ap.add_argument("--c", default=None, help="override C's line_labels.parquet path (testing)")
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "integration", "out", a.phase, "submission.csv.gz")

    lines = pd.read_csv(txns_path(a.phase), dtype=str, keep_default_na=False)
    log(f"{len(lines):,} bank lines in {a.phase}")
    B = load_optional(a.b or os.path.join(ROOT, "matching", "out", a.phase, "allocations.parquet"), "B allocations")
    C = load_optional(a.c or os.path.join(ROOT, "classification", "out", a.phase, "line_labels.parquet"),
                  "C line labels")

    c = C.set_index("bank_txn_id") if C is not None else pd.DataFrame(columns=["p_items", "category", "exception"])
    if B is not None and len(B):
        B = B[B.bank_txn_id.isin(lines.bank_txn_id)].copy()
        b_lines = pd.Index(B.bank_txn_id.unique())
        p = c.p_items.reindex(b_lines).astype(float)
        cat = c.category.reindex(b_lines)
        keep = b_lines[(p.isna() | (p >= a.p_items)) | cat.isna() | (cat == "")]
        items = B[B.bank_txn_id.isin(keep)].copy()
        if a.overpay_max > 0:
            items = drop_wild_overpay(items, lines, a.phase, a.overpay_max)
        log(f"B keeps {len(keep):,}/{len(b_lines):,} of its lines ({len(items):,} rows)")
        if not a.no_c_exceptions:
            b_amt = items["amount"].copy()
            b_exc = items["exception"] if "exception" in items else pd.Series(None, index=items.index)
            items["b_exc"] = b_exc.astype(object).where(b_exc.notna() & (b_exc.astype(str).str.strip() != ""), None)
            exc = exceptions_from_c(items, lines, a.phase, not a.no_amount_fix)
            if exc is not None:
                items["exception"] = exc.where(exc.notna(), items.get("exception")) if a.b_exc_fallback else exc
                # matchers that KNOW the item is cleared (last instalment, dispute settlement): keep their "no
                # exception" (the ledger misses earlier unmatched payments and would say PARTIAL)
                if "method" in items and a.trust_methods:
                    tm = set(TRUST_METHODS if a.trust_methods == "default" else a.trust_methods.split(","))
                    t2 = items.method.astype(str).isin(tm)
                    ch = int((t2 & (items.exception.fillna("") != items.b_exc.fillna(""))).sum())
                    items.loc[t2, "exception"] = items.loc[t2, "b_exc"]
                    items.loc[t2, "amount"] = b_amt[t2]
                    log(f"exceptions: trusted the matcher on {int(t2.sum()):,} rows ({ch:,} exceptions restored)")
                if "method" in items:
                    trust = items.method.astype(str).isin(KEEP_EMPTY_METHODS) & items.b_exc.isna()
                    items.loc[trust, "exception"] = None
                    log(f"exceptions: kept the matcher's empty exception on {int(trust.sum()):,} rows "
                        f"({', '.join(sorted(KEEP_EMPTY_METHODS))})")
        items["category"] = ""
    else:
        items = pd.DataFrame(columns=SUB_COLS)
    covered = set(items.bank_txn_id)

    rest = lines[~lines.bank_txn_id.isin(covered)][["bank_txn_id", "amount"]].copy()
    ccat = c.category.reindex(rest.bank_txn_id).values if len(c) else [None] * len(rest)
    cexc = c.exception.reindex(rest.bank_txn_id).values if len(c) else [None] * len(rest)
    receipt = rest.amount.astype(float).gt(0)
    ccat = pd.Series(ccat, index=rest.index)
    has_c_cat = ccat.notna() & (ccat != "")
    pc = pd.to_numeric(c.p_items, errors="coerce").reindex(rest.bank_txn_id).values if len(c) else [float("nan")] * len(rest)
    pc = pd.Series(pc, index=rest.index)
    c_says_items = rest.bank_txn_id.isin(c.index) & ~has_c_cat & (pc >= a.unid_p)  # C: settles items; B missed it
    # entities never seen in train (FINAL: ZA01): C's p_items is not calibrated there. A leftover receipt there is
    # UNIDENTIFIED only when nothing suggests it settles items (an invoice-like reference of that entity, or a payer
    # known to that entity); otherwise CUSTOMER_ADVANCE (no exception). --unid-evidence all: everywhere.
    ent = rest.bank_txn_id.map(dict(zip(lines.bank_txn_id, lines.entity)))
    scope = ~ent.isin(train_entities()) if a.unid_evidence == "new" else pd.Series(a.unid_evidence == "all",
                                                                                     index=rest.index)
    scope &= receipt & ~(has_c_cat & ~ccat.isin(["CUSTOMER_ADVANCE", "UNIDENTIFIED"]))
    if scope.any():
        sub_lines = lines[lines.bank_txn_id.isin(rest.bank_txn_id[scope])]
        ev = unid_evidence(sub_lines, a.phase)
        evid = rest.bank_txn_id.map(ev).fillna(False).astype(bool) & scope
        log(f"unid evidence ({a.unid_evidence}): {int(scope.sum()):,} leftover receipts in scope "
            f"({sorted(set(ent[scope]))[:8]}), {int(evid.sum()):,} with evidence → CUSTOMER_ADVANCE, "
            f"{int((scope & ~evid).sum()):,} → UNIDENTIFIED")
        c_says_items = (c_says_items & ~scope) | evid
        ccat = ccat.where(~scope, None)
        has_c_cat = ccat.notna() & (ccat != "")
    # Lines nobody explains score as wrong whatever we write, so the only thing that matters is not to emit a false
    # UNIDENTIFIED exception (it costs UNIDENTIFIED precision). No C opinion at all → starter rule.
    default = pd.Series("BANK_FEES", index=rest.index)
    default[receipt & ~c_says_items] = "UNIDENTIFIED"
    default[receipt & c_says_items] = "CUSTOMER_ADVANCE"
    rest["category"] = ccat.where(has_c_cat, default)
    rest["exception"] = ""
    rest.loc[rest.category == "UNIDENTIFIED", "exception"] = "UNIDENTIFIED"
    rest["item_id"], rest["amount"] = "", ""
    log(f"no-item rows: {len(rest):,} (from C: {int(has_c_cat.sum()):,}; C says items but B missed: "
        f"{int(c_says_items.sum()):,}; no opinion: {int((~has_c_cat & ~c_says_items).sum()):,})")

    if len(items):
        items["amount"] = items.amount.astype(float).map(lambda x: f"{x:.2f}")
    sub = pd.concat([items.reindex(columns=SUB_COLS), rest[SUB_COLS]], ignore_index=True).fillna("")
    sub["exception"] = sub.exception.replace({"None": "", "nan": ""})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sub.to_csv(out, index=False, compression="gzip")
    log(f"wrote {len(sub):,} rows → {os.path.relpath(out, ROOT)}")

    from integration.validate import check
    ok, probs, warns = check(out, a.phase)
    for w in warns:
        log("WARN " + w)
    for p in probs:
        log("ERROR " + p)
    log("VALID ✓" if ok else "INVALID ✗ — run integration/validate.py --repair")
    if ok and has_golden(a.phase):
        os.system(f'"{sys.executable}" "{os.path.join(ROOT, "integration", "score.py")}" "{out}" --phase {a.phase} '
                  f'--label run --segments 10')
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
