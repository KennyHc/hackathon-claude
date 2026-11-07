"""Helpers for the payment_runs matcher: load our outgoing payment files and resolve their documents to AP items."""
from __future__ import annotations

import glob
from collections import defaultdict
import os
import re
import unicodedata

import numpy as np
import pandas as pd

from ingestion import lib

FACTORY_SUBS = {"ES00": {"ES00", "ES01", "ES02", "PT01"}}   # POLICIES §5.2: ES00 pays invoices of its subsidiaries

_LEGAL = re.compile(r"\b(S\.?L\.?U?|S\.?A\.?U?|S\.?COOP|SARL|SAS|GMBH|LTD|LLC|INC|CORP|CO|BV|SPA|SRL|PTY|LIMITED|"
                    r"SP\.? Z O\.?O|LTDA|SA DE CV|EFC|AG|NV|LDA)\b\.?")


_FACTOR = re.compile(r"\b(CONFIRMING|EFC|FACTOR\w*|PAGOS)\b")


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def alnum(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def phase_window(ctx) -> tuple[str, str]:
    d = ctx.bank.booking_date
    return d.min(), d.max()


def _month_add(ym: str, k: int) -> str:
    y, m = int(ym[:4]), int(ym[4:])
    m += k
    while m < 1:
        y, m = y - 1, m + 12
    while m > 12:
        y, m = y + 1, m - 12
    return f"{y:04d}{m:02d}"


def load_runs(ctx, months_before: int = 2, months_after: int = 1) -> pd.DataFrame:
    """All payment-run rows whose file month is in [phase start - months_before, phase end + months_after], from every
    phase folder (a run file can sit in the neighbouring phase's folder)."""
    start, end = phase_window(ctx)
    lo, hi = _month_add(start[:4] + start[5:7], -months_before), _month_add(end[:4] + end[5:7], months_after)
    files = {}
    for folder in ("train", "live", "final"):
        for f in glob.glob(os.path.join(lib.DATA, folder, "payment_runs", "*.csv")):
            ym = os.path.basename(f).split("_")[1]
            if lo <= ym <= hi:
                files.setdefault(os.path.basename(f), f)   # same file in two folders -> keep first
    parts = [pd.read_csv(f, dtype=str, keep_default_na=False).assign(run_file=os.path.basename(f))
             for f in sorted(files.values())]
    if not parts:
        return pd.DataFrame()
    r = pd.concat(parts, ignore_index=True)
    r = r.drop_duplicates(["run_id", "end_to_end_id", "document_number", "document_type", "document_amount"])
    r["transfer_amount"] = r.transfer_amount.astype(float)
    r["document_amount"] = r.document_amount.astype(float)
    r["debtor_entity"] = r.debtor_account_id.str[:4]
    r["doc_alnum"] = r.document_number.map(alnum)
    r["benef_norm"] = r.beneficiary.map(norm_name)
    return r


def all_ap_items(ctx, docs: set[str]) -> pd.DataFrame:
    """Every AP item (also already-settled ones) of the phase's item folders whose document alnum is in `docs`."""
    parts = []
    for folder in lib.PHASES[ctx.phase][3]:
        d = os.path.join(lib.DATA, folder)
        if not os.path.exists(os.path.join(d, "ap_invoices.csv.gz")):
            continue
        ap = lib.read_csv(os.path.join(d, "ap_invoices.csv.gz"))
        parts.append(pd.DataFrame({"item_id": ap.item_id, "item_type": "API", "entity": ap.entity,
                                   "party_id": ap.supplier_id, "party_name": ap.supplier_name,
                                   "document_number": ap.supplier_invoice_number, "currency": ap.currency,
                                   "total_amount": ap.total_amount.astype(float), "due_date": ap.due_date}))
        cn = lib.read_csv(os.path.join(d, "credit_notes.csv.gz"))
        cn = cn[cn.side == "AP"]
        parts.append(pd.DataFrame({"item_id": cn.item_id, "item_type": "APC", "entity": cn.entity,
                                   "party_id": cn.party_id, "party_name": cn.party_name,
                                   "document_number": cn.document_number, "currency": cn.currency,
                                   "total_amount": cn.total_amount.astype(float), "due_date": ""}))
    if not parts:
        return pd.DataFrame()
    a = pd.concat(parts, ignore_index=True).drop_duplicates("item_id", keep="last")
    a["doc_alnum"] = a.document_number.map(alnum)
    a = a[a.doc_alnum.isin(docs)].copy()
    a["party_name_norm"] = a.party_name.map(norm_name)
    a["open_amount"] = 0.0
    return a


def _sup_match(t, r, sup_iban) -> bool:
    if r.beneficiary_iban and sup_iban.get(t.party_id, "") == r.beneficiary_iban:
        return True
    pn = norm_name(t.party_name) if isinstance(t.party_name, str) else ""
    return bool(pn) and (pn == r.benef_norm or (len(pn) >= 8 and r.benef_norm.startswith(pn)))


def _choose(cands, r, sup_iban, amt):
    """Pick one candidate item for a run row, or None."""
    sup = [t for t in cands if _sup_match(t, r, sup_iban)]
    if sup:
        cands = sup
    exact = [t for t in cands if abs(t.open_amount - amt) < 0.01 or abs(t.total_amount - amt) < 0.01]
    feas = exact or [t for t in cands if t.open_amount >= amt - 0.01]
    if not sup and not exact:
        return None   # neither the supplier nor the amount confirms it
    if len(feas) > 1:
        f2 = [t for t in feas if t.currency == r.currency]
        feas = f2 or feas
    if len(feas) > 1:
        f2 = [t for t in feas if t.open_amount > 0.005]
        feas = f2 or feas
    return feas[0] if len(feas) == 1 else None


def resolve_items(ctx, runs: pd.DataFrame, items: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """-> (run row index -> item_id ('' when not resolvable uniquely), already-settled items used).

    Candidates: AP items of the right type (INVOICE->API, CREDIT_NOTE->APC) whose supplier document number equals the
    run document number, in the debtor entity (or a payment-factory subsidiary). A candidate must be confirmed by the
    supplier (beneficiary IBAN / name) or by the amount. Open items first; then items already settled before the phase
    (re-paid after a returned transfer, duplicates), which need supplier + exact total. Documents still ambiguous (paid
    to a factor / confirming bank) take the supplier of the other documents of the same transfer."""
    ap = items[items.side == "AP"]
    ap = ap.assign(doc_alnum=ap.document_number.map(alnum))
    sup = ctx.masters["suppliers"]
    sup_iban = dict(zip(sup.supplier_id, sup.iban))
    by_doc: dict[str, list] = {}
    for t in ap.itertuples(index=False):
        by_doc.setdefault(t.doc_alnum, []).append(t)
    out, pending = {}, {}
    closed = pd.DataFrame(columns=["item_id", "currency", "party_id"])
    for idx, r in zip(runs.index, runs.itertuples(index=False)):
        want = "APC" if r.document_type == "CREDIT_NOTE" else "API"
        ents = FACTORY_SUBS.get(r.debtor_entity, {r.debtor_entity})
        cands = [t for t in by_doc.get(r.doc_alnum, ()) if t.item_type == want and t.entity in ents]
        c = _choose(cands, r, sup_iban, abs(r.document_amount)) if cands else None
        out[idx] = c.item_id if c is not None else ""
        if c is None:
            pending[idx] = cands
    # settled items (not in `items`): supplier + exact total
    if pending:
        docs = {runs.doc_alnum[i] for i in pending}
        closed = all_ap_items(ctx, docs)
        if closed.empty:
            closed = pd.DataFrame(columns=["item_id", "item_type", "entity", "party_id", "party_name", "currency",
                                           "total_amount", "doc_alnum", "open_amount"])
        closed = closed[~closed.item_id.isin(set(ap.item_id))]
        cby: dict[str, list] = {}
        for t in closed.itertuples(index=False):
            cby.setdefault(t.doc_alnum, []).append(t)
        for idx in list(pending):
            r = runs.loc[idx]
            want = "APC" if r.document_type == "CREDIT_NOTE" else "API"
            ents = FACTORY_SUBS.get(r.debtor_entity, {r.debtor_entity})
            amt = abs(r.document_amount)
            cc = [t for t in cby.get(r.doc_alnum, ()) if t.item_type == want and t.entity in ents
                  and _sup_match(t, r, sup_iban) and abs(t.total_amount - amt) < 0.01]
            if len(cc) == 1:
                out[idx] = cc[0].item_id
                del pending[idx]
            else:
                pending[idx] = pending[idx] + cc
    # still ambiguous: the supplier of the other documents of the same transfer
    if pending:
        party = dict(zip(ap.item_id, ap.party_id))
        party.update(zip(closed.item_id, closed.party_id))
        key = runs.debtor_account_id + "|" + runs.end_to_end_id
        res = pd.Series(out)
        res = res[res != ""]
        parties = res.map(party).groupby(key.reindex(res.index)).agg(lambda s: set(s))
        for idx, cands in pending.items():
            ps = parties.get(key[idx], set())
            if len(ps) != 1:
                continue
            amt = abs(runs.document_amount[idx])
            c2 = [t for t in cands if t.party_id in ps and t.open_amount >= amt - 0.01]
            if len(c2) == 1:
                out[idx] = c2[0].item_id
    # remainder of a split payment: another run row of the same document resolved to a candidate whose open amount
    # minus that earlier payment equals this one
    if pending:
        res = pd.Series(out)
        paid_rows = defaultdict(list)
        for i, iid in res[res != ""].items():
            paid_rows[iid].append((i, abs(runs.document_amount[i])))
        for idx, cands in pending.items():
            if out.get(idx):
                continue
            amt = abs(runs.document_amount[idx])
            ex = runs.execution_date[idx]
            c2 = [t for t in cands if any(runs.execution_date[j] < ex
                                          and abs(t.open_amount - a - amt) <= max(0.01, 0.005 * t.open_amount)
                                          for j, a in paid_rows.get(t.item_id, ()))]
            if len(c2) == 1:
                out[idx] = c2[0].item_id
    # paid to a factor / confirming bank (no supplier IBAN): the only open candidate that can absorb the amount
    for idx, cands in pending.items():
        if out.get(idx):
            continue
        r = runs.loc[idx]
        if r.beneficiary_iban and not _FACTOR.search(r.benef_norm):
            continue
        amt = abs(r.document_amount)
        c2 = [t for t in cands if t.open_amount >= amt - 0.01 and t.currency == r.currency]
        if len(c2) == 1:
            out[idx] = c2[0].item_id
    res = pd.Series(out, dtype=object).reindex(runs.index).fillna("")
    return res, closed[closed.item_id.isin(set(res))]
