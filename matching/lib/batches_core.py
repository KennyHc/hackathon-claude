"""Core helpers for the `batches` matcher: item resolution of batch-file entries (receipts / orders / cheques) and parsing
of bank-line texts. No labels are used anywhere (all rules are structural), so nothing here leaks val/live golden."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

TOL = 0.011


def norm_doc(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper()) if s is not None and s == s else ""


def digits(s) -> str:
    return re.sub(r"\D", "", str(s)) if s is not None and s == s else ""


def _tokens(name) -> set:
    if name is None or name != name:
        return set()
    t = re.sub(r"[^a-z0-9 ]", " ", str(name).lower().replace("&", " and "))
    stop = {"sl", "slu", "sa", "pty", "ltd", "inc", "llc", "gmbh", "sas", "sarl", "srl", "spa", "bv", "co", "de", "the", "and", "y", "el", "la", "corp", "eurl", "sp", "zoo", "plc", "scoop", "s"}
    return {w for w in t.split() if len(w) > 1 and w not in stop}


def name_sim(a, b) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def prep_items(items: pd.DataFrame, cust_names: dict | None = None) -> pd.DataFrame:
    """AR invoices only (ARI), with the lookup keys. Keeps memory small."""
    a = items[(items.side == "AR") & (items.item_type == "ARI")][
        ["item_id", "entity", "party_id", "party_name", "document_number", "issue_date", "currency", "total_amount", "open_amount"]].copy()
    a["docnorm"] = a.document_number.map(norm_doc)
    a["dig"] = a.document_number.map(digits)
    a["d5"] = a.dig.str[-5:]
    if cust_names:
        a["pname"] = a.party_name.where(a.party_name.notna(), a.party_id.map(cust_names))
    else:
        a["pname"] = a.party_name
    a["total_amount"] = a.total_amount.astype(float)
    a["open_amount"] = a.open_amount.astype(float)
    return a.reset_index(drop=True)


def resolve(R: pd.DataFrame, A: pd.DataFrame, amount_mode: str = "le") -> pd.DataFrame:
    """R columns: rid, entity, ccy, ref, amount, name, asof(datetime). Returns one row per rid that resolved to a unique best
    item: rid, item_id, kind ('exact'/'suffix'), score, margin.  amount_mode: 'le' = amount <= open (or total) amount,
    'eq' = amount equals open or total amount."""
    R = R.copy()
    R["rn"] = R.ref.map(norm_doc)
    R["rd"] = R.ref.map(digits)
    R["rd5"] = R.rd.str[-5:]
    parts = []
    ex = R[R.rn.str.len() >= 5][["rid", "entity", "ccy", "rn", "rd", "amount", "name", "adate"]].merge(
        A, left_on=["entity", "rn"], right_on=["entity", "docnorm"], how="inner")
    ex["kind"] = "exact"; parts.append(ex)
    sf = R[R.rd.str.len() >= 5][["rid", "entity", "ccy", "rn", "rd", "rd5", "amount", "name", "adate"]].merge(
        A, left_on=["entity", "rd5"], right_on=["entity", "d5"], how="inner")
    if len(sf):
        ok = np.fromiter((d.endswith(r) or r.endswith(d) for d, r in zip(sf.dig.values, sf.rd.values)), bool, len(sf))
        sf = sf[ok & (sf.docnorm != sf.rn)].copy(); sf["kind"] = "suffix"; parts.append(sf)
    C = pd.concat([p[["rid", "item_id", "ccy", "currency", "amount", "name", "adate", "issue_date", "total_amount", "open_amount", "pname", "kind"]]
                   for p in parts], ignore_index=True)
    if C.empty:
        return pd.DataFrame(columns=["rid", "item_id", "kind", "score", "margin"])
    C = C[(C.ccy == C.currency) & (C.issue_date <= C.adate + pd.Timedelta(days=1))]
    ae = ((C.amount - C.open_amount).abs() <= TOL) | ((C.amount - C.total_amount).abs() <= TOL)
    al = (C.amount <= C.open_amount + TOL) | (C.amount <= C.total_amount + TOL)
    C = C[ae] if amount_mode == "eq" else C[al]
    if C.empty:
        return pd.DataFrame(columns=["rid", "item_id", "kind", "score", "margin"])
    ae = ((C.amount - C.open_amount).abs() <= TOL) | ((C.amount - C.total_amount).abs() <= TOL)
    ns = np.fromiter((name_sim(a, b) for a, b in zip(C.name.values, C.pname.values)), float, len(C))
    age = (C.adate - C.issue_date).dt.days.clip(lower=0, upper=1000)
    C["score"] = np.where(C.kind == "exact", 100.0, 60.0) + np.where(ae, 30.0, 0.0) + 15.0 * ns - age / 1000.0
    C = C.sort_values(["rid", "score"], ascending=[True, False]).drop_duplicates(["rid", "item_id"])
    C["rk"] = C.groupby("rid").cumcount()
    top = C[C.rk == 0].set_index("rid")
    sec = C[C.rk == 1].set_index("rid").score
    out = top[["item_id", "kind", "score"]].copy()
    out["margin"] = (out.score - sec.reindex(out.index)).fillna(99.0)
    return out.reset_index()


def resolve_by_name(R: pd.DataFrame, A: pd.DataFrame, min_sim: float = 0.6) -> pd.DataFrame:
    """For entries with no usable reference (cheque stub missing): same entity/currency, amount equal to the item's open or total
    amount, issued on/before the entry date, payer name similar to the customer. Needs a unique best candidate."""
    R = R[R.name.notna()].copy()
    if R.empty:
        return pd.DataFrame(columns=["rid", "item_id", "kind", "score", "margin"])
    R["ak"] = (R.amount * 100).round().astype("int64")
    Ak = pd.concat([A.assign(ak=(A.open_amount * 100).round().astype("int64")), A.assign(ak=(A.total_amount * 100).round().astype("int64"))]
                   ).drop_duplicates(["item_id", "ak"])
    C = R[["rid", "entity", "ccy", "ak", "name", "adate"]].merge(Ak[["entity", "ak", "item_id", "currency", "issue_date", "pname"]], on=["entity", "ak"])
    C = C[(C.ccy == C.currency) & (C.issue_date <= C.adate + pd.Timedelta(days=1))]
    if C.empty:
        return pd.DataFrame(columns=["rid", "item_id", "kind", "score", "margin"])
    C["sim"] = np.fromiter((name_sim(a, b) for a, b in zip(C.name.values, C.pname.values)), float, len(C))
    C = C[C.sim >= min_sim]
    C["score"] = C.sim - (C.adate - C.issue_date).dt.days.clip(lower=0, upper=1000) / 100000.0
    C = C.sort_values(["rid", "score"], ascending=[True, False])
    C["rk"] = C.groupby("rid").cumcount()
    top = C[C.rk == 0].set_index("rid")
    sec = C[C.rk == 1].set_index("rid").sim
    out = top[["item_id", "score", "sim"]].copy()
    out["kind"] = "name"
    out["margin"] = (out.sim - sec.reindex(out.index)).fillna(99.0)
    return out.reset_index()[["rid", "item_id", "kind", "score", "margin"]]
