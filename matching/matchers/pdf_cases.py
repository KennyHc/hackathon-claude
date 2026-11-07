"""pdf_cases matcher: customer receipts that only the invoice PDF explains (POLICIES s3, s5.7, s5.8).

    python -m matching.matchers.pdf_cases --phase val

Clauses come from `matching.lib.pdf_clauses` (cached per phase). From them every open AR invoice with a clause gets the
cash amounts the PDF announces ("expected payments"):
  instalment       each printed instalment amount (k of n)                         -> PARTIAL unless the balance is cleared
  retention_first  total - retained amount ("Importe a pagar al vencimiento")       -> PARTIAL (the retention stays open)
  retention_rel    the retained amount, once the open balance is exactly it         -> no exception
  advance_net      total - advance of the proforma ("Importe neto a pagar")         -> PARTIAL (advance stays open)
  replaced         open amount of the REPLACEMENT invoice, evidenced by a quote of the cancelled one (or of itself)
A credit bank line (same entity and currency, not claimed earlier) is claimed only when ONE expected payment equals the
cash to the cent and the line also evidences it:
  _ref    the text quotes the invoice (or, for `replaced`, the cancelled invoice) - refs_text.find_refs level >= 2
  _payer  the payer is the invoice customer (party index: IBAN / name / learned alias, or the IBAN of a bank-account notice
          PDF) and that customer has no other open item of the same amount
Plus `replaced_multi`: the text quotes several of our numbers, at least one is a cancelled invoice, and after replacing
each cancelled number by its replacement the quoted open amounts sum to the cash.
Ledger: open_items_at_start balance - earlier matchers' allocations (`prior`) - our own claims, in booking order.
No hard-coded ids/names/dates; no labels used (the party index learns aliases from train golden before `cut`, like refs).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from matching.lib import data, evalutil, pdf_clauses, party as partylib, refs_text as rt  # noqa: E402

CUT = {"val": "2025-01-01", "sample": "2025-01-01"}
DEFAULT_CUT = "2025-07-01"
COLS = ["bank_txn_id", "item_id", "amount", "exception", "confidence", "method"]
# measured pair precision decides what is emitted (>= 97% on val); weaker methods get confidence < 0.65 and are dropped here
CONF = {"ref": 0.95, "payer": 0.9, "multi": 0.95}
EMIT_MIN = 0.65
_KN = re.compile(r"(?:RATE|PLAZO|CUOTA|VTO|VENC\w*|INSTAL\w*|ECH\w*|PARC\w*|PREST\w*|RATA|TERMIJN|INST)\.?\s*(?:N[O.º]?\s*)?"
                 r"(\d{1,2})\s*(?:/|OF|DE|DI|VON|SUR|SU)\s*(\d{1,2})\b")
_RELEASE = re.compile(r"(RETEN|GARANT|RELEASE|LIBERA|RETENUE|EINBEHALT|RITENUTA)")


def _c(x) -> int:
    return int(round(float(x) * 100))


def _items(phase: str) -> pd.DataFrame:
    cols = ["item_id", "item_type", "side", "entity", "party_id", "document_number", "issue_date", "currency", "open_amount", "total_amount"]
    cp = os.path.join(data.CACHE, phase, "items.parquet")
    if not os.path.exists(cp):
        data.items(phase)
    it = pd.read_parquet(cp, columns=cols, filters=[("side", "==", "AR")])
    it = it[it.item_type.isin(["ARI", "ARC"])].reset_index(drop=True)
    it["document_number"] = it.document_number.astype(str)
    return it


def expected_payments(items: pd.DataFrame, cl: pd.DataFrame, stats: Counter) -> pd.DataFrame:
    """One row per (item, kind, cash cents) announced by an invoice PDF."""
    inv = items[items.item_type == "ARI"]
    key = inv.entity + "|" + inv.document_number
    dup = key.duplicated(keep=False)
    pos_of = pd.Series(inv.index[~dup], index=key[~dup])
    out = []
    c = cl[cl.document_type == "invoice"]
    for r in c.itertuples(index=False):
        k = f"{r.entity}|{r.document_number}"
        if k not in pos_of.index:
            stats["clause_item_missing"] += 1
            continue
        p = int(pos_of[k]); it = items.loc[p]
        tot = _c(it.total_amount)
        if r.clause_type == "INSTALMENTS" and r.amounts is not None:
            am = [_c(a) for a in r.amounts]
            if abs(sum(am) - tot) > 2:
                stats["inst_sum_mismatch"] += 1
                continue
            for j, a in enumerate(am):
                out.append((p, "instalment", a, j + 1, len(am)))
        elif r.clause_type == "RETENTION" and r.amount and r.amount == r.amount:
            ret = _c(r.amount)
            pay = tot - ret
            if r.payable == r.payable and r.payable and abs(_c(r.payable) - pay) > 2:
                stats["ret_payable_mismatch"] += 1
            out.append((p, "retention_first", pay, 1, 2))
            out.append((p, "retention_rel", ret, 2, 2))
        elif r.clause_type == "ADVANCE_DEDUCT" and r.amount and r.amount == r.amount:
            net = tot - _c(r.amount)
            if net > 0:
                out.append((p, "advance_net", net, 1, 1))
        elif r.clause_type == "REPLACES" and isinstance(r.ref_document, str):
            out.append((p, "replaced", -1, 1, 1))  # cash = the replacement's open amount at run time
    df = pd.DataFrame(out, columns=["pos", "kind", "cents", "k", "n"]).drop_duplicates(["pos", "kind", "cents"])
    return df


def match(phase: str, claimed: set | None = None, prior: pd.DataFrame | None = None, p_items: pd.Series | None = None,
          cut: str | None = None, verbose: bool = False, all_methods: bool = False) -> pd.DataFrame:
    t0 = time.time()
    stats: Counter = Counter()
    claimed = set(claimed or ())
    cut = cut or CUT.get(phase, DEFAULT_CUT)
    cl = pdf_clauses.clauses(phase, verbose=verbose)
    items = _items(phase)
    ex = expected_payments(items, cl, stats)
    if ex.empty:
        return pd.DataFrame(columns=COLS)

    # ledger
    open_c = np.round(items.open_amount.astype(float).values * 100).astype("int64")
    pos_of = pd.Series(np.arange(len(items)), index=items.item_id.values)
    if prior is not None and len(prior):
        pa = prior[prior.item_id.isin(pos_of.index) & (prior.exception.fillna("") != "RETURNED")].groupby("item_id").amount.sum()
        ii = pos_of[pa.index].values
        open_c[ii] = open_c[ii] - np.round(pa.values * 100).astype("int64")
    total_c = np.round(items.total_amount.astype(float).values * 100).astype("int64")
    ent = items.entity.values; cur = items.currency.values; party = items.party_id.values
    docs = items.document_number.values; issue = items.issue_date.values; typ = items.item_type.values

    # cancelled invoice number -> replacement (from the replacement's REPLACES clause). The cancelled invoice is usually NOT an
    # open item any more (cancelled by a credit note), so the link is kept on its printed number.
    repl_key: dict[tuple, int] = {}       # (entity, full or digits key of the cancelled number) -> replacement pos
    cancelled_doc: dict[int, str] = {}    # replacement pos -> cancelled number
    rp = ex[ex.kind == "replaced"].pos.unique()
    rc = cl[cl.clause_type == "REPLACES"].drop_duplicates(["entity", "document_number"])
    rdoc = dict(zip(zip(rc.entity, rc.document_number), rc.ref_document))
    for p in rp:
        x = rdoc.get((ent[p], docs[p]))
        if not isinstance(x, str):
            continue
        cancelled_doc[p] = x
        k = rt.doc_keys(x)
        for kk in {k["full"], k["digits"]}:
            if len(kk) >= 8:
                repl_key[(ent[p], kk)] = -1 if (ent[p], kk) in repl_key and repl_key[(ent[p], kk)] != p else p
    repl_key = {k: v for k, v in repl_key.items() if v >= 0}
    stats["replaced_links"] = len(cancelled_doc)
    # entity-wide index of our numbers (for multi-reference lines)
    ref_idx: dict[tuple, list[int]] = defaultdict(list)
    for i in range(len(items)):
        k = rt.doc_keys(docs[i])
        if len(k["full"]) >= 8:
            ref_idx[(ent[i], k["full"])].append(i)
        if len(k["digits"]) >= 8 and k["digits"] != k["full"]:
            ref_idx[(ent[i], k["digits"])].append(i)

    # same-amount open items per customer (to refuse a payer-only claim when another open item has the cash amount)
    by_party = defaultdict(list)
    for i in np.flatnonzero(open_c > 0):
        by_party[party[i]].append(i)

    # bank-account notices: IBAN -> customer (both the new and the old account are valid, POLICIES s5.11)
    notice_iban: dict[str, set] = defaultdict(set)
    for r in cl[cl.clause_type == "NEW_IBAN"].itertuples(index=False):
        for ib in (r.iban, r.old_iban):
            if isinstance(ib, str):
                notice_iban[ib.upper()].add(r.party_id)

    bank = data.bank(phase)
    b = bank[(bank.amount > 0) & ~bank.bank_txn_id.isin(claimed)].sort_values(["booking_date", "bank_txn_id"])
    b = b.assign(cents=np.round(b.amount.values * 100).astype("int64"))
    # candidate index: (entity, currency, cents) -> expected rows; 'replaced' cents resolved per line from the ledger
    fixed = ex[ex.kind != "replaced"]
    cand_idx = defaultdict(list)
    for r in fixed.itertuples(index=False):
        cand_idx[(ent[r.pos], cur[r.pos], r.cents)].append((r.pos, r.kind, r.k, r.n))
    for p in set(ex[ex.kind == "replaced"].pos):  # the replacement's open amount (only our own claims change it later)
        if open_c[p] > 0:
            cand_idx[(ent[p], cur[p], int(open_c[p]))].append((p, "replaced", 1, 1))

    px = partylib.PartyIndex(phase, cut=cut)
    if verbose:
        print(f"[pdf_cases] {len(cl)} clauses, {len(ex)} expected payments, {len(cancelled_doc)} replacement links, "
              f"{len(b)} credit lines, setup {time.time() - t0:.0f}s")
    rows = []
    for r in b.itertuples(index=False):
        e, c, v = r.entity, r.currency, r.cents
        # +-1 cent: printed amounts are total - round(retention) while payers often pay round(total * (1 - pct))
        cands = list(dict.fromkeys(x for d in (0, -1, 1) for x in cand_idx.get((e, c, v + d), ())))
        rowd = r._asdict()
        text = rt.bank_text(rowd)
        toks = rt.text_tokens(text)
        bd = np.datetime64(r.booking_date)
        # --- multi-reference lines quoting a cancelled invoice: replace it by its replacement, the quoted set must sum to the cash
        q, subst = set(), False
        for t in toks:
            rpos = repl_key.get((e, t))
            if rpos is not None:
                q.add(rpos); subst = True
                continue
            for i in ref_idx.get((e, t), ()):
                q.add(i)
        if subst and len(q) > 1:
            q2 = {i for i in q if open_c[i] > 0 and issue[i] <= bd and cur[i] == c}
            sgn = sum((open_c[i] if typ[i] == "ARI" else -open_c[i]) for i in q2)
            if len(q2) == len(q) and abs(sgn - v) <= 1 and len({party[i] for i in q2}) == 1:
                for i in q2:
                    rows.append((r.bank_txn_id, items.item_id.values[i], open_c[i] / 100, None, CONF["multi"], "replaced_multi"))
                    open_c[i] = 0
                stats["claim:replaced_multi"] += 1
                continue
        if not cands:
            continue
        stats["lines_with_amount_cand"] += 1
        # filter by ledger and dates
        ok = []
        for (p, kind, k, n) in cands:
            if issue[p] > bd:
                continue
            o = open_c[p]
            if kind == "instalment" and o >= v - 2:
                ok.append((p, kind, k, n))
            elif kind in ("retention_first", "advance_net") and abs(o - total_c[p]) <= 1 and v < o:
                ok.append((p, kind, k, n))
            elif kind == "retention_rel" and abs(o - v) <= 1 and o < total_c[p]:
                ok.append((p, kind, k, n))
            elif kind == "replaced" and abs(o - v) <= 1:
                ok.append((p, kind, k, n))
        if not ok:
            stats["cand_ledger_reject"] += 1
            continue
        # evidence
        kn = _KN.search(rt.strip_accents(text).upper())
        kn = (int(kn.group(1)), int(kn.group(2))) if kn else None
        pc = None
        ev = []
        for (p, kind, k, n) in ok:
            dl = [docs[p]] + ([cancelled_doc[p]] if p in cancelled_doc else [])
            lvl = max(rt.find_refs(toks, dl).values(), default=0)
            if kn and kind == "instalment" and kn[1] != n:
                continue
            if lvl >= 2:
                ev.append((p, kind, "ref", lvl))
                continue
            if pc is None:
                try:
                    pc = {x.customer_id for x in px.candidates(rowd)}
                except Exception:
                    pc = set()
                ib = str(rowd.get("counterparty_account") or "").replace(" ", "").upper()
                pc |= notice_iban.get(ib, set())
                for ib2 in partylib.ibans_in(rowd.get("text")):
                    pc |= notice_iban.get(ib2, set())
            if party[p] in pc:
                others = [i for i in by_party[party[p]] if i != p and open_c[i] > 0 and abs(open_c[i] - v) <= 1 and issue[i] <= bd]
                if not others:
                    ev.append((p, kind, "payer", 0))
                else:
                    stats["payer_same_amount_other"] += 1
        if not ev:
            stats["no_evidence"] += 1
            continue
        best = [x for x in ev if x[2] == "ref"] or ev
        if best[0][2] == "ref":
            top = max(x[3] for x in best)
            best = [x for x in best if x[3] == top]
        if len({x[0] for x in best}) != 1:
            stats["ambiguous"] += 1
            continue
        p, kind, evk, _ = best[0]
        if kind == "retention_rel" and evk == "payer" and not _RELEASE.search(rt.strip_accents(text).upper()):
            pass  # the release amount is exact and the payer is known: keep
        o = open_c[p]
        exc = None
        if kind in ("retention_first", "advance_net"):
            exc = "PARTIAL"
        elif kind == "instalment" and o - v > max(1, int(0.005 * o)):
            exc = "PARTIAL"
        method = f"{kind}_{evk}"
        rows.append((r.bank_txn_id, items.item_id.values[p], v / 100, exc, CONF[evk], method))
        open_c[p] = o - v
        stats["claim:" + method] += 1
    out = pd.DataFrame(rows, columns=COLS)
    if not all_methods:
        out = out[out.confidence >= EMIT_MIN]
    if verbose:
        print(f"[pdf_cases] {out.bank_txn_id.nunique()} lines / {len(out)} rows in {time.time() - t0:.0f}s")
        for k, n in sorted(stats.items()):
            print(f"   {k:28s} {n}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="sample")
    ap.add_argument("--standalone", action="store_true", help="no prior allocations (default: batches/ap/remittance outputs if present)")
    a = ap.parse_args()
    prior = None
    if not a.standalone:
        ps = [os.path.join("matching", "out", a.phase, f"{n}.parquet") for n in ("batches", "ap", "remittance")]
        ps = [p for p in ps if os.path.exists(p)]
        if ps:
            prior = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True)
    claimed = set(prior.bank_txn_id) if prior is not None else set()
    df = match(a.phase, claimed, prior=prior, verbose=True, all_methods=True)
    os.makedirs(os.path.join("matching", "out", a.phase), exist_ok=True)
    df[df.confidence >= EMIT_MIN].to_parquet(os.path.join("matching", "out", a.phase, "pdf_cases.parquet"), index=False)
    if a.phase in ("sample", "val", "train") and len(df):
        evalutil.report(df, a.phase, name="pdf_cases")
        g = data.golden(a.phase)
        gi = g[g.item_id.notna()].rename(columns={"amount": "gamt"})[["bank_txn_id", "item_id", "gamt", "segment"]]
        m = df.merge(gi, on=["bank_txn_id", "item_id"], how="left")
        m["ok"] = m.gamt.notna() & ((m.amount - m.gamt).abs() <= np.maximum(0.01, 0.005 * m.gamt.abs().fillna(0)))
        print(m.groupby("method").ok.agg(["size", "mean"]).rename(columns={"size": "pairs", "mean": "precision"}).round(4).to_string())
        sys.path.insert(0, os.path.join("matching", "tests", "pdf"))
        from seg_report import seg_table
        print(seg_table(df, a.phase).to_string())
