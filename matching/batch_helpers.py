"""Helpers shared by the dd_batches and cards_cheques matchers: side-file loaders and an item resolver.

Side files (direct_debit_batches/, card_settlements/, cheque_deposits/) are read from the phase folder and the previous
phase folders (a return / chargeback / returned cheque can refer to a collection of the previous month, which for the
first month of LIVE/FINAL lives in the previous phase's folder).
"""
from __future__ import annotations

import glob
import os
import re
import pandas as pd

from ingestion import lib, text

FOLDER_ORDER = ["train", "live", "final"]
METHODS = {"dd", "card", "cheque"}   # every receipt of a batch / settlement / deposit is an item with one of these


def _month_of(path: str) -> str:
    m = re.search(r"_(\d{6})\.csv$", path)
    return m.group(1) if m else ""


def _phase_months(ctx) -> tuple[str, str]:
    d = ctx.bank.booking_date
    lo = pd.Timestamp(d.min()) - pd.DateOffset(months=3)
    hi = pd.Timestamp(d.max()) + pd.DateOffset(months=1)
    return lo.strftime("%Y%m"), hi.strftime("%Y%m")


def side_files(ctx, sub: str) -> pd.DataFrame:
    """All rows of <folder>/<sub>/*.csv (current and previous phase folders) for months near the phase window."""
    lo, hi = _phase_months(ctx)
    upto = FOLDER_ORDER.index(ctx.folder) if ctx.folder in FOLDER_ORDER else 0
    seen: dict[str, str] = {}
    for folder in FOLDER_ORDER[: upto + 1]:
        for p in sorted(glob.glob(os.path.join(lib.DATA, folder, sub, "*.csv"))):
            mo = _month_of(p)
            if lo <= mo <= hi:
                seen[os.path.basename(p)] = p   # a later folder wins on duplicate names
    parts = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in seen.values()]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def all_items(ctx) -> pd.DataFrame:
    """Every AR item (dd / card / cheque) of every folder up to the phase (not only open ones, issued in the last 2
    years) with normalised references."""
    folders = lib.PHASES[ctx.phase][3]
    it = pd.concat([lib.items_of(f) for f in folders], ignore_index=True)
    it = it[(it.side == "AR") & it.payment_method.isin(METHODS)].drop_duplicates("item_id", keep="last")
    since = (pd.Timestamp(ctx.bank.booking_date.min()) - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    it = it[it.issue_date >= since].copy()
    it["document_alnum"] = text.alnum_series(it.document_number.fillna(""))
    it["document_number_norm"] = text.doc_number_norm_series(it.document_number.fillna(""))
    it["party_name_norm"] = text.norm_name_series(it.party_name.fillna(""))
    it["total_amount"] = it.total_amount.astype(float).round(2)
    return it.reset_index(drop=True)


def get_resolver(ctx) -> "Resolver":
    """One Resolver per Ctx (shared by dd_batches and cards_cheques)."""
    r = getattr(ctx, "_batch_resolver", None)
    if r is None:
        r = Resolver(ctx)
        ctx._batch_resolver = r
    return r


class Resolver:
    """Find the AR item behind one receipt line of a side file (reference + amount + payer)."""

    def __init__(self, ctx, pool: pd.DataFrame | None = None):
        self.ctx = ctx
        it = ctx.items[(ctx.items.side == "AR") & ctx.items.payment_method.isin(METHODS)]
        self.open_amt = dict(zip(it.item_id, it.open_amount.astype(float)))
        self.pool = pool if pool is not None else all_items(ctx)
        # items open in the phase but issued before the pool horizon
        extra = it[~it.item_id.isin(set(self.pool.item_id))]
        if len(extra):
            self.pool = pd.concat([self.pool, extra[[c for c in self.pool.columns if c in extra.columns]]],
                                  ignore_index=True)
        self.pool_by_id = self.pool.set_index("item_id", drop=False)
        P = self.pool
        self.tot = dict(zip(P.item_id, P.total_amount.astype(float)))
        self.party = dict(zip(P.item_id, P.party_id))
        self.dn = dict(zip(P.item_id, P.document_number_norm))
        self.issue = dict(zip(P.item_id, P.issue_date))
        self.by_alnum: dict[tuple, list[str]] = {}
        self.by_dn: dict[tuple, list[str]] = {}
        self.by_party: dict[tuple, list[str]] = {}
        self.by_suffix: dict[tuple, list[str]] = {}
        self.name_parties: dict[tuple, set[str]] = {}
        for iid, e, al, dn, pid, nn in zip(P.item_id, P.entity, P.document_alnum, P.document_number_norm, P.party_id,
                                           P.party_name_norm):
            self.by_alnum.setdefault((e, al), []).append(iid)
            self.by_dn.setdefault((e, dn), []).append(iid)
            self.by_party.setdefault((e, pid), []).append(iid)
            self.name_parties.setdefault((e, nn), set()).add(pid)
            if len(dn) >= 5:   # suffix index for truncated stubs / 'RECIBO <last digits>'
                self.by_suffix.setdefault((e, dn[-5:]), []).append(iid)
        cust = ctx.masters["customers"]
        self.mandate_to_cust = {m: c for m, c in zip(cust.sepa_mandate_id, cust.customer_id) if m}
        for c, e, n, tn in zip(cust.customer_id, cust.entity, cust.name, cust.trade_name):
            for nm in (n, tn):
                if nm:
                    self.name_parties.setdefault((e, text.norm_name(nm)), set()).add(c)
        self._nn_cache: dict[str, str] = {}

    # ---- amounts --------------------------------------------------------------------------------------------------
    def open_amount(self, iid: str) -> float | None:
        return self.open_amt.get(iid)

    def total(self, iid: str) -> float:
        return self.tot[iid]

    def _amount_ok(self, iid: str, amt: float, open_only: bool) -> bool:
        oa = self.open_amt.get(iid)
        if oa is not None and abs(oa - amt) < 0.006:
            return True
        if open_only:
            return False
        return iid in self.tot and abs(self.tot[iid] - amt) < 0.006

    # ---- reference candidates -------------------------------------------------------------------------------------
    def ref_candidates(self, entity: str, ref: str) -> list[str]:
        ref = (ref or "").upper()
        al = re.sub(r"[^A-Z0-9]", "", ref)
        out: list[str] = []
        if al:
            out += self.by_alnum.get((entity, al), [])
        digits = re.sub(r"\D", "", ref)
        dn = digits.lstrip("0")
        if not out and len(dn) >= 3:
            out += self.by_dn.get((entity, dn), [])
            # 'RECIBO 0209279' / stub '107534' = trailing digits of the document number
            if not out and len(digits) >= 5:
                out += [c for c in self.by_suffix.get((entity, digits[-5:]), []) if self.dn[c].endswith(dn)]
        return list(dict.fromkeys(out))

    def parties_for(self, entity: str, mandate: str = "", name: str = "") -> set[str]:
        ps: set[str] = set()
        if mandate and mandate in self.mandate_to_cust:
            ps.add(self.mandate_to_cust[mandate])
        if name:
            nn = self._nn_cache.get(name)
            if nn is None:
                nn = self._nn_cache[name] = text.norm_name(name)
            ps |= self.name_parties.get((entity, nn), set())
        return ps

    def resolve(self, entity: str, ref: str, amount: float, parties: set[str] | None = None,
                used: set[str] | None = None) -> tuple[str | None, str]:
        """-> (item_id, how) or (None, reason)."""
        used = used or set()
        parties = parties or set()
        cands = [c for c in self.ref_candidates(entity, ref) if c not in used]
        if cands:
            for open_only in (True, False):
                ok = [c for c in cands if self._amount_ok(c, amount, open_only)]
                if len(ok) > 1 and parties:
                    ok = [c for c in ok if self.party[c] in parties] or ok
                if len(ok) == 1:
                    return ok[0], "ref"
                if len(ok) > 1:
                    return None, "ref_ambiguous"
            # reference found but amount differs: accept if it is the only candidate and the party agrees
            same_party = [c for c in cands if not parties or self.party[c] in parties]
            if len(same_party) == 1:
                return same_party[0], "ref_amount_diff"
        # no usable reference: payer + amount (oldest open first)
        if parties:
            pc = []
            for p in parties:
                pc += self.by_party.get((entity, p), [])
            pc = [c for c in pc if c not in used]
            for open_only in (True, False):
                ok = [c for c in pc if self._amount_ok(c, amount, open_only)]
                if ok:
                    ok.sort(key=lambda c: (self.issue[c], c))
                    return ok[0], "party_amount" if len(ok) == 1 else "party_amount_fifo"
        return None, "none"
