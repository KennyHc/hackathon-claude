"""Who paid? bank line -> customer candidates (owner: refs matcher; reusable by other matchers).

    from matching.lib import party
    px = party.PartyIndex(phase, cut="2025-01-01")        # cut: aliases are learned from train golden booked BEFORE cut only
    cands = px.candidates(bank_row)                       # list[Cand] best first
    px.group_members(customer_id)                         # customer + its children (parent_customer_id) + siblings in the group

Evidence kinds (Cand.kind, strength in Cand.score, 0..1):
  iban      counterparty_account / IBAN found in the text equals the customer's master IBAN or a learned paying IBAN   (0.95)
  name      normalised customer / trade / invoice-printed name found in the bank text (full, or a truncated prefix >= 10 chars) (0.5..0.9)
  alias     learned: the same payer string was booked to this customer in train before `cut`                              (0.85)
Everything is restricted to customers of the bank line's entity (customers.csv `entity`).
No ids / names are hard-coded; `learn_aliases` is the only code that looks at labels and takes a `cut`.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

from . import data
from .refs_text import compact, name_key_tokens, name_keys, norm, strip_accents

ALIAS_VERSION = 1  # bump when learning code changes (invalidates the cached alias table)


@dataclass
class Cand:
    customer_id: str
    score: float
    kind: str
    detail: str = ""


_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")


def ibans_in(s) -> list[str]:
    if s is None or (isinstance(s, float) and s != s):
        return []
    t = strip_accents(str(s)).upper().replace(" ", "")
    out = []
    for m in _IBAN.finditer(t):
        out.append(m.group(0))
    return out


def train_item_party(ids=None) -> pd.DataFrame:
    """item_id -> customer_id for AR invoices / credit notes of TRAIN (cached; inputs only). `ids`: only these items."""
    p = os.path.join(data.CACHE, "refs_item_party.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p, filters=[("item_id", "in", list(ids))] if ids is not None else None)
    tr = data.phase_dir("train")
    ar = pd.read_csv(os.path.join(tr, "ar_invoices.csv.gz"), usecols=["item_id", "customer_id"])
    cn = pd.read_csv(os.path.join(tr, "credit_notes.csv.gz"), usecols=["item_id", "side", "party_id"])
    cn = cn[cn.side == "AR"].rename(columns={"party_id": "customer_id"})[["item_id", "customer_id"]]
    out = pd.concat([ar, cn], ignore_index=True)
    out.to_parquet(p, index=False)
    return out


def _ngrams(row, sizes=(2, 3)):
    """Token n-grams (digit-bearing tokens break runs) of the line's text / counterparty name."""
    out = set()
    for c in ("counterparty_name", "text", "remittance_info"):
        v = row.get(c)
        if not isinstance(v, str) or not v:
            continue
        for part in v.split(" | ") if c == "text" else [v]:
            run = []
            for t in norm(part).split() + ["1"]:
                if any(ch.isdigit() for ch in t):
                    for n in sizes:
                        for i in range(len(run) - n + 1):
                            g = tuple(run[i:i + n])
                            if sum(map(len, g)) >= 9:
                                out.add(g)
                    run = []
                else:
                    run.append(t)
    return out


def _inv_names(phase: str) -> pd.DataFrame:
    """(customer_id, printed name) pairs from the invoices we issued (old names of renamed customers). Inputs, not labels."""
    p = os.path.join(data.CACHE, "refs_invnames.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    parts = []
    for ph in ("train", "live", "final"):
        f = os.path.join(data.phase_dir(ph), "ar_invoices.csv.gz")
        if os.path.exists(f):
            d = pd.read_csv(f, usecols=["customer_id", "customer_name", "issue_date"])
            d = d.groupby(["customer_id", "customer_name"], as_index=False).issue_date.min()
            parts.append(d)
    out = pd.concat(parts).groupby(["customer_id", "customer_name"], as_index=False).issue_date.min() if parts else pd.DataFrame(columns=["customer_id", "customer_name", "issue_date"])
    out.to_parquet(p, index=False)
    return out


class PartyIndex:
    def __init__(self, phase: str, cut: str | None = None, learn: bool = True, customers: pd.DataFrame | None = None):
        self.phase = phase
        self.cut = cut
        c = customers if customers is not None else data.customers()
        self.cust = c.set_index("customer_id")
        self.entity_of = c.set_index("customer_id").entity.to_dict()
        # group structure
        self.children = defaultdict(list)
        self.group = defaultdict(list)
        for r in c[["customer_id", "parent_customer_id", "group_id"]].itertuples(index=False):
            if isinstance(r.parent_customer_id, str):
                self.children[r.parent_customer_id].append(r.customer_id)
            if isinstance(r.group_id, str):
                self.group[r.group_id].append(r.customer_id)
        self.group_of = c.set_index("customer_id").group_id.to_dict()
        self.parent_of = c.set_index("customer_id").parent_customer_id.to_dict()
        # names: customer_id -> set(keys)
        self.names: dict[str, set[str]] = defaultdict(set)
        for r in c[["customer_id", "name", "trade_name"]].itertuples(index=False):
            for nm in (r.name, r.trade_name):
                if isinstance(nm, str):
                    self.names[r.customer_id] |= name_keys(nm)
        try:
            inv = _inv_names(phase)
            for cid, nm in zip(inv.customer_id, inv.customer_name):
                if cid in self.entity_of and isinstance(nm, str):
                    self.names[cid] |= name_keys(nm)
        except Exception:
            pass
        # ibans
        self.iban_cust: dict[str, set[str]] = defaultdict(set)
        for cid, ib in zip(c.customer_id, c.iban):
            if isinstance(ib, str):
                self.iban_cust[ib.replace(" ", "").upper()].add(cid)
        self.alias_text: dict[tuple, dict] = {}
        self._build_index()
        if learn and cut:
            try:
                self.learn_aliases(cut)
            except Exception as e:  # never crash the matcher
                print("party.learn_aliases failed:", repr(e))

    # ------------------------------------------------------------------ index
    def _build_index(self):
        """token trie per entity: root[entity] -> {'c': {token: node}, 'end': {customer_id,...}}"""
        self.root: dict = {}
        for cid, keys in self.names.items():
            for k in keys:
                self._add_key(cid, k)

    def _add_key(self, cid: str, toks: tuple):
        ent = self.entity_of.get(cid)
        node = self.root.setdefault(ent, {"c": {}, "end": set()})
        for t in toks:
            node = node["c"].setdefault(t, {"c": {}, "end": set()})
        node["end"].add(cid)

    def add_alias(self, cid: str, name_or_tokens):
        toks = tuple(name_or_tokens) if not isinstance(name_or_tokens, str) else name_key_tokens(name_or_tokens)
        if toks and sum(map(len, toks)) >= 6:
            self.names[cid].add(toks)
            self._add_key(cid, toks)

    # ------------------------------------------------------------------ learning (labels: train before cut ONLY)
    def learn_aliases(self, cut: str, lookback_days: int = 400, min_n: int = 2, purity: float = 0.9, verbose: bool = False,
                      rebuild: bool = False):
        """Learned payer evidence from TRAIN golden booked in [cut - lookback, cut) ONLY (no leakage: val -> cut 2025-01-01;
        live/final -> cut 2025-07-01 = all of train).  The table is cached in matching/out/cache/aliases_cut<..>.parquet
        (columns kind, key, customer_id) and only rebuilt when ALIAS_VERSION changes or rebuild=True.

        1. iban: camt counterparty_account -> customer (one customer explains >= 3x the runner-up).
        2. name: token bigrams/trigrams (digit tokens break runs) of bank texts the current name keys did NOT explain (renamed
           customers, other payer names) that point to ONE customer in >= `purity` of the training lines containing them."""
        tag = cut[:4] if cut.endswith("-01-01") else cut
        path = os.path.join(data.CACHE, f"aliases_cut{tag}.parquet")
        tab = None
        if os.path.exists(path) and not rebuild:
            tab = pd.read_parquet(path)
            if "version" not in tab or (len(tab) and tab.version.iloc[0] != ALIAS_VERSION):
                tab = None
        if tab is None:
            tab = self._learn_table(cut, lookback_days, min_n, purity, verbose)
            tab["version"] = ALIAS_VERSION
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tab.to_parquet(path, index=False)
        for kind, key, cid in zip(tab.kind, tab.key, tab.customer_id):
            if cid not in self.entity_of:
                continue
            if kind == "iban":
                self.iban_cust[key].add(cid)
            else:
                self.add_alias(cid, tuple(key.split(" ")))
        if verbose:
            print(f"learn_aliases: {int((tab.kind == 'iban').sum())} iban + {int((tab.kind == 'name').sum())} name aliases ({path})")

    def _learn_table(self, cut, lookback_days, min_n, purity, verbose) -> pd.DataFrame:
        tr = data.phase_dir("train")
        lo = (pd.Timestamp(cut) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        parts = []
        for ch in pd.read_csv(os.path.join(tr, "bank_transactions.csv.gz"), chunksize=200_000,
                              usecols=["bank_txn_id", "booking_date", "amount", "counterparty_account", "counterparty_name", "text", "remittance_info"]):
            ch = ch[(ch.booking_date >= lo) & (ch.booking_date < cut) & (ch.amount > 0)]
            parts.append(ch.drop(columns=["booking_date", "amount"]))
        b = pd.concat(parts, ignore_index=True)
        del parts
        ids = set(b.bank_txn_id)
        skip = "remittance|dispute|dd_|card|cheque|third_party|ap_|fx"
        gs = []
        for ch in pd.read_csv(os.path.join(tr, "golden_allocations.csv.gz"), usecols=["bank_txn_id", "item_id", "segment"], chunksize=500_000):
            ch = ch[ch.item_id.notna() & ch.bank_txn_id.isin(ids)]
            gs.append(ch[~ch.segment.str.contains(skip)][["bank_txn_id", "item_id"]])
        g = pd.concat(gs, ignore_index=True)
        del gs
        ip = train_item_party(set(g.item_id))
        g = g.merge(ip, on="item_id").drop_duplicates(["bank_txn_id", "customer_id"])
        del ip
        g = g[g.groupby("bank_txn_id").customer_id.transform("nunique") == 1][["bank_txn_id", "customer_id"]]
        g = g.merge(b, on="bank_txn_id")
        del b
        out = []
        gi = g[g.counterparty_account.notna()]
        cnt = gi.groupby(["counterparty_account", "customer_id"]).size().reset_index(name="n")
        for ib, grp in cnt.groupby("counterparty_account"):
            grp = grp.sort_values("n", ascending=False)
            if len(grp) == 1 or grp.n.iloc[0] >= 3 * grp.n.iloc[1]:
                out.append(("iban", str(ib).replace(" ", "").upper(), grp.customer_id.iloc[0]))
        from collections import Counter
        cnts: dict = defaultdict(Counter)
        allgrams = []
        for r in g.itertuples(index=False):
            cid = r.customer_id
            ent = self.entity_of.get(cid)
            row = {"text": r.text, "counterparty_name": r.counterparty_name, "remittance_info": r.remittance_info}
            grams = _ngrams(row)
            allgrams.append((ent, cid, grams))
            if any(c.customer_id == cid for c in self.candidates(row, ent)):
                continue
            for key in grams:
                cnts[(ent, key)][cid] += 1
        keys = {k for k, c in cnts.items() if sum(c.values()) >= min_n}
        other: dict = defaultdict(Counter)
        for ent, cid, grams in allgrams:
            for key in grams:
                if (ent, key) in keys:
                    other[(ent, key)][cid] += 1
        for k in keys:
            c = other[k]
            tot = sum(c.values())
            cid, top = c.most_common(1)[0]
            if tot >= min_n and top / tot >= purity:
                out.append(("name", " ".join(k[1]), cid))
        if verbose:
            print(f"_learn_table: {len(g)} train lines in [{lo}, {cut}), {len(keys)} candidate name keys")
        return pd.DataFrame(out, columns=["kind", "key", "customer_id"])

    def _walk(self, root, toks, cands):
        n = len(toks)
        for i in range(n):
            node = root
            j = i
            clen = 0
            while True:
                if node["end"] and j > i:
                    sc = 0.9 if clen >= 12 else 0.7
                    for cid in node["end"]:
                        old = cands.get(cid)
                        if old is None or sc > old.score:
                            cands[cid] = Cand(cid, sc, "name", " ".join(toks[i:j]))
                t = toks[j] if j < n else None
                nxt = node["c"].get(t) if t is not None else None
                if nxt is not None:
                    node = nxt
                    clen += len(t)
                    j += 1
                    continue
                # walk stops: text ended, or the text token is a truncated name token, or the name was cut here.
                cs = None
                if t is not None and len(t) >= 2 and j > i:
                    sub = [ch for k, ch in node["c"].items() if k.startswith(t)]
                    if sub and clen + len(t) >= 8:
                        cs = set()
                        for ch in sub:
                            c2 = self._subtree(ch, 12, include_self=True)
                            if c2 is None:
                                cs = None
                                break
                            cs |= c2
                        clen += len(t)
                        j += 1
                if not cs and clen >= 10 and node["c"] and (t is None or j > i):
                    cs = self._subtree(node, 12)
                if cs:
                    for cid in cs:
                        if cid not in cands:
                            cands[cid] = Cand(cid, 0.45 + min(clen, 25) / 100, "name", " ".join(toks[i:j]) + "~")
                break

    @staticmethod
    def _subtree(node, limit, include_self=False):
        """customers whose key lies in the subtree of `node`; None if more than `limit` (too ambiguous)."""
        out = set(node["end"]) if include_self else set()
        stack = [node]
        while stack:
            nd = stack.pop()
            out |= nd["end"]
            if len(out) > limit:
                return None
            stack.extend(nd["c"].values())
        return out

    # ------------------------------------------------------------------ queries
    def group_members(self, cid: str) -> list[str]:
        """customer itself + customers whose parent is it (+ siblings under the same parent)."""
        out = [cid] + list(self.children.get(cid, []))
        par = self.parent_of.get(cid)
        if isinstance(par, str):
            out += [par] + list(self.children.get(par, []))
        return list(dict.fromkeys(out))

    def candidates(self, row, entity: str | None = None, _alias: bool = True) -> list[Cand]:
        """Customers of the line's entity whose IBAN or name is evidenced in the bank line (best first)."""
        ent = entity or row["entity"]
        cands: dict[str, Cand] = {}
        # --- IBAN
        ibs = ibans_in(row.get("counterparty_account")) + ibans_in(row.get("text")) + ibans_in(row.get("remittance_info"))
        for ib in ibs:
            for cid in self.iban_cust.get(ib, ()):
                if self.entity_of.get(cid) == ent:
                    cands[cid] = Cand(cid, 0.95, "iban", ib)
        # --- names (token trie; fields = counterparty name, remittance info, N43 ' | ' records, text)
        root = self.root.get(ent if _alias else ent)
        if root is not None:
            fields = []
            cn = row.get("counterparty_name")
            if isinstance(cn, str) and cn:
                fields.append(norm(cn).split())
            for c in ("text", "remittance_info"):
                v = row.get(c)
                if isinstance(v, str) and v:
                    parts = v.split(" | ")
                    fields.append(norm(v.replace(" | ", " ")).split() if len(parts) > 1 else norm(v).split())
                    if len(parts) > 1:
                        fields.extend(norm(pt).split() for pt in parts)
            for toks in fields:
                self._walk(root, toks, cands)
        out_all = list(cands.values())
        out = sorted(cands.values(), key=lambda c: -c.score)
        return out
