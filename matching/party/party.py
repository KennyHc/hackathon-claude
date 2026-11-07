"""Who paid: alias keys from bank texts, learned from TRAIN golden (+ masters), resolved to a party or a group.

Keys are length-preserving normalisations (accents stripped, upper case) of the string that follows the bank's
boilerplate prefix (DIRECT CREDIT, /ORDP/, PIX RECEBIDO, TRANSFERENCIA A, …): its prefixes at fixed lengths and at word
boundaries, plus counterparty_name / counterparty IBAN (camt). A key is usable when ≥ PURITY of its training lines map
to the same party (or, failing that, the same customer group).
"""
from __future__ import annotations

import os
import re
import sys
import time
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from integration.lib.masters import read_master  # noqa: E402

DATA = os.environ.get("BANKREC_DATA", os.path.join(ROOT, "data"))
CACHE = os.path.join(ROOT, "matching", "party", "out")
PLENS = (40, 32, 26, 22, 20, 18, 16, 14, 12, 10)
NWORDS = (6, 5, 4, 3, 2)
PURITY = 0.9
MAX_PARTIES = 8

# boilerplate before the payer / beneficiary name (applied on the normalised upper-case text)
_PREFIXES = [
    r"DIRECT CREDIT ", r"ACH CREDIT ", r"ACH DEBIT ", r"/ORDP/", r"/TRTP/[^/]*/NAME/", r"PIX RECEBIDO ",
    r"PIX ENVIADO ", r"LIQUIDACAO BOLETO \S+ ", r"TT IN ", r"TT OUT ", r"FP ", r"BONIFICO DA ",
    r"BONIFICO A FAVORE DI ", r"VIR SEPA RECU /DE ", r"VIR SEPA EMIS /A ", r"TRANSFERENCIAS ", r"TRANSFERENCIA DE ",
    r"TRANSFERENCIA A ", r"TRANSF\. A SU FAVOR \| (?:\d+-)?", r"TRANSFER INMEDIATA \| (?:\d+-)?",
    r"TRANSFERENCIA SEPA \| (?:\d+-)?", r"PRZELEW PRZYCH\. ", r"PRZELEW WYCH\. ", r"SEPA-UEBERWEISUNG AN ",
    r"SPEI ENVIADO ", r"WIRE IN /ORG/", r"WIRE OUT /BNF/", r"PAGO TRANSFERENCIAS \| ", r"TRANSFER\. EN DIV\. (?:\| )?",
]
_PREFIX_RE = re.compile(r"^(?:" + "|".join(_PREFIXES) + ")")
_ORD_RE = re.compile(r"ORD\. (.+)$")
_EXPTE_RE = re.compile(r"EXPTE \S+ (.+)$")
_ABWA_RE = re.compile(r"ABWA\+(.+)$")
_SPEI_RE = re.compile(r"^SPEI RECIBIDO CLAVE \S+ REF \S+ CONCEPTO (.*)$")


def norm(s: str) -> str:
    """Upper case, accents stripped; same length as the input for latin text (so truncation lengths survive)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s.upper())
    return "".join(c for c in s if not unicodedata.combining(c))


def name_strings(text: str, cp_name: str, strip: bool = True) -> list[tuple[str, str]]:
    """[(tag, string)] candidate strings that start with a party name. tag: 'r' bank text, 'c' counterparty."""
    t = norm(text)
    out = []
    for rx, tag in ((_ORD_RE, "o"), (_EXPTE_RE, "o"), (_ABWA_RE, "o")):
        m = rx.search(t)
        if m:
            out.append((tag, m.group(1).strip()))
    m = _SPEI_RE.match(t) if strip else None
    if m:
        out.append(("s", m.group(1).strip()))
    else:
        r = _PREFIX_RE.sub("", t, count=1).strip() if strip else t.strip()
        if r:
            out.append(("r", r))
    if cp_name:
        out.append(("c", norm(cp_name).strip()))
    return out


NO_PREFIX: set = set()  # entities whose bank boilerplate we pretend not to know (LOEO test)


def keys_of(text: str, cp_name: str, cp_acct: str, strip: bool = True) -> list[tuple[str, int]]:
    """Alias keys of a bank line, each with a specificity rank (higher = more specific)."""
    keys = []
    if cp_acct:
        keys.append(("I:" + cp_acct.replace(" ", ""), 100))
    for tag, s in name_strings(text, cp_name, strip):
        if tag == "s":  # SPEI: name at the END of the concept → prefixes from every word position
            w = s.split()
            for i in range(min(len(w), 7)):
                r = " ".join(w[i:])
                for L in PLENS:
                    if len(r) >= L:
                        keys.append((f"P{L}:" + r[:L], L - 1))
                for k in NWORDS:
                    if len(w) - i >= k:
                        keys.append(("PW:" + " ".join(w[i:i + k]), 4 * k - 1))
            continue
        pre = "O" if tag == "o" else "P"
        if tag == "c":
            keys.append(("C:" + s, 90))
        for L in PLENS:
            if len(s) >= L:
                keys.append((f"{pre}{L}:" + s[:L], L))
        w = s.split()
        for k in NWORDS:
            if len(w) >= k:
                keys.append((f"{pre}W:" + " ".join(w[:k]), 4 * k))
    return keys


def master_keys(name: str) -> list[str]:
    s = norm(name).strip()
    if not s:
        return []
    ks = ["C:" + s]
    for L in PLENS:
        if len(s) >= L:
            ks.append(f"P{L}:" + s[:L])
            ks.append(f"O{L}:" + s[:L])
    w = s.split()
    for k in NWORDS:
        if len(w) >= k:
            ks.append("PW:" + " ".join(w[:k]))
            ks.append("OW:" + " ".join(w[:k]))
            ks.append("S:" + " ".join(w[-k:]))
    return ks


# ---------------------------------------------------------------- learning
FACTORY = {"ES01", "ES02", "PT01"}  # ES00 pays some of their supplier invoices (POLICIES §5.2)


def _scopes(entity: str, supplier: bool) -> list[str]:
    return [entity, "ES00"] if supplier and entity in FACTORY else [entity]

def _train_items_party() -> pd.DataFrame:
    out = []
    for f, pid in (("ar_invoices", "customer_id"), ("ap_invoices", "supplier_id"), ("credit_notes", "party_id")):
        out.append(pd.read_csv(os.path.join(DATA, "train", f + ".csv.gz"), dtype=str, keep_default_na=False,
                               usecols=["item_id", pid]).rename(columns={pid: "party_id"}))
    return pd.concat(out, ignore_index=True).drop_duplicates("item_id")


def group_map(phase: str) -> dict:
    cu = read_master("customers.csv", phase)
    g = {}
    for cid, gid, par in zip(cu.customer_id, cu.group_id, cu.parent_customer_id):
        g[cid] = gid or par or cid
    return g


def _stamp(phase: str) -> str:
    """Short hash of the mtimes/sizes of everything the alias table depends on (train data + masters)."""
    import hashlib
    from integration.lib.masters import masters_path
    fs = [os.path.join(DATA, "train", f) for f in ("bank_transactions.csv.gz", "golden_allocations.csv.gz",
                                                    "ar_invoices.csv.gz", "ap_invoices.csv.gz", "credit_notes.csv.gz")]
    fs += [masters_path(n, phase) for n in ("customers.csv", "suppliers.csv")]
    fs += [os.path.join(DATA, phase, "masters_final", n) for n in ("customers.csv", "suppliers.csv")]
    h = hashlib.md5()
    for f in fs:
        if os.path.exists(f):
            st = os.stat(f)
            h.update(f"{f}:{st.st_mtime_ns}:{st.st_size}".encode())
    return h.hexdigest()[:10]


def learn_aliases(phase: str, start: str, cutoff: str, log=print, loeo: str = "") -> pd.DataFrame:
    """key → parties with counts and purity. Train lines with start ≤ booking_date < cutoff (minus entity `loeo`)."""
    os.makedirs(CACHE, exist_ok=True)
    cf = os.path.join(CACHE, f"aliases4_{phase}_{start}_{cutoff}_{loeo or 'all'}_{_stamp(phase)}.parquet")
    if os.path.exists(cf):
        return pd.read_parquet(cf)
    t0 = time.time()
    b = pd.read_csv(os.path.join(DATA, "train", "bank_transactions.csv.gz"), dtype=str, keep_default_na=False,
                    usecols=["bank_txn_id", "booking_date", "entity", "amount", "text", "counterparty_name",
                             "counterparty_account"])
    b = b[(b.booking_date >= start) & (b.booking_date < cutoff)]
    if loeo:
        b = b[b.entity != loeo]
    g = pd.read_csv(os.path.join(DATA, "train", "golden_allocations.csv.gz"), dtype=str, keep_default_na=False,
                    usecols=["bank_txn_id", "item_id"])
    g = g[(g.item_id != "") & g.bank_txn_id.isin(b.bank_txn_id)]
    ip = _train_items_party()
    g = g.merge(ip, on="item_id", how="left")
    lp = g.groupby("bank_txn_id").party_id.agg(["nunique", "first"])
    lp = lp[lp["nunique"] == 1]["first"]
    b = b[b.bank_txn_id.isin(lp.index)].copy()
    b["party_id"] = lp.reindex(b.bank_txn_id).values
    log(f"aliases: {len(b):,} single-party train lines {start}…{cutoff} ({time.time() - t0:.0f}s)")
    gm = group_map(phase)
    rows_k, rows_p, rows_e = [], [], []
    for txt, cpn, cpa, pid, ent in zip(b.text, b.counterparty_name, b.counterparty_account, b.party_id, b.entity):
        for k, _ in keys_of(txt, cpn, cpa):
            rows_k.append(ent + "|" + k)
            rows_p.append(pid)
            rows_e.append(ent)
    # masters: customers and suppliers (name, trade_name, iban, account_number)
    cu = read_master("customers.csv", phase)
    su = read_master("suppliers.csv", phase)
    for df, idc, cols in ((cu, "customer_id", ("name", "trade_name")), (su, "supplier_id", ("name",))):
        for c in cols:
            for pid, nm, ent in zip(df[idc], df[c], df.entity):
                for e2 in _scopes(ent, df is su):
                    for k in master_keys(nm):
                        rows_k.append(e2 + "|" + k)
                        rows_p.append(pid)
                        rows_e.append(ent)
        for c in ("iban", "account_number"):
            if c in df:
                for pid, v, ent in zip(df[idc], df[c], df.entity):
                    if v:
                        for e2 in _scopes(ent, df is su):
                            rows_k.append(e2 + "|I:" + v.replace(" ", ""))
                            rows_p.append(pid)
                            rows_e.append(ent)
    kp = pd.DataFrame({"key": rows_k, "party_id": rows_p})
    log(f"aliases: {len(kp):,} key rows ({time.time() - t0:.0f}s)")
    c = kp.groupby(["key", "party_id"]).size().rename("n").reset_index()
    c["tot"] = c.groupby("key").n.transform("sum")
    c["nparty"] = c.groupby("key").n.transform("size")
    c["purity"] = c.n / c.tot
    out = c[c.nparty <= MAX_PARTIES].reset_index(drop=True)
    out.to_parquet(cf)
    log(f"aliases: {out.key.nunique():,} usable keys, {len(out):,} key-party rows ({time.time() - t0:.0f}s)")
    return out


class Resolver:
    def __init__(self, aliases: pd.DataFrame, group_members: dict):
        a = aliases.sort_values(["key", "n"], ascending=[True, False])
        self.p = {}
        for k, pid, n, pur in zip(a.key.values, a.party_id.values, a.n.values, a.purity.values):
            self.p.setdefault(k, []).append((pid, int(n), float(pur)))
        self.members = group_members

    def add(self, key: str, party_id: str, group_id: str = "", purity: float | None = None):
        """Extra alias (e.g. invoice party_name of the phase's items; harvested keys pass purity=0 so they never
        unlock the weak fits, which need purity ≥ 0.9)."""
        v = self.p.setdefault(key, [])
        if all(x[0] != party_id for x in v) and len(v) < MAX_PARTIES:
            v.append((party_id, 1, 1.0 / (len(v) + 1) if purity is None else purity))

    def resolve(self, entity: str, text: str, cp_name: str, cp_acct: str, cap: int = 6):
        """→ candidate parties [(party_id, rank, purity, key)], most specific key first (empty if none)."""
        hits = []
        for k, rank in keys_of(text, cp_name, cp_acct, strip=entity not in NO_PREFIX):
            v = self.p.get(entity + "|" + k)
            if v:
                hits.append((rank, k, v))
        hits.sort(key=lambda h: -h[0])
        out, seen = [], set()
        for rank, k, v in hits:
            for pid, n, pur in v:
                if pid not in seen:
                    seen.add(pid)
                    out.append((pid, rank, pur, k))
            if len(out) >= cap:
                break
        return out[:cap]


def group_members(phase: str) -> dict:
    gm = group_map(phase)
    mem = defaultdict(list)
    for cid, gid in gm.items():
        mem[gid].append(cid)
    return dict(mem)


# ---------------------------------------------------------------- history-free: master names anywhere in the text
_IBAN_RE = re.compile(r"\b([A-Z]{2}\d{2}[A-Z0-9]{11,30})\b")
MIN_OVERLAP = 14


class Anywhere:
    """Entity-scoped index of party names (masters + phase invoices) by their first two words. A party is a candidate
    when the text, from some word position, shares ≥ min(len(name), MIN_OVERLAP) leading chars with its name."""

    def __init__(self, names):  # iterable of (scope_entity, party_id, name)
        self.ix = defaultdict(list)
        seen = set()
        for ent, pid, nm in names:
            s = norm(nm).strip()
            w = s.split()
            if len(w) < 2 or len(s) < 8 or (ent, pid, s) in seen:
                continue
            seen.add((ent, pid, s))
            self.ix[(ent, w[0] + " " + w[1])].append((pid, s))

    def find(self, entity: str, *texts, cap: int = 6):
        """[(party_id, overlap)] best first."""
        best = {}
        for t in texts:
            t = norm(t).replace(" | ", "").replace("/", " ").replace("+", " ")  # /ORDP/NAME, SVWZ+NAME
            if not t:
                continue
            words = [(m.start(), m.group()) for m in re.finditer(r"\S+", t)]
            for i in range(len(words) - 1):
                v = self.ix.get((entity, words[i][1] + " " + words[i + 1][1]))
                if not v:
                    continue
                rest = t[words[i][0]:]
                for pid, nm in v:
                    need = min(len(nm), MIN_OVERLAP)
                    n = 0
                    for a, b in zip(rest, nm):
                        if a != b:
                            break
                        n += 1
                    if n >= need and n > best.get(pid, 0):
                        best[pid] = n
        return sorted(best.items(), key=lambda x: -x[1])[:cap]


def ibans_in(text: str):
    return _IBAN_RE.findall(norm(text).replace(" | ", ""))
