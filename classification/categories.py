"""Category model for bank lines (issue #20): which lines settle no item, and their no-item category.

    label(bank, phase) -> DataFrame[bank_txn_id, p_items, category, exception, confidence, method]

One multinomial logistic regression over 22 classes (ITEM + the 21 categories of masters/categories.csv) on sparse
binary features: bank code / subcode (camt and N43 views, so Banco del Moncayo's N43 -> camt.053 switch keeps its N43
code), sign, amount shape, text / remittance / counterparty tokens, counterparty type (own account of the same or another
entity, customer, supplier, group entity) and same-day / pair context (opposite movement on another own account, bank
error + "ANULACION OPERACION INDEBIDA"). Trained on TRAIN golden (ITEM = line with item rows), time-split by phase to avoid
leakage; cached under classification/out/model_<cutoff>.npz.

Outputs: p_items = P(ITEM); category = best no-item class; confidence = P(category) (joint, so confidence >= 0.9 implies
p_items <= 0.1); exception = UNIDENTIFIED iff category == UNIDENTIFIED. A deterministic rule pairs bank errors with their
"ANULACION OPERACION INDEBIDA" reversal (both lines BANK_REVERSAL).
"""
from __future__ import annotations

import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from ingestion import lib  # noqa: E402
from ingestion.text import norm_name, strip_accents  # noqa: E402

CUTOFF = {"sample": "2025-06-01", "val": "2025-01-01", "hold": "2024-01-01", "live": "9999-12-31", "final": "9999-12-31"}
ITEM = "ITEM"
MODEL_VERSION = 1
N_ITEM_TRAIN = 300_000          # ITEM lines subsampled for training (re-weighted to their true frequency)
MIN_DF = 3

_WORD = re.compile(r"[A-Z0-9]+")


# ----------------------------------------------------------------------------------------------------------- features
def _shape(tok: str) -> str:
    """'PF24' -> 'PF#', '2025022' -> '#', 'AB12CD' -> 'AB#CD#' (letters kept, digit runs collapsed)."""
    return re.sub(r"\d+", "#", tok)


def _clean(s: str) -> str:
    return strip_accents(str(s).upper().replace("ß", "SS"))


class Lookups:
    """Counterparty knowledge from the masters."""

    def __init__(self, m: dict[str, pd.DataFrame]):
        ba = m["bank_accounts"]
        self.acc_entity = dict(zip(ba.account_id, ba.entity))
        self.own_iban = {}
        self.own_last4 = defaultdict(set)
        for _, r in ba.iterrows():
            for v in (r.iban, r.account_number):
                v = str(v).replace(" ", "")
                if v:
                    self.own_iban[v] = r.entity
                    self.own_last4[v[-4:]].add(r.entity)
        self.entity_codes = set(m["entities"].entity)
        ent_names = {norm_name(x) for x in m["entities"].legal_name}
        cu, su = m["customers"], m["suppliers"]
        self.cust_iban = {v.replace(" ", "") for v in cu.iban if v}
        self.supp_iban = {v.replace(" ", "") for v in su.iban if v} | {v for v in su.account_number if v}
        self.name_type: dict[str, set] = defaultdict(set)
        self.bigram_type: dict[str, set] = defaultdict(set)
        for typ, names in (("CUST", list(cu.name) + [x for x in cu.trade_name if x]), ("SUPP", list(su.name)),
                           ("ENT", list(m["entities"].legal_name))):
            for nm in names:
                nn = norm_name(nm)
                if not nn:
                    continue
                self.name_type[nn].add(typ)
                t = nn.split()
                if len(t) >= 2:
                    self.bigram_type[t[0] + " " + t[1]].add(typ)
        self.ent_names = ent_names


def _features(bank: pd.DataFrame, lk: Lookups) -> list[list[str]]:
    """Sparse binary feature names per bank line (raw bank_transactions columns only)."""
    n = len(bank)
    amt = bank.amount.astype(float).values
    sgn = np.where(amt > 0, "P", "N")
    ctx = _context(bank, lk)
    cols = {c: bank[c].fillna("").astype(str).values for c in
            ["bank", "bank_code", "bank_subcode", "statement_format", "text", "remittance_info", "counterparty_name",
             "counterparty_account", "end_to_end_id", "ref2", "entity", "account_id", "currency"]}
    cache_text: dict[str, list[str]] = {}
    cache_name: dict[str, list[str]] = {}
    out = []
    for i in range(n):
        s = sgn[i]
        bk, code, sub = cols["bank"][i], cols["bank_code"][i], cols["bank_subcode"][i]
        f = [f"S:{s}", f"FMT:{cols['statement_format'][i]}|{s}", f"CUR:{cols['currency'][i]}|{s}"]
        # bank codes: camt (A/B/C), N43 (family + subcode; camt.053 Moncayo keeps the N43 code in bank_subcode)
        n43fam = n43sub = ""
        if "/" in code:
            parts = code.split("/")
            f += [f"C:{code}|{s}", f"C2:{parts[0]}/{parts[1]}|{s}", f"C3:{parts[-1]}|{s}"]
            if len(sub) == 5 and sub.isdigit():
                n43fam, n43sub = sub[:2], sub[2:]
        elif code.isdigit():
            n43fam, n43sub = code, sub
        else:
            f += [f"C:{code}|{s}", f"CB:{bk}|{code}|{s}"]          # MT940 transaction type (NTRF, NMSC...)
        if n43fam:
            f.append(f"F:{n43fam}|{s}")
            if n43sub and n43sub != "000":
                f.append(f"FS:{bk}|{n43fam}|{n43sub}|{s}")
        # amount shape
        a = abs(amt[i])
        f.append(f"AMT:{int(np.log10(a + 1) * 2)}|{s}")
        if a >= 1000 and a % 1000 == 0:
            f.append(f"ROUND1000|{s}")
        elif a >= 100 and a % 100 == 0:
            f.append(f"ROUND100|{s}")
        elif a % 1 == 0:
            f.append(f"ROUND1|{s}")
        # text tokens (+ bigrams), remittance tokens
        txt = cols["text"][i]
        tt = cache_text.get(txt)
        if tt is None:
            w = [_shape(t) for t in _WORD.findall(_clean(txt))]
            w = [t for t in w if len(t) >= 2 or t == "#"]
            tt = [f"T:{t}" for t in set(w)] + [f"TB:{a_}_{b_}" for a_, b_ in set(zip(w, w[1:]))]
            if not w:
                tt = ["T:<empty>"]
            cache_text[txt] = tt
        f += [x + "|" + s for x in tt]
        rem = cols["remittance_info"][i]
        if rem:
            rt = cache_text.get("R" + rem)
            if rt is None:
                w = [_shape(t) for t in _WORD.findall(_clean(rem))]
                rt = [f"R:{t}" for t in set(w) if len(t) >= 2 or t == "#"]
                cache_text["R" + rem] = rt
            f += [x + "|" + s for x in rt]
        else:
            f.append(f"R:<empty>|{s}")
        # counterparty name tokens + type
        cpn = cols["counterparty_name"][i]
        nt = cache_name.get(cpn)
        if nt is None:
            nn = norm_name(cpn)
            nt = [f"N:{t}" for t in set(nn.split())] if nn else ["N:<empty>"]
            typ = lk.name_type.get(nn, set()) if nn else set()
            nt += [f"NT:{t}" for t in typ]
            cache_name[cpn] = nt
        f += [x + "|" + s for x in nt]
        e2e = cols["end_to_end_id"][i]
        f.append(f"E2E:{'empty' if not e2e else e2e if e2e in ('NOTPROVIDED', 'NONREF') else _shape(e2e)[:6]}|{s}")
        f += [x + "|" + s for x in ctx[i]]
        out.append(f)
    return out


def _context(bank: pd.DataFrame, lk: Lookups) -> list[list[str]]:
    """Counterparty-type and pair features."""
    n = len(bank)
    res: list[list[str]] = [[] for _ in range(n)]
    ent = bank.entity.values
    ca = bank.counterparty_account.fillna("").str.replace(" ", "", regex=False).values
    for i in range(n):
        c = ca[i]
        if not c:
            continue
        if c in lk.own_iban:
            res[i].append("CP:OWN_SAME" if lk.own_iban[c] == ent[i] else "CP:OWN_OTHER")
        elif c in lk.cust_iban:
            res[i].append("CP:CUST_IBAN")
        elif c in lk.supp_iban:
            res[i].append("CP:SUPP_IBAN")
        else:
            res[i].append("CP:OTHER_IBAN")
    # text mentions: own account last-4 / entity code / group or party name bigram
    texts = (bank.text.fillna("") + " " + bank.remittance_info.fillna("")).values
    cache: dict[tuple, list[str]] = {}
    for i in range(n):
        key = (texts[i], ent[i])
        r = cache.get(key)
        if r is None:
            u = _clean(texts[i])
            toks = _WORD.findall(u)
            r = []
            for t in toks:
                if len(t) == 4 and t.isdigit() and t in lk.own_last4:
                    r.append("TX:OWNACC_SAME" if ent[i] in lk.own_last4[t] else "TX:OWNACC_OTHER")
                elif t in lk.entity_codes:
                    r.append("TX:ENT_SAME" if t == ent[i] else "TX:ENT_OTHER")
            nn = norm_name(texts[i]).split()
            for a_, b_ in zip(nn, nn[1:]):
                for typ in lk.bigram_type.get(a_ + " " + b_, ()):
                    r.append(f"TX:NAME_{typ}")
            r = sorted(set(r))
            cache[key] = r
        res[i] += r
    # opposite movement on another own account the same day (internal transfer / cash pool / IC / FX legs)
    d = pd.DataFrame({"i": np.arange(n), "date": bank.booking_date.values, "amt": bank.amount.round(2).values,
                      "acc": bank.account_id.values, "ent": ent, "cur": bank.currency.values})
    opp = d.assign(amt=-d.amt).merge(d, on=["date", "amt"], suffixes=("", "_o"))
    opp = opp[opp.acc != opp.acc_o]
    for i, same in opp.groupby("i").apply(lambda g: (g.ent == g.ent_o).any(), include_groups=False).items():
        res[i].append("PAIR:OPP_SAME_ENT" if same else "PAIR:OPP_OTHER_ENT")
    # FX legs: opposite-sign line in another currency, same entity, same day
    big = d[d.amt.abs() >= 10_000]
    fx = big.merge(big, on=["date", "ent"], suffixes=("", "_o"))
    fx = fx[(fx.cur != fx.cur_o) & (np.sign(fx.amt) == -np.sign(fx.amt_o))]
    for i in fx.i.unique():
        res[i].append("PAIR:FXLEG")
    return res


# ---------------------------------------------------------------------------------------------------- bank reversals
_ANUL = re.compile(r"ANULACION OPERACION INDEBIDA N\s*(\d{8})")


def reversal_pairs(bank: pd.DataFrame) -> set[str]:
    """Lines of bank error + 'ANULACION OPERACION INDEBIDA N <yyyymmdd...>' reversal pairs (both BANK_REVERSAL)."""
    txt = bank.text.fillna("").map(_clean).str.replace(r"\s*\|\s*", "", regex=True)   # N43 wraps at 38 chars
    m = txt.str.extract(_ANUL)[0]
    out = set()
    rev = bank[m.notna()]
    if rev.empty:
        return out
    by_acc_amt = defaultdict(list)
    for tid, acc, amt, dt in zip(bank.bank_txn_id, bank.account_id, bank.amount.round(2), bank.booking_date):
        by_acc_amt[(acc, amt)].append((dt, tid))
    for idx, r in rev.iterrows():
        out.add(r.bank_txn_id)
        ds = m[idx]
        orig_date = f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}"
        cands = [(dt, tid) for dt, tid in by_acc_amt.get((r.account_id, round(-r.amount, 2)), [])
                 if tid != r.bank_txn_id and dt <= r.booking_date and tid not in out]
        if cands:
            exact = [c for c in cands if c[0] == orig_date]
            out.add((exact or sorted(cands, key=lambda c: abs(pd.Timestamp(c[0]) - pd.Timestamp(orig_date)))) [0][1])
    return out


# -------------------------------------------------------------------------------------------------------- training
def _labels() -> tuple[pd.DataFrame, pd.Series]:
    bank = lib.bank_of("train")
    g = lib.golden_train()
    has_item = g.item_id.ne("").groupby(g.bank_txn_id).any()
    cat = g[g.item_id == ""].drop_duplicates("bank_txn_id").set_index("bank_txn_id").category
    y = pd.Series(np.where(has_item, ITEM, cat.reindex(has_item.index).fillna(ITEM)), index=has_item.index)
    bank = bank[bank.bank_txn_id.isin(y.index)].reset_index(drop=True)
    return bank, y.reindex(bank.bank_txn_id).values


def _matrix(feats: list[list[str]], vocab: dict[str, int]) -> sp.csr_matrix:
    indptr, idx = [0], []
    for f in feats:
        idx.extend(j for j in (vocab.get(x) for x in f) if j is not None)
        indptr.append(len(idx))
    data = np.ones(len(idx), dtype=np.float32)
    X = sp.csr_matrix((data, np.array(idx, dtype=np.int32), np.array(indptr, dtype=np.int64)),
                      shape=(len(feats), len(vocab)))
    X.sum_duplicates()
    X.data[:] = 1.0
    return X


def train(cutoff: str, verbose: bool = True) -> dict:
    from sklearn.linear_model import LogisticRegression
    t = time.time()
    bank, y = _labels()
    keep = (bank.booking_date < cutoff).values
    bank, y = bank[keep].reset_index(drop=True), y[keep]
    rng = np.random.default_rng(0)
    is_item = y == ITEM
    n_item = int(is_item.sum())
    take = np.where(is_item)[0]
    if n_item > N_ITEM_TRAIN:
        take = rng.choice(take, N_ITEM_TRAIN, replace=False)
    sel = np.sort(np.concatenate([take, np.where(~is_item)[0]]))
    w = np.where(is_item[sel], n_item / len(take), 1.0)
    lk = Lookups(lib.masters())
    # context features need the full day of lines (pairs) -> compute on everything, then select
    feats_all = _features(bank, lk)
    feats = [feats_all[i] for i in sel]
    del feats_all
    df = defaultdict(int)
    for f in feats:
        for x in set(f):
            df[x] += 1
    vocab = {x: j for j, x in enumerate(sorted(k for k, v in df.items() if v >= MIN_DF))}
    X = _matrix(feats, vocab)
    ys = y[sel]
    if verbose:
        print(f"[categories] train cutoff {cutoff}: {X.shape[0]:,} rows x {X.shape[1]:,} feats "
              f"({time.time() - t:.0f}s features)", flush=True)
    clf = LogisticRegression(C=4.0, max_iter=300, tol=1e-4)
    clf.fit(X, ys, sample_weight=w)
    if verbose:
        print(f"[categories] fitted in {time.time() - t:.0f}s", flush=True)
    return {"vocab": np.array(sorted(vocab, key=vocab.get)), "coef": clf.coef_.astype(np.float32),
            "intercept": clf.intercept_.astype(np.float32), "classes": clf.classes_.astype(str)}


def _model_path(cutoff: str) -> str:
    return os.path.join(ROOT, "classification", "out", f"model_v{MODEL_VERSION}_{cutoff[:10]}.npz")


def load_model(cutoff: str) -> dict:
    p = _model_path(cutoff)
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    m = train(cutoff)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, **m)
    return m


def predict_proba(model: dict, bank: pd.DataFrame, lk: Lookups) -> np.ndarray:
    vocab = {x: j for j, x in enumerate(model["vocab"])}
    X = _matrix(_features(bank, lk), vocab)
    z = X @ model["coef"].T + model["intercept"]
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------------- keyword rules
# Rare categories the model has (almost) no examples of; (regex on cleaned text+remittance, sign, category)
RULES = [
    (re.compile(r"DEVOLUCION AEAT|DEVOLUCION (DE )?IMPUESTOS|TAX REFUND|VAT REFUND|REMBOURSEMENT (DE )?TVA|"
                r"STEUERERSTATTUNG|RIMBORSO IVA"), 1, "TAX_REFUND"),
    (re.compile(r"DISPOSICION (POLIZA|PRESTAMO)|LOAN DRAWDOWN"), 1, "LOAN_DRAWDOWN"),
]


def keyword_rules(bank: pd.DataFrame) -> pd.Series:
    txt = (bank.text.fillna("") + " " + bank.remittance_info.fillna("")).map(_clean).values
    sgn = np.sign(bank.amount.values)
    out = np.full(len(bank), "", dtype=object)
    for rx, s, cat in RULES:
        hit = np.array([bool(rx.search(t)) for t in txt]) & (sgn == s) & (out == "")
        out[hit] = cat
    return pd.Series(out, index=bank.index)


# ----------------------------------------------------------------------------------------------------------- label
def label(bank: pd.DataFrame, phase: str) -> pd.DataFrame:
    model = load_model(CUTOFF[phase])
    folder = lib.PHASES[phase][0]
    lk = Lookups(lib.masters(folder))
    P = predict_proba(model, bank, lk)
    classes = list(model["classes"])
    ji = classes.index(ITEM)
    p_items = P[:, ji]
    Pn = P.copy()
    Pn[:, ji] = -1
    best = Pn.argmax(axis=1)
    cat = np.array(classes)[best]
    conf = P[np.arange(len(P)), best]
    method = np.full(len(bank), "lr", dtype=object)
    # keyword rules for rare categories
    kw = keyword_rules(bank).values
    hit = kw != ""
    cat = np.where(hit, kw, cat)
    p_items = np.where(hit, np.minimum(p_items, 0.05), p_items)
    conf = np.where(hit, np.maximum(conf, 0.95), conf)
    method[hit] = "rule_kw"
    # bank error + its reversal
    rev = bank.bank_txn_id.isin(reversal_pairs(bank)).values
    cat = np.where(rev, "BANK_REVERSAL", cat)
    p_items = np.where(rev, 0.01, p_items)
    conf = np.where(rev, 0.99, conf)
    method[rev] = "reversal_pair"
    return pd.DataFrame({
        "bank_txn_id": bank.bank_txn_id.values,
        "p_items": p_items.round(4),
        "category": cat,
        "exception": np.where(cat == "UNIDENTIFIED", "UNIDENTIFIED", ""),
        "confidence": conf.round(4),
        "method": method,
    })
