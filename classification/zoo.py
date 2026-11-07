"""Model zoo + ensemble for p_items (does this bank line settle items?), on top of the category LR.

Why p_items only: the 22-class LR already gets the category of no-item lines right 99.98-99.99 % of the time; the score
levers left for the classifier are all driven by p_items (the integration veto at p_items < 0.2, the UNIDENTIFIED gate
at 0.97). So every zoo member predicts P(ITEM); the category distribution among no-item classes stays the LR's
conditional one: P(cat) = (1 - p_items) * P_lr(cat) / (1 - P_lr(ITEM)).

Members (same sparse binary features as categories._features unless noted):
  lr      the current 22-class LR (categories.py, reference; its P(ITEM))
  lgb     LightGBM binary on the sparse CSR (+ dense extras)
  xgb     XGBoost binary, hist, on the sparse CSR (+ dense extras)
  chr     char n-gram TF-IDF of text | remittance | counterparty (digits -> 0) + the sparse features, binary LR
Dense extras: log |amount|, cents/round flags, day of month, weekday, and history of the counterparty account / name
(prior lines that were ITEM / UNIDENTIFIED / other category), computed ONLY from labelled lines dated before
min(line date, history end) -- training rows see their own past, prediction rows see history up to the cutoff.

Leakage: for a phase with cutoff C, members are trained on lines < C - 6 months, the stacker (LR on member logits) is
fit on [C - 6 months, C), then members are refit on < C and the stacker applied to the phase. Nothing is fit on the
phase window. live/final: C = 2025-07-01 (end of TRAIN), so the stacker slice is TRAIN 2025-01..06.
"""
from __future__ import annotations

import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from classification import categories as C  # noqa: E402
from ingestion import lib  # noqa: E402

ZOO_VERSION = 1
ZDIR = os.path.join(ROOT, "classification", "out", "zoo")
TRAIN_END = "2025-07-01"
TREE_MIN_DF = 20
N_JOBS = 3


class _FitLock:
    """One zoo fit at a time across processes (the locked pipeline run and an off-lock trainer)."""

    def __enter__(self):
        import fcntl
        os.makedirs(ZDIR, exist_ok=True)
        self.f = open(os.path.join(ZDIR, ".fit.lock"), "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def log(*a):
    print("[zoo]", *a, flush=True)


def shift6(d: str) -> str:
    return str((pd.Timestamp(d) - pd.DateOffset(months=6)).date())


def eff_cutoff(phase_or_cut: str) -> str:
    c = C.CUTOFF.get(phase_or_cut, phase_or_cut)
    return min(c, TRAIN_END)


# ------------------------------------------------------------------------------------------ train feature cache
BANK_COLS = ["bank_txn_id", "account_id", "entity", "bank", "statement_format", "booking_date", "amount", "currency",
             "bank_code", "bank_subcode", "ref2", "text", "counterparty_name", "counterparty_account", "end_to_end_id",
             "remittance_info"]


def _labels_light() -> tuple[pd.DataFrame, np.ndarray]:
    """categories._labels with Arrow-backed strings and only the needed columns (much less memory)."""
    for f in ("bank_transactions.csv.gz", "golden_allocations.csv.gz"):              # make sure the parquet caches exist
        if not os.path.exists(os.path.join(lib.CACHE, f"train__{f}.parquet")):
            lib.read_csv(os.path.join(lib.DATA, "train", f))
    b = pd.read_parquet(os.path.join(lib.CACHE, "train__bank_transactions.csv.gz.parquet"), columns=BANK_COLS,
                        dtype_backend="pyarrow")
    b["amount"] = b.amount.astype(float).astype(np.float64)
    b = b.sort_values(["booking_date", "bank_txn_id"], kind="stable").reset_index(drop=True)
    g = pd.read_parquet(os.path.join(lib.CACHE, "train__golden_allocations.csv.gz.parquet"),
                        columns=["bank_txn_id", "item_id", "category"], dtype_backend="pyarrow")
    has_item = g.item_id.ne("").groupby(g.bank_txn_id).any()
    cat = g[g.item_id == ""].drop_duplicates("bank_txn_id").set_index("bank_txn_id").category
    y = pd.Series(np.where(has_item.values, C.ITEM, cat.reindex(has_item.index).fillna(C.ITEM).astype(str).values),
                  index=has_item.index.astype(str))
    del g
    b = b[b.bank_txn_id.isin(y.index)].reset_index(drop=True)
    return b, y.reindex(b.bank_txn_id.astype(str)).values.astype(str)


def _plain(df: pd.DataFrame) -> pd.DataFrame:
    """Arrow strings -> object (what categories._features expects)."""
    out = df.copy()
    for c in out.columns:
        if c != "amount":
            out[c] = out[c].astype(object).where(out[c].notna(), "")
    return out


_TC = {}


def train_cache() -> dict:
    """All TRAIN lines: sparse features over a global vocab (df >= 2), labels, dates; cached once."""
    if "tc" not in _TC:
        if os.path.exists(os.path.join(ZDIR, f"train_feats_v{ZOO_VERSION}.pkl")):
            _TC["tc"] = _train_cache()
        else:
            with _FitLock():
                _TC["tc"] = _train_cache()
    return _TC["tc"]


def _train_cache() -> dict:
    p = os.path.join(ZDIR, f"train_feats_v{ZOO_VERSION}.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    t = time.time()
    bank, y = _labels_light()
    lk = C.Lookups(lib.masters())
    # month by month (pair context features are same-day, so months are independent): keeps memory low
    vocab: dict[str, int] = {}
    parts = []
    month = bank.booking_date.str[:7].values
    for mo in np.unique(month):
        ix = np.where(month == mo)[0]
        feats = C._features(_plain(bank.iloc[ix].reset_index(drop=True)), lk)
        for f in feats:
            for x in f:
                if x not in vocab:
                    vocab[x] = len(vocab)
        parts.append(C._matrix(feats, vocab))
        del feats
    n = len(vocab)
    X = sp.vstack([sp.csr_matrix((q.data, q.indices, q.indptr), shape=(q.shape[0], n)) for q in parts], format="csr")
    del parts
    log(f"features for {len(bank):,} train lines in {time.time() - t:.0f}s ({n:,} raw features)")
    df = np.bincount(X.indices, minlength=n)
    keep = np.where(df >= 2)[0]
    X = X[:, keep].tocsr()
    X.data = X.data.astype(np.float32)
    names = np.empty(n, dtype=object)
    for x, j in vocab.items():
        names[j] = x
    del vocab
    vocab_arr = names[keep]
    from ingestion.run import fx_eur
    bank = _plain(bank[["bank_txn_id", "booking_date", "amount", "currency", "text", "remittance_info",
                        "counterparty_name", "counterparty_account"]])
    bank["amount_eur"] = fx_eur(bank, lib.masters()["fx_rates"]).values
    d = {"X": X, "vocab": vocab_arr, "y": np.asarray(y).astype(str),
         "date": bank.booking_date.values.astype(str), "ids": bank.bank_txn_id.values.astype(str),
         "bank": bank[["bank_txn_id", "booking_date", "amount", "amount_eur", "currency", "text", "remittance_info",
                       "counterparty_name", "counterparty_account"]].reset_index(drop=True)}
    os.makedirs(ZDIR, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(d, f, protocol=4)
    log(f"train cache {X.shape} nnz {X.nnz:,} in {time.time() - t:.0f}s -> {p}")
    return d


# ---------------------------------------------------------------------------------------------- dense features
def _acc_key(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace(" ", "", regex=False)


def _name_key(s: pd.Series) -> pd.Series:
    from ingestion.text import norm_name
    u = s.fillna("").astype(str)
    m = {v: norm_name(v) for v in u.unique()}
    return u.map(m)


def _hist_tables(hb: pd.DataFrame, y: np.ndarray, hist_end: str) -> dict:
    """Per key: cumulative counts (n, item, unid) by date, from labelled TRAIN lines dated < hist_end."""
    keep = hb.booking_date.values < hist_end
    h = pd.DataFrame({"date": pd.to_datetime(hb.booking_date.values[keep]),
                      "item": (y[keep] == C.ITEM).astype(np.int32),
                      "unid": (y[keep] == "UNIDENTIFIED").astype(np.int32)})
    out = {}
    for kname, kser in (("acc", _acc_key(hb.counterparty_account)), ("nam", _name_key(hb.counterparty_name))):
        hk = h.assign(key=kser.values[keep])
        hk = hk[hk.key != ""]
        g = hk.groupby(["key", "date"], sort=True).agg(n=("item", "size"), item=("item", "sum"), unid=("unid", "sum"))
        g = g.groupby(level=0).cumsum().reset_index()
        out[kname] = g
    return out


def dense(bank: pd.DataFrame, tables: dict, hist_end: str) -> np.ndarray:
    """Dense extras for `bank` lines; history = labelled lines dated before min(line date, hist_end)."""
    amt = bank.amount.astype(float).values
    a = np.abs(amt)
    if "amount_eur" in bank.columns:             # currency-blind log amount is wrong for ZAR & co.
        ae = np.abs(pd.to_numeric(bank.amount_eur, errors="coerce").values)
        a = np.where(np.isfinite(ae), ae, a)
    dt = pd.to_datetime(bank.booking_date.values)
    cols = [np.log10(a + 1), (np.round(a * 100) % 100 == 0).astype(float), (a % 100 == 0).astype(float),
            dt.day.values.astype(float), dt.weekday.values.astype(float), np.sign(amt)]
    qd = pd.Series(np.minimum(dt.values, np.datetime64(pd.Timestamp(min(hist_end, "2100-01-01")), "ns")))
    for kname, kser in (("acc", _acc_key(bank.counterparty_account)), ("nam", _name_key(bank.counterparty_name))):
        g = tables[kname]
        q = pd.DataFrame({"key": kser.values, "date": qd.values, "_i": np.arange(len(bank))})
        q = q.sort_values("date")
        m = pd.merge_asof(q, g.sort_values("date"), on="date", by="key", direction="backward", allow_exact_matches=False)
        m = m.sort_values("_i")
        n = m.n.fillna(0).values
        it = m["item"].fillna(0).values
        un = m.unid.fillna(0).values
        has = (kser.values != "").astype(float)
        nc = np.minimum(n, 30.0)                 # capped: early TRAIN rows have little history (covariate shift)
        cols += [has, np.log1p(nc), (it + 0.9) / (n + 1), (un + 0.05) / (n + 1), ((n - it - un) + 0.05) / (n + 1)]
    return np.column_stack(cols).astype(np.float32)


# ------------------------------------------------------------------------------------------------- char text
_DIG = re.compile(r"\d")


def char_docs(bank: pd.DataFrame) -> list[str]:
    s = (bank.text.fillna("").astype(str) + " | " + bank.remittance_info.fillna("").astype(str) + " | "
         + bank.counterparty_name.fillna("").astype(str)).str.upper()
    return [_DIG.sub("0", C._clean(x)) for x in s.values]


# ------------------------------------------------------------------------------------------------- training
def _train_rows(tc: dict, cutoff: str):
    """Same selection/weights as categories.train: ITEM subsampled to 300k (rng 0), weighted to true frequency."""
    keep = tc["date"] < cutoff
    idx = np.where(keep)[0]
    y = tc["y"][idx]
    is_item = y == C.ITEM
    n_item = int(is_item.sum())
    take = np.where(is_item)[0]
    rng = np.random.default_rng(0)
    if n_item > C.N_ITEM_TRAIN:
        take = rng.choice(take, C.N_ITEM_TRAIN, replace=False)
    sel = np.sort(np.concatenate([take, np.where(~is_item)[0]]))
    w = np.where(is_item[sel], n_item / len(take), 1.0)
    return idx[sel], w


def _cols_df(X: sp.csr_matrix, rows: np.ndarray, min_df: int) -> np.ndarray:
    df = np.asarray((X[rows] > 0).sum(axis=0)).ravel()
    return np.where(df >= min_df)[0]


def fit_members(cutoff: str, members=("lgb", "xgb", "chr"), tc: dict | None = None) -> dict:
    """Train the p_items members on TRAIN lines < cutoff. Cached per (cutoff, member, version)."""
    if all(os.path.exists(_mpath(cutoff, m)) for m in members):
        return _fit_members(cutoff, members, tc)
    tc = tc or train_cache()
    with _FitLock():
        return _fit_members(cutoff, members, tc)


def _fit_members(cutoff: str, members, tc: dict | None = None) -> dict:
    os.makedirs(ZDIR, exist_ok=True)
    out = {}
    need = [m for m in members if not os.path.exists(_mpath(cutoff, m))]
    for m in members:
        if m not in need:
            with open(_mpath(cutoff, m), "rb") as f:
                out[m] = pickle.load(f)
    if not need:
        return out
    tc = tc or train_cache()
    rows, w = _train_rows(tc, cutoff)
    yb = (tc["y"][rows] == C.ITEM).astype(np.int32)
    tables = _hist_tables(tc["bank"], tc["y"], cutoff)
    D = dense(tc["bank"].iloc[rows], tables, "9999-12-31")      # training rows: their own past only
    for m in need:
        t = time.time()
        if m in ("lgb", "xgb", "lgbr"):
            r_, w_, y_, D_ = rows, w, yb, D
            if m == "lgbr":      # receipts that are ITEM or UNIDENTIFIED only: the UNIDENTIFIED gate's population
                yr = tc["y"][rows]
                k = (tc["bank"].amount.values[rows] > 0) & ((yr == C.ITEM) | (yr == "UNIDENTIFIED"))
                r_, w_, y_, D_ = rows[k], w[k], yb[k], D[k]
            cols = _cols_df(tc["X"], r_, TREE_MIN_DF)
            Xtr = sp.hstack([tc["X"][r_][:, cols], sp.csr_matrix(D_)], format="csr")
            if m in ("lgb", "lgbr"):
                import lightgbm as lgb
                clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                                         colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=1.0,
                                         n_jobs=N_JOBS, verbose=-1, random_state=0)
            else:
                import xgboost as xgb
                clf = xgb.XGBClassifier(n_estimators=400, learning_rate=0.08, max_depth=8, min_child_weight=2,
                                        subsample=0.8, colsample_bytree=0.5, tree_method="hist", n_jobs=N_JOBS,
                                        random_state=0, eval_metric="logloss")
            clf.fit(Xtr, y_, sample_weight=w_)
            mod = {"kind": m, "cols": tc["vocab"][cols], "clf": clf}
        elif m in ("svm", "cnb"):
            # linear SVM / complement NB on the sparse binary features, sigmoid-calibrated by 3-fold CV inside the
            # training rows (all dated < cutoff)
            from sklearn.calibration import CalibratedClassifierCV
            cols = _cols_df(tc["X"], rows, C.MIN_DF)
            Xtr = tc["X"][rows][:, cols].tocsr()
            if m == "svm":
                from sklearn.svm import LinearSVC
                base = LinearSVC(C=0.1, loss="squared_hinge", dual=True, max_iter=2000, tol=1e-4, random_state=0)
            else:
                from sklearn.naive_bayes import ComplementNB
                base = ComplementNB(alpha=0.5)
            clf = CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=1)
            clf.fit(Xtr, yb, sample_weight=w)
            mod = {"kind": m, "cols": tc["vocab"][cols], "clf": clf}
        elif m == "chr":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            docs = char_docs(tc["bank"].iloc[rows])
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5, max_features=200_000,
                                  sublinear_tf=True, dtype=np.float32)
            T = vec.fit_transform(docs)
            cols = _cols_df(tc["X"], rows, C.MIN_DF)
            Xtr = sp.hstack([T, tc["X"][rows][:, cols]], format="csr")
            clf = LogisticRegression(C=4.0, max_iter=300, tol=1e-4)
            clf.fit(Xtr, yb, sample_weight=w)
            mod = {"kind": m, "cols": tc["vocab"][cols], "vec": vec, "clf": clf}
        else:
            raise ValueError(m)
        log(f"{m} cutoff {cutoff}: {Xtr.shape[0]:,} rows fitted in {time.time() - t:.0f}s")
        with open(_mpath(cutoff, m), "wb") as f:
            pickle.dump(mod, f, protocol=4)
        out[m] = mod
    return out


def _mpath(cutoff: str, m: str) -> str:
    return os.path.join(ZDIR, f"{m}_v{ZOO_VERSION}_{cutoff[:10]}.pkl")


def _remap(X: sp.csr_matrix, vocab_from: np.ndarray, vocab_to: np.ndarray) -> sp.csr_matrix:
    """Columns of X (over vocab_from) re-indexed to vocab_to (absent -> dropped)."""
    pos = {x: j for j, x in enumerate(vocab_to)}
    m = np.array([pos.get(x, -1) for x in vocab_from])
    X = X.tocoo()
    k = m[X.col] >= 0
    return sp.csr_matrix((X.data[k], (X.row[k], m[X.col[k]])), shape=(X.shape[0], len(vocab_to)))


def predict_member(mod: dict, Xs: sp.csr_matrix, vocab: np.ndarray, D: np.ndarray, bank: pd.DataFrame) -> np.ndarray:
    Xm = _remap(Xs, vocab, mod["cols"])
    if mod["kind"] == "chr":
        Xm = sp.hstack([mod["vec"].transform(char_docs(bank)), Xm], format="csr")
    elif mod["kind"] in ("svm", "cnb"):
        pass
    else:
        Xm = sp.hstack([Xm, sp.csr_matrix(D)], format="csr")
    return mod["clf"].predict_proba(Xm)[:, 1]


# ------------------------------------------------------------------------------------------------- LR via cache
def lr_model(cutoff: str) -> dict:
    """The reference 22-class LR for `cutoff` (categories.py's own cached model when it exists)."""
    return C.load_model(cutoff)


def lr_proba(model: dict, Xs: sp.csr_matrix, vocab: np.ndarray) -> np.ndarray:
    Xm = _remap(Xs, vocab, model["vocab"])
    z = (Xm @ model["coef"].astype(np.float64).T) + model["intercept"].astype(np.float64)
    z = np.asarray(z)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def lr_train_cached(cutoff: str, tc: dict) -> dict:
    """Same as categories.train(cutoff) but from the train feature cache (fast); used for the stacker slice."""
    p = os.path.join(ZDIR, f"lr_v{ZOO_VERSION}_{cutoff[:10]}.npz")
    if not os.path.exists(p):
        with _FitLock():
            return _lr_train_cached(cutoff, tc, p)
    return _lr_train_cached(cutoff, tc, p)


def _lr_train_cached(cutoff: str, tc: dict, p: str) -> dict:
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    from sklearn.linear_model import LogisticRegression
    rows, w = _train_rows(tc, cutoff)
    cols = _cols_df(tc["X"], rows, C.MIN_DF)
    t = time.time()
    clf = LogisticRegression(C=4.0, max_iter=300, tol=1e-4)
    clf.fit(tc["X"][rows][:, cols], tc["y"][rows], sample_weight=w)
    m = {"vocab": tc["vocab"][cols].astype(str), "coef": clf.coef_.astype(np.float32), "intercept": clf.intercept_.astype(np.float32),
         "classes": clf.classes_.astype(str)}
    log(f"lr cutoff {cutoff} fitted in {time.time() - t:.0f}s")
    np.savez_compressed(p, **m)
    return m


# ------------------------------------------------------------------------------------------------ the ensemble
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def member_probs(cutoff: str, members, Xs, vocab, bank, lr: dict, tc: dict | None = None) -> dict:
    """{name: P(ITEM)} for the lines of `bank` from members trained < cutoff (history up to cutoff)."""
    tc_ = tc
    out = {"lr": lr_proba(lr, Xs, vocab)[:, list(lr["classes"]).index(C.ITEM)]}
    if members:
        mods = fit_members(cutoff, members, tc_)
        tc_ = tc_ or train_cache()
        tables = _hist_tables(tc_["bank"], tc_["y"], cutoff)
        D = dense(bank, tables, cutoff)
        for m in members:
            out[m] = predict_member(mods[m], Xs, vocab, D, bank)
    return out


def pop_mask(P_lr: np.ndarray, classes: list, amount: np.ndarray) -> np.ndarray:
    """The UNIDENTIFIED gate's population: receipts whose best no-item class is UNIDENTIFIED."""
    ji = classes.index(C.ITEM)
    Pn = P_lr.copy()
    Pn[:, ji] = -1
    return (np.asarray(amount) > 0) & (np.array(classes)[Pn.argmax(axis=1)] == "UNIDENTIFIED")


def fit_stacker(cutoff: str, members, tc: dict, pop: bool = False) -> dict:
    """Stacker (binary LR on member logits) fit on TRAIN [cutoff - 6 months, cutoff) with members trained before it.
    pop=True: fit only on the UNIDENTIFIED-gate population of the slice (applied only to that population)."""
    p = os.path.join(ZDIR, f"stack_v{ZOO_VERSION}_{cutoff[:10]}_{'-'.join(members)}{'_pop' if pop else ''}.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    from sklearn.linear_model import LogisticRegression
    end = eff_cutoff(cutoff)
    c0 = shift6(end)
    sl = np.where((tc["date"] >= c0) & (tc["date"] < end))[0]
    bank = tc["bank"].iloc[sl].reset_index(drop=True)
    lr = C.load_model(c0) if os.path.exists(C._model_path(c0)) else lr_train_cached(c0, tc)
    pm = member_probs(c0, members, tc["X"][sl], tc["vocab"], bank, lr, tc)
    y = (tc["y"][sl] == C.ITEM).astype(int)
    names = ["lr", *members]
    Z = np.column_stack([_logit(pm[n]) for n in names])
    k = np.ones(len(y), bool)
    if pop:
        k = pop_mask(lr_proba(lr, tc["X"][sl], tc["vocab"]), list(lr["classes"]), bank.amount.values)
    st = LogisticRegression(C=1.0, max_iter=1000).fit(Z[k], y[k])
    res = {"names": names, "coef": st.coef_.ravel(), "intercept": float(st.intercept_[0]), "slice": (c0, end),
           "slice_probs": {n: pm[n] for n in names}, "slice_y": y, "slice_ids": tc["ids"][sl], "slice_pop": k}
    log(f"stacker {cutoff}{' pop' if pop else ''} on [{c0},{end}) n={int(k.sum()):,}: "
        f"{dict(zip(names, np.round(res['coef'], 3)))} b={res['intercept']:.3f}")
    with open(p, "wb") as f:
        pickle.dump(res, f, protocol=4)
    return res


def qmap_fit(src: np.ndarray, ref: np.ndarray) -> dict:
    """Monotone map sending the slice distribution of `src` onto that of `ref` (keeps the LR's operating points
    at the fixed integration thresholds 0.2 / 0.97 while using the new ranking)."""
    qs = np.linspace(0, 1, 2001)
    return {"x": np.quantile(src, qs), "y": np.quantile(ref, qs)}


def qmap_apply(q: dict, p: np.ndarray) -> np.ndarray:
    x, y = q["x"], q["y"]
    x = x + np.arange(len(x)) * 1e-12            # strictly increasing for interp
    return np.interp(p, x, y)


def apply_stacker(st: dict, pm: dict) -> np.ndarray:
    z = np.column_stack([_logit(pm[n]) for n in st["names"]]) @ st["coef"] + st["intercept"]
    return 1 / (1 + np.exp(-z))


def compose(P_lr: np.ndarray, classes: list, p_items: np.ndarray) -> np.ndarray:
    """22-class matrix with P(ITEM) = p_items and the LR's conditional no-item distribution."""
    ji = classes.index(C.ITEM)
    P = P_lr.astype(np.float64).copy()
    P[:, ji] = 0.0
    rest = P.sum(axis=1)
    P /= np.where(rest > 0, rest, 1.0)[:, None]          # conditional no-item distribution (sums to 1)
    P *= (1 - p_items)[:, None]
    P[:, ji] = p_items
    return P
