"""classification.categories.features: Feature extraction pipeline for bank lines.

Extracts:
1. Word TF-IDF (1-2 gram) on combined text + remittance_info + counterparty_name.
2. Character n-gram TF-IDF (2-5 gram char_wb) on combined text.
3. One-hot encoding on bank_code, bank_subcode, currency, entity, statement_format.
4. Binned and continuous log|amount| features + sign indicator.
5. Counterparty priors: historical settles_items probability and 21-category distribution
   per counterparty account / normalized name, learned strictly on training rows.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from .rules import ALL_CATEGORIES

# Common corporate legal suffixes to strip during normalization
LEGAL_SUFFIXES_RE = re.compile(
    r"\b(SL|SA|SLU|SAU|GMBH|LLC|INC|CORP|PTY|LTD|EURL|SARL|SP|Z|OO|BV|NV|CO|KG|SCOOP|EIREL)\b",
    re.IGNORECASE,
)
NON_ALPHANUM_RE = re.compile(r"[^A-Z0-9\s]")


def normalize_name(name: str) -> str:
    """Normalizes a company or counterparty name."""
    if not name or not isinstance(name, str):
        return ""
    name_clean = name.upper()
    name_clean = NON_ALPHANUM_RE.sub(" ", name_clean)
    name_clean = LEGAL_SUFFIXES_RE.sub(" ", name_clean)
    return " ".join(name_clean.split())


def extract_cp_keys(df: pd.DataFrame) -> np.ndarray:
    """Extracts unique counterparty identifiers from account or normalized name."""
    accounts = df["counterparty_account"].fillna("").astype(str).values if "counterparty_account" in df.columns else np.full(len(df), "")
    names = df["counterparty_name"].fillna("").astype(str).values if "counterparty_name" in df.columns else np.full(len(df), "")

    keys = []
    for acct, name in zip(accounts, names):
        acct = acct.strip()
        name = name.strip()
        if acct:
            keys.append("A:" + acct.upper())
        elif name:
            n = normalize_name(name)
            keys.append("N:" + n if n else "")
        else:
            keys.append("")
    return np.array(keys, dtype=object)


class CounterpartyPriorEncoder:
    """Computes empirical counterparty priors from training lines with Laplace smoothing."""

    def __init__(self, alpha: float = 5.0):
        self.alpha = alpha
        self.global_p_settles: float = 0.90
        self.global_cat_priors: np.ndarray = np.zeros(len(ALL_CATEGORIES), dtype=np.float32)
        self.cat_to_idx: Dict[str, int] = {c: i for i, c in enumerate(ALL_CATEGORIES)}
        self.p_settles_map: Dict[str, float] = {}
        self.cat_priors_map: Dict[str, np.ndarray] = {}

    def fit(self, df: pd.DataFrame, y_settles: np.ndarray, y_category: np.ndarray) -> "CounterpartyPriorEncoder":
        keys = extract_cp_keys(df)

        # Global estimates
        self.global_p_settles = float(np.mean(y_settles)) if len(y_settles) > 0 else 0.90

        no_item_mask = (y_settles == 0)
        cat_counts = np.zeros(len(ALL_CATEGORIES), dtype=np.float32)
        for cat in y_category[no_item_mask]:
            if cat in self.cat_to_idx:
                cat_counts[self.cat_to_idx[cat]] += 1.0
        total_no_item = cat_counts.sum()
        self.global_cat_priors = (cat_counts + 1.0) / (total_no_item + len(ALL_CATEGORIES))

        # Group by counterparty key
        non_empty = np.where(keys != "")[0]
        if len(non_empty) == 0:
            return self

        df_sub = pd.DataFrame({
            "key": keys[non_empty],
            "settles": y_settles[non_empty],
            "category": y_category[non_empty],
        })

        for key, grp in df_sub.groupby("key"):
            k = str(key)
            n_tot = len(grp)
            n_settles = grp["settles"].sum()
            p_s = (n_settles + self.alpha * self.global_p_settles) / (n_tot + self.alpha)
            self.p_settles_map[k] = float(p_s)

            # Category shares among no-item instances
            c_counts = np.zeros(len(ALL_CATEGORIES), dtype=np.float32)
            grp_no_item = grp[grp["settles"] == 0]
            for c in grp_no_item["category"]:
                if c in self.cat_to_idx:
                    c_counts[self.cat_to_idx[c]] += 1.0

            n_no_item = len(grp_no_item)
            c_priors = (c_counts + self.alpha * self.global_cat_priors) / (n_no_item + self.alpha)
            self.cat_priors_map[k] = c_priors

        return self

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        keys = extract_cp_keys(df)
        n = len(df)
        p_settles_col = np.empty((n, 1), dtype=np.float32)
        cat_matrix = np.empty((n, len(ALL_CATEGORIES)), dtype=np.float32)

        for i, key in enumerate(keys):
            if key and key in self.p_settles_map:
                p_settles_col[i, 0] = self.p_settles_map[key]
                cat_matrix[i, :] = self.cat_priors_map[key]
            else:
                p_settles_col[i, 0] = self.global_p_settles
                cat_matrix[i, :] = self.global_cat_priors

        features = np.hstack([p_settles_col, cat_matrix])
        return csr_matrix(features, dtype=np.float32)


class FeaturePipeline:
    """End-to-end feature extraction pipeline producing a unified sparse matrix."""

    AMOUNT_BINS = [0.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 25000.0, 50000.0, 100000.0, 500000.0, 1e9]
    CAT_COLUMNS = ["bank_code", "bank_subcode", "currency", "entity", "statement_format"]

    def __init__(
        self,
        max_word_features: int = 15000,
        max_char_features: int = 25000,
    ):
        self.max_word_features = max_word_features
        self.max_char_features = max_char_features

        self.vec_word = TfidfVectorizer(
            max_features=self.max_word_features,
            ngram_range=(1, 2),
            min_df=3,
            dtype=np.float32,
        )
        self.vec_char = TfidfVectorizer(
            max_features=self.max_char_features,
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=5,
            dtype=np.float32,
        )
        self.ohe_cat = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
        self.ohe_bins = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
        self.cp_encoder = CounterpartyPriorEncoder(alpha=5.0)

    @staticmethod
    def _prepare_text(df: pd.DataFrame) -> pd.Series:
        text = df["text"].fillna("").astype(str)
        rem = df["remittance_info"].fillna("").astype(str) if "remittance_info" in df.columns else ""
        cp = df["counterparty_name"].fillna("").astype(str) if "counterparty_name" in df.columns else ""
        return text + " " + rem + " " + cp

    def fit(self, df: pd.DataFrame, y_settles: np.ndarray, y_category: np.ndarray) -> "FeaturePipeline":
        combined_text = self._prepare_text(df)
        self.vec_word.fit(combined_text)
        self.vec_char.fit(combined_text)

        # Categorical columns
        cat_df = df[self.CAT_COLUMNS].fillna("").astype(str)
        self.ohe_cat.fit(cat_df)

        # Amount bins
        amt = np.abs(df["amount"].astype(float).values)
        amt_bins = np.digitize(amt, self.AMOUNT_BINS).reshape(-1, 1)
        self.ohe_bins.fit(amt_bins)

        # Counterparty priors
        self.cp_encoder.fit(df, y_settles, y_category)
        return self

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        combined_text = self._prepare_text(df)
        X_w = self.vec_word.transform(combined_text)
        X_c = self.vec_char.transform(combined_text)

        cat_df = df[self.CAT_COLUMNS].fillna("").astype(str)
        X_cat = self.ohe_cat.transform(cat_df)

        amt = df["amount"].astype(float).values
        sign = np.sign(amt).reshape(-1, 1).astype(np.float32)
        log_amt = np.log1p(np.abs(amt)).reshape(-1, 1).astype(np.float32)

        amt_bins = np.digitize(np.abs(amt), self.AMOUNT_BINS).reshape(-1, 1)
        X_bins = self.ohe_bins.transform(amt_bins)

        X_cp = self.cp_encoder.transform(df)

        return hstack([
            X_w,
            X_c,
            X_cat,
            csr_matrix(sign),
            csr_matrix(log_amt),
            X_bins,
            X_cp,
        ], format="csr")
