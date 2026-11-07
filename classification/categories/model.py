"""classification.categories.model: Machine learning line classifier models A and B.

Combines:
- Rules from classification.categories.rules (override when fired).
- Model A: Binary settles_items probability estimator (calibrated probability -> p_items).
- Model B: 21-class category classifier among no-item lines (LogisticRegression/SGD with balanced weights).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from .features import FeaturePipeline
from .rules import ALL_CATEGORIES, apply_rules


class LineClassifier:
    """Two-stage classifier: Model A (settles_items) + Model B (category)."""

    def __init__(
        self,
        max_word_features: int = 15000,
        max_char_features: int = 25000,
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.feature_pipeline = FeaturePipeline(
            max_word_features=max_word_features,
            max_char_features=max_char_features,
        )
        self.model_a = SGDClassifier(
            loss="log_loss",
            max_iter=35,
            tol=1e-3,
            random_state=self.random_state,
        )
        self.model_b = SGDClassifier(
            loss="log_loss",
            max_iter=45,
            tol=1e-3,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self.classes_b_: Optional[np.ndarray] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        y_settles: np.ndarray,
        y_category: np.ndarray,
        sample_pos_size: int = 250000,
    ) -> "LineClassifier":
        """Fits feature pipeline, Model A, and Model B on training transactions.

        Args:
            train_df: DataFrame of bank transactions.
            y_settles: Binary array (1 if line settles items, 0 if no item).
            y_category: Category string array for no-item lines (empty string for item lines).
            sample_pos_size: Number of positive (settles_items) rows to subsample for fast training.
        """
        n_total = len(train_df)
        pos_indices = np.where(y_settles == 1)[0]
        neg_indices = np.where(y_settles == 0)[0]

        # Subsample positive class if larger than sample_pos_size
        if len(pos_indices) > sample_pos_size:
            rng = np.random.RandomState(self.random_state)
            sub_pos = rng.choice(pos_indices, size=sample_pos_size, replace=False)
            sub_indices = np.concatenate([sub_pos, neg_indices])
            rng.shuffle(sub_indices)
        else:
            sub_indices = np.arange(n_total)

        sub_df = train_df.iloc[sub_indices].reset_index(drop=True)
        sub_settles = y_settles[sub_indices]
        sub_cat = y_category[sub_indices]

        # Calculate unbiased sample weights for Model A
        weights = np.ones(len(sub_indices), dtype=np.float32)
        if len(pos_indices) > sample_pos_size:
            weights[sub_settles == 1] = float(len(pos_indices)) / float(sample_pos_size)

        # 1. Fit feature pipeline on subsampled data
        self.feature_pipeline.fit(sub_df, sub_settles, sub_cat)
        X_sub = self.feature_pipeline.transform(sub_df)

        # 2. Fit Model A (binary settles_items)
        self.model_a.fit(X_sub, sub_settles, sample_weight=weights)

        # 3. Fit Model B (category among no-item lines)
        no_item_mask = (sub_settles == 0)
        X_no_item = X_sub[no_item_mask]
        y_no_item = sub_cat[no_item_mask]

        self.model_b.fit(X_no_item, y_no_item)
        self.classes_b_ = self.model_b.classes_

        return self

    def predict(
        self,
        df: pd.DataFrame,
        p_cutoff: float = 0.5,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Predicts line labels, probabilities, categories, exceptions, confidences.

        Args:
            df: Bank transactions dataframe of the target phase.
            p_cutoff: Probability cutoff above which a line is treated as settling items.

        Returns:
            result_df: DataFrame matching the line_labels contract:
                bank_txn_id, p_items, category, exception, confidence, method
            cat_b_all: Model B's category prediction for every line (for evaluation).
        """
        n = len(df)
        bank_txn_id = df["bank_txn_id"].values

        # 1. Apply high-precision deterministic rules
        rule_fired, rule_cat, rule_conf, rule_method = apply_rules(df)

        # 2. Extract features and predict with Model A and Model B
        X = self.feature_pipeline.transform(df)
        p_items_raw = self.model_a.predict_proba(X)[:, 1]
        cat_b_pred = self.model_b.predict(X)
        prob_b_max = np.max(self.model_b.predict_proba(X), axis=1)

        # 3. Combine rules and models
        p_items = p_items_raw.copy()
        # Rules override model: if rule fired, the line definitely settles NO item
        p_items[rule_fired] = 0.0

        category = np.full(n, None, dtype=object)
        exception = np.full(n, None, dtype=object)
        confidence = np.zeros(n, dtype=np.float64)
        method = np.full(n, "", dtype=object)

        # Vectorized assembly
        for i in range(n):
            if rule_fired[i]:
                category[i] = str(rule_cat[i])
                exception[i] = "UNIDENTIFIED" if rule_cat[i] == "UNIDENTIFIED" else None
                confidence[i] = float(rule_conf[i])
                method[i] = str(rule_method[i])
            else:
                p_val = float(p_items[i])
                if p_val >= p_cutoff:
                    # Settles items: category and exception must be null
                    category[i] = None
                    exception[i] = None
                    confidence[i] = round(p_val, 4)
                    method[i] = "model:settles"
                else:
                    # Does not settle items: assign Model B category
                    c_pred = str(cat_b_pred[i])
                    category[i] = c_pred
                    exception[i] = "UNIDENTIFIED" if c_pred == "UNIDENTIFIED" else None
                    confidence[i] = round(max(float(1.0 - p_val), float(prob_b_max[i])), 4)
                    method[i] = f"model:{c_pred.lower()}"

        result_df = pd.DataFrame({
            "bank_txn_id": bank_txn_id,
            "p_items": p_items.astype(np.float64),
            "category": category,
            "exception": exception,
            "confidence": confidence,
            "method": method,
        })

        # All-rows combined predicted category for no-item evaluation
        cat_b_all = np.where(rule_fired, rule_cat, cat_b_pred)

        return result_df, cat_b_all
