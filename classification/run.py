"""classification.run: CLI entry point for line classification pipeline.

Usage:
    .venv/bin/python classification/run.py --phase {sample,val,train,live,final}

Outputs:
    classification/out/<phase>/line_labels.parquet
    classification/out/models/model_val.joblib (cached model for val/sample)
    classification/out/models/model_all.joblib (cached model for live/final/train)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from classification.categories.model import LineClassifier
from classification.categories.rules import ALL_CATEGORIES


def load_train_labels(
    golden_path: str,
    train_txn_ids: set,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Loads golden allocations and extracts settles_items and category per bank line."""
    ga = pd.read_csv(
        golden_path,
        dtype=str,
        keep_default_na=False,
        usecols=["bank_txn_id", "item_id", "category"],
    )
    # Filter to transactions in the target training slice
    if train_txn_ids:
        ga = ga[ga["bank_txn_id"].isin(train_txn_ids)]

    # A line settles items if any row has item_id != ''
    item_rows = ga[ga["item_id"] != ""]
    settles_set = set(item_rows["bank_txn_id"].unique())

    # No-item lines have item_id == '' and category set
    no_item_rows = ga[ga["item_id"] == ""][["bank_txn_id", "category"]].drop_duplicates("bank_txn_id")
    category_map = dict(zip(no_item_rows["bank_txn_id"], no_item_rows["category"]))

    settles_map = {txn: (1 if txn in settles_set else 0) for txn in train_txn_ids}
    return settles_map, category_map


def train_or_load_model(
    data_dir: str,
    phase: str,
    cache_dir: str,
    force_retrain: bool = False,
) -> LineClassifier:
    """Trains a LineClassifier or loads a cached model."""
    os.makedirs(cache_dir, exist_ok=True)

    is_val_window = (phase in ("val", "sample"))
    model_name = "model_val.joblib" if is_val_window else "model_all.joblib"
    model_path = os.path.join(cache_dir, model_name)

    if os.path.exists(model_path) and not force_retrain:
        print(f"Loading cached model from {model_path}...")
        return joblib.load(model_path)

    print(f"Fitting new model for phase '{phase}'...")
    train_bt_path = os.path.join(data_dir, "train", "bank_transactions.csv.gz")
    train_ga_path = os.path.join(data_dir, "train", "golden_allocations.csv.gz")

    t0 = time.time()
    train_bt = pd.read_csv(train_bt_path, dtype=str, keep_default_na=False)

    if is_val_window:
        # Strict no-leakage: val models learn ONLY from rows booked < 2025-01-01
        print("Restricting training data to booking_date < 2025-01-01 (no leakage)...")
        mask = train_bt["booking_date"] < "2025-01-01"
        train_bt = train_bt[mask].reset_index(drop=True)

    print(f"Loaded {len(train_bt):,} bank lines in {time.time() - t0:.2f}s.")

    # Load targets from golden allocations
    train_txn_ids = set(train_bt["bank_txn_id"])
    t1 = time.time()
    settles_map, category_map = load_train_labels(train_ga_path, train_txn_ids)
    print(f"Loaded golden targets in {time.time() - t1:.2f}s.")

    y_settles = np.array([settles_map.get(t, 0) for t in train_bt["bank_txn_id"]], dtype=int)
    y_category = np.array([category_map.get(t, "") for t in train_bt["bank_txn_id"]], dtype=object)

    sample_pos_size = 50000 if phase == "sample" else 250000

    classifier = LineClassifier(
        max_word_features=15000,
        max_char_features=25000,
        random_state=42,
    )
    t_fit = time.time()
    classifier.fit(train_bt, y_settles, y_category, sample_pos_size=sample_pos_size)
    print(f"Model fitting complete in {time.time() - t_fit:.2f}s.")

    joblib.dump(classifier, model_path)
    print(f"Model successfully saved to {model_path}.")
    return classifier


def evaluate_predictions(
    val_bt: pd.DataFrame,
    preds_df: pd.DataFrame,
    cat_b_all: np.ndarray,
    golden_path: str,
):
    """Computes and prints evaluation metrics vs golden allocations."""
    print("\n" + "=" * 60)
    print("EVALUATION VERSUS GOLDEN ALLOCATIONS")
    print("=" * 60)

    val_ga = pd.read_csv(
        golden_path,
        dtype=str,
        keep_default_na=False,
        usecols=["bank_txn_id", "item_id", "category"],
    )

    val_txns = set(val_bt["bank_txn_id"])
    val_ga = val_ga[val_ga["bank_txn_id"].isin(val_txns)]

    # True settles_items
    item_rows = val_ga[val_ga["item_id"] != ""]
    true_settles_set = set(item_rows["bank_txn_id"].unique())
    true_settles = np.array([1 if t in true_settles_set else 0 for t in val_bt["bank_txn_id"]], dtype=int)

    # True no-item categories
    no_item_rows = val_ga[val_ga["item_id"] == ""][["bank_txn_id", "category"]].drop_duplicates("bank_txn_id")
    true_cat_map = dict(zip(no_item_rows["bank_txn_id"], no_item_rows["category"]))

    # 1. AUC and Accuracy of settles_items
    p_items = preds_df["p_items"].values
    pred_settles = (p_items >= 0.5).astype(int)

    auc_score = roc_auc_score(true_settles, p_items)
    acc_settles = accuracy_score(true_settles, pred_settles)

    print(f"settles_items ROC AUC:             {auc_score:.4f}")
    print(f"settles_items Accuracy (p >= 0.5):  {acc_settles:.4f} ({acc_settles * 100:.2f}%)")

    # 2. Category Accuracy and Macro-F1 on true no-item lines
    true_no_item_mask = (true_settles == 0)
    true_no_item_ids = val_bt.loc[true_no_item_mask, "bank_txn_id"].values
    y_true_no_item = np.array([true_cat_map.get(t, "UNIDENTIFIED") for t in true_no_item_ids])

    # Model category prediction on true no-item lines
    y_pred_no_item = cat_b_all[true_no_item_mask]

    cat_acc = accuracy_score(y_true_no_item, y_pred_no_item)
    cat_macro_f1 = f1_score(y_true_no_item, y_pred_no_item, average="macro", zero_division=0)

    print(f"Category Accuracy (true no-item):  {cat_acc:.4f} ({cat_acc * 100:.2f}%)")
    print(f"Category Macro-F1 (true no-item):  {cat_macro_f1:.4f}")

    print("\nPer-Category Precision / Recall Table (True No-Item Lines):")
    print("-" * 60)
    report = classification_report(
        y_true_no_item,
        y_pred_no_item,
        zero_division=0,
        digits=4,
    )
    print(report)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Bank line category & settlement classifier")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["sample", "val", "train", "live", "final"],
        help="Data phase to predict",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force model retraining even if cached model exists",
    )
    parser.add_argument(
        "--data",
        default="data",
        help="Root path to data directory (default: 'data')",
    )
    parser.add_argument(
        "--out",
        default="classification/out",
        help="Root path to output directory (default: 'classification/out')",
    )
    args = parser.parse_args()

    phase = args.phase
    data_dir = args.data
    out_dir = args.out
    cache_dir = os.path.join(out_dir, "models")
    phase_out_dir = os.path.join(out_dir, phase)
    os.makedirs(phase_out_dir, exist_ok=True)

    print(f"=== Running Classification Pipeline: Phase '{phase}' ===")
    t_start = time.time()

    # 1. Train or load model
    classifier = train_or_load_model(
        data_dir=data_dir,
        phase=phase,
        cache_dir=cache_dir,
        force_retrain=args.retrain,
    )

    # 2. Load target phase bank transactions
    target_csv = os.path.join(data_dir, phase, "bank_transactions.csv.gz")
    if not os.path.exists(target_csv):
        print(f"Error: Target bank transactions file not found: {target_csv}", file=sys.stderr)
        sys.exit(1)

    t_load = time.time()
    print(f"Loading target transactions from {target_csv}...")
    df_target = pd.read_csv(target_csv, dtype=str, keep_default_na=False)
    print(f"Loaded {len(df_target):,} bank lines in {time.time() - t_load:.2f}s.")

    # 3. Predict
    t_pred = time.time()
    print("Predicting line labels, probabilities, and categories...")
    preds_df, cat_b_all = classifier.predict(df_target, p_cutoff=0.5)
    print(f"Predictions completed in {time.time() - t_pred:.2f}s.")

    # 4. Save output parquet
    out_parquet = os.path.join(phase_out_dir, "line_labels.parquet")
    preds_df.to_parquet(out_parquet, engine="pyarrow", index=False)
    print(f"Saved {len(preds_df):,} lines to {out_parquet} ({os.path.getsize(out_parquet):,} bytes).")

    # 5. Evaluate if golden allocations exist for this phase
    golden_csv = os.path.join(data_dir, phase, "golden_allocations.csv.gz")
    if os.path.exists(golden_csv):
        evaluate_predictions(
            val_bt=df_target,
            preds_df=preds_df,
            cat_b_all=cat_b_all,
            golden_path=golden_csv,
        )

    print(f"=== Phase '{phase}' completed in {time.time() - t_start:.2f}s ===\n")


if __name__ == "__main__":
    main()
