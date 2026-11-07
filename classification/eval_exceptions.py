"""Evaluate assign_exceptions on golden item rows of a phase.

Usage:
    .venv/bin/python classification/eval_exceptions.py --phase val
    .venv/bin/python classification/eval_exceptions.py --phase sample
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from classification.exceptions import assign_exceptions

EXCEPTION_CODES = [
    "PARTIAL",
    "OVERPAYMENT",
    "SHORT_DISCOUNT",
    "SHORT_BANK_CHARGES",
    "SHORT_WITHHOLDING",
    "SHORT_DISPUTE",
    "DUPLICATE_PAYMENT",
    "UNIDENTIFIED",
    "RETURNED",
    "FX_DIFFERENCE",
]


def load_phase_items(phase_dir: str) -> pd.DataFrame:
    """Build open_items table for the phase."""
    dfs: list[pd.DataFrame] = []

    oi_path = os.path.join(phase_dir, "open_items_at_start.csv.gz")
    has_oi = os.path.exists(oi_path)
    oi_ids: set[str] = set()

    if has_oi:
        oi_df = pd.read_csv(oi_path, dtype=str, keep_default_na=False)
        if "payment_terms" not in oi_df.columns:
            oi_df["payment_terms"] = ""
        if "payment_method" not in oi_df.columns:
            oi_df["payment_method"] = ""
        dfs.append(oi_df)
        oi_ids = set(oi_df["item_id"])

    # Load AR invoices
    ar_path = os.path.join(phase_dir, "ar_invoices.csv.gz")
    if os.path.exists(ar_path):
        ar = pd.read_csv(ar_path, dtype=str, keep_default_na=False)
        ar = ar[~ar["item_id"].isin(oi_ids)]
        df_ar = pd.DataFrame({
            "item_id": ar["item_id"],
            "item_type": "ARI",
            "entity": ar["entity"],
            "party_id": ar["customer_id"],
            "document_number": ar["document_number"],
            "issue_date": ar["issue_date"],
            "due_date": ar["due_date"],
            "currency": ar["currency"],
            "total_amount": ar["total_amount"],
            "open_amount": ar["total_amount"],
            "payment_terms": ar.get("payment_terms", ""),
            "payment_method": ar.get("payment_method", "")
        })
        dfs.append(df_ar)

    # Load AP invoices
    ap_path = os.path.join(phase_dir, "ap_invoices.csv.gz")
    if os.path.exists(ap_path):
        ap = pd.read_csv(ap_path, dtype=str, keep_default_na=False)
        ap = ap[~ap["item_id"].isin(oi_ids)]
        df_ap = pd.DataFrame({
            "item_id": ap["item_id"],
            "item_type": "API",
            "entity": ap["entity"],
            "party_id": ap["supplier_id"],
            "document_number": ap["supplier_invoice_number"],
            "issue_date": ap["issue_date"],
            "due_date": ap["due_date"],
            "currency": ap["currency"],
            "total_amount": ap["total_amount"],
            "open_amount": ap["total_amount"],
            "payment_terms": ap.get("payment_terms", ""),
            "payment_method": ap.get("payment_method", "")
        })
        dfs.append(df_ap)

    # Load credit notes
    cn_path = os.path.join(phase_dir, "credit_notes.csv.gz")
    if os.path.exists(cn_path):
        cn = pd.read_csv(cn_path, dtype=str, keep_default_na=False)
        cn = cn[~cn["item_id"].isin(oi_ids)]
        df_cn = pd.DataFrame({
            "item_id": cn["item_id"],
            "item_type": cn["item_id"].str[:3],
            "entity": cn["entity"],
            "party_id": cn["party_id"],
            "document_number": cn["document_number"],
            "issue_date": cn["issue_date"],
            "due_date": cn["issue_date"],
            "currency": cn["currency"],
            "total_amount": cn["total_amount"],
            "open_amount": cn["total_amount"],
            "payment_terms": "",
            "payment_method": ""
        })
        dfs.append(df_cn)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate exception classification on golden item rows.")
    parser.add_argument("--phase", default="val", help="Phase to evaluate (default: val)")
    parser.add_argument("--data", default="data", help="Data root directory (default: data)")
    parser.add_argument("--masters", default="data/masters", help="Masters directory (default: data/masters)")
    args = parser.parse_args()

    phase = args.phase
    phase_dir = os.path.join(args.data, phase)
    out_dir = os.path.join("classification", "out", phase)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Exception Evaluation: phase={phase} ===")
    t0 = time.time()

    # Load bank lines
    bl_path = os.path.join(phase_dir, "bank_transactions.csv.gz")
    print(f"Loading bank lines from {bl_path}...")
    bank_lines = pd.read_csv(bl_path, dtype=str, keep_default_na=False)
    print(f"  Loaded {len(bank_lines):,} bank transactions.")

    # Load golden labels
    golden_path = os.path.join(phase_dir, "golden_allocations.csv.gz")
    print(f"Loading golden allocations from {golden_path}...")
    golden = pd.read_csv(golden_path, dtype=str, keep_default_na=False)
    golden_items = golden[golden["item_id"] != ""].copy()
    print(f"  Loaded {len(golden_items):,} golden item allocations.")

    # Load open items
    print(f"Building open items table for phase {phase}...")
    open_items = load_phase_items(phase_dir)
    print(f"  Total items in phase table: {len(open_items):,}")

    # Prepare allocations with simulated matching guesses
    # (keep RETURNED / DUPLICATE_PAYMENT / SHORT_DISPUTE)
    allocations = golden_items[["bank_txn_id", "item_id", "amount"]].copy()
    guess_mask = golden_items["exception"].isin({"RETURNED", "DUPLICATE_PAYMENT", "SHORT_DISPUTE"})
    allocations["exception"] = np.where(guess_mask, golden_items["exception"], "")

    print(f"Running assign_exceptions...")
    t_start = time.time()
    pred_series = assign_exceptions(allocations, bank_lines, open_items, masters_dir=args.masters)
    t_infer = time.time() - t_start
    print(f"  Inference finished in {t_infer:.2f}s ({len(pred_series)/max(t_infer, 1e-4):,.0f} rows/s)")

    # Align labels
    actual = golden_items["exception"].fillna("").astype(str).values
    predicted = pred_series.fillna("").astype(str).values

    # Compute metrics per exception code
    metrics_rows = []
    f1_list = []

    print("\nPer-Exception Metrics:")
    print(f"{'Exception':<22} {'Support':>8} {'TP':>8} {'FP':>8} {'FN':>8} {'Prec (%)':>9} {'Rec (%)':>9} {'F1 (%)':>9}")
    print("-" * 88)

    for code in EXCEPTION_CODES:
        tp = int(np.sum((actual == code) & (predicted == code)))
        fp = int(np.sum((actual != code) & (predicted == code)))
        fn = int(np.sum((actual == code) & (predicted != code)))
        support = int(np.sum(actual == code))

        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if support > 0:
            f1_list.append(f1)

        print(f"{code:<22} {support:>8d} {tp:>8d} {fp:>8d} {fn:>8d} {prec:>9.2f} {rec:>9.2f} {f1:>9.2f}")
        metrics_rows.append({
            "exception": code,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1": round(f1, 2)
        })

    # Empty exception metrics
    tp_emp = int(np.sum((actual == "") & (predicted == "")))
    fp_emp = int(np.sum((actual != "") & (predicted == "")))
    fn_emp = int(np.sum((actual == "") & (predicted != "")))
    sup_emp = int(np.sum(actual == ""))
    prec_emp = (tp_emp / (tp_emp + fp_emp)) * 100.0 if (tp_emp + fp_emp) > 0 else 0.0
    rec_emp = (tp_emp / (tp_emp + fn_emp)) * 100.0 if (tp_emp + fn_emp) > 0 else 0.0
    f1_emp = (2 * prec_emp * rec_emp / (prec_emp + rec_emp)) if (prec_emp + rec_emp) > 0 else 0.0

    print("-" * 88)
    print(f"{'(empty/clean)':<22} {sup_emp:>8d} {tp_emp:>8d} {fp_emp:>8d} {fn_emp:>8d} {prec_emp:>9.2f} {rec_emp:>9.2f} {f1_emp:>9.2f}")

    macro_f1 = float(np.mean(f1_list)) if f1_list else 100.0
    print("-" * 88)
    print(f"{'MACRO-F1 (10 codes)':<22} {sum(r['support'] for r in metrics_rows):>8d} "
          f"{'':>8} {'':>8} {'':>8} {'':>9} {'':>9} {macro_f1:>9.2f}")

    # Confusion matrix
    print("\nConfusion Matrix (rows = actual, cols = predicted):")
    labels_order = EXCEPTION_CODES + [""]
    labels_display = [c if c else "(empty)" for c in labels_order]
    cm = pd.crosstab(
        pd.Series(np.where(actual == "", "(empty)", actual), name="Actual"),
        pd.Series(np.where(predicted == "", "(empty)", predicted), name="Predicted")
    )
    print(cm)

    # Save to CSV
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.loc[len(metrics_df)] = {
        "exception": "MACRO_F1",
        "support": sum(r["support"] for r in metrics_rows),
        "tp": sum(r["tp"] for r in metrics_rows),
        "fp": sum(r["fp"] for r in metrics_rows),
        "fn": sum(r["fn"] for r in metrics_rows),
        "precision": round(float(np.mean([r["precision"] for r in metrics_rows if r["support"] > 0])), 2),
        "recall": round(float(np.mean([r["recall"] for r in metrics_rows if r["support"] > 0])), 2),
        "f1": round(macro_f1, 2)
    }

    out_csv = os.path.join(out_dir, "exceptions_eval.csv")
    metrics_df.to_csv(out_csv, index=False)
    print(f"\nWrote evaluation report to {out_csv}")
    print(f"Total time elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
