#!/usr/bin/env python3
"""Quick eval of alloc_refs vs golden (val/sample): per-method line precision, per-segment claimed/correct."""
import sys, os
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
phase = sys.argv[1] if len(sys.argv) > 1 else "val"
g = pd.read_csv(f"{ROOT}/data/{phase}/golden_allocations.csv.gz", dtype=str, keep_default_na=False)
r = pd.read_parquet(f"{ROOT}/matching/out/{phase}/alloc_refs.parquet")
seg = g.groupby("bank_txn_id").segment.first()
gi = g[g.item_id != ""].copy(); gi["amount"] = gi.amount.astype(float)
gset = gi.groupby("bank_txn_id").apply(lambda d: dict(zip(d.item_id, d.amount)))
def ok(d):
    s = gset.get(d.name)
    if s is None or set(s) != set(d.item_id): return False
    return all(abs(a - s[i]) <= max(0.01, 0.005 * s[i]) for i, a in zip(d.item_id, d.amount))
lr = r.groupby("bank_txn_id").apply(ok).rename("ok").to_frame()
lr["method"] = r.groupby("bank_txn_id").method.first(); lr["segment"] = seg.reindex(lr.index).values
# pair precision
m = r.merge(gi[["bank_txn_id", "item_id", "amount"]], on=["bank_txn_id", "item_id"], how="left", suffixes=("", "_g"))
m["tp"] = (m.amount - m.amount_g).abs() <= (0.005 * m.amount_g).clip(lower=0.01)
print(f"lines claimed {len(lr):,}  line-exact {lr.ok.mean():.3f}  pairs {len(m):,} pair-prec {m.tp.mean():.3f}  pair-hit(any amt) {m.amount_g.notna().mean():.3f}")
print(lr.groupby("method").ok.agg(["size", "mean"]).sort_values("size", ascending=False).to_string())
tot = seg.value_counts()
s = lr.groupby("segment").ok.agg(["size", "sum"]); s["seg_lines"] = tot.reindex(s.index); s["acc_of_seg"] = s["sum"] / s.seg_lines
s["prec"] = s["sum"] / s["size"]
print(s.sort_values("size", ascending=False).head(40).to_string())
