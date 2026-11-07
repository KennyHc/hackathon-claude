"""Precision by method of the refs output, measured only through evalutil.report.
    PYTHONUTF8=1 .venv/Scripts/python.exe matching/tests/refs/by_method.py val"""
import sys
sys.path.insert(0, ".")
import pandas as pd
from matching.lib import evalutil

ph = sys.argv[1] if len(sys.argv) > 1 else "val"
a = pd.read_parquet(f"matching/out/{ph}/refs.parquet")
rows = []
for m, d in a.groupby("method"):
    r = evalutil.report(d, ph, quiet=True)
    rows.append((m, r["lines"], r["pairs_pred"], round(r["precision"], 4), round(r["exact_lines"] / max(r["lines"], 1), 4), round(r["exc_agree"], 4), round(d.confidence.mean(), 3)))
print(pd.DataFrame(rows, columns=["method", "lines", "pairs", "pair_prec", "line_exact", "exc_agree", "conf"]).sort_values("lines", ascending=False).to_string(index=False))
