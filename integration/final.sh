#!/usr/bin/env bash
# FINAL runbook: one command from extracted data to a validated submission + drift report. It never submits.
#
#   integration/final.sh                          # data already extracted to data/final/
#   integration/final.sh FINAL.zip <sha256>       # also verify the download's checksum first
#
# Steps: checksum -> wipe every cached `final` artefact (a dress rehearsal or an earlier attempt must never leak into
# the real run) -> ingestion (builds the PDF text caches) -> matching + classification + integration -> validate ->
# drift report vs LIVE -> print the submit command. Log: integration/out/final_run.log
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=integration/out/final_run.log
mkdir -p integration/out
exec > >(tee "$LOG") 2>&1
PY=${PY:-python3}
say() { printf '\n=== %s  [%s]\n' "$1" "$(date +%H:%M:%S)"; }

if [ $# -ge 2 ]; then
  say "checksum"
  got=$(shasum -a 256 "$1" | cut -d' ' -f1)
  if [ "$got" != "$2" ]; then echo "SHA-256 MISMATCH: got $got expected $2 — stop, re-download"; exit 1; fi
  echo "SHA-256 OK"
fi

DATA=${RECON_DATA:-data}
[ -f "$DATA/final/bank_transactions.csv.gz" ] || { echo "no $DATA/final/bank_transactions.csv.gz — extract the FINAL zip into $DATA/final/ first"; exit 1; }
[ -d "$DATA/final/masters_final" ] && echo "masters_final: $(ls "$DATA/final/masters_final" | tr '\n' ' ')"

say "masters check (rows vs base, FX rates end date)"
$PY - "$DATA" <<'EOF'
import os, sys, pandas as pd
d = sys.argv[1]; base, fin = os.path.join(d, "masters"), os.path.join(d, "final", "masters_final")
if os.path.isdir(fin):
    for f in sorted(os.listdir(fin)):
        if f.endswith(".csv") and os.path.exists(os.path.join(base, f)):
            nb, nf = len(pd.read_csv(os.path.join(base, f))), len(pd.read_csv(os.path.join(fin, f)))
            print(f"{f:28s} base {nb:7,}  final {nf:7,}" + ("   <-- FEWER ROWS: partial file? lib.masters replaces whole files" if nf < nb else ""))
fx = next((p for p in (os.path.join(fin, "fx_rates.csv"), os.path.join(base, "fx_rates.csv")) if os.path.exists(p)), None)
if fx:
    r = pd.read_csv(fx); dc = next(c for c in r.columns if "date" in c.lower())
    end = r.groupby(next((c for c in r.columns if c.lower() in ("currency", "ccy", "quote_currency")), r.columns[1]))[dc].max()
    print(f"fx_rates ({fx}): last date per currency:\n{end.to_string()}")
    if end.min() < "2026-06-01":
        print(">>> FX rates stop before June 2026: rate-based FX checks use stale rates (remittance falls back to the bank's orig_amount)")
EOF

say "wipe cached final artefacts"
rm -rf ingestion/out/final matching/out/final classification/out/final integration/out/final
rm -f ingestion/out/_cache/final__* ingestion/out/_cache/pdftext_final*
# build the PDF text caches HERE (pdftext.path() otherwise falls back to a legacy data/cache/pdftext_final* that any
# teammate's older code may have written; ensure() only checks its file count, not how it was extracted)
$PY ingestion/pdftext.py final || { echo "PDF TEXT EXTRACTION FAILED"; exit 1; }

say "ingestion (+ PDF text)"
$PY ingestion/run.py --phase final || { echo "INGESTION FAILED"; exit 1; }
$PY - <<'EOF2'
import pandas as pd
b = pd.read_parquet("ingestion/out/final/bank_lines.parquet", columns=["entity", "currency", "amount_eur"])
o = pd.read_parquet("ingestion/out/final/open_items.parquet", columns=["entity", "open_amount"])
print("bank lines per entity:", b.entity.value_counts().to_dict())
print("open items per entity:", o[o.open_amount > 0.005].entity.value_counts().to_dict())
if b.amount_eur.isna().any():
    print(f"WARNING amount_eur missing for {int(b.amount_eur.isna().sum())} lines, currencies {sorted(b.currency[b.amount_eur.isna()].unique())} (no fx rate)")
for e in sorted(set(b.entity) - set(o.entity)):
    print(f"WARNING entity {e} has bank lines but no open items")
EOF2

say "matching + classification + integration"
$PY integration/run.py --phase final --all || { echo "PIPELINE FAILED"; exit 1; }
grep -E "CRASHED|WARNING|ERROR|FAILED|: 0 lines" "$LOG" && echo ">>> a matcher crashed / produced 0 lines, a fallback fired (assign_exceptions/refine FAILED) or a parser warned: read above before submitting"

say "validate"
$PY integration/validate.py --phase final || { echo "VALIDATION FAILED — do not submit this file"; exit 1; }

say "drift vs LIVE"
$PY integration/drift.py --phase final --ref live || echo ">>> drift flags above: look at them before submitting"

say "add-on: sd_noadv on top (LIVE 96.42 with it vs 96.34 without; the safe file is kept)"
cp integration/out/final/submission.csv.gz integration/out/final/submission_safe.csv.gz
cp integration/out/final/methods.parquet integration/out/final/methods_safe.parquet
if $PY matching/addon.py --phase final --stages sd_noadv \
   && ALLOC_FILE=allocations_plus.parquet $PY integration/run.py --phase final \
   && $PY integration/validate.py --phase final; then
  echo "submission.csv.gz = WITH sd_noadv (fallback: submission_safe.csv.gz)"
else
  cp integration/out/final/submission_safe.csv.gz integration/out/final/submission.csv.gz
  cp integration/out/final/methods_safe.parquet integration/out/final/methods.parquet
  echo ">>> ADD-ON FAILED: submission.csv.gz = the safe file"
fi

say "ready"
ls -la integration/out/final/submission.csv.gz
cat <<EOF

To submit the ONE FINAL shot (type the token, it is not echoed):
  read -rs HK_TOKEN && export HK_TOKEN && $PY integration/submit.py FINAL integration/out/final/submission.csv.gz --i-am-sure
EOF
