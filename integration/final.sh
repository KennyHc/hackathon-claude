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

say "ingestion (+ PDF text)"
$PY ingestion/run.py --phase final || { echo "INGESTION FAILED"; exit 1; }

say "matching + classification + integration"
$PY integration/run.py --phase final --all || { echo "PIPELINE FAILED"; exit 1; }
grep -E "CRASHED|WARNING no" "$LOG" && echo ">>> a matcher crashed or a PDF cache is missing: read above before submitting"

say "validate"
$PY integration/validate.py --phase final || { echo "VALIDATION FAILED — do not submit this file"; exit 1; }

say "drift vs LIVE"
$PY integration/drift.py --phase final --ref live || echo ">>> drift flags above: look at them before submitting"

say "ready"
ls -la integration/out/final/submission.csv.gz
cat <<EOF

To submit the ONE FINAL shot (type the token, it is not echoed):
  read -rs HK_TOKEN && export HK_TOKEN && $PY integration/submit.py FINAL integration/out/final/submission.csv.gz --i-am-sure
EOF
