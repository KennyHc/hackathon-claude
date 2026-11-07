#!/usr/bin/env bash
# Whole pipeline for one phase: classification → matching cascade → merge + exceptions → validate (→ score on val).
#   integration/pipeline.sh val|live|final [extra args for integration/run.py, e.g. --p-items 0.3]
# Env: SKIP_C=1 reuses classification/out/P/line_labels.parquet if present; STAGES=batch,refs,party,k4_sbc (cascade order);
#      MATCH_ARGS="--fallback-files" extra args for matching/run.py.
# Logs: integration/out/P/pipeline_<time>.log. Exit code ≠ 0 when the submission is not valid.
set -uo pipefail
P="${1:?phase}"; shift
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
PY=.venv/bin/python
mkdir -p "integration/out/$P"
LOG="integration/out/$P/pipeline_$(date +%H%M%S).log"
exec > >(tee "$LOG") 2>&1
T0=$(date +%s)
step() { echo; echo "=== [$(( $(date +%s) - T0 ))s] $*"; }

DATA="${BANKREC_DATA:-data}"
step "data check ($DATA)"
[ -f "$DATA/$P/bank_transactions.csv.gz" ] || { echo "$DATA/$P/bank_transactions.csv.gz missing"; exit 2; }
[ "$P" = final ] && { ls "$DATA/final/masters_final" 2>/dev/null || echo "WARN: no $DATA/final/masters_final"; }

# Phase caches (remittance parse, pdf clauses, stage outputs, party aliases, merged masters) are reused without a
# freshness check. FRESH=1 (the default for final) deletes them so FINAL never runs on mock/stale caches.
if [ "${FRESH:-$([ "$P" = final ] && echo 1 || echo 0)}" = 1 ]; then
  step "purging $P caches"
  rm -rf "matching/out/$P" "integration/out/masters_merged/$P"
  rm -f matching/party/out/*_"$P"_*.parquet
fi

if [ "${SKIP_C:-0}" = 1 ] && [ -f "classification/out/$P/line_labels.parquet" ]; then
  step "classification: reusing classification/out/$P/line_labels.parquet"
else
  step "classification/run.py --phase $P"
  $PY classification/run.py --phase "$P" || echo "WARN: classification failed; integration falls back to defaults"
fi

step "matching/run.py --phase $P --stages ${STAGES:-batch,refs,party,k4_sbc} ${MATCH_ARGS:-}"
$PY matching/run.py --phase "$P" --stages "${STAGES:-batch,refs,party,k4_sbc}" ${MATCH_ARGS:-} \
  || echo "WARN: matching cascade failed; integration runs without allocations"

step "integration/run.py --phase $P $*"
$PY integration/run.py --phase "$P" "$@"
RC=$?
step "done in $(( $(date +%s) - T0 ))s, exit $RC → integration/out/$P/submission.csv.gz (log $LOG)"
exit $RC
