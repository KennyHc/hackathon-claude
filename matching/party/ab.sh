#!/usr/bin/env bash
# matching/party/ab.sh <tag> [run.py flags...]: party on val with cascade_batch+refs as claimed, combo score.
set -uo pipefail
cd "$(dirname "$0")/../.."
T=$1; shift
O=matching/party/out/ab_$T
mkdir -p $O
.venv/bin/python matching/party/run.py --phase val --out $O/alloc_party.parquet "$@" 2>&1 | grep -v "aliases:" | tail -4
.venv/bin/python - <<PY
import pandas as pd
a=[pd.read_parquet(f'matching/out/val/cascade_{s}.parquet') for s in ['batch','refs']]+[pd.read_parquet('$O/alloc_party.parquet')]
c=pd.concat(a,ignore_index=True).drop_duplicates(['bank_txn_id','item_id'])
c.to_parquet('$O/combo.parquet')
PY
.venv/bin/python integration/run.py --phase val --b $O/combo.parquet --out $O/sub.csv.gz > $O/score.log 2>&1
grep "SCORE" $O/score.log
cp matching/party/out/debug_val.parquet $O/debug.parquet 2>/dev/null
