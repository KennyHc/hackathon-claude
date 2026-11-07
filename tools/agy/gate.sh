#!/usr/bin/env bash
# tools/agy/gate.sh <name>: files a Flash worker created/changed/deleted, and which are outside its allowed paths.
set -uo pipefail
AGY_HOME="${AGY_HOME:-/private/tmp/claude-501/-Users-luigizheng-Desktop-Close-the-tabs/agy_runs}"
D="$AGY_HOME/$1"; ok=1
cd "$D" && diff -rq base ws -x data -x .venv -x out -x __pycache__ 2>/dev/null \
  | sed -E 's#^Files base/(.*) and ws/.* differ$#M \1#; s#^Only in ws/?(.*): (.*)$#A \1/\2#; s#^Only in base/?(.*): (.*)$#D \1/\2#; s#^([AD]) /#\1 #' > changes.txt
while read -r k f; do
  in=0; while read -r a; do [[ "$f" == "$a" || "$f" == "$a"/* ]] && in=1; done < allowed.txt
  [ $in -eq 1 ] && echo "  ok      $k $f" || { echo "  OUTSIDE $k $f"; ok=0; }
done < changes.txt
[ -s changes.txt ] || echo "  (no changes)"
[ $ok -eq 1 ] && echo "GATE PASS: $1" || echo "GATE FAIL: $1 (accept.sh copies only the allowed files)"
