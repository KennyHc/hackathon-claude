#!/usr/bin/env bash
# tools/agy/accept.sh <name>: copy a worker's created/changed files that are inside its allowed paths into the repo.
# Deletions and anything outside the allowed paths are NOT applied (they are printed).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AGY_HOME="${AGY_HOME:-/private/tmp/claude-501/-Users-luigizheng-Desktop-Close-the-tabs/agy_runs}"
D="$AGY_HOME/$1"
while read -r k f; do
  in=0; while read -r a; do [[ "$f" == "$a" || "$f" == "$a"/* ]] && in=1; done < "$D/allowed.txt"
  if [ $in -eq 0 ]; then echo "  skip (outside) $k $f"; continue; fi
  if [ "$k" = D ]; then echo "  skip (deletion) $f"; continue; fi
  if [ -d "$D/ws/$f" ]; then mkdir -p "$REPO/$f"; rsync -a --exclude out --exclude __pycache__ "$D/ws/$f/" "$REPO/$f/"; else mkdir -p "$(dirname "$REPO/$f")"; cp "$D/ws/$f" "$REPO/$f"; fi
  echo "  copied $k $f"
done < "$D/changes.txt"
