#!/usr/bin/env bash
# tools/agy/argon.sh <prompt-file> [out-file]: read-only review by Gemini 4 Argon (high), run from the repo root.
# No edits, no commands (print mode denies them). Prints the review (also to out-file if given). ~2-8 min.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
OUT="${2:-/dev/null}"
agy -p "You are a critical senior reviewer. Read-only: do not edit files, do not run commands.
$(cat "$1")" --model gemini-4-argon-high --print-timeout "${AGY_TIMEOUT:-540}s" 2>&1 | tee "$OUT"
