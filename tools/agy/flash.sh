#!/usr/bin/env bash
# Run one Gemini 3.8 Flash (high) worker on a snapshot of the current checkout.
#   tools/agy/flash.sh <name> <brief.md> <allowed-path> [<allowed-path> ...]
# - snapshot = copy of the repo (no data/, .venv/, */out/, .git) in $AGY_HOME/<name>/ws; data, .venv and every
#   <piece>/out are symlinked (read the real ones).
# - allow-list: .venv/bin/python, ls, cat, head, grep, wc; reads of the repo. Anything else is denied and ends the run.
# - when it ends: tools/agy/gate.sh <name> <allowed...> lists what changed; tools/agy/accept.sh <name> copies it in.
# Log: $AGY_HOME/<name>/agy.log. Run it in the background; it can take up to $AGY_TIMEOUT (default 1500s).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AGY_HOME="${AGY_HOME:-/private/tmp/claude-501/-Users-luigizheng-Desktop-Close-the-tabs/agy_runs}"
N="$1"; BRIEF="$2"; shift 2; ALLOWED=("$@")
[ ${#ALLOWED[@]} -gt 0 ] || { echo "need at least one allowed path"; exit 2; }
D="$AGY_HOME/$N"; rm -rf "$D"; mkdir -p "$D"
EXC=(--exclude .git --exclude data --exclude .venv --exclude '*/out' --exclude __pycache__ --exclude .claude/me)
rsync -a "${EXC[@]}" "$REPO/" "$D/ws/"; rsync -a "${EXC[@]}" "$REPO/" "$D/base/"
ln -sfn "$REPO/data" "$D/ws/data"; ln -sfn "$REPO/.venv" "$D/ws/.venv"
for p in ingestion matching classification integration; do mkdir -p "$REPO/$p/out"; ln -sfn "$REPO/$p/out" "$D/ws/$p/out"; done
printf '%s\n' "${ALLOWED[@]}" > "$D/allowed.txt"
S="{\"permissions\":{\"allow\":[\"command(.venv/bin/python)\",\"command(ls)\",\"command(cat)\",\"command(head)\",\"command(grep)\",\"command(wc)\",\"read_file($REPO)\"]}}"
RULES="$(cat "$REPO/tools/agy/worker_rules.md")
YOU OWN (create/edit only these paths): ${ALLOWED[*]}"
cd "$D/ws" && agy -p "$RULES

$(cat "$BRIEF")" --model gemini-3.8-flash-high --mode accept-edits --settings "$S" \
  --print-timeout "${AGY_TIMEOUT:-1500}s" > "$D/agy.log" 2>&1
echo "exit=$? $(date +%T)" >> "$D/agy.log"
"$REPO/tools/agy/gate.sh" "$N"
