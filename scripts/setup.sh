#!/usr/bin/env bash
# One-time setup after cloning. Run from anywhere inside the repo:  scripts/setup.sh
set -euo pipefail
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"

command -v gh >/dev/null || { echo "✗ Install the GitHub CLI first: brew install gh"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "✗ Run: gh auth login"; exit 1; }
rm -f .claude/me
ME="$(python3 -c 'import sys; sys.path.insert(0, "scripts"); import owners; print(owners.whoami())')"
echo "✓ You are @$ME"

echo "✓ No folder ownership: every folder is open (coordinate on the board, see CLAUDE.md)"

missing=0
for d in masters sample train train_lite live; do
  [ -d "data/$d" ] || { echo "✗ missing data/$d"; missing=1; }
done
[ $missing -eq 0 ] && echo "✓ data/ looks complete" \
  || echo "  → Put the extracted dataset folders into data/ (see README.md). They are gitignored."

chmod +x scripts/*.sh
echo "✓ Done. Start Claude Code from the repo root: claude"
