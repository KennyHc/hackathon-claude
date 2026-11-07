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

echo "✓ Your folders:"
python3 - "$ME" <<'PY'
import sys; sys.path.insert(0, "scripts"); import owners
me = sys.argv[1]
mine = [p for p, o in owners.load() if me in o and p != "*"]
print("\n".join("   " + p for p in mine) or "   (none yet: ask the admin to add you to .github/CODEOWNERS)")
PY

missing=0
for d in masters sample train train_lite live; do
  [ -d "data/$d" ] || { echo "✗ missing data/$d"; missing=1; }
done
[ $missing -eq 0 ] && echo "✓ data/ looks complete" \
  || echo "  → Put the extracted dataset folders into data/ (see README.md). They are gitignored."

echo "… Python environment (.venv, Python 3.12, requirements.txt)"
if command -v uv >/dev/null; then
  [ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv
  uv pip install -q --python .venv/bin/python -r requirements.txt
elif command -v python3.12 >/dev/null; then
  [ -x .venv/bin/python ] || python3.12 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
else
  echo "✗ Need uv (curl -LsSf https://astral.sh/uv/install.sh | sh) or python3.12 (brew install python@3.12)"; exit 1
fi
echo "✓ $(.venv/bin/python --version) ready: run code with .venv/bin/python (or: source .venv/bin/activate)"

chmod +x scripts/*.sh
echo "✓ Done. Start Claude Code from the repo root: claude"
