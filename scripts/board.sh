#!/usr/bin/env bash
# Team message board = one pinned GitHub issue (number in .github/BOARD_ISSUE).
#
#   scripts/board.sh done "<what landed, where (path), score if any, what's next>"
#   scripts/board.sh need @user "<what you need from them, and whether it blocks you>"
#   scripts/board.sh info "<heads-up that changes things for others>"
#   scripts/board.sh read [N]     # last N posts (default 15)
#   scripts/board.sh mine         # posts that tag you
#   scripts/board.sh brief        # used by the SessionStart hook
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
ISSUE="$(tr -d '[:space:]' < .github/BOARD_ISSUE 2>/dev/null || true)"
if [ -z "$ISSUE" ]; then
  [ "${1:-}" = "brief" ] && exit 0
  echo "No board issue configured (.github/BOARD_ISSUE is empty)." >&2; exit 1
fi

ME="$(python3 -c 'import sys; sys.path.insert(0, "scripts"); import owners; print(owners.whoami())')"
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo '-')"

post() { gh issue comment "$ISSUE" --body "$1" >/dev/null && echo "Posted to board (issue #$ISSUE)."; }

fmt='"── @\(.author.login) · \(.createdAt[0:16] | sub("T";" ")) UTC\n\(.body)\n"'

case "${1:-}" in
  done)
    shift; [ $# -ge 1 ] || { echo "usage: board.sh done \"message\"" >&2; exit 1; }
    post "$(printf '🤖 **[DONE]** agent of **%s** · `%s`\n\n%s' "$ME" "$SHA" "$*")" ;;
  need)
    shift; TO="${1:-}"
    [[ "$TO" == @* ]] && [ $# -ge 2 ] || { echo "usage: board.sh need @user \"message\"" >&2; exit 1; }
    shift
    post "$(printf '🤖 **[NEED]** agent of **%s** → %s · `%s`\n\n%s' "$ME" "$TO" "$SHA" "$*")" ;;
  info)
    shift; [ $# -ge 1 ] || { echo "usage: board.sh info \"message\"" >&2; exit 1; }
    post "$(printf '🤖 **[INFO]** agent of **%s** · `%s`\n\n%s' "$ME" "$SHA" "$*")" ;;
  read)
    N="${2:-15}"
    gh issue view "$ISSUE" --json comments --jq ".comments[-$N:][] | $fmt" ;;
  mine)
    gh issue view "$ISSUE" --json comments --jq ".comments[] | select(.body | test(\"@$ME\\\\b\"; \"i\")) | $fmt" ;;
  brief)
    echo "=== TEAM BOARD (issue #$ISSUE) — you are @$ME. Last 8 posts: ==="
    gh issue view "$ISSUE" --json comments --jq ".comments[-8:][] | $fmt" 2>/dev/null \
      || echo "(could not reach GitHub)"
    echo "=== Answer any [NEED] that tags @$ME. Post with scripts/board.sh (see CLAUDE.md). ===" ;;
  *)
    sed -n '2,10p' "$0"; exit 1 ;;
esac
