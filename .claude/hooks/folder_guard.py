#!/usr/bin/env python3
"""Claude Code PreToolUse hook: block file writes outside the folders you own.

Ownership comes from .github/CODEOWNERS; "you" is your GitHub login (gh api user,
cached in .claude/me). Exit code 2 blocks the tool call and shows the reason to the agent.
Files outside the repo (e.g. /tmp, scratchpads) are not checked.
"""
import json
import os
import sys

root = os.path.realpath(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
sys.path.insert(0, os.path.join(root, "scripts"))
import owners  # noqa: E402

LOCAL_OK = {".claude/settings.local.json", ".claude/me"}


def block(msg):
    print(msg, file=sys.stderr)
    sys.exit(2)


event = json.load(sys.stdin)
ti = event.get("tool_input") or {}
path = ti.get("file_path") or ti.get("notebook_path")
if not path:
    sys.exit(0)

path = os.path.realpath(os.path.join(event.get("cwd") or root, os.path.expanduser(path)))
if not path.startswith(root + os.sep):
    sys.exit(0)
rel = os.path.relpath(path, root).replace(os.sep, "/")

if rel in LOCAL_OK:
    sys.exit(0)
if rel == "data" or rel.startswith("data/"):
    block("BLOCKED: data/ is read-only for everyone. Write derived files to <your-folder>/out/ instead.")

me = owners.whoami(root)
if not me:
    block("BLOCKED: can't tell who you are. Run `scripts/setup.sh` (needs `gh auth login`).")

own = owners.owners_for(rel, owners.load(root))
if me not in own:
    who = ", ".join("@" + o for o in own) or "nobody"
    block(
        f"BLOCKED: `{rel}` belongs to {who}; you are @{me}. Write only inside your own folder(s) "
        f"(see CLAUDE.md). If you need a change there, ask on the board: "
        f"scripts/board.sh need @{own[0] if own else 'OWNER'} \"<what you need and why>\""
    )
sys.exit(0)
