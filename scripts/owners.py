#!/usr/bin/env python3
"""Folder ownership, read from .github/CODEOWNERS (the single source of truth).

Used by the Claude Code hook (.claude/hooks/folder_guard.py) and by CI
(.github/workflows/folder-guard.yml). Supports the subset of CODEOWNERS syntax
this repo uses: `*`, `/dir/` and `/path/to/file`. The last matching line wins.

CI mode:  python3 scripts/owners.py ci   (env: ACTOR, BASE, HEAD, GH_TOKEN)
Lookup:   python3 scripts/owners.py who <path>
"""
import os
import posixpath
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_FILE_MB = 5


def parse(text):
    rules = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pat, *owners = line.split()
        rules.append((pat, [o.lstrip("@").lower() for o in owners]))
    return rules


def load(root=ROOT, rev=None):
    if rev:
        text = subprocess.run(["git", "-C", root, "show", f"{rev}:.github/CODEOWNERS"],
                              capture_output=True, text=True, check=True).stdout
    else:
        path = os.path.join(root, ".github", "CODEOWNERS")
        if not os.path.exists(path):   # ownership removed: nobody is restricted
            return []
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    return parse(text)


def _norm(rel):
    rel = posixpath.normpath(rel.replace(os.sep, "/"))
    return rel[2:] if rel.startswith("./") else rel


def _match(pat, rel):
    if pat == "*":
        return True
    p = pat.lstrip("/")
    if p.endswith("/"):
        return rel.startswith(p)
    return rel == p or rel.startswith(p + "/")


def owners_for(rel, rules):
    rel = _norm(rel)
    found = []
    for pat, owners in rules:
        if _match(pat, rel):
            found = owners
    return found


def whoami(root=ROOT):
    """GitHub login of the person running this checkout (cached in .claude/me)."""
    cache = os.path.join(root, ".claude", "me")
    try:
        with open(cache, encoding="utf-8") as fh:
            me = fh.read().strip().lower()
            if me:
                return me
    except OSError:
        pass
    try:
        me = subprocess.run(["gh", "api", "user", "-q", ".login"], capture_output=True,
                            text=True, timeout=15).stdout.strip().lower()
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if me:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w", encoding="utf-8") as fh:
            fh.write(me + "\n")
    return me


def _git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True).stdout


def ci():
    actor = os.environ["ACTOR"].lower()
    base, head = os.environ.get("BASE", ""), os.environ["HEAD"]
    if not base or set(base) == {"0"}:
        print("First push of the branch: nothing to compare, skipping.")
        return 0
    try:
        rules = load(rev=base)  # rules as they were BEFORE this push (can't grant yourself rights)
    except subprocess.CalledProcessError:
        print("No .github/CODEOWNERS before this push (bootstrap commit): skipping.")
        return 0
    files = [f for f in _git("diff", "--name-only", "--no-renames", f"{base}...{head}").splitlines() if f]

    outside, data, big = [], [], []
    for f in files:
        if f.startswith("data/"):
            data.append(f)
        own = owners_for(f, rules)
        if actor not in own:
            outside.append((f, own))
        size = _git("cat-file", "-s", f"{head}:{f}").strip()
        if size.isdigit() and int(size) > MAX_FILE_MB * 1024 * 1024:
            big.append((f, int(size) / 1024 / 1024))

    print(f"{len(files)} changed file(s) by @{actor}")
    if not (outside or data or big):
        print("OK: every change is inside a folder owned by the pusher.")
        return 0

    lines = [f"🚨 **[GUARD]** push by @{actor} (`{head[:7]}`) breaks the folder rules:", ""]
    for f, own in outside:
        lines.append(f"- `{f}` belongs to {', '.join('@' + o for o in own) or 'nobody'}")
    for f in data:
        lines.append(f"- `{f}` is under `data/`: datasets must never be committed")
    for f, mb in big:
        lines.append(f"- `{f}` is {mb:.1f} MB (limit {MAX_FILE_MB} MB): outputs/data don't belong in git")
    lines += ["", "Please revert these files (`git revert` or restore them) and push again."]
    body = "\n".join(lines)
    print(body)

    issue_file = os.path.join(ROOT, ".github", "BOARD_ISSUE")
    if os.path.exists(issue_file):
        issue = open(issue_file, encoding="utf-8").read().strip()
        if issue:
            subprocess.run(["gh", "issue", "comment", issue, "--body", body], cwd=ROOT)
    return 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "ci":
        sys.exit(ci())
    if cmd == "who" and len(sys.argv) > 2:
        print(" ".join("@" + o for o in owners_for(sys.argv[2], load())) or "(nobody)")
        sys.exit(0)
    print(__doc__)
    sys.exit(1)
