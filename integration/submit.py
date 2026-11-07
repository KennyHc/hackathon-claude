#!/usr/bin/env python3
"""Submit a file to the leaderboard (reference flow from https://hackathon.usetalky.com/enviar).

    read -rs HK_TOKEN && export HK_TOKEN            # token in an env var ONLY, never in code/files/commits
    python3 integration/submit.py LIVE integration/out/live/submission.csv.gz
    python3 integration/submit.py FINAL integration/out/final/submission.csv.gz --i-am-sure   # ONE shot

Validates locally first (integration/validate.py) and refuses to send an invalid file.
"""
from __future__ import annotations

import json
import os
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from integration.validate import validate  # noqa: E402

API = "https://hackathon.usetalky.com/api"
DONE = {"scored", "received", "invalid", "error"}


def main(phase: str, path: str, sure: bool) -> None:
    if phase == "FINAL" and not sure:
        sys.exit("FINAL is a single shot: re-run with --i-am-sure after the checks in issue #27.")
    if not validate(phase.lower(), path):
        sys.exit("local validation failed: not sending")
    token = os.environ.get("HK_TOKEN")
    if not token:
        sys.exit("HK_TOKEN is not set: run `read -rs HK_TOKEN && export HK_TOKEN` in your own terminal")
    headers = {"Authorization": f"Bearer {token}"}
    name = os.path.basename(path)
    r = requests.post(f"{API}/submissions/upload-url", headers=headers, json={"phase": phase, "filename": name}, timeout=30)
    if r.status_code != 201:
        sys.exit(f"upload-url failed: {r.status_code} {r.text}")
    up = r.json()
    sub_id = up["sub_id"]
    with open(path, "rb") as f:
        s3 = requests.post(up["upload"]["url"], data=up["upload"]["fields"], files={"file": (name, f)}, timeout=900)
    if s3.status_code not in (200, 201, 204):
        sys.exit(f"S3 upload failed: {s3.status_code} {s3.text[:500]}")
    r = requests.post(f"{API}/submissions/{sub_id}/score", headers=headers, json={}, timeout=30)
    if r.status_code >= 400:
        sys.exit(f"score failed: {r.status_code} {r.text}")
    sub = r.json()
    deadline = time.time() + 600
    while sub.get("status") not in DONE and time.time() < deadline:
        time.sleep(5)   # the whole venue shares one IP: never poll faster
        sub = requests.get(f"{API}/submissions/{sub_id}", headers=headers, timeout=30).json()
    print(json.dumps(sub, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(args) != 2 or args[0] not in ("LIVE", "FINAL"):
        sys.exit("usage: python3 integration/submit.py LIVE|FINAL file.csv[.gz] [--i-am-sure]")
    main(args[0], args[1], "--i-am-sure" in sys.argv)
