#!/usr/bin/env python3
"""Upload a submission to the leaderboard (integration owner only; run it yourself, in your own terminal).

    read -rs HK_TOKEN && export HK_TOKEN          # token only ever lives in this env var
    .venv/bin/python integration/submit.py integration/out/live/submission.csv.gz --phase LIVE
    .venv/bin/python integration/submit.py integration/out/final/submission.csv.gz --phase FINAL --final-i-am-sure

Refuses to upload unless integration/validate.py passes for that phase. Local quota guard (1 per 3 min, 20 per day per
phase; requesting an upload URL already burns a slot) from integration/out/submissions.csv, which logs every attempt
(never the token). FINAL is one shot: needs --final-i-am-sure, and refuses if a FINAL attempt was already accepted.
API: docs/HACKATHON.md.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integration.scoring import ROOT  # noqa: E402
from integration.validate import check  # noqa: E402

API = "https://hackathon.usetalky.com/api"
LOG = os.path.join(ROOT, "integration", "out", "submissions.csv")
DONE = {"scored", "received", "invalid", "error"}


def log_row(**kw):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    old = pd.read_csv(LOG, dtype=str) if os.path.exists(LOG) else None
    new = pd.DataFrame([{k: str(v) for k, v in kw.items()}])
    (new if old is None or not len(old) else pd.concat([old, new], ignore_index=True)).to_csv(LOG, index=False)


def quota_ok(phase, force):
    if not os.path.exists(LOG):
        return True
    h = pd.read_csv(LOG, dtype=str)
    h = h[h.phase == phase]
    if phase == "FINAL" and (h.status.isin(["scored", "received"])).any():
        print("✗ A FINAL submission was already accepted (one shot). Refusing.")
        return False
    if not len(h) or force:
        return True
    times = pd.to_datetime(h.time)
    now = dt.datetime.now()
    since = (now - times.max()).total_seconds()
    today = (times.dt.date == now.date()).sum()
    if since < 180:
        print(f"✗ last {phase} upload-url request was {since:.0f}s ago (limit: 1 per 3 min). Wait {180 - since:.0f}s.")
        return False
    if today >= 20:
        print(f"✗ {today} {phase} requests today (limit 20). Use --force-quota only if you know the server count differs.")
        return False
    print(f"  quota: {today} {phase} requests today, last {since / 60:.1f} min ago")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--phase", required=True, choices=["LIVE", "FINAL"])
    ap.add_argument("--final-i-am-sure", action="store_true")
    ap.add_argument("--force-quota", action="store_true", help="ignore the local quota guard")
    ap.add_argument("--dry-run", action="store_true", help="validate + quota check only, no network")
    ap.add_argument("--note", default="", help="what this file is, for the log")
    a = ap.parse_args()

    if a.phase == "FINAL" and not a.final_i_am_sure:
        sys.exit("✗ FINAL is one shot. Re-run with --final-i-am-sure.")
    print(f"validating {a.submission} for {a.phase} …")
    ok, probs, warns = check(a.submission, a.phase.lower())
    for w in warns:
        print("  WARN ", w)
    for p in probs:
        print("  ERROR", p)
    if not ok:
        sys.exit("✗ invalid: not uploading (fix it or use integration/validate.py --repair)")
    print("✓ valid")
    if not quota_ok(a.phase, a.force_quota):
        sys.exit(1)
    if a.dry_run:
        print("dry run: stopping before any request")
        return
    token = os.environ.get("HK_TOKEN")
    if not token:
        sys.exit("✗ HK_TOKEN is not set (read -rs HK_TOKEN && export HK_TOKEN)")
    hdr = {"Authorization": f"Bearer {token}"}
    name = os.path.basename(a.submission)
    rec = {"time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "phase": a.phase, "file": a.submission,
           "note": a.note, "sub_id": "", "status": "requested", "score": ""}

    r = requests.post(f"{API}/submissions/upload-url", json={"phase": a.phase, "filename": name}, headers=hdr, timeout=60)
    if r.status_code != 201:
        rec["status"] = f"upload-url {r.status_code}"
        log_row(**rec)
        sys.exit(f"✗ upload-url: HTTP {r.status_code}: {r.text[:500]}")
    j = r.json()
    sub_id, up = j["sub_id"], j["upload"]
    rec["sub_id"] = sub_id
    print(f"  sub_id {sub_id}: uploading …")
    with open(a.submission, "rb") as fh:
        u = requests.post(up["url"], data=up.get("fields", {}), files={"file": (name, fh)}, timeout=300)
    if u.status_code >= 300:
        rec["status"] = f"s3 {u.status_code}"
        log_row(**rec)
        sys.exit(f"✗ S3 upload: HTTP {u.status_code}: {u.text[:500]}")
    s = requests.post(f"{API}/submissions/{sub_id}/score", json={}, headers=hdr, timeout=60)
    if s.status_code >= 300:
        rec["status"] = f"score {s.status_code}"
        log_row(**rec)
        sys.exit(f"✗ score request: HTTP {s.status_code}: {s.text[:500]}")
    print("  scoring … (polling every 6 s)")
    res = {}
    for _ in range(200):
        time.sleep(6)
        g = requests.get(f"{API}/submissions/{sub_id}", headers=hdr, timeout=60)
        if g.status_code != 200:
            print(f"  poll HTTP {g.status_code}")
            continue
        res = g.json()
        if res.get("status") in DONE:
            break
    rec["status"] = res.get("status", "timeout")
    rec["score"] = res.get("score", "")
    log_row(**rec)
    print(json.dumps(res, indent=1)[:3000])
    print(f"→ {rec['status']}  score={rec['score']}   (logged to {os.path.relpath(LOG, ROOT)})")


if __name__ == "__main__":
    main()
