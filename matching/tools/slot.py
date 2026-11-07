"""Run a heavy command only when one of N machine-wide slots is free (16 GB box: a val run peaks ~1.5 GB, >2 at once crashed it).

    python matching/tools/slot.py -- python matching/run.py --phase val
    python matching/tools/slot.py --slots 2 --label fx -- python matching/tests/shortpay/weak_precision.py

Slots are directories under SLOT_DIR (default ~/.recon_slots), created atomically with os.mkdir; a slot whose owner PID is
gone is reclaimed. Waits (printing once a minute) until a slot is free, runs the command, frees the slot, exits with its code.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

SLOT_DIR = os.environ.get("RECON_SLOT_DIR", os.path.join(os.path.expanduser("~"), ".recon_slots"))


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire(n: int, label: str) -> str:
    os.makedirs(SLOT_DIR, exist_ok=True)
    waited, last = 0, -60
    while True:
        for k in range(n):
            d = os.path.join(SLOT_DIR, f"slot{k}")
            try:
                os.mkdir(d)
                with open(os.path.join(d, "owner"), "w") as fh:
                    fh.write(f"{os.getpid()} {label} {time.strftime('%H:%M:%S')}")
                return d
            except FileExistsError:
                try:
                    with open(os.path.join(d, "owner")) as fh:
                        pid = int(fh.read().split()[0])
                except (OSError, ValueError, IndexError):
                    pid = -1
                    if time.time() - os.path.getmtime(d) < 30:  # being created right now
                        continue
                if not _alive(pid):
                    _release(d)
        if waited - last >= 60:
            print(f"[slot] {label}: all {n} slots busy, waiting ({waited}s)", flush=True)
            last = waited
        time.sleep(5)
        waited += 5


def _release(d: str) -> None:
    try:
        os.remove(os.path.join(d, "owner"))
    except OSError:
        pass
    try:
        os.rmdir(d)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=int(os.environ.get("RECON_SLOTS", "2")))
    ap.add_argument("--label", default=os.path.basename(os.getcwd()))
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command")
    d = _acquire(a.slots, a.label)
    t0 = time.time()
    try:
        rc = subprocess.call(cmd)
    finally:
        _release(d)
    print(f"[slot] {a.label}: done in {time.time() - t0:.0f}s (exit {rc})", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
