"""Pre-train zoo members + stackers for a list of phases (memory-checked; one fit at a time).

    nice -n 10 .venv/bin/python classification/zoo_fit.py --phases val,hold --members lgb,lgbr
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from classification import categories as C  # noqa: E402
from classification import zoo as Z  # noqa: E402


def free_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pg = 16384
    v = {}
    for line in out.splitlines():
        if ":" in line:
            k, x = line.split(":", 1)
            try:
                v[k.strip()] = int(x.strip().rstrip("."))
            except ValueError:
                pass
    return (v.get("Pages free", 0) + v.get("Pages inactive", 0)) * pg / 2**30


def wait_mem(min_gb: float = 2.0):
    while (g := free_gb()) < min_gb:
        Z.log(f"free+inactive {g:.1f} GB < {min_gb} GB: pausing 30 s")
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="val,hold")
    ap.add_argument("--members", default="lgb,lgbr")
    ap.add_argument("--stackers", default="1")
    a = ap.parse_args()
    members = [m for m in a.members.split(",") if m]
    wait_mem()
    tc = Z.train_cache()
    for ph in a.phases.split(","):
        cut = C.CUTOFF[ph]
        end = Z.eff_cutoff(cut)
        c0 = Z.shift6(end)
        for c in (c0, cut):
            for m in members:
                wait_mem()
                Z.fit_members(c, [m], tc)
        if a.stackers == "1":
            wait_mem()
            for ms in ([], *[[m] for m in members]):
                Z.fit_stacker(cut, ms, tc, pop=True)
            for m in members:
                if m != "lgbr":
                    Z.fit_stacker(cut, [m], tc)
            full = [m for m in members if m != "lgbr"]
            if len(full) > 1:
                Z.fit_stacker(cut, full, tc)
        Z.log(f"{ph}: done (free {free_gb():.1f} GB)")


if __name__ == "__main__":
    main()
