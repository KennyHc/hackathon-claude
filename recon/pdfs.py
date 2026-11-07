"""Extract text from all PDFs of a phase once (pdftotext -layout) and cache it as one gzipped JSONL file.

    python -m recon.pdfs live train      # writes data/cache/pdftext_<phase>.jsonl.gz
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE = os.path.join(DATA, "cache")


def _text(path: str) -> str:
    try:
        return subprocess.run(["pdftotext", "-layout", "-q", path, "-"], capture_output=True, timeout=30).stdout.decode("utf-8", "replace")
    except Exception:
        return ""


def extract(phase: str, folder: str = "pdfs", workers: int = 10) -> str:
    root = os.path.join(DATA, phase, folder)
    out = os.path.join(CACHE, f"pdftext_{phase}{'' if folder == 'pdfs' else '_' + folder}.jsonl.gz")
    os.makedirs(CACHE, exist_ok=True)
    files = []
    for dp, _, fns in os.walk(root):
        files += [os.path.relpath(os.path.join(dp, f), root) for f in fns if f.lower().endswith(".pdf")]
    files.sort()
    tmp = out + ".tmp"
    with ThreadPoolExecutor(workers) as ex, gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for i, (f, txt) in enumerate(zip(files, ex.map(lambda f: _text(os.path.join(root, f)), files, chunksize=64))):
            fh.write(json.dumps({"file": f, "text": txt}, ensure_ascii=False) + "\n")
            if i % 10000 == 0:
                print(f"{phase}/{folder}: {i}/{len(files)}", flush=True)
    os.replace(tmp, out)
    print(f"{phase}/{folder}: {len(files)} PDFs -> {out}", flush=True)
    return out


def load(phase: str, folder: str = "pdfs") -> dict[str, str]:
    p = os.path.join(CACHE, f"pdftext_{phase}{'' if folder == 'pdfs' else '_' + folder}.jsonl.gz")
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return {(r := json.loads(line))["file"]: r["text"] for line in fh}


if __name__ == "__main__":
    for ph in sys.argv[1:] or ["live", "train"]:
        extract(ph, "pdfs")
        extract(ph, "remittance_advices")
