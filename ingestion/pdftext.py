"""PDF text cache: extract the text of every PDF of a data folder once (pdftotext -layout), as one gzipped JSONL.

    python ingestion/pdftext.py final            # pdfs/ and remittance_advices/ of data/final
    python ingestion/pdftext.py live train       # several folders

`ingestion/run.py` calls `ensure(folder)` for the phase's folders, so FINAL gets its cache automatically. New caches go
to `ingestion/out/_cache/` (data/ stays read-only); `path()` also finds the older ones in `data/cache/`.
Readers: `matching/pdf_clauses.py` (sub "pdfs") and `matching/remittance_parse.py` (sub "remittance_advices").
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ingestion import lib  # noqa: E402

OUT_CACHE = os.path.join(ROOT, "ingestion", "out", "_cache")
SUBS = ("pdfs", "remittance_advices")


def _name(folder: str, sub: str) -> str:
    return f"pdftext_{folder}{'' if sub == 'pdfs' else '_' + sub}.jsonl.gz"


def path(folder: str, sub: str = "pdfs") -> str:
    """The cache file for (folder, sub): ingestion/out/_cache first, then the legacy data/cache; else the new path."""
    new = os.path.join(OUT_CACHE, _name(folder, sub))
    old = os.path.join(lib.DATA, "cache", _name(folder, sub))
    return new if os.path.exists(new) or not os.path.exists(old) else old


def _text(p: str) -> str:
    try:
        return subprocess.run(["pdftotext", "-layout", "-q", p, "-"], capture_output=True, timeout=30).stdout.decode("utf-8", "replace")
    except Exception:
        return ""


def _pdfs(root: str) -> list[str]:
    files = []
    for dp, _, fns in os.walk(root):
        files += [os.path.relpath(os.path.join(dp, f), root) for f in fns if f.lower().endswith(".pdf")]
    return sorted(files)


def extract(folder: str, sub: str = "pdfs", workers: int = 10, out: str | None = None) -> str | None:
    root = os.path.join(lib.DATA, folder, sub)
    if not os.path.isdir(root):
        return None
    if shutil.which("pdftotext") is None:
        print("[pdftext] ERROR: pdftotext not found (brew install poppler) — PDF clauses and PDF advices will be EMPTY")
        return None
    out = out or os.path.join(OUT_CACHE, _name(folder, sub))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    files, t = _pdfs(root), time.time()
    tmp = out + ".tmp"
    empty = 0
    with ThreadPoolExecutor(workers) as ex, gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for f, txt in zip(files, ex.map(lambda f: _text(os.path.join(root, f)), files, chunksize=64)):
            empty += not txt.strip()
            fh.write(json.dumps({"file": f, "text": txt}, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    print(f"[pdftext] {folder}/{sub}: {len(files):,} PDFs ({empty} without text) -> {out}  {time.time() - t:.0f}s")
    return out


def _cached_count(p: str) -> int:
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def ensure(folder: str) -> None:
    """Build the caches of a data folder that has PDFs but no cache yet, or a cache whose file count doesn't match the
    folder (a stale cache from a rehearsal or an earlier download)."""
    for sub in SUBS:
        root = os.path.join(lib.DATA, folder, sub)
        if not os.path.isdir(root):
            continue
        p = path(folder, sub)
        if not os.path.exists(p):
            extract(folder, sub)
        elif (n := len(_pdfs(root))) != (c := _cached_count(p)):
            print(f"[pdftext] {folder}/{sub}: cache has {c:,} files, folder has {n:,} -> rebuilding")
            extract(folder, sub)


if __name__ == "__main__":
    for fo in sys.argv[1:]:
        for s in SUBS:
            extract(fo, s)
