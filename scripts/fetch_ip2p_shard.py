from __future__ import annotations

"""Fetch ONE ~500 MB shard of the OFFICIAL InstructPix2Pix clip-filtered data.

Why this shard
--------------
`docs/一图多目标数据集调研.md` ranked InstructPix2Pix as the most reproducible
"one source image + instruction -> target image" paradigm. The full
clip-filtered release is 436 GB / 313,010 examples, but HuggingFace exposes it
as 262 individual parquet shards of ~490-505 MB each. Downloading ONE shard
gives a REAL, public, ~500 MB dataset without touching the rest.

    https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered

Triplet columns (official):
    original_prompt, original_image, edit_prompt, edited_prompt, edited_image

    D:/Anaconda/envs/ChatAI/python.exe scripts/fetch_ip2p_shard.py
    D:/Anaconda/envs/ChatAI/python.exe scripts/fetch_ip2p_shard.py --shard 1

Writes a resumable .incomplete file, then os.replace() (a RENAME, never a
delete) to stay sandbox-friendly. Progress -> outputs/logs/ip2p_fetch_progress.json
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT

REPO = "timbrooks/instructpix2pix-clip-filtered"
# Exact filenames verified 2026-11-13 via the HuggingFace tree API.
SHARDS = {
    0: "data/train-00000-of-00262-57cebf95b4a9170c.parquet",
    1: "data/train-00001-of-00262-b4c37264e7cef478.parquet",
    2: "data/train-00002-of-00262-b42012f7e4ae416f.parquet",
}
SIZES = {0: 499_830_000, 1: 489_230_000, 2: 503_850_000}  # approximate, API metadata
UA = {"User-Agent": "Mozilla/5.0 (WorkBuddy IP2P subset builder)"}


def progress(stage: str, got: int, total: int, msg: str) -> None:
    p = ROOT / "outputs" / "logs" / "ip2p_fetch_progress.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "stage": stage, "bytes": got, "total_bytes": total,
        "pct": round(100.0 * got / max(total, 1), 2),
        "msg": msg, "ts": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False), encoding="utf-8")
    print(f"[{stage}] {got / 1e6:8.2f}/{total / 1e6:.2f} MB "
          f"({100.0 * got / max(total, 1):5.1f}%) {msg}", flush=True)


def fetch(idx: int, out: Path, retries: int = 6) -> int:
    rel = SHARDS[idx]
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{rel}"
    total = SIZES[idx]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_size > total * 0.98:
        progress("CACHED", out.stat().st_size, total, out.name)
        return 0

    part = out.with_name(out.name + ".incomplete")
    existing = part.stat().st_size if part.exists() else 0
    t0 = time.time()

    for attempt in range(1, retries + 1):
        try:
            headers = dict(UA)
            if existing:
                headers["Range"] = f"bytes={existing}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                partial = (r.status == 206)
                cl = int(r.headers.get("Content-Length", 0) or 0)
                if cl and not partial:
                    total = existing + cl
                mode = "ab" if (existing and partial) else "wb"
                with open(part, mode) as f:
                    if mode == "wb":
                        existing = 0
                    last_report = existing
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        existing += len(chunk)
                        if existing - last_report >= 8 << 20:
                            last_report = existing
                            progress("DOWNLOAD", existing, total,
                                     f"{existing / 1e6 / max(time.time() - t0, 1):.1f} MB/s")
            if part.stat().st_size == 0:
                raise RuntimeError("empty response")
            os.replace(part, out)
            progress("DONE", out.stat().st_size, out.stat().st_size,
                     f"{out.name} ({time.time() - t0:.1f}s)")
            return 0
        except Exception as e:                                # noqa: BLE001
            existing = part.stat().st_size if part.exists() else 0
            progress("RETRY", existing, total,
                     f"attempt {attempt}/{retries}: {type(e).__name__}: {str(e)[:90]}")
            if attempt == retries:
                progress("FAILED", existing, total, str(e)[:150])
                return 1
            time.sleep(2.0 * attempt)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0, choices=sorted(SHARDS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = (Path(a.out) if a.out else
           ROOT / "data" / "ip2p_500m" / "raw" /
           Path(SHARDS[a.shard]).name)
    print(f"[fetch] shard={a.shard} -> {out}", flush=True)
    return fetch(a.shard, out)


if __name__ == "__main__":
    sys.exit(main())
