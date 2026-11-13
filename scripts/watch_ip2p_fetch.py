from __future__ import annotations

"""Live terminal panel for scripts/fetch_ip2p_shard.py.

    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_ip2p_fetch.py
    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_ip2p_fetch.py --once
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT

P = ROOT / "outputs" / "logs" / "ip2p_fetch_progress.json"
BAR = 42


def render() -> str:
    if not P.is_file():
        return "waiting for download to start ..."
    d = json.loads(P.read_text(encoding="utf-8"))
    got, total = d.get("bytes", 0), d.get("total_bytes", 1)
    pct = d.get("pct", 0.0)
    n = int(BAR * min(pct, 100) / 100)
    lines = [
        "=" * 62,
        " InstructPix2Pix clip-filtered  shard 0  (~500 MB)",
        "=" * 62,
        f" stage   : {d.get('stage')}",
        f" progress: |{'#' * n}{'.' * (BAR - n)}| {pct:5.1f}%",
        f" bytes   : {got / 1e6:.2f} / {total / 1e6:.2f} MB",
        f" note    : {d.get('msg', '')}",
        f" updated : {d.get('ts', '')}",
        "=" * 62,
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=3.0)
    a = ap.parse_args()
    if a.once:
        print(render(), flush=True)
        return 0
    try:
        while True:
            os.system("cls") if sys.platform == "win32" else os.system("clear")
            print(render(), flush=True)
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
