from __future__ import annotations

"""Pollable live panel for the 500 MB InstructPix2Pix training run.

    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_ip2p_train.py --once
    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_ip2p_train.py --interval 5

Read-only. Reads outputs/logs/ip2p_train.log, checkpoint dir and nvidia-smi.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT

LOG = ROOT / "outputs" / "logs" / "ip2p_train.log"
CK = ROOT / "outputs" / "checkpoints" / "ip2p_500m"
DATA = ROOT / "outputs" / "logs" / "ip2p_500m_example.json"


def gpu() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=6,
        ).strip().splitlines()[0]
        used, free, util = [x.strip() for x in out.split(",")[:3]]
        return f"VRAM used {used} MiB / free {free} MiB | util {util}%"
    except Exception as e:                              # noqa: BLE001
        return f"n/a ({type(e).__name__})"


def render() -> str:
    lines = ["=" * 66,
             " IP2P 500MB shard -> local training (toy, CUDA)",
             "=" * 66]
    if DATA.is_file():
        d = json.loads(DATA.read_text(encoding="utf-8"))
        lines.append(f" dataset : {d['shard_bytes'] / 1e6:.2f} MB / "
                     f"{d['shard_rows']} rows / using {d['rows_used_for_training']} "
                     f"/ {d['distinct_edit_prompts']} distinct prompts")
    if LOG.is_file():
        steps = [l.strip() for l in LOG.read_text(
            encoding="utf-8", errors="replace").splitlines()
            if "[train] step" in l or "[train] HIGH-NOISE" in l
            or "[train] done" in l]
        lines.append(" " + gpu())
        if steps:
            lines.append(" last steps:")
            lines.extend("   " + s for s in steps[-6:])
        else:
            lines.append(" (loading dataset / building system ...)")
    else:
        lines.append(" waiting for log ...")
    n = len(list(CK.glob("cond_step*.pt"))) if CK.is_dir() else 0
    lines.append(f" checkpoints written: {n}")
    lines.append("=" * 66)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    a = ap.parse_args()
    if a.once:
        print(render(), flush=True)
        return 0
    import os
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
