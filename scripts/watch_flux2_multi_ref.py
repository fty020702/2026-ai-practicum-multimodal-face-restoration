from __future__ import annotations

"""Live panel for the FLUX.2 multi-reference-image verification.

    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_flux2_multi_ref.py --once
    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_flux2_multi_ref.py --interval 5
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT

PROGRESS = ROOT / "outputs/logs/flux2_multi_ref_progress.json"
LOG = ROOT / "outputs/logs/flux2_multi_ref_demo.log"


def gpu() -> str:
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=6,
        ).strip().splitlines()[0]
        u, f, x = [s.strip() for s in o.split(",")[:3]]
        return f"VRAM used {u} MiB / free {f} MiB | util {x}%"
    except Exception as e:                                   # noqa: BLE001
        return f"n/a ({type(e).__name__})"


def render() -> str:
    L = ["=" * 76,
         " FLUX.2 [klein] 4B Base  |  MULTI-REFERENCE verification",
         " is the 2nd reference image actually READ?",
         "=" * 76]
    if PROGRESS.is_file():
        try:
            d = json.loads(PROGRESS.read_text(encoding="utf-8"))
            L.append(f" status : {d.get('status')}   size={d.get('size')} "
                     f"steps={d.get('steps')} guidance={d.get('guidance')} "
                     f"seed={d.get('seed')}")
            if d.get("load_seconds"):
                L.append(f" load   : {d['load_seconds']}s")
            L.append(f" cases  : {len(d.get('cases', []))}/5")
            for c in d.get("cases", []):
                refs = "+".join(f"r{i}" for i in c["refs"])
                L.append(f"   {c['id']:<12} image=[{refs:<5}] "
                         f"{c['seconds']:6.1f}s  {Path(c['output']).name}")
            if d.get("pairwise_rms"):
                L.append(" RMS (output pixel differences):")
                for k, v in d["pairwise_rms"].items():
                    L.append(f"   {k:<28} {v:.5f}")
            if d.get("grid"):
                L.append(f" grid   : {d['grid']}")
            for e in d.get("errors", []):
                L.append(f" ! {e}")
        except Exception as e:                               # noqa: BLE001
            L.append(f" progress parse: {type(e).__name__}")
    else:
        L.append(" waiting for progress file ...")
    L.append(" " + gpu())
    if LOG.is_file():
        rows = [l.strip() for l in LOG.read_text(
            encoding="utf-8", errors="replace").splitlines()
            if "[multi-ref]" in l or "Traceback" in l or "Error" in l]
        L.extend("   " + r for r in rows[-7:])
    L.append("=" * 76)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
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
