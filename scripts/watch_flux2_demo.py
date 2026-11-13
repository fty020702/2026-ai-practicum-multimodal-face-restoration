from __future__ import annotations

"""Live panel for the real FLUX.2 old-photo inference.

    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_flux2_demo.py --once
    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_flux2_demo.py \
        --tag old_photo_edit --interval 5
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


def render(tag: str) -> str:
    progress = ROOT / "outputs" / "logs" / f"flux2_{tag}_progress.json"
    log = ROOT / "outputs" / "logs" / f"flux2_{tag}_demo.log"
    L = ["=" * 70,
         f" REAL FLUX.2 [klein] 4B Base  |  tag={tag}",
         "=" * 70]
    if progress.is_file():
        try:
            d = json.loads(progress.read_text(encoding="utf-8"))
            L.append(f" status : {d.get('status')}   prompt_set={d.get('prompt_set')} "
                     f"steps={d.get('num_inference_steps')} "
                     f"size={d.get('size')} guidance={d.get('guidance_scale')}")
            if d.get("load_seconds"):
                L.append(f" load   : {d['load_seconds']}s")
            L.append(f" images : {len(d.get('cases', []))}")
            for c in d.get("cases", []):
                L.append(f"   {c['id']:<16} {c['seconds']:6.1f}s  {c['output']}")
            if d.get("grid"):
                L.append(f" grid   : {d['grid']}")
            for e in d.get("errors", []):
                L.append(f" ! {e}")
        except Exception as e:                               # noqa: BLE001
            L.append(f" progress parse: {type(e).__name__}")
    else:
        L.append(" waiting for progress file ...")
    L.append(" " + gpu())
    if log.is_file():
        rows = [l.strip() for l in log.read_text(
            encoding="utf-8", errors="replace").splitlines()
            if "[flux2-demo]" in l or "Traceback" in l or "Error" in l]
        L.extend("   " + r for r in rows[-5:])
    L.append("=" * 70)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--tag", default="old_photo")
    a = ap.parse_args()
    if a.once:
        print(render(a.tag), flush=True)
        return 0
    try:
        while True:
            os.system("cls") if sys.platform == "win32" else os.system("clear")
            print(render(a.tag), flush=True)
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
