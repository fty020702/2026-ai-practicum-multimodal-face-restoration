from __future__ import annotations

"""Pollable live panel: generated OLD-PHOTO multi-target training + A/B probe.

    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_old_photo_train.py --once
    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_old_photo_train.py --interval 5

Read-only. Watches two training logs, two checkpoint dirs, the dataset card
and the ambiguity diagnostic. Never starts / modifies a training run.
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

LOGS = ROOT / "outputs" / "logs"
CKPT = ROOT / "outputs" / "checkpoints"
INFO = ROOT / "data" / "old_photo_multitarget" / "dataset_info.json"

RUNS = [
    ("A-class  (K distinct GTs)", "old_photo_multitarget_train.log",
     "old_photo_multitarget"),
    ("CONTROL  (collapsed GT)  ", "old_photo_multitarget_control_train.log",
     "old_photo_multitarget_control"),
]
DIAG = LOGS / "ambiguity_old_photo_ab_g32.json"   # prefer larger probe set
DIAG_FALLBACK = LOGS / "ambiguity_old_photo_ab.json"


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


def last_steps(path: Path, n: int = 3) -> list[str]:
    if not path.is_file():
        return ["  (waiting for log ...)"]
    rows = [l.strip() for l in path.read_text(
        encoding="utf-8", errors="replace").splitlines()
        if "[train] step" in l or "[train] done" in l or "Traceback" in l]
    return ["  " + s for s in rows[-n:]] if rows else [
        "  (building system / loading JPEGs ...)"]


def render() -> str:
    L = ["=" * 72,
         " OLD PHOTO (generated) + TEXT -> 4 HQ TARGETS  |  train + strict A/B",
         "=" * 72]
    if INFO.is_file():
        d = json.loads(INFO.read_text(encoding="utf-8"))
        L.append(f" dataset : {d['n_groups']} groups x K={d['K_targets_per_group']} "
                 f"= {d['n_samples']} samples | "
                 f"{d['jpg_bytes_total'] / 1e6:.1f} MB | "
                 f"inter-target RMS {d['inter_target_rms_mean']}")
    L.append(" " + gpu())
    for name, log, ck in RUNS:
        p = LOGS / log
        cdir = CKPT / ck
        n = len(list(cdir.glob("cond_step*.pt"))) if cdir.is_dir() else 0
        L.append(f"- {name}  checkpoints={n}")
        L.extend(last_steps(p))
    diag = DIAG if DIAG.is_file() else DIAG_FALLBACK
    if diag.is_file():
        try:
            d = json.loads(diag.read_text(encoding="utf-8"))
            L.append(" diagnostic: " + str(d.get("verdict", "?"))[:120])
            for k, v in d.get("checkpoints", {}).items():
                m = v["means"]
                L.append(f"   {Path(k).parent.name:<34} "
                         f"penalty/gap={m['penalty_to_gap_ratio'] * 100:7.2f}%  "
                         f"selectivity={m['style_selectivity'] * 100:5.2f}%")
        except Exception as e:                          # noqa: BLE001
            L.append(f" diagnostic: parsing ({type(e).__name__})")
    else:
        L.append(" diagnostic: pending")
    L.append("=" * 72)
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
