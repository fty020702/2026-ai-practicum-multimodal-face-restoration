from __future__ import annotations

"""Live terminal dashboard for REAL GPU training on FLUX.2 [klein] 4B.

    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_gpu_train.py --interval 5
    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_gpu_train.py --once --log gpu_train_smoke.log

Read-only: it never starts, stops or modifies the training process.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "logs"


def gpu_line() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=6,
        ).strip().splitlines()[0]
        used, free, util = [x.strip() for x in out.split(",")[:3]]
        return f"{used:>6} / {free:>6} MiB free | util {util:>3}%"
    except Exception as e:
        return f"n/a ({type(e).__name__})"


def history_line(out_dir: Path) -> str:
    p = out_dir / "train_history.json"
    if not p.is_file():
        return "pending"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        h = d.get("history", [])
        if not h:
            return "running (no logged step yet)"
        last = h[-1]
        return (f"step {last.get('step')}/{d.get('steps')}  "
                f"loss={last.get('loss')}  elapsed={last.get('elapsed_s')}s")
    except Exception:
        return "writing..."


def ckpt_count(out_dir: Path) -> str:
    if not out_dir.is_dir():
        return "0"
    return str(len(list(out_dir.glob("cond_step*.pt"))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--log", default="gpu_train_smoke.log",
                    help="log filename under outputs/logs/")
    ap.add_argument("--out-dir", default="outputs/checkpoints/flux2_klein_smoke",
                    help="checkpoint dir (relative to project root)")
    args = ap.parse_args()
    log_path = LOGS / args.log
    out_dir = ROOT / args.out_dir
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    while True:
        lines = [
            "=" * 78,
            "  FLUX.2 [klein] 4B Base - REAL GPU TRAINING (live, read-only)",
            "  " + time.strftime("%Y-%m-%d %H:%M:%S"),
            f"  log: {log_path.name}",
            "=" * 78,
            f"  GPU used / free / util : {gpu_line()}",
            f"  checkpoints            : {ckpt_count(out_dir)}",
            f"  train_history.json     : {history_line(out_dir)}",
            "-" * 78,
        ]
        if log_path.is_file():
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-14:]
            lines.extend("  | " + t for t in tail)
        else:
            lines.append("  | (run log not created yet)")
        lines.append("=" * 78)
        print("\n".join(lines), flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
