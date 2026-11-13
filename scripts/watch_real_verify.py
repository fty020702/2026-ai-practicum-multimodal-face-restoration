from __future__ import annotations

"""Live terminal dashboard for the real-weight FLUX.2 verification chain.

    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_real_verify.py --interval 3
    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_real_verify.py --once

Reads the stage reports + the run log written by the background chain.
Read-only: it never starts, stops or modifies any verification task.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "logs" / "real_verify_run.log"
REPORTS = [
    ("download", ROOT / "outputs" / "logs" / "download_state.json"),
    ("inspect", ROOT / "outputs" / "logs" / "backbone_inspect.json"),
    ("precompute", ROOT / "outputs" / "logs" / "precompute_prompts.json"),
    ("latent", ROOT / "outputs" / "logs" / "latent_roundtrip.json"),
    ("forward", ROOT / "outputs" / "logs" / "verify_real_forward.json"),
    ("warmup", ROOT / "outputs" / "logs" / "text_warmup.json"),
    ("text-effect", ROOT / "outputs" / "logs" / "real_text_effect.json"),
    ("preflight", ROOT / "outputs" / "logs" / "verify_real_backend.json"),
]


def status(p: Path) -> str:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return "pending"
    for k in ("status", "verdict"):
        if k in d:
            return str(d[k])
    return "OK" if d.get("ok") else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--log", default="real_verify_run.log",
                    help="log filename under outputs/logs/ (default: real_verify_run.log)")
    args = ap.parse_args()
    log_path = ROOT / "outputs" / "logs" / args.log
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    while True:
        lines = [
            "=" * 74,
            "  FLUX.2 [klein] 4B Base - REAL-WEIGHT VERIFICATION (live)",
            "  " + time.strftime("%Y-%m-%d %H:%M:%S"),
            f"  log: {log_path.name}",
            "=" * 74,
        ]
        for name, p in REPORTS:
            lines.append(f"  {name:<12} {status(p):<30} {p.name}")
        lines.append("-" * 74)
        if log_path.is_file():
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-16:]
            lines.extend("  | " + t for t in tail)
        else:
            lines.append("  | (run log not created yet)")
        lines.append("=" * 74)
        print("\n".join(lines), flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
