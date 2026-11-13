from __future__ import annotations

"""Live terminal panel for the FLUX.2 [klein] 4B Base weight download.

    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_download.py --once
    D:/Anaconda/envs/ChatAI/python.exe scripts/watch_download.py --interval 5

Reads outputs/logs/download_manifest.json + download_state.json (written by
scripts/download_models.py), scans the target folder (including huggingface_hub
*.incomplete files), and writes outputs/logs/download_progress.json so the
conversation can poll the same numbers.

Windows terminal rules (verified in this workspace): reconfigure stdout to
utf-8, clear with `cls` each refresh, print ONE full panel instead of appending.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT, save_json  # noqa: E402

STATE = ROOT / "outputs" / "logs" / "download_state.json"
MANIFEST = ROOT / "outputs" / "logs" / "download_manifest.json"
PROGRESS = ROOT / "outputs" / "logs" / "download_progress.json"

_PREV: tuple[float, int] | None = None


def _read_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _gb(n: float) -> str:
    return f"{n/1e9:.2f} GB"


def _hhmmss(sec: float) -> str:
    sec = max(0, int(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _gpu() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=8,
            creationflags=0x08000000 if os.name == "nt" else 0,
        ).strip().splitlines()[0]
        used, free, util = [x.strip() for x in out.split(",")[:3]]
        return f"used {used} MiB / free {free} MiB / util {util}%"
    except Exception:
        return "n/a"


def _disk() -> str:
    try:
        t, u, f = shutil.disk_usage(str(ROOT)[:3])
        return f"{f/1e9:.1f} GB free of {t/1e9:.1f} GB"
    except Exception:
        return "n/a"


def scan(state: dict, manifest: dict) -> dict:
    dst = Path(state.get("dst") or (ROOT / "models" / "FLUX.2-klein-4B-Base"))
    rows, done_bytes = [], 0
    for f in manifest.get("files", []):
        p = dst / f["path"]
        sz = p.stat().st_size if p.exists() else 0
        exp = int(f["size"])
        if exp > 0 and sz >= exp * 0.999:
            st = "DONE"
        elif sz > 0:
            st = "PARTIAL"
        else:
            st = "PENDING"
        done_bytes += min(sz, exp)
        rows.append((f["path"], sz, exp, st))

    inflight_bytes, n_inflight = 0, 0
    if dst.exists():
        for p in dst.rglob("*.incomplete"):
            if ".cache" in p.parts:      # leftovers from the aborted hub engine
                continue
            try:
                inflight_bytes += p.stat().st_size
                n_inflight += 1
            except OSError:
                pass

    total = int(manifest.get("total_bytes", 0)) or 1
    cur = min(total, done_bytes + inflight_bytes)
    pct = 100.0 * cur / total
    return {"dst": str(dst), "rows": rows, "done_bytes": done_bytes,
            "inflight_bytes": inflight_bytes, "n_inflight": n_inflight,
            "total_bytes": total, "cur_bytes": cur, "pct": pct}


def _rate_eta(state: dict, s: dict) -> tuple[float, float]:
    """Instantaneous rate from the previous refresh; fall back to average."""
    global _PREV
    now = time.time()
    cur = s["cur_bytes"]
    rate, eta = 0.0, float("inf")
    if _PREV is not None:
        dt = now - _PREV[0]
        db = cur - _PREV[1]
        if dt > 0 and db >= 0:
            rate = db / dt
            if rate > 0:
                eta = (s["total_bytes"] - cur) / rate
    if rate <= 0:
        t0 = state.get("started_ts")
        if t0:
            dt = now - float(t0)
            if dt > 0:
                rate = cur / dt
                if rate > 0:
                    eta = (s["total_bytes"] - cur) / rate
    _PREV = (now, cur)
    return rate, eta


def render(state: dict, manifest: dict, verify_note: bool = True) -> tuple[str, dict]:
    s = scan(state, manifest)
    rate, eta = _rate_eta(state, s)
    # The direct downloader writes its own instantaneous estimate; prefer it,
    # because a single --once poll has no previous sample to difference against.
    if state.get("rate_MBps") is not None:
        rate = float(state["rate_MBps"]) * 1e6
    if state.get("eta_s") is not None:
        eta = float(state["eta_s"])
    status = state.get("status", "NOT STARTED")
    t0 = state.get("started_ts")
    elapsed = (time.time() - float(t0)) if t0 else 0.0
    if state.get("status") == "done" and state.get("finished_ts") and t0:
        elapsed = float(state["finished_ts"]) - float(t0)

    bar_len = 48
    filled = int(bar_len * s["pct"] / 100)
    bar = "#" * filled + "-" * (bar_len - filled)

    L = []
    L.append("=" * 88)
    L.append(f" FLUX.2 [klein] 4B Base  |  REAL WEIGHTS  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("=" * 88)
    L.append(f" status   : {status.upper():<12} repo: {manifest.get('repo', state.get('repo', '-'))}")
    L.append(f" target   : {s['dst']}")
    L.append(f" elapsed  : {_hhmmss(elapsed)}      started {state.get('started_at', '-')}")
    L.append("-" * 88)
    L.append(f" [{bar}] {s['pct']:5.1f}%")
    L.append(f"   {_gb(s['cur_bytes'])} / {_gb(s['total_bytes'])}      "
             f"completed {sum(1 for r in s['rows'] if r[3]=='DONE')}/{len(s['rows'])} files"
             f"   inflight {s['n_inflight']} ({_gb(s['inflight_bytes'])})")
    eta_s = _hhmmss(eta) if eta != float("inf") else "n/a"
    L.append(f"   rate {rate/1e6:6.2f} MB/s      ETA {eta_s}      "
             f"engine {state.get('engine', '-')}")
    if state.get("current_file"):
        cf = state["current_file"]
        cfb = float(state.get("current_file_bytes", 0))
        cfe = float(state.get("current_file_expected", 0)) or 1.0
        L.append(f"   now: {cf}   {cfb/1e9:.2f}/{cfe/1e9:.2f} GB "
                 f"({100*cfb/cfe:.1f}%)   attempt {state.get('attempt', 1)}")
    if state.get("last_error"):
        L.append(f"   last error: {state['last_error']}")
    L.append("-" * 88)
    L.append(f" {'FILE':<56} {'ON DISK':>10} {'EXPECT':>10}  STATE")
    for path, sz, exp, st in s["rows"][:12]:
        L.append(f" {path:<56} {sz/1e9:9.2f}G {exp/1e9:9.2f}G  {st}")
    if len(s["rows"]) > 12:
        L.append(f" {'... ' + str(len(s['rows'])-12) + ' smaller config/tokenizer files':<56}")
    L.append("-" * 88)
    L.append(f" GPU      : {_gpu()}")
    L.append(f" DISK     : {_disk()}")
    if verify_note:
        L.append(" NOTE     : BF16 full diffusers pipeline. FP8 single-file repo is NOT loadable")
        L.append("            (no model_index.json / text_encoder / vae; ComfyUI-native FP8 keys).")
    if state.get("error"):
        L.append(f" ERROR    : {state['error']}")
    if state.get("verify"):
        v = state["verify"]
        L.append(f" VERIFY   : ok={v.get('ok')} missing={len(v.get('missing', []))} "
                 f"mismatch={len(v.get('size_mismatch', []))}")
    L.append("=" * 88)
    L.append(" refresh: --interval N  |  Ctrl+C closes this window only; download runs separately")
    panel = "\n".join(L)

    rep = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status, "repo": manifest.get("repo"), "dst": s["dst"],
        "elapsed_s": round(elapsed, 1),
        "pct": round(s["pct"], 2),
        "cur_bytes": s["cur_bytes"], "total_bytes": s["total_bytes"],
        "done_bytes": s["done_bytes"], "inflight_bytes": s["inflight_bytes"],
        "n_inflight": s["n_inflight"],
        "n_files": len(s["rows"]),
        "n_done": sum(1 for r in s["rows"] if r[3] == "DONE"),
        "rate_MBps": round(rate / 1e6, 2),
        "eta_s": None if eta == float("inf") else round(eta, 1),
        "gpu": _gpu(), "disk": _disk(),
        "error": state.get("error"),
    }
    return panel, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="print one panel and exit")
    ap.add_argument("--interval", type=int, default=5)
    args = ap.parse_args()

    state = _read_json(STATE, {})
    manifest = _read_json(MANIFEST, {})
    if not manifest:
        print("[watch] no download_manifest.json yet -- run download_models.py --dry-run first")
        return 1

    if args.once:
        panel, rep = render(state, manifest)
        print(panel)
        save_json(rep, PROGRESS)
        print(f"[watch] -> {PROGRESS}")
        return 0

    try:
        while True:
            state = _read_json(STATE, {})
            if os.name == "nt":
                os.system("cls")
            panel, rep = render(state, manifest)
            print(panel, flush=True)
            save_json(rep, PROGRESS)
            time.sleep(max(1, args.interval))
    except KeyboardInterrupt:
        print("\n[watch] closed (download is unaffected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
