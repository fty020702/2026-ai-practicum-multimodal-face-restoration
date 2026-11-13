from __future__ import annotations

"""Download FLUX.2 [klein] 4B Base weights into models/ (project-local).

    D:/Anaconda/envs/ChatAI/python.exe scripts/download_models.py --dry-run
    D:/Anaconda/envs/ChatAI/python.exe scripts/download_models.py
    D:/Anaconda/envs/ChatAI/python.exe scripts/download_models.py --fp8-singlefile

=============================================================================
[CORRECTION 2026-11-07] Verified against HF API files_metadata, both model
cards, the FP8 safetensors header (HTTP Range) and installed diffusers 0.37.1.
The earlier "FP8 repo == loadable pipeline, ~11 GB" assumption was WRONG.
=============================================================================

[FACT] FP8 repo black-forest-labs/FLUX.2-klein-base-4b-fp8 has EXACTLY 7 files:
    flux-2-klein-base-4b-fp8.safetensors   4.09 GB
    editing.jpg / others.jpg / realism.jpg
    README.md / LICENSE.md / .gitattributes
  NO model_index.json, NO scheduler/, NO transformer/config.json,
  NO text_encoder/, NO vae/, NO tokenizer/.
  HF tag: "Diffusion Single File". Model card: "No code snippets available yet".

[FACT] FP8 checkpoint keys are BFL/ComfyUI NATIVE, not diffusers:
    double_blocks.0.img_attn.qkv.weight_scale / input_scale
    single_blocks.0.linear1
    __metadata__._quantization_metadata -> "float8_e4m3fn" + per-layer scales
  305 tensors. Flux2KleinPipeline inherits DiffusionPipeline and has NEITHER
  from_single_file NOR FP8 quantisation support.
  => Flux2KleinPipeline.from_pretrained(<fp8 repo>) CANNOT work.

[FACT] The ONLY loadable pipeline is the BF16 Base repo (23.74 GB):
    black-forest-labs/FLUX.2-klein-base-4B
  model_index.json, scheduler/, transformer/ (7.75 GB),
  text_encoder/ (Qwen3-4B, 8.05 GB), vae/ (0.17 GB), tokenizer/,
  AND a DUPLICATE ComfyUI single file flux-2-klein-base-4b.safetensors
  (7.75 GB). We skip the duplicate + jpgs with allow_patterns:
  -> ~15.98 GB to download, ~23.7 GB if you take everything.

[FACT] Official usage (BF16 model card, quoted):
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
  Undistilled, num_inference_steps=50. Hardware: ~13 GB VRAM.

[FACT] Measured on this machine 2026-11-07:
    G: 56 GB free. HF official CDN via local proxy 127.0.0.1:55599:
    HTTP 200, 20 MB Range probe -> 4.32 MB/s. hf-mirror: 3.13 MB/s
    (via proxy 3.19 MB/s), so official HF is FASTER here. HF_ENDPOINT is
    not set; HTTPS_PROXY/HTTP_PROXY are inherited by huggingface_hub.
    Estimated wall time for ~15.98 GB: roughly 60 min single-connection.

--fp8-singlefile downloads the 4.09 GB FP8 checkpoint into a SEPARATE folder
purely as research material. It is [UNVERIFIED] and is NOT consumed by ANY
loader in this project: using it would require writing a reverse-quantisation,
BFL->diffusers key remap and an FP8 linear layer. Do NOT mistake it for the
main path.

Everything lands under the project root, per the user's instruction that all
data / models / code live here. Progress is written to
outputs/logs/download_state.json and read by scripts/watch_download.py.

[ENGINE] Default engine is "direct": a sequential, Range-resumable downloader
that writes straight to <target>.incomplete and finishes with os.replace()
(a RENAME, not a delete). It deliberately avoids snapshot_download, which
creates AND DELETES one .lock + one tmp_* per file: the workspace sandbox
intercepts bulk deletions (SAFE_DELETE_BULK_CONFIRM_REQUIRED, 50/turn) and
stalled that path at 1/21 files after 5 minutes. --engine hub keeps
huggingface_hub.snapshot_download as an opt-in fallback.
"""

import argparse
import json
import os
import sys
import time
from fnmatch import fnmatch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT, save_json

BF16_REPO = "black-forest-labs/FLUX.2-klein-base-4B"
FP8_REPO = "black-forest-labs/FLUX.2-klein-base-4b-fp8"
DST = "FLUX.2-klein-4B-Base"
FP8_DST = "FLUX.2-klein-4B-Base-FP8-singlefile"

# Only the files Flux2KleinPipeline.from_pretrained() needs from a LOCAL folder.
# Deliberately excludes the duplicate ComfyUI single file and the demo jpgs.
ALLOW_FULL = [
    "model_index.json",
    "scheduler/*",
    "transformer/*",
    "text_encoder/*",
    "tokenizer/*",
    "vae/*",
    "README.md",
    "LICENSE.md",
    ".gitattributes",
]
ALLOW_FP8 = ["*.safetensors", "README.md", "LICENSE.md", ".gitattributes"]

STATE = ROOT / "outputs" / "logs" / "download_state.json"
MANIFEST = ROOT / "outputs" / "logs" / "download_manifest.json"


def build_manifest(repo: str, allow: list[str]) -> dict:
    """Ask the HF API for exact byte sizes; used by the watch window for %.

    The local proxy (127.0.0.1:55599) intermittently answers 502 Bad Gateway
    on huggingface.co API calls, so retry before giving up.
    """
    from huggingface_hub import HfApi

    last = None
    info = None
    for attempt in range(1, 4):
        try:
            info = HfApi().model_info(repo, files_metadata=True)
            break
        except Exception as e:  # ProxyError / ConnectionError are transient here
            last = e
            print(f"[plan] HF API attempt {attempt}/3 failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            if attempt < 3:
                time.sleep(3 * attempt)
    if info is None:
        raise RuntimeError(f"HF API unavailable after 3 attempts: {last}")

    files = [
        {"path": s.rfilename, "size": int(s.size or 0)}
        for s in info.siblings
        if any(fnmatch(s.rfilename, p) for p in allow)
    ]
    files.sort(key=lambda f: -f["size"])
    return {
        "repo": repo,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_files": len(files),
        "total_bytes": sum(f["size"] for f in files),
        "total_GB": round(sum(f["size"] for f in files) / 1e9, 2),
        "files": files,
    }


def load_manifest(repo: str, allow: list[str], refresh: bool = False) -> dict:
    """Reuse the manifest written by --dry-run; the API is needed only ONCE.

    A fresh API request at download start is a liability on a flaky proxy,
    and the manifest is immutable for a given repo revision.
    """
    if not refresh and MANIFEST.exists():
        try:
            m = json.loads(MANIFEST.read_text(encoding="utf-8"))
            if m.get("repo") == repo and m.get("files"):
                print(f"[plan] reusing manifest ({m['n_files']} files, "
                      f"{m['total_GB']} GB) from {MANIFEST.name}")
                return m
        except Exception:
            pass
    m = build_manifest(repo, allow)
    save_json(m, MANIFEST)
    return m


def write_state(**kw):
    try:
        cur = {}
        if STATE.exists():
            cur = json.loads(STATE.read_text(encoding="utf-8"))
        cur.update(kw)
        save_json(cur, STATE)
    except Exception as e:
        print(f"[dl] WARN could not write state: {e}")


def verify(dst: Path, manifest: dict) -> dict:
    missing, wrong = [], []
    for f in manifest["files"]:
        p = dst / f["path"]
        if not p.exists():
            missing.append(f["path"])
        elif f["size"] > 0 and abs(p.stat().st_size - f["size"]) > max(1024, f["size"] * 1e-6):
            wrong.append({"path": f["path"], "on_disk": p.stat().st_size, "expected": f["size"]})
    return {"n_files": len(manifest["files"]), "missing": missing, "size_mismatch": wrong,
            "ok": not missing and not wrong}


def download_all(dst: Path, rid: str, manifest: dict, retries: int = 3) -> None:
    """Sequential, Range-resumable download; ZERO temp-file deletions.

    See module docstring [ENGINE]. `requests` follows the 302 to the LFS CDN
    and inherits HTTPS_PROXY from the environment -- the same path that
    measured 4.32 MB/s on this machine with a curl Range probe.
    """
    import requests

    files = manifest["files"]
    total = manifest["total_bytes"] or 1
    t_start = time.time()
    hist: list[tuple[float, int]] = []

    def completed_bytes() -> int:
        n = 0
        for f in files:
            p = dst / f["path"]
            if p.is_file():
                n += min(p.stat().st_size, f["size"])
        return n

    def n_done() -> int:
        return sum(
            1 for f in files
            if (dst / f["path"]).is_file()
            and (f["size"] == 0 or
                 (dst / f["path"]).stat().st_size >= f["size"] * 0.999)
        )

    def report(cf=None, cfb=0, cfe=0, attempt=1, last_error=None) -> dict:
        done = completed_bytes()
        cur = min(total, done + cfb)
        now = time.time()
        hist.append((now, cur))
        while len(hist) > 2 and now - hist[0][0] > 15:
            hist.pop(0)
        rate = 0.0
        if len(hist) >= 2 and hist[-1][0] > hist[0][0]:
            rate = (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0])
        eta = (total - cur) / rate if rate > 0 else None
        st = {
            "status": "running", "repo": rid, "dst": str(dst),
            "variant": dst.name, "engine": "direct",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
            "started_ts": t_start, "elapsed_s": round(now - t_start, 1),
            "total_bytes": total, "done_bytes": done, "cur_bytes": cur,
            "pct": round(100.0 * cur / total, 2),
            "n_files": len(files), "n_done": n_done(),
            "current_file": cf, "current_file_bytes": cfb,
            "current_file_expected": cfe,
            "rate_MBps": round(rate / 1e6, 2),
            "eta_s": None if eta is None else round(eta, 1),
            "attempt": attempt, "last_error": last_error,
        }
        write_state(**st)
        return st

    for idx, f in enumerate(files, 1):
        rel, expected = f["path"], int(f["size"])
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and (expected == 0 or target.stat().st_size == expected):
            print(f"[dl] ({idx:2d}/{len(files)}) SKIP  {rel}", flush=True)
            continue

        part = dst / (rel + ".incomplete")
        url = f"https://huggingface.co/{rid}/resolve/main/{rel}"
        ok, last = False, None
        for attempt in range(1, retries + 1):
            try:
                existing = part.stat().st_size if part.exists() else 0
                headers = {"Range": f"bytes={existing}-"} if existing else {}
                r = requests.get(url, headers=headers, stream=True, timeout=(20, 180))
                if r.status_code not in (200, 206):
                    r.close()
                    raise RuntimeError(f"HTTP {r.status_code} for {rel}")
                mode = "ab" if (r.status_code == 206 and existing) else "wb"
                if mode == "wb":
                    existing = 0
                written = existing
                last_report = 0.0
                with open(part, mode) as fh:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        written += len(chunk)
                        now = time.time()
                        if now - last_report >= 2.0:
                            report(rel, written, expected, attempt=attempt)
                            last_report = now
                r.close()
                size = part.stat().st_size
                if expected > 0 and size != expected:
                    raise RuntimeError(f"size mismatch {rel}: {size} != {expected}")
                os.replace(part, target)     # RENAME, never a delete
                ok = True
                break
            except Exception as e:
                last = e
                print(f"[dl] ({idx:2d}/{len(files)}) attempt {attempt}/{retries} "
                      f"failed {rel}: {type(e).__name__}: {e}", flush=True)
                report(rel, part.stat().st_size if part.exists() else 0, expected,
                       attempt=attempt, last_error=f"{type(e).__name__}: {e}")
                if attempt < retries:
                    time.sleep(4 * attempt)
        if not ok:
            raise RuntimeError(f"failed to download {rel}: {last}")
        print(f"[dl] ({idx:2d}/{len(files)}) DONE  {rel}  {expected/1e9:.3f} GB",
              flush=True)
    report()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fp8-singlefile", action="store_true",
                    help="download ONLY the experimental FP8 single file (not loadable)")
    ap.add_argument("--repo", default=None, help="override repo id")
    ap.add_argument("--token", default=None)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--engine", choices=("direct", "hub"), default="direct",
                    help="direct = sandbox-safe sequential downloader (default); "
                         "hub = huggingface_hub.snapshot_download (may stall on bulk deletes)")
    ap.add_argument("--refresh-manifest", action="store_true",
                    help="force a fresh HF API call instead of reusing the cached manifest")
    args = ap.parse_args()

    if args.fp8_singlefile:
        repo, sub, allow = FP8_REPO, FP8_DST, ALLOW_FP8
        note = ("[UNVERIFIED] BFL/ComfyUI-native scaled-FP8 single file; "
                "NOT loadable by Flux2KleinPipeline. Research material only.")
    else:
        repo, sub, allow = BF16_REPO, DST, ALLOW_FULL
        note = "Full diffusers pipeline (duplicate ComfyUI single file + jpgs excluded)."
    rid = args.repo or repo
    dst = ROOT / "models" / sub
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[plan] repo      : {rid}")
    print(f"[plan] target    : {dst}")
    print(f"[plan] note      : {note}")
    print(f"[plan] HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '(unset -> huggingface.co)')}")
    print(f"[plan] proxy      : {os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY') or '(none)'}")

    manifest = load_manifest(rid, allow, refresh=args.refresh_manifest)
    print(f"[plan] files     : {manifest['n_files']}  total {manifest['total_GB']} GB")
    for f in manifest["files"][:8]:
        print(f"        {f['size']/1e9:8.3f} GB  {f['path']}")
    if len(manifest["files"]) > 8:
        print(f"        ... {len(manifest['files'])-8} smaller config/tokenizer files")
    if args.dry_run:
        print("[plan] dry-run: nothing downloaded")
        return 0

    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
    t0 = time.time()
    write_state(status="running", repo=rid, dst=str(dst), variant=sub,
                started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                started_ts=t0, pid=os.getpid(),
                total_bytes=manifest["total_bytes"], n_files=manifest["n_files"])
    print(f"[dl] START {time.strftime('%H:%M:%S')}  {rid} -> {dst}", flush=True)

    print(f"[dl] engine={args.engine}", flush=True)
    try:
        if args.engine == "direct":
            download_all(dst, rid, manifest)
        else:
            from huggingface_hub import snapshot_download
            # Opt-in fallback. Resumes from *.incomplete, but creates + deletes
            # many .lock/tmp_* files, which the sandbox may throttle.
            snapshot_download(
                repo_id=rid,
                local_dir=str(dst),
                allow_patterns=allow,
                token=args.token,
                max_workers=args.max_workers,
            )
    except Exception as e:
        write_state(status="failed", error=f"{type(e).__name__}: {e}",
                    finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    finished_ts=time.time())
        print(f"[dl] FAILED: {type(e).__name__}: {e}", flush=True)
        return 1

    v = verify(dst, manifest)
    write_state(status="done", finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                finished_ts=time.time(), elapsed_s=round(time.time() - t0, 1),
                verify=v)
    print(f"[dl] DONE {time.strftime('%H:%M:%S')}  verify_ok={v['ok']} "
          f"missing={len(v['missing'])} mismatch={len(v['size_mismatch'])}", flush=True)
    if not v["ok"]:
        print(f"[dl] missing: {v['missing'][:5]}", flush=True)
        return 1
    print("[dl] Next: scripts/inspect_backbone.py -> scripts/precompute_prompts.py "
          "-> scripts/verify_real_backend.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
