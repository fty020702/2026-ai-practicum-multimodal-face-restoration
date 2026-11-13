from __future__ import annotations

"""Build a REAL old-photo-restoration dataset inside the project directory.

Sources (all verified reachable on 2026-11-07)
---------------------------------------------
GT / clean  : mAiello00/DIV2K  -> DIV2K_train_HR/NNNN.png
              (HuggingFace, 1200 real 2K photographs; DIV2K is exactly the
               benchmark Wan et al. CVPR 2020 used for synthetic test pairs)
Captions    : ScooterTaylor/DIV2K_captioned_subset -> imgNNNN.txt
              (REAL human captions -> content-level prompts, not invented)
Real old    : microsoft/Bringing-Old-Photos-Back-to-Life/test_images/*
              (the paper's own real old-photo test images, NO ground truth)
              + froilo/old_family_photo_dataset (real aged family photos)

Why this shape
--------------
Real historical photos have NO pixel-aligned clean GT. Following the papers,
we build PAIRED supervision by applying a physical old-photo degradation
model to real modern photographs, and keep real old photos as a SEPARATE
unpaired evaluation set for no-reference IQA / user study.

    D:/Anaconda/envs/ChatAI/python.exe scripts/build_old_photo_data.py --gt 24 --size 256
    D:/Anaconda/envs/ChatAI/python.exe scripts/build_old_photo_data.py --offline  # use data/raw/

Outputs
-------
    data/old_photo/hr/            cropped real GT (256x256)
    data/old_photo/degraded/      paper-style degraded pairs
    data/old_photo/hr_raw/        original downloaded DIV2K images
    data/old_photo/real_old/      real old photos (NO GT, evaluation only)
    data/old_photo/train.jsonl    paired training manifest
    data/old_photo/eval_real.jsonl
    data/old_photo/captions.json  real caption pool
    data/old_photo/dataset_info.json
    outputs/logs/old_photo_build_progress.json   (pollable progress)
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

import numpy as np
from PIL import Image

from src.data.old_photo import ALL_DEFECTS, degrade_old_photo
from src.utils import ROOT, load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Verified public data sources
# ---------------------------------------------------------------------------
DIV2K_HR = (
    "https://huggingface.co/datasets/mAiello00/DIV2K/resolve/main/"
    "DIV2K_train_HR/{:04d}.png"
)
CAPTION = (
    "https://huggingface.co/datasets/ScooterTaylor/DIV2K_captioned_subset/"
    "resolve/main/img{:04d}.txt"
)
CAPTION_FIRST_ID = 193          # img0193.txt ... img0384.txt (192 files)
CAPTION_COUNT = 192

MS_OLD = (
    "https://raw.githubusercontent.com/microsoft/"
    "Bringing-Old-Photos-Back-to-Life/master/test_images/{sub}/{name}"
)
MS_SCRATCHED = ["a.png", "b.png", "c.png", "d.png"]   # verified via GitHub API
HF_OLD = (
    "https://huggingface.co/datasets/froilo/old_family_photo_dataset/"
    "resolve/main/datasets/dataset_0/images/{name}"
)
HF_OLD_FILES = [
    "17966798_web1_EA_UoZ0W4AA7NmM.jpeg",
    "26dd3d80a239422a344e4fe5fa92347d.jpg",
    "4454ff377d18fb4a59b84f649af5c9e5.jpg",
    "566.jpg",
    "607.jpg",
    "7d33f96a60802249623674d5f339e4d5.jpg",
    "Sept09.jpg",
    "one_year_old_family_session15-scaled.jpg",
    "pngtree-an-old-black-family-with-black-children-standing-image_2961397.png",
]

UA = {"User-Agent": "Mozilla/5.0 (WorkBuddy dataset builder)"}


# ---------------------------------------------------------------------------
# Range-resumable download. Writes .incomplete then os.replace (a RENAME,
# never a delete) so it stays friendly to sandboxed filesystems.
# ---------------------------------------------------------------------------
def download(url: str, dst: Path, retries: int = 4, timeout: int = 90) -> str:
    if dst.is_file() and dst.stat().st_size > 0:
        return "cached"
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".incomplete")
    existing = part.stat().st_size if part.exists() else 0

    for attempt in range(1, retries + 1):
        try:
            headers = dict(UA)
            if existing:
                headers["Range"] = f"bytes={existing}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                partial = (r.status == 206)
                mode = "ab" if (existing and partial) else "wb"
                with open(part, mode) as f:
                    if mode == "wb":
                        existing = 0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        existing += len(chunk)
            if part.stat().st_size == 0:
                raise RuntimeError("empty response")
            os.replace(part, dst)
            return "ok"
        except Exception as e:                       # noqa: BLE001
            existing = part.stat().st_size if part.exists() else 0
            if attempt == retries:
                return f"FAILED: {type(e).__name__}: {str(e)[:110]}"
            time.sleep(1.5 * attempt)
    return "FAILED"


def gh_list_old_dir() -> list[str]:
    """Best-effort listing of the official `test_images/old` folder."""
    u = ("https://api.github.com/repos/microsoft/"
         "Bringing-Old-Photos-Back-to-Life/contents/test_images/old")
    for _ in range(3):
        try:
            req = urllib.request.Request(u, headers=UA)
            d = json.load(urllib.request.urlopen(req, timeout=25))
            return [x["name"] for x in d if x.get("name", "").lower().endswith(
                (".png", ".jpg", ".jpeg"))]
        except Exception:                            # noqa: BLE001
            time.sleep(2.0)
    return []                                        # rate-limited -> skip gracefully


def progress(stage: str, done: int, total: int, msg: str = "") -> None:
    line = f"[{stage} {done}/{total}] {msg}"
    print(line, flush=True)
    p = ROOT / "outputs" / "logs" / "old_photo_build_progress.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "stage": stage, "done": done, "total": total, "msg": msg,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
def fetch_sources(n_gt: int, n_cap: int, n_real: int, dirs: dict) -> dict:
    rep = {"gt": [], "captions": [], "real_old": []}

    # --- 1. real GT photographs (DIV2K HR) ---
    for i in range(1, n_gt + 1):
        name = f"{i:04d}.png"
        dst = dirs["hr_raw"] / name
        st = download(DIV2K_HR.format(i), dst)
        rep["gt"].append({"file": name, "status": st,
                          "bytes": dst.stat().st_size if dst.is_file() else 0})
        progress("GT", i, n_gt, f"{name} {st}")

    # --- 2. real human captions (content-level prompts) ---
    cap_ids = [CAPTION_FIRST_ID + j for j in range(min(n_cap, CAPTION_COUNT))]
    for j, cid in enumerate(cap_ids, 1):
        name = f"img{cid:04d}.txt"
        dst = dirs["captions_raw"] / name
        st = download(CAPTION.format(cid), dst)
        rep["captions"].append({"file": name, "status": st})
        progress("CAP", j, len(cap_ids), f"{name} {st}")

    # --- 3. real old photographs, NO ground truth (evaluation only) ---
    targets: list[tuple[str, str, str]] = []
    for f in MS_SCRATCHED:
        targets.append(("ms_scratch_" + f, "ms", MS_OLD.format(sub="old_w_scratch", name=f)))
    for f in gh_list_old_dir()[: max(0, n_real - len(MS_SCRATCHED) - len(HF_OLD_FILES))]:
        targets.append(("ms_" + f, "ms", MS_OLD.format(sub="old", name=f)))
    for f in HF_OLD_FILES:
        targets.append(("hf_" + f, "hf", HF_OLD.format(name=f)))
    targets = targets[:n_real] if n_real > 0 else targets

    for j, (local, tag, url) in enumerate(targets, 1):
        dst = dirs["real_old"] / local
        st = download(url, dst)
        rep["real_old"].append({"file": local, "source": tag, "status": st,
                                "bytes": dst.stat().st_size if dst.is_file() else 0})
        progress("REAL_OLD", j, len(targets), f"{local} {st}")
    return rep


def random_crop(im: Image.Image, size: int, rng) -> Image.Image:
    w, h = im.size
    if w < size or h < size:
        s = max(size, w, h)
        im = im.resize((s, s), Image.BILINEAR)
        w = h = s
    x = int(rng.integers(0, w - size + 1))
    y = int(rng.integers(0, h - size + 1))
    return im.crop((x, y, x + size, y + size))


def build(cfg, args) -> int:
    d = cfg["data"]
    size = int(args.size or d.get("old_photo_size", 256))
    seed = int(cfg["project"]["seed"])
    base = ROOT / "data" / "old_photo"
    dirs = {
        "hr_raw": base / "hr_raw",
        "captions_raw": base / "captions_raw",
        "hr": base / "hr",
        "degraded": base / "degraded",
        "real_old": base / "real_old",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    fetch = {"gt": [], "captions": [], "real_old": []}
    if not args.offline:
        print("[build] downloading REAL source images (poll progress below) ...", flush=True)
        fetch = fetch_sources(args.gt, args.captions, args.real_old, dirs)
    else:
        print("[build] --offline: using images already present in data/raw/", flush=True)

    # ---- caption pool (real human captions) ----
    def clip_words(text: str, limit: int = 180) -> str:
        """Word-wise truncation: never cut a caption in the middle of a word."""
        out, n = [], 0
        for w in text.split():
            if n + len(w) + 1 > limit:
                break
            out.append(w)
            n += len(w) + 1
        return " ".join(out)

    captions: list[str] = []
    for p in sorted(dirs["captions_raw"].glob("*.txt")):
        txt = p.read_text(encoding="utf-8", errors="replace").strip()
        if txt:
            captions.append(clip_words(" ".join(txt.split())))
    if not captions:
        captions = ["a historical photograph with natural scene detail"]
    (base / "captions.json").write_text(
        json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- GT images: downloaded DIV2K, or user-supplied data/raw/ ----
    if args.offline:
        gt_files = [p for p in sorted((ROOT / "data" / "raw").glob("*"))
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")]
    else:
        gt_files = [p for p in sorted(dirs["hr_raw"].glob("*.png"))]
    if not gt_files:
        raise RuntimeError(
            "[build] no real GT images. Re-run without --offline, or put real "
            "photographs in data/raw/. Synthetic toy data will NOT be generated."
        )
    if args.pairs:
        gt_files = gt_files[: args.pairs]

    # ---- build paired (clean, degraded) samples ----
    lines = []
    t0 = time.time()
    for i, gp in enumerate(gt_files):
        rng = np.random.default_rng(seed + i * 104729)
        try:
            im = Image.open(gp).convert("RGB")
            clean = np.asarray(random_crop(im, size, rng), dtype=np.uint8)
        except Exception as e:                        # noqa: BLE001
            print(f"[build] skip {gp.name}: {e}", flush=True)
            continue
        degraded, defects = degrade_old_photo(clean, rng)
        stem = f"{i:05d}"
        Image.fromarray(clean).save(dirs["hr"] / f"{stem}.png")
        Image.fromarray(degraded).save(dirs["degraded"] / f"{stem}.png")
        # caption is deliberately drawn from the pool, NOT tied to this image
        cap = captions[(i * 7 + 3) % len(captions)]
        lines.append({
            "clean": f"data/old_photo/hr/{stem}.png",
            "degraded": f"data/old_photo/degraded/{stem}.png",
            "defects": defects,
            "caption": cap,
            "gt_source": gp.name,
        })
        progress("PAIRS", i + 1, len(gt_files),
                 f"{stem}.png defects={','.join(defects)} "
                 f"({time.time() - t0:.1f}s)")

    (base / "train.jsonl").write_text(
        "".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines),
        encoding="utf-8")

    # ---- real old-photo evaluation manifest (NO clean GT) ----
    eval_lines = []
    for p in sorted(dirs["real_old"].glob("*")):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        eval_lines.append({
            "image": f"data/old_photo/real_old/{p.name}",
            "has_gt": False,
            "prompt": "restore this old photo: remove scratches and dust, "
                      "restore faded contrast; preserve facial identity and "
                      "keep the original sepia tone",
        })
    (base / "eval_real.jsonl").write_text(
        "".join(json.dumps(l, ensure_ascii=False) + "\n" for l in eval_lines),
        encoding="utf-8")

    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "task": "text-guided old photo restoration",
        "design": "paired supervision from REAL photos + paper-style physical "
                  "degradation; real old photos are UNPAIRED eval only",
        "benchmarked_against": [
            "Wan et al., Bringing Old Photos Back to Life, CVPR 2020",
            "InstructIR, ECCV 2024 (text instructions, random prompt sampling)",
            "Time-Travel Rephotography, ACM TOG 2020 / LL-Bench (no-reference IQA)",
        ],
        "gt_source": {
            "repo": "mAiello00/DIV2K (DIV2K_train_HR)",
            "url": "https://huggingface.co/datasets/mAiello00/DIV2K",
            "n_downloaded": len(fetch["gt"]),
            "n_used": len(lines),
        },
        "caption_source": {
            "repo": "ScooterTaylor/DIV2K_captioned_subset",
            "n": len(captions),
        },
        "real_old_source": {
            "microsoft_official": "Bringing-Old-Photos-Back-to-Life/test_images",
            "hf": "froilo/old_family_photo_dataset",
            "n": len(eval_lines),
            "note": "NO pixel-aligned ground truth; for NIQE/BRISQUE/CLIP-IQA "
                    "and user study, never used as (clean, degraded) training pairs",
        },
        "image_size": size,
        "n_pairs": len(lines),
        "prompts_per_image": int(d.get("prompts_per_image", 3)),
        "defect_taxonomy": list(ALL_DEFECTS),
        "anti_confound": "random defect combinations; task prompt is a noisy "
                         "subset with distractors; each image has multiple "
                         "different texts; captions decoupled from degradation",
        "license_note": "Research / coursework use only. Respect each source "
                        "dataset's own license (DIV2K, Microsoft, HF dataset).",
        "fetch_report": fetch,
    }
    (base / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    ok_gt = sum(1 for x in fetch["gt"] if x["status"] in ("ok", "cached"))
    ok_old = sum(1 for x in fetch["real_old"] if x["status"] in ("ok", "cached"))
    print(f"\n[build] DONE  pairs={len(lines)}  real_old_eval={len(eval_lines)} "
          f" captions={len(captions)}  size={size}", flush=True)
    print(f"[build] downloaded GT ok={ok_gt}/{len(fetch['gt'])}  "
          f"real-old ok={ok_old}/{len(fetch['real_old'])}", flush=True)
    print(f"[build] -> {base}", flush=True)
    progress("DONE", len(lines), len(lines), "dataset built")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=int, default=24, help="number of DIV2K HR photos")
    ap.add_argument("--captions", type=int, default=48, help="number of real captions")
    ap.add_argument("--real-old", type=int, default=14, help="number of real old photos (eval)")
    ap.add_argument("--pairs", type=int, default=0, help="cap number of pairs (0=all)")
    ap.add_argument("--size", type=int, default=256, help="crop size (paper uses 256)")
    ap.add_argument("--offline", action="store_true",
                    help="no network; use images already in data/raw/")
    ap.add_argument("--override", default=None)
    args = ap.parse_args()
    cfg = load_config(override=args.override)
    return build(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
