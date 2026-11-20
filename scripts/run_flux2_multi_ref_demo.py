from __future__ import annotations

"""REAL FLUX.2 [klein] 4B Base: multi-reference-image verification.

    D:/Anaconda/envs/ChatAI/python.exe scripts/run_flux2_multi_ref_demo.py \
        --size 512 --steps 20 --seed 20261120

Question: does FLUX.2 actually READ the 2nd reference image, or silently drop it?

Controlled cases (SAME prompt, SAME seed, SAME steps where noted):
  single_0    image=[img0]          baseline
  single_1    image=[img1]          input changed -> output should change (trivial)
  double_01   image=[img0, img1]    2nd image added -> output should change (KEY)
  double_02   image=[img0, img2]    2nd image swapped -> output should change (KEY)
  fuse_01     image=[img0, img1]    semantic fusion instruction (different prompt)

Outputs:
    outputs/samples/flux2_multi_ref/<case>.png
    outputs/samples/flux2_multi_ref/grid.png
    outputs/logs/flux2_multi_ref_progress.json
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from src.utils import ROOT

WEIGHTS = ROOT / "models" / "FLUX.2-klein-4B-Base"
OUT = ROOT / "outputs/samples/flux2_multi_ref"
PROGRESS = ROOT / "outputs/logs/flux2_multi_ref_progress.json"
REFS = [
    ROOT / "data/old_photo_multitarget/old/00000.jpg",
    ROOT / "data/old_photo_multitarget/old/00001.jpg",
    ROOT / "data/old_photo_multitarget/old/00002.jpg",
]

RESTORE_PROMPT = (
    "Restore these old photographs into clean, natural-color images. "
    "Remove scratches, dust and film grain. Keep each person's identity, "
    "the background and the clothing EXACTLY as they are."
)
FUSE_PROMPT = (
    "Using the TWO reference photographs: take the FACE AND IDENTITY from "
    "the FIRST image and the CLOTHING AND BACKGROUND from the SECOND image, "
    "and create ONE clean restored portrait that combines both. "
    "Remove all scratches and grain."
)

CASES = [
    {"id": "single_0", "refs": [0], "prompt": RESTORE_PROMPT},
    {"id": "single_1", "refs": [1], "prompt": RESTORE_PROMPT},
    {"id": "double_01", "refs": [0, 1], "prompt": RESTORE_PROMPT},
    {"id": "double_02", "refs": [0, 2], "prompt": RESTORE_PROMPT},
    {"id": "fuse_01", "refs": [0, 1], "prompt": FUSE_PROMPT},
]


def log(m: str) -> None:
    print(f"[multi-ref] {m}", flush=True)


def write(rep: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                        encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--limit", type=int, default=5)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    imgs = [Image.open(p).convert("RGB") for p in REFS]
    for i, im in enumerate(imgs):
        log(f"ref[{i}] = {REFS[i].name}  {im.size}")

    rep = {
        "status": "LOADING",
        "model": "black-forest-labs/FLUX.2-klein-base-4B (local BF16)",
        "question": "is the 2nd reference image actually read by the model?",
        "size": a.size, "steps": a.steps, "guidance": a.guidance,
        "seed": a.seed, "refs": [str(p.relative_to(ROOT)) for p in REFS],
        "cases": [], "errors": [],
    }
    write(rep)

    from diffusers import Flux2KleinPipeline
    t0 = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(
        str(WEIGHTS), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.enable_model_cpu_offload(gpu_id=0)
    rep["load_seconds"] = round(time.time() - t0, 1)
    rep["status"] = "GENERATING"
    log(f"pipeline loaded in {rep['load_seconds']}s; cpu offload on")
    write(rep)

    for c in CASES[: a.limit]:
        refs = [imgs[i] for i in c["refs"]]
        log(f"[{c['id']}] {len(refs)} ref(s): {[REFS[i].name for i in c['refs']]}")
        t1 = time.time()
        try:
            gen = torch.Generator(device="cuda").manual_seed(a.seed)
            out = pipe(
                prompt=c["prompt"], image=refs,
                height=a.size, width=a.size,
                num_inference_steps=a.steps,
                guidance_scale=a.guidance, generator=gen,
            ).images[0]
        except torch.cuda.OutOfMemoryError:
            log("  OOM -> retry with guidance=1.0")
            torch.cuda.empty_cache()
            gc.collect()
            gen = torch.Generator(device="cuda").manual_seed(a.seed)
            out = pipe(prompt=c["prompt"], image=refs,
                       height=a.size, width=a.size,
                       num_inference_steps=a.steps,
                       guidance_scale=1.0, generator=gen).images[0]
            rep["errors"].append(f"{c['id']}: OOM retried with guidance=1.0")
        p = OUT / f"{c['id']}.png"
        out.save(p)
        dt = round(time.time() - t1, 1)
        log(f"  -> {p.name}  {dt}s")
        rep["cases"].append({
            "id": c["id"], "refs": c["refs"],
            "prompt": c["prompt"], "output": str(p.relative_to(ROOT)),
            "seconds": dt,
        })
        write(rep)
        torch.cuda.empty_cache()

    # ---- pairwise image difference: proves 2nd ref changed the output ---- #
    try:
        import numpy as np
        diff = {}
        arr = {c["id"]: np.asarray(Image.open(ROOT / c["output"]).convert("RGB"),
                                  dtype=np.float32) / 255.0
               for c in rep["cases"]}
        for x, y in (("single_0", "single_1"),
                     ("single_0", "double_01"),
                     ("double_01", "double_02"),
                     ("double_01", "fuse_01")):
            if x in arr and y in arr:
                diff[f"{x}__{y}"] = round(
                    float(np.sqrt(np.mean((arr[x] - arr[y]) ** 2))), 5)
        rep["pairwise_rms"] = diff
        write(rep)
        log("pairwise RMS: " + json.dumps(diff, ensure_ascii=False))
    except Exception as e:                                       # noqa: BLE001
        rep["errors"].append(f"diff: {type(e).__name__}: {e}")

    # ---- grid ------------------------------------------------------------ #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                           "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        n = len(rep["cases"])
        fig = plt.figure(figsize=(3.05 * (n + 3), 6.8))
        gs = fig.add_gridspec(2, n + 3, hspace=0.14, wspace=0.10)
        fig.suptitle("FLUX.2 [klein] 4B Base  |  原生多参考图输入验证\n"
                     "same seed / same steps — only the reference-image LIST changes",
                     fontsize=13.5, fontweight="bold", y=0.985)

        for k, im in enumerate(imgs):
            ax = fig.add_subplot(gs[0, k])
            ax.imshow(im)
            ax.set_title(f"ref[{k}]\n{REFS[k].name}", fontsize=10,
                         fontweight="bold", color="#a33")
            ax.axis("off")
        ax = fig.add_subplot(gs[1, 0])
        ax.text(0.5, 0.5,
                "SAME PROMPT (except fuse_01)\n\n"
                "Restore these old photographs\n"
                "into clean, natural-color images.\n"
                "Keep identity / background\n"
                "EXACTLY as they are.",
                ha="center", va="center", fontsize=7.8, family="monospace",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.35", fc="#f2f5fa", ec="#9fb3c8"))
        ax.set_title("shared instruction", fontsize=9.5, color="#555")
        ax.axis("off")
        ax2 = fig.add_subplot(gs[1, 1:3])
        ax2.text(0.5, 0.5,
                 "KEY CONTROL\n\n"
                 "single_0 -> double_01 : add ref[1]\n"
                 "double_01 -> double_02 : swap ref[1]\n\n"
                 "If RMS ~ 0, the 2nd image was ignored.",
                 ha="center", va="center", fontsize=8.0, family="monospace",
                 transform=ax2.transAxes,
                 bbox=dict(boxstyle="round,pad=0.35", fc="#fff6e8", ec="#e0b878"))
        ax2.axis("off")

        for j, row in enumerate(rep["cases"]):
            ax = fig.add_subplot(gs[:, j + 3])
            ax.imshow(Image.open(ROOT / row["output"]).convert("RGB"))
            tag = "+".join(f"r{i}" for i in row["refs"])
            extra = " [FUSE]" if row["id"].startswith("fuse") else ""
            ax.set_title(f"{row['id']}\nimage=[{tag}]{extra}", fontsize=10,
                         fontweight="bold")
            ax.axis("off")

        fig.text(0.5, 0.012,
                 f"size={a.size}  steps={a.steps}  guidance={a.guidance}  "
                 f"seed={a.seed}  |  pairwise RMS = "
                 + json.dumps(rep.get("pairwise_rms", {}), ensure_ascii=False),
                 ha="center", fontsize=8.0)
        grid = OUT / "grid.png"
        fig.savefig(grid, dpi=125, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        rep["grid"] = str(grid.relative_to(ROOT))
        write(rep)
        log(f"grid -> {grid}")
    except Exception as e:                                       # noqa: BLE001
        rep["errors"].append(f"grid: {type(e).__name__}: {e}")

    rep["status"] = "DONE"
    rep["total_seconds"] = round(time.time() - t0, 1)
    write(rep)
    log(f"DONE in {rep['total_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
