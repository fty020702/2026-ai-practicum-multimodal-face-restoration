from __future__ import annotations

"""REAL FLUX.2 [klein] 4B Base inference: aged old photo + text -> restoration.

This is a genuine IMAGE-RESTORATION demo on the local 15.98 GB BF16 pipeline,
NOT the toy trunk. The native Qwen3 text encoder and the DiT are used AS-IS
(frozen, zero training). The reference image is passed through the official
`image=` input, which VAE-encodes it and concatenates the packed latent into
the IMAGE stream -- the text stream stays untouched.

    D:/Anaconda/envs/ChatAI/python.exe scripts/run_flux2_old_photo_demo.py \
        --steps 20 --size 512 --limit 4

Outputs (written incrementally, so you can look at them while it runs)
    outputs/samples/flux2_old_photo/<target_id>.png
    outputs/samples/flux2_old_photo/grid.png
    outputs/logs/flux2_old_photo_progress.json
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
OUT = ROOT / "outputs" / "samples" / "flux2_old_photo"
PROGRESS = ROOT / "outputs" / "logs" / "flux2_old_photo_progress.json"
MANIFEST = ROOT / "data" / "old_photo_multitarget" / "train.jsonl"

# 2026-11-13 user request: "prompt 也要有啊，比如背景改为蓝天、人脸长胡子".
# These are CONTENT-EDIT instructions (not restoration styles), so the only
# way the model can know what to change is by reading the text. Same input
# image, same seed, same steps -- the text is the single variable.
EDIT_CASES = [
    {
        "id": "baseline",
        "prompt": (
            "Restore this old photograph: remove all scratches, dust spots and "
            "film grain, repair the faded tones. Keep the original background, "
            "the person and the face EXACTLY as they are."
        ),
        "description": "repair only, change nothing", "gt": None,
    },
    {
        "id": "blue_sky",
        "prompt": (
            "Restore this old photograph: remove all scratches, dust spots and "
            "film grain, and CHANGE THE BACKGROUND to a clear bright blue sky "
            "with a few white clouds. Keep the person, the face and the clothing "
            "unchanged."
        ),
        "description": "background -> blue sky", "gt": None,
    },
    {
        "id": "beard",
        "prompt": (
            "Restore this old photograph: remove all scratches, dust spots and "
            "film grain, and ADD A THICK DARK BEARD AND MOUSTACHE to the person's "
            "face. Keep the original background and clothing unchanged."
        ),
        "description": "add beard + moustache", "gt": None,
    },
    {
        "id": "sky_and_beard",
        "prompt": (
            "Restore this old photograph: remove all scratches, dust spots and "
            "film grain, CHANGE THE BACKGROUND to a clear bright blue sky with "
            "white clouds, and ADD A THICK DARK BEARD AND MOUSTACHE to the "
            "person's face."
        ),
        "description": "blue sky + beard at once", "gt": None,
    },
]


def log(m: str) -> None:
    print(f"[flux2-demo] {m}", flush=True)


def write_progress(rep: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                        encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="data/old_photo_multitarget/old/00000.jpg")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--offload", choices=["model", "sequential", "none"],
                    default="model")
    ap.add_argument("--tag", default="old_photo",
                    help="output subdir / progress file suffix")
    ap.add_argument("--prompt-set", choices=["restore", "edit"],
                    default="restore",
                    help="restore = 4 restoration styles from our dataset; "
                         "edit = background/face content-edit instructions")
    a = ap.parse_args()

    global OUT, PROGRESS
    OUT = ROOT / "outputs" / "samples" / f"flux2_{a.tag}"
    PROGRESS = ROOT / "outputs" / "logs" / f"flux2_{a.tag}_progress.json"
    OUT.mkdir(parents=True, exist_ok=True)
    img_path = ROOT / a.image
    if not img_path.is_file():
        raise FileNotFoundError(img_path)
    if not WEIGHTS.is_dir():
        raise FileNotFoundError(f"BF16 pipeline missing: {WEIGHTS}")

    if a.prompt_set == "edit":
        cases = [dict(c) for c in EDIT_CASES][: a.limit]
    else:
        recs = [json.loads(l) for l in MANIFEST.read_text(
            encoding="utf-8").splitlines() if l.strip()]
        rec = recs[a.group]
        cases = [
            {"id": t["id"], "prompt": t["prompts"][0], "gt": t["clean"],
             "description": t["description"]}
            for t in rec["targets"]
        ][: a.limit]

    rep = {
        "status": "LOADING",
        "backend": "black-forest-labs/FLUX.2-klein-base-4B (local BF16)",
        "mode": "NATIVE reference image + NATIVE frozen Qwen3 text; NO training",
        "input_image": a.image,
        "group": a.group, "seed": a.seed,
        "tag": a.tag, "prompt_set": a.prompt_set,
        "num_inference_steps": a.steps, "size": a.size,
        "guidance_scale": a.guidance, "offload": a.offload,
        "cases": [], "errors": [],
    }
    write_progress(rep)

    img = Image.open(img_path).convert("RGB")
    log(f"input = {img_path.relative_to(ROOT)}  {img.size}")
    for c in cases:
        log(f"  case {c['id']:<16} {c['prompt'][:78]}")

    from diffusers import Flux2KleinPipeline

    t0 = time.time()
    log("loading BF16 pipeline (17 GB from disk, ~30-60 s) ...")
    pipe = Flux2KleinPipeline.from_pretrained(
        str(WEIGHTS), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    log(f"pipeline loaded in {time.time() - t0:.1f}s")

    if a.offload == "model":
        # accelerate keeps the DiT on GPU and swaps text_encoder / VAE on demand.
        # On a 12 GB 4070 Ti the DiT (7.75 GB) + Qwen3 (8.05 GB) cannot coexist.
        pipe.enable_model_cpu_offload(gpu_id=0)
        log("enable_model_cpu_offload() applied")
    elif a.offload == "sequential":
        pipe.enable_sequential_cpu_offload(gpu_id=0)
        log("enable_sequential_cpu_offload() applied")
    else:
        pipe.to("cuda")
        log("pipeline moved to cuda (no offload)")

    rep["status"] = "GENERATING"
    rep["load_seconds"] = round(time.time() - t0, 1)
    write_progress(rep)

    for i, c in enumerate(cases):
        log(f"[{i + 1}/{len(cases)}] {c['id']} ...")
        t1 = time.time()
        try:
            gen = torch.Generator(device="cuda").manual_seed(a.seed)
            out = pipe(
                prompt=c["prompt"],
                image=[img],
                height=a.size, width=a.size,
                num_inference_steps=a.steps,
                guidance_scale=a.guidance,
                generator=gen,
            ).images[0]
        except torch.cuda.OutOfMemoryError as e:                 # noqa: BLE001
            log(f"  OOM: {e} -> retry with guidance_scale=1.0 (no CFG)")
            torch.cuda.empty_cache()
            gc.collect()
            gen = torch.Generator(device="cuda").manual_seed(a.seed)
            out = pipe(
                prompt=c["prompt"], image=[img],
                height=a.size, width=a.size,
                num_inference_steps=a.steps,
                guidance_scale=1.0, generator=gen,
            ).images[0]
            rep["errors"].append(f"{c['id']}: OOM retried with guidance=1.0")

        p = OUT / f"{c['id']}.png"
        out.save(p)
        dt = time.time() - t1
        log(f"  -> {p.name}  {dt:.1f}s")
        rep["cases"].append({
            "id": c["id"], "prompt": c["prompt"], "gt": c["gt"],
            "output": str(p.relative_to(ROOT)),
            "seconds": round(dt, 1),
        })
        write_progress(rep)
        torch.cuda.empty_cache()

    # ---- comparison grid: INPUT | 4 OUTPUTS  /  GT references --------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(rep["cases"])
        is_edit = a.prompt_set == "edit"
        header = ("FLUX.2 [klein] 4B Base  |  ONE aged old photo + DIFFERENT TEXT\n"
                  "same image / same seed / same steps -- TEXT is the only variable"
                  if is_edit else
                  "FLUX.2 [klein] 4B Base  |  ONE aged old photo + text -> restoration\n"
                  "native frozen Qwen3 text + native reference-image input, ZERO training")
        fig = plt.figure(figsize=(3.1 * (n + 1), 7.0))
        gs = fig.add_gridspec(2, n + 1, hspace=0.16, wspace=0.10)
        fig.suptitle(header, fontsize=13.5, fontweight="bold", y=0.98)

        ax = fig.add_subplot(gs[:, 0])
        ax.imshow(img)
        ax.set_title("SHARED INPUT\naged old photo", fontsize=10.5,
                     fontweight="bold", color="#a33")
        ax.axis("off")

        for j, row in enumerate(rep["cases"]):
            a1 = fig.add_subplot(gs[0, j + 1])
            a1.imshow(Image.open(ROOT / row["output"]).convert("RGB"))
            a1.set_title(f"OUTPUT: {row['id']}", fontsize=10.5, fontweight="bold")
            a1.axis("off")
            a2 = fig.add_subplot(gs[1, j + 1])
            if row.get("gt"):
                a2.imshow(Image.open(ROOT / row["gt"]).convert("RGB"))
                a2.set_title("dataset GT (reference)", fontsize=9, color="#555")
            else:
                # show the instruction itself: the whole point of the edit set
                words, lines, cur = row["prompt"].split(), [], ""
                for w in words:
                    if len(cur) + len(w) + 1 <= 26:
                        cur = (cur + " " + w).strip()
                    else:
                        if cur:
                            lines.append(cur)
                        cur = w
                if cur:
                    lines.append(cur)
                a2.text(0.0, 1.0, "TEXT:\n" + "\n".join(lines[:9]),
                        va="top", ha="left", fontsize=7.4, family="monospace",
                        transform=a2.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="#f2f5fa", ec="#9fb3c8"))
            a2.axis("off")
        fig.text(0.5, 0.012,
                 f"steps={a.steps}  size={a.size}  guidance={a.guidance}  "
                 f"seed={a.seed}  |  top = model output, "
                 f"bottom = {'the instruction' if is_edit else 'our dataset target'}",
                 ha="center", fontsize=8.6)
        grid = OUT / "grid.png"
        fig.savefig(grid, dpi=125, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        rep["grid"] = str(grid.relative_to(ROOT))
        log(f"grid -> {grid}")
    except Exception as e:                                       # noqa: BLE001
        rep["errors"].append(f"grid: {type(e).__name__}: {e}")

    rep["status"] = "DONE"
    rep["total_seconds"] = round(time.time() - t0, 1)
    write_progress(rep)
    log(f"DONE in {rep['total_seconds']}s, {len(rep['cases'])} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
