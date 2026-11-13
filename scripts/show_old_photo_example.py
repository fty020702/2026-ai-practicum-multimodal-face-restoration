from __future__ import annotations

"""Visualise the REAL old-photo dataset and the unpaired real-old eval set.

    D:/Anaconda/envs/ChatAI/python.exe scripts/show_old_photo_example.py

Outputs
    outputs/samples/old_photo_example.png      paired clean / degraded / prompt
    outputs/samples/old_photo_real_eval.png    real old photos (NO ground truth)
    outputs/logs/old_photo_example.json        structured example record
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.data.old_photo_dataset import OldPhotoDataset
from src.utils import ROOT, load_config


def wrap(text: str, width: int = 54) -> str:
    """Word-wise wrapping (never cut a word in the middle)."""
    lines, cur = [], ""
    for word in text.split():
        if len(cur) + len(word) + 1 <= width:
            cur = (cur + " " + word).strip()
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def main() -> int:
    cfg = load_config(override=str(ROOT / "configs" / "old_photo.yaml"))
    ds = OldPhotoDataset(cfg, root=ROOT)
    base = ROOT / "data" / "old_photo"
    recs = [json.loads(l) for l in
            (base / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    n = min(4, len(recs))
    out_samples = ROOT / "outputs" / "samples"
    out_logs = ROOT / "outputs" / "logs"
    out_samples.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    # ---------------- paired samples ----------------
    fig, axes = plt.subplots(n, 3, figsize=(15.5, 3.6 * n))
    if n == 1:
        axes = axes[None, :]
    json_out = []
    for i in range(n):
        rec = recs[i]
        b = ds[i * ds.k]                      # variant 0 of the i-th REAL image
        cfull = Image.open(ROOT / rec["clean"]).convert("RGB")
        dfull = Image.open(ROOT / rec["degraded"]).convert("RGB")

        axes[i, 0].imshow(cfull)
        axes[i, 0].set_title(f"[{i}] CLEAN GT  (real DIV2K crop)", fontsize=10)
        axes[i, 1].imshow(dfull)
        axes[i, 1].set_title("DEGRADED  (paper-style old photo)", fontsize=10)

        info = (
            "TASK + USER TEXT\n"
            + wrap(b["prompt"], 54)
            + "\n\nAPPLIED DEFECTS\n"
            + wrap(", ".join(rec["defects"]), 54)
            + "\n\nGT SOURCE: " + rec.get("gt_source", "?")
            + "\nVARIANTS (same image, different text): " + str(ds.k)
        )
        axes[i, 2].axis("off")
        axes[i, 2].text(
            0.0, 0.98, info, va="top", ha="left", fontsize=8.6,
            family="monospace", transform=axes[i, 2].transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f4f4", edgecolor="#cccccc"),
        )
        for a in (axes[i, 0], axes[i, 1]):
            a.axis("off")

        # show that variant 1 changes ONLY the text
        b1 = ds[i * ds.k + 1]
        json_out.append({
            "index": i,
            "clean": rec["clean"],
            "degraded": rec["degraded"],
            "defects": rec["defects"],
            "gt_source": rec.get("gt_source"),
            "prompt_variant_0": b["prompt"],
            "prompt_variant_1": b1["prompt"],
            "image_identical_across_variants": True,
            "source": ds.source,
        })

    fig.suptitle(
        "REAL old-photo dataset - clean GT from DIV2K, degradation from the "
        "CVPR 2020 paper recipe, prompts de-confounded",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p1 = out_samples / "old_photo_example.png"
    fig.savefig(p1, dpi=125)
    plt.close(fig)

    # ---------------- unpaired real old photos (NO GT) ----------------
    eval_lines = []
    ep = base / "eval_real.jsonl"
    if ep.is_file():
        eval_lines = [json.loads(l) for l in ep.read_text(encoding="utf-8").splitlines()
                      if l.strip()]
    m = min(8, len(eval_lines))
    if m:
        cols = 4
        rows = (m + cols - 1) // cols
        fig2, ax2 = plt.subplots(rows, cols, figsize=(13, 3.2 * rows))
        ax2 = list(ax2.flat) if hasattr(ax2, "flat") else [ax2]
        for j, a in enumerate(ax2):
            a.axis("off")
            if j < m:
                im = Image.open(ROOT / eval_lines[j]["image"]).convert("RGB")
                a.imshow(im)
                a.set_title(Path(eval_lines[j]["image"]).name, fontsize=8)
        fig2.suptitle(
            "REAL old photographs - NO pixel-aligned ground truth\n"
            "(evaluation only: NIQE / BRISQUE / CLIP-IQA / user study, "
            "following Wan et al. 2020 and LL-Bench)",
            fontsize=11,
        )
        fig2.tight_layout(rect=(0, 0, 1, 0.93))
        p2 = out_samples / "old_photo_real_eval.png"
        fig2.savefig(p2, dpi=125)
        plt.close(fig2)
    else:
        p2 = None

    (out_logs / "old_photo_example.json").write_text(
        json.dumps({
            "source": ds.source,
            "n_pairs": len(recs),
            "prompts_per_image": ds.k,
            "n_real_old_eval": len(eval_lines),
            "examples": json_out,
            "images": [str(p1)] + ([str(p2)] if p2 else []),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[example] pairs={len(recs)} real_old={len(eval_lines)} -> {p1}")
    if p2:
        print(f"[example] -> {p2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
