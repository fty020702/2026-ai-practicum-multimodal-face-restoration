from __future__ import annotations

"""Visualise the REAL official InstructPix2Pix triplets from the 500 MB shard.

    D:/Anaconda/envs/ChatAI/python.exe scripts/show_ip2p_example.py

Outputs
    outputs/samples/ip2p_500m_example.png
    outputs/logs/ip2p_500m_example.json
    data/ip2p_500m/train_subset.jsonl   (manifest of the subset actually used)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import make_dataset
from src.utils import ROOT, load_config


def wrap(text: str, width: int = 40) -> str:
    lines, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def main() -> int:
    cfg = load_config(override=ROOT / "configs" / "ip2p_500m.yaml")
    ds = make_dataset(cfg, root=ROOT)
    ds.write_manifest()

    n = min(4, len(ds))
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 3.5 * n))
    if n == 1:
        axes = axes[None, :]
    fig.suptitle(
        "OFFICIAL InstructPix2Pix clip-filtered shard 0  (499.83 MB / 1,195 rows)\n"
        "format: original_image + edit_prompt -> edited_image",
        fontsize=13, fontweight="bold",
    )
    rep = []
    for i in range(n):
        b = ds[i]
        axes[i, 0].imshow(b["degraded"].permute(1, 2, 0).numpy())
        axes[i, 0].set_title(f"[{i}] INPUT  official original_image", fontsize=9.5)
        axes[i, 1].imshow(b["clean"].permute(1, 2, 0).numpy())
        axes[i, 1].set_title(f"[{i}] TARGET  official edited_image", fontsize=9.5)
        axes[i, 2].axis("off")
        axes[i, 2].text(0, 1.0,
                        "edit_prompt (user text):\n" + wrap(b["prompt"], 38)
                        + "\n\noriginal_prompt:\n" + wrap(b["task"], 38),
                        va="top", ha="left", fontsize=9, family="monospace",
                        transform=axes[i, 2].transAxes,
                        bbox=dict(boxstyle="round,pad=0.4", fc="#f3f5f9",
                                  ec="#9fb3c8"))
        for j in (0, 1):
            axes[i, j].axis("off")
        rep.append({
            "row": i, "edit_prompt": b["prompt"],
            "original_prompt": b["task"],
            "ip2p_hash": b["ip2p_hash"],
        })

    outdir = ROOT / "outputs" / "samples"
    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / "ip2p_500m_example.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130, facecolor="white")
    plt.close(fig)

    info = {
        "dataset": "timbrooks/instructpix2pix-clip-filtered shard 0",
        "shard_bytes": (ROOT / "data" / "ip2p_500m" / "raw" /
                        "train-00000-of-00262-57cebf95b4a9170c.parquet").stat().st_size,
        "shard_rows": ds.n_shard_rows,
        "rows_used_for_training": len(ds),
        "distinct_edit_prompts": len(ds.all_prompts()),
        "image_size_fed_to_model": cfg["data"]["image_size"],
        "a_class_native": False,
        "honesty": "SD+Prompt-to-Prompt generated image pairs; one target per "
                   "input, NOT one-input-many-targets.",
        "examples": rep,
    }
    out = ROOT / "outputs" / "logs" / "ip2p_500m_example.json"
    out.write_text(json.dumps(info, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[example] rows={len(ds)}/{ds.n_shard_rows} "
          f"distinct_prompts={len(ds.all_prompts())} -> {out_png}")
    print(f"[example] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
