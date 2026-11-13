from __future__ import annotations

"""Show EXACTLY what the current GPU training consumes, side by side with the
REAL old-photo dataset that is on disk but NOT wired into this run.

    D:/Anaconda/envs/ChatAI/python.exe scripts/show_train_data_used.py

Writes outputs/samples/train_data_used.png and outputs/logs/train_data_used.json.
CPU only, no model loading.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image

from src.utils import ROOT, load_config, save_json
from src.data.dataset import make_dataset


def wrap(text: str, width: int = 58) -> str:
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return "\n".join(out)


def show_image(ax, arr, title):
    ax.imshow(arr)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    cfg = load_config(override=ROOT / "configs" / "flux2_klein.yaml")
    ds = make_dataset(cfg, root=cfg["_root"])
    manifest = ROOT / "data" / "old_photo" / "train.jsonl"
    real_recs = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines()]

    n_syn, n_real = 4, 2
    fig = plt.figure(figsize=(15.5, 10.5))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[2.0, 1.15], hspace=0.32)

    # ---------------- TOP: what the GPU run ACTUALLY uses ----------------
    gs_top = GridSpecFromSubplotSpec(n_syn, 3, subplot_spec=outer[0],
                                    width_ratios=[1.0, 1.0, 2.5], hspace=0.42,
                                    wspace=0.12)
    fig.text(0.5, 0.965,
             "CURRENT GPU TRAINING DATA  -  configs/flux2_klein.yaml inherits default.yaml",
             ha="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.940,
             f"RestorationDataset | source={ds.source} | {len(ds)} pairs | "
             f"{cfg['data']['image_size']}x{cfg['data']['image_size']} | "
             f"{len(cfg['data']['degradations'])} synthetic degradations  "
             "(NOT real photographs)",
             ha="center", fontsize=10, color="#A32D2D")

    syn_meta = []
    for i in range(n_syn):
        b = ds[i]
        clean = b["clean"].permute(1, 2, 0).numpy()
        degraded = b["degraded"].permute(1, 2, 0).numpy()
        show_image(fig.add_subplot(gs_top[i, 0]), clean, f"[{i}] clean GT (synthetic)")
        show_image(fig.add_subplot(gs_top[i, 1]), degraded, "degraded")
        ax = fig.add_subplot(gs_top[i, 2])
        ax.axis("off")
        info = (f"task   : {b['task']}\n"
                f"prompt : {wrap(b['prompt'], 58)}\n"
                f"tensor : [3, {cfg['data']['image_size']}, "
                f"{cfg['data']['image_size']}] float32 in [0,1]\n"
                "FLUX.2 VAE -> packed latent tokens [1, 16, 128]")
        ax.text(0.01, 0.9, info, va="top", ha="left", fontsize=8.6,
                family="monospace", transform=ax.transAxes)
        syn_meta.append({"index": i, "task": b["task"], "prompt": b["prompt"]})

    # ---------------- BOTTOM: real data on disk, NOT used ----------------
    gs_bot = GridSpecFromSubplotSpec(n_real, 3, subplot_spec=outer[1],
                                    width_ratios=[1.0, 1.0, 2.5], hspace=0.45,
                                    wspace=0.12)
    fig.text(0.5, 0.455,
             "ON DISK BUT NOT USED BY THIS RUN  -  data/old_photo/  "
             "(configs/old_photo.yaml has backend.name=toy)",
             ha="center", fontsize=12, fontweight="bold", color="#185FA5")
    fig.text(0.5, 0.435,
             "24 paired REAL DIV2K 256x256 crops + 48 real captions + "
             "14 real old photos (no GT) | 11 paper-style ageing defects",
             ha="center", fontsize=10, color="#185FA5")

    real_meta = []
    for i in range(n_real):
        rec = real_recs[i]
        clean = Image.open(ROOT / rec["clean"]).convert("RGB")
        degraded = Image.open(ROOT / rec["degraded"]).convert("RGB")
        show_image(fig.add_subplot(gs_bot[i, 0]), clean, "real clean GT (DIV2K crop)")
        show_image(fig.add_subplot(gs_bot[i, 1]), degraded, "paper-style aged")
        ax = fig.add_subplot(gs_bot[i, 2])
        ax.axis("off")
        info = (f"gt_source : {rec.get('gt_source')}\n"
                f"defects   : {', '.join(rec.get('defects', []))}\n"
                f"caption   : {wrap(rec.get('caption', ''), 58)}\n"
                "status    : NOT loaded by this GPU run")
        ax.text(0.01, 0.9, info, va="top", ha="left", fontsize=8.4,
                family="monospace", transform=ax.transAxes)
        real_meta.append({"gt_source": rec.get("gt_source"),
                          "defects": rec.get("defects")})

    out_png = ROOT / "outputs" / "samples" / "train_data_used.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    rep = {
        "question": "what data does the CURRENT GPU training use?",
        "current_run": {
            "config_override": "configs/flux2_klein.yaml",
            "backend": cfg["backend"]["name"],
            "dataset_class": type(ds).__name__,
            "source": ds.source,
            "n_pairs": len(ds),
            "image_size": cfg["data"]["image_size"],
            "degradations": cfg["data"]["degradations"],
            "distinct_prompts": len({b["prompt"] for b in ds}),
            "prompt_confound": "degradation type and task prompt are strictly "
                               "1-to-1 coupled (6 tasks -> 6 prompts)",
            "sample_tensor": "[3,64,64] float32 [0,1]",
            "flux2_packed_tokens": "[1,16,128] after the real VAE",
            "text_condition": "offline Qwen3 cache [1,512,7680] for the "
                              "trainable branch; frozen constant "
                              "NEUTRAL_PROMPT stream for the native text stream",
            "examples": syn_meta,
        },
        "on_disk_but_unused": {
            "config": "configs/old_photo.yaml",
            "backend_in_that_config": "toy",
            "why_unused": "train.py accepts ONE --override; old_photo.yaml sets "
                          "data.source but leaves backend.name=toy, so it "
                          "cannot be stacked with flux2_klein.yaml as-is",
            "n_paired": len(real_recs),
            "examples": real_meta,
        },
        "png": str(out_png),
    }
    save_json(rep, ROOT / "outputs" / "logs" / "train_data_used.json")
    print(json.dumps(rep["current_run"], ensure_ascii=False, indent=2)[:1200])
    print(f"\n[show-train-data] -> {out_png}")


if __name__ == "__main__":
    main()
