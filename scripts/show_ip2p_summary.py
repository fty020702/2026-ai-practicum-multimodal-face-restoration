from __future__ import annotations

"""One-page chart: official 500 MB shard + A/B text-necessity result.

    D:/Anaconda/envs/ChatAI/python.exe scripts/show_ip2p_summary.py

Outputs
    outputs/samples/ip2p_500m_summary.png
    outputs/logs/ip2p_500m_summary.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import ROOT

LOGS = ROOT / "outputs" / "logs"


def main() -> int:
    ab = json.loads((LOGS / "ambiguity_ip2p_ab.json").read_text(encoding="utf-8"))
    diag = json.loads((LOGS / "ip2p_text_diagnosis.json").read_text(encoding="utf-8"))
    info = json.loads((ROOT / "data" / "ip2p_500m" / "multitarget" /
                       "dataset_info.json").read_text(encoding="utf-8"))
    shard = json.loads((LOGS / "ip2p_shard_inspect.json").read_text(
        encoding="utf-8"))

    ms = {k: v["means"] for k, v in ab["checkpoints"].items()}
    ctrl = next(v for k, v in ms.items() if "control" in k)
    multi = next(v for k, v in ms.items() if "control" not in k)

    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.72, 1.0], hspace=0.52,
                          wspace=0.28)
    fig.suptitle("Official InstructPix2Pix 500 MB shard -> local training",
                 fontsize=15, fontweight="bold", y=0.98)

    # ---- dataset card ----
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    card = (
        f"DATASET  timbrooks/instructpix2pix-clip-filtered  shard 0\n"
        f"  {shard['file_bytes'] / 1e6:.2f} MB  |  {shard['n_rows']} rows  |  "
        f"512x512 JPEG  |  triplet: original_image / edit_prompt / edited_image\n"
        f"  trained on 800 rows / 748 distinct real edit_prompts  |  "
        f"A-class view: {info['n_groups']} groups x {info['K_targets_per_group']} "
        f"targets = {info['n_samples']} samples, inter-target RMS "
        f"{info['inter_target_rms_mean']}\n"
        f"  [FACT] images are SD+Prompt-to-Prompt generated, NOT real photos; "
        f"shard has a_class_native=false (1 target per input)"
    )
    ax.text(0.01, 0.95, card, va="top", ha="left", fontsize=9.6,
            family="monospace", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc="#eef2f7", ec="#9fb3c8"))

    # ---- A: penalty / gap ----
    a = fig.add_subplot(gs[1, 0])
    vals = [ctrl["penalty_to_gap_ratio"] * 100,
            multi["penalty_to_gap_ratio"] * 100,
            diag["means"]["wrong_penalty_relative"] * 100]
    labels = ["control\n(single GT)", "A-class\n(4 GTs)", "official triplet\n(no A-class view)"]
    colors = ["#8fa5bd", "#d1495b", "#c9a227"]
    bars = a.bar(range(3), vals, color=colors, edgecolor="white", linewidth=1.2)
    a.axhline(10, ls="--", lw=1.2, color="#444", label="support threshold 10%")
    a.set_ylabel("wrong-text penalty / theoretical gap  (%)")
    a.set_title("A. Does the objective NEED the specific text?", fontsize=10.5)
    a.set_xticks(range(3))
    a.set_xticklabels(labels, fontsize=8.5)
    a.legend(fontsize=8)
    a.grid(axis="y", alpha=0.3)
    a.set_axisbelow(True)
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2,
               v + (1.2 if v >= 0 else -2.4), f"{v:.2f}%",
               ha="center", fontsize=9, fontweight="bold")

    # ---- B: style selectivity ----
    b_ax = fig.add_subplot(gs[1, 1])
    sel = [ctrl["style_selectivity"] * 100, multi["style_selectivity"] * 100,
           diag["means"]["style_selectivity"] * 100]
    bars2 = b_ax.bar(range(3), sel, color=colors, edgecolor="white", linewidth=1.2)
    b_ax.set_ylabel("d_text(wrong) / d_text(on-off)  (%)")
    b_ax.set_title("B. Style selectivity of the text branch", fontsize=10.5)
    b_ax.set_xticks(range(3))
    b_ax.set_xticklabels(labels, fontsize=8.5)
    b_ax.grid(axis="y", alpha=0.3)
    b_ax.set_axisbelow(True)
    for b, v in zip(bars2, sel):
        b_ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}%",
                  ha="center", fontsize=9, fontweight="bold")

    fig.text(0.5, 0.015,
             "1200 steps / CUDA / toy trunk / t~U(0.9,1) / lr 5e-4. "
             "Official triplet trains fine but shows NO text-specific response; "
             "adding multiple targets restores it.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))

    png = ROOT / "outputs" / "samples" / "ip2p_500m_summary.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=140, facecolor="white")
    plt.close(fig)

    summary = {
        "dataset": {
            "repo": "timbrooks/instructpix2pix-clip-filtered",
            "shard_bytes": shard["file_bytes"],
            "shard_rows": shard["n_rows"],
            "a_class_native": shard["a_class_native"],
            "image_note": "SD + Prompt-to-Prompt generated, NOT real photos",
            "rows_used_triplet": 800,
            "distinct_edit_prompts": 748,
            "a_class_groups": info["n_groups"],
            "a_class_samples": info["n_samples"],
            "inter_target_rms": info["inter_target_rms_mean"],
            "total_dir_MB": round(sum(
                p.stat().st_size for p in
                (ROOT / "data" / "ip2p_500m").rglob("*") if p.is_file()
            ) / 1e6, 2),
        },
        "training": {
            "triplet": {"steps": 1200, "device": "cuda",
                        "loss_start": 1.453311, "loss_end": 1.030900,
                        "elapsed_s": 28},
            "a_class": {"steps": 1200, "device": "cuda"},
            "control": {"steps": 1200, "device": "cuda"},
        },
        "results": {
            "official_triplet_wrong_penalty_relative":
                diag["means"]["wrong_penalty_relative"],
            "official_triplet_verdict": diag["verdict"],
            "control_penalty_to_gap_pct":
                round(ctrl["penalty_to_gap_ratio"] * 100, 2),
            "a_class_penalty_to_gap_pct":
                round(multi["penalty_to_gap_ratio"] * 100, 2),
            "control_style_selectivity_pct":
                round(ctrl["style_selectivity"] * 100, 2),
            "a_class_style_selectivity_pct":
                round(multi["style_selectivity"] * 100, 2),
            "a_class_text_needed_by_loss": multi["text_needed_by_loss"],
            "verdict": ab["verdict"],
        },
        "honesty": [
            "toy trunk (random, frozen) + HashTextEncoder; NOT language "
            "understanding, NOT restoration quality, NOT real FLUX.2",
            "real FLUX.2 would need a fresh Qwen3 offline cache for these "
            "748+ new edit_prompts (CPU ~3h / 10 prompts)",
            "y1..y3 of the A-class view are deterministic tone transforms of "
            "the official edited_image, not independent photographs",
        ],
    }
    out = LOGS / "ip2p_500m_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[summary] -> {png}")
    print(f"[summary] -> {out}")
    print(json.dumps(summary["results"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
