from __future__ import annotations

"""Summary card for the generated OLD-PHOTO multi-target dataset + strict A/B.

    D:/Anaconda/envs/ChatAI/python.exe scripts/show_old_photo_summary.py

Outputs
    outputs/samples/old_photo_multitarget_summary.png
    outputs/logs/old_photo_multitarget_summary.json

HONESTY: this run did NOT reach the 0.10 penalty/gap threshold. The figure
shows the measured numbers, not a passed verdict. See report["verdict"].
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

LOG = ROOT / "outputs" / "logs"
CK = ROOT / "outputs" / "checkpoints"
SAMPLE = ROOT / "outputs" / "samples"


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def pick_diag():
    """Prefer the larger 32-group probe; fall back to the 8-group one."""
    for name in ("ambiguity_old_photo_ab_g32.json", "ambiguity_old_photo_ab.json"):
        p = LOG / name
        if p.is_file():
            return p, jload(p)
    raise RuntimeError("run scripts/diagnose_ambiguity.py first")


def main() -> int:
    info = jload(ROOT / "data" / "old_photo_multitarget" / "dataset_info.json")
    diag_path, diag = pick_diag()
    h_a = jload(CK / "old_photo_multitarget" / "train_history.json")
    h_c = jload(CK / "old_photo_multitarget_control" / "train_history.json")

    ck = diag["checkpoints"]
    ctrl = next(v["means"] for k, v in ck.items() if "control" in k)
    multi = next(v["means"] for k, v in ck.items() if "control" not in k)
    gap = diag["data_gap"]["no_text_mse_floor_mean"]

    report = {
        "dataset": {
            "n_groups": info["n_groups"],
            "K": info["K_targets_per_group"],
            "n_samples": info["n_samples"],
            "jpg_MB": round(info["jpg_bytes_total"] / 1e6, 2),
            "inter_target_rms": info["inter_target_rms_mean"],
            "gt_pool": info["gt_pool"]["source"],
            "old_photo_defects": info["old_photo_input"]["defects"],
            "honesty": info["honesty"],
        },
        "training": {
            "steps": h_a["steps"], "device": h_a["device"],
            "a_class_loss_start": h_a["history"][0]["loss"],
            "a_class_loss_end": h_a["history"][-1]["loss"],
            "a_class_elapsed_s": h_a["history"][-1]["elapsed_s"],
            "control_loss_start": h_c["history"][0]["loss"],
            "control_loss_end": h_c["history"][-1]["loss"],
            "control_elapsed_s": h_c["history"][-1]["elapsed_s"],
            "t_range": [0.90, 1.0],
        },
        "diagnostic": {
            "file": str(diag_path.relative_to(ROOT)),
            "n_groups_probed": diag["n_groups_probed"],
            "timestep_t": diag["timestep_t"],
            "no_text_mse_floor": gap,
            "control_penalty_to_gap_pct": round(ctrl["penalty_to_gap_ratio"] * 100, 3),
            "a_class_penalty_to_gap_pct": round(multi["penalty_to_gap_ratio"] * 100, 3),
            "control_style_selectivity_pct": round(ctrl["style_selectivity"] * 100, 3),
            "a_class_style_selectivity_pct": round(multi["style_selectivity"] * 100, 3),
            "threshold_pct": 10.0,
            "control_text_needed": ctrl["text_needed_by_loss"],
            "a_class_text_needed": multi["text_needed_by_loss"],
        },
        "verdict": diag["verdict"],
        "comparison_with_ip2p_a_class": {
            "ip2p_control_penalty_to_gap_pct": -1.0,
            "ip2p_a_class_penalty_to_gap_pct": 36.73,
            "note": "the generated OLD-PHOTO variant is HARDER: 300 groups x 3 "
                    "paraphrases = 3600 distinct prompts, only ~1 pass per "
                    "sample in 1200 steps, and the 4 targets differ mostly in "
                    "global tone, which 64x64 downsampling compresses. "
                    "Do NOT claim the old-photo dataset reproduces the IP2P "
                    "36.73% result.",
        },
        "next_steps": [
            "fix prompt dilution: train with prompts[0] only (or 3000+ steps)",
            "raise t_min to 0.95 and keep the high-noise bias",
            "reduce num_pairs to 200 so each (group,target) is seen >=1.5x",
            "re-run the SAME strict A/B before claiming TEXT-NECESSITY-SUPPORTED",
        ],
    }

    SAMPLE.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "old_photo_multitarget_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    d = report["diagnostic"]
    fig = plt.figure(figsize=(13.5, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.30)
    fig.suptitle(
        "Generated OLD-PHOTO multi-target dataset  |  strict A/B  |  "
        "honest result: threshold NOT reached",
        fontsize=13.5, fontweight="bold", y=0.97,
    )

    # ---- left: dataset card ------------------------------------------- #
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    card = (
        "DATASET (self-built, no public equivalent)\n"
        "--------------------------------------------\n"
        f"groups           {info['n_groups']}\n"
        f"targets / group  K = {info['K_targets_per_group']}\n"
        f"samples          {info['n_samples']}\n"
        f"prompt variants  {info['n_prompt_variants']}\n"
        f"JPEG on disk     {info['jpg_bytes_total'] / 1e6:.1f} MB\n"
        f"inter-target RMS {info['inter_target_rms_mean']}\n"
        f"image size       {info['image_size']}\n"
        "\n"
        "ONE shared aged old photo  ->  4 HQ targets\n"
        "  natural_color / monochrome /\n"
        "  sepia_tone / minimal_repair\n"
        "\n"
        "shared defects: " + ", ".join(info["old_photo_input"]["defects"][:3]) + ",\n"
        "                " + ", ".join(info["old_photo_input"]["defects"][3:]) + "\n"
        "\n"
        "TRAINING (CUDA, toy trunk)\n"
        "--------------------------------------------\n"
        f"steps            {report['training']['steps']}\n"
        f"t range          [0.90, 1.0] (high noise)\n"
        f"A-class loss     {report['training']['a_class_loss_start']:.4f} -> "
        f"{report['training']['a_class_loss_end']:.4f}\n"
        f"control loss     {report['training']['control_loss_start']:.4f} -> "
        f"{report['training']['control_loss_end']:.4f}\n"
        f"elapsed          {report['training']['a_class_elapsed_s']}s / "
        f"{report['training']['control_elapsed_s']}s\n"
        f"probe groups     {d['n_groups_probed']}\n"
    )
    ax.text(0.0, 1.0, card, va="top", ha="left", fontsize=9.6,
            family="monospace", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.55", fc="#f4f7fb", ec="#9fb3c8"))

    # ---- right: two bar charts ---------------------------------------- #
    gs2 = gs[1].subgridspec(2, 1, hspace=0.48)
    labels = ["CONTROL\n(1 shared GT)", "A-CLASS\n(4 distinct GTs)"]
    colors = ["#8fa6bd", "#c0504d"]

    a1 = fig.add_subplot(gs2[0])
    vals = [d["control_penalty_to_gap_pct"], d["a_class_penalty_to_gap_pct"]]
    bars = a1.bar(labels, vals, color=colors, width=0.55)
    a1.axhline(10.0, color="#333333", ls="--", lw=1.2)
    a1.text(1.42, 10.6, "support threshold 10%", fontsize=8.5, ha="right")
    a1.set_ylabel("wrong-text penalty / gap (%)")
    a1.set_title("Does the wrong text cost loss?  (primary metric)",
                 fontsize=10.5, fontweight="bold")
    a1.set_ylim(0, max(40, max(vals) * 1.35))
    for b, v in zip(bars, vals):
        a1.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.2f}%",
                ha="center", fontsize=10, fontweight="bold")
    a1.grid(axis="y", alpha=0.3)

    a2 = fig.add_subplot(gs2[1])
    vals2 = [d["control_style_selectivity_pct"],
             d["a_class_style_selectivity_pct"]]
    bars2 = a2.bar(labels, vals2, color=colors, width=0.55)
    a2.set_ylabel("style selectivity (%)")
    a2.set_title("Does the text branch emit DIFFERENT deltas per style?",
                 fontsize=10.5, fontweight="bold")
    a2.set_ylim(0, max(12, max(vals2) * 1.5))
    for b, v in zip(bars2, vals2):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}%",
                ha="center", fontsize=10, fontweight="bold")
    a2.grid(axis="y", alpha=0.3)

    fig.text(0.5, 0.015,
             "VERDICT: NOT-YET-SUPPORTED - direction is right "
             f"({d['control_penalty_to_gap_pct']:.2f}% -> "
             f"{d['a_class_penalty_to_gap_pct']:.2f}%) but below the 10% bar; "
             "IP2P A-class reached 36.73%. Toy trunk + hash encoder, NOT language understanding.",
             ha="center", fontsize=8.8, color="#8a2f2f")
    out = SAMPLE / "old_photo_multitarget_summary.png"
    fig.savefig(out, dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[summary] {out}")
    print(f"[summary] diagnostic = {diag_path.name}")
    print(f"[summary] penalty/gap: control {d['control_penalty_to_gap_pct']}% "
          f"-> A-class {d['a_class_penalty_to_gap_pct']}% "
          f"(threshold {d['threshold_pct']}%)")
    print(f"[summary] verdict: {report['verdict'][:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
