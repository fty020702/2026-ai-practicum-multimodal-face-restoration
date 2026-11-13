from __future__ import annotations

"""Summarise the two A/B rounds into one reproducible chart + JSON report.

Round 1 (outputs/logs/ambiguity_round1.json): mild target differences,
400 steps, t ~ U(0,1) and t ~ U(0.8,1).
Round 2 (outputs/logs/ambiguity_round2.json): amplified target differences,
1200 steps, lr 5e-4, t ~ U(0.9,1).

    D:/Anaconda/envs/ChatAI/python.exe scripts/show_ambiguity_ab.py

Outputs
    outputs/samples/ambiguity_ab.png
    outputs/logs/ambiguity_summary.json
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

LABELS = {
    "amb_single_target": "single t~U(0,1)",
    "amb_multi_target": "MULTI t~U(0,1)",
    "amb_single_hn": "single t~U(.8,1)",
    "amb_multi_hn": "MULTI t~U(.8,1)",
    "amb_single_strong": "single STRONG",
    "amb_multi_strong": "MULTI STRONG",
}
ORDER = [
    "amb_single_target", "amb_multi_target",
    "amb_single_hn", "amb_multi_hn",
    "amb_single_strong", "amb_multi_strong",
]


def short_key(k: str) -> str:
    for full, label in LABELS.items():
        if full in k.replace("/", "\\"):
            return full
    return k


def main() -> int:
    logs = ROOT / "outputs" / "logs"
    r1 = json.loads((logs / "ambiguity_round1.json").read_text(encoding="utf-8"))
    r2 = json.loads((logs / "ambiguity_round2.json").read_text(encoding="utf-8"))

    merged = {}
    for rep in (r1, r2):
        for k, v in rep["checkpoints"].items():
            merged[short_key(k)] = v["means"]

    present = [k for k in ORDER if k in merged]
    gap1 = r1["data_gap"]["no_text_mse_floor_mean"]
    gap2 = r2["data_gap"]["no_text_mse_floor_mean"]

    penal = [merged[k]["penalty_to_gap_ratio"] * 100 for k in present]
    sel = [merged[k]["style_selectivity"] * 100 for k in present]
    colors = ["#d1495b" if "MULTI" in LABELS[k] else "#8fa5bd" for k in present]
    x = np.arange(len(present))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    fig.suptitle(
        "Does the training objective actually NEED the text?\n"
        "ONE degraded image -> K different clean targets vs. strict single-target control",
        fontsize=13, fontweight="bold",
    )

    a = axes[0]
    bars = a.bar(x, penal, color=colors, edgecolor="white", linewidth=1.2)
    a.axhline(10, ls="--", lw=1.3, color="#444", label="support threshold (10% of gap)")
    a.set_ylabel("wrong-text penalty / theoretical gap  (%)")
    a.set_title("A. loss penalty when the WRONG target's text is given", fontsize=10.5)
    a.set_xticks(x)
    a.set_xticklabels([LABELS[k] for k in present], rotation=28, ha="right", fontsize=8.5)
    a.legend(fontsize=8)
    for b, v in zip(bars, penal):
        a.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%",
               ha="center", fontsize=8.5, fontweight="bold")

    b = axes[1]
    bars2 = b.bar(x, sel, color=colors, edgecolor="white", linewidth=1.2)
    b.set_ylabel("d_text(wrong) / d_text(on-off)  (%)")
    b.set_title("B. style selectivity of the text branch", fontsize=10.5)
    b.set_xticks(x)
    b.set_xticklabels([LABELS[k] for k in present], rotation=28, ha="right", fontsize=8.5)
    for bb, v in zip(bars2, sel):
        b.text(bb.get_x() + bb.get_width() / 2, v + 0.15, f"{v:.1f}%",
               ha="center", fontsize=8.5, fontweight="bold")

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    fig.text(0.5, 0.015,
             "Red = MULTI-target (K=4 different clean GTs)   Grey = strict control "
             "(same 4 texts, ONE shared GT).  Round1 gap=%.5f, Round2 gap=%.5f"
             % (gap1, gap2),
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))

    out_png = ROOT / "outputs" / "samples" / "ambiguity_ab.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, facecolor="white")
    plt.close(fig)

    multi_s = merged.get("amb_multi_strong", {})
    single_s = merged.get("amb_single_strong", {})
    ratio = (multi_s.get("wrong_penalty", 0) /
             max(single_s.get("wrong_penalty", 1e-12), 1e-12))
    summary = {
        "question": "does one-image-many-targets data force the text branch "
                    "to use the specific instruction?",
        "round1": {
            "config": "mild tone differences, 400 steps, lr 1e-4",
            "gap": gap1,
            "note": "direction correct but magnitude too small; t leakage and "
                    "weak target signal",
            "verdict": r1["verdict"],
        },
        "round2": {
            "config": "amplified tone differences, 1200 steps, lr 5e-4, "
                      "t ~ U(0.9, 1.0)",
            "gap": gap2,
            "verdict": r2["verdict"],
        },
        "per_config": {LABELS[k]: merged[k] for k in present},
        "headline": {
            "multi_target_penalty_to_gap_pct": round(
                multi_s.get("penalty_to_gap_ratio", 0) * 100, 2),
            "single_target_penalty_to_gap_pct": round(
                single_s.get("penalty_to_gap_ratio", 0) * 100, 2),
            "wrong_penalty_multi_over_single": round(ratio, 2),
            "multi_style_selectivity_pct": round(
                multi_s.get("style_selectivity", 0) * 100, 2),
            "single_style_selectivity_pct": round(
                single_s.get("style_selectivity", 0) * 100, 2),
        },
        "honesty": [
            "[RESULT] measured on the TOY trunk (random, frozen) with a hash "
            "text encoder. This proves the DATA DESIGN makes text necessary "
            "and the training objective can exploit it; it does NOT prove "
            "language understanding or restoration quality.",
            "[FACT] real FLUX.2 [klein] 4B multi-target training is NOT yet "
            "run: the new 288 prompts need a fresh Qwen3-4B offline cache "
            "(~3 h for 10 prompts on CPU).",
            "[INFERENCE] t_min=0.9 is an experimental high-noise sampling "
            "choice, not the standard U(0,1) schedule. Default configs keep "
            "(0,1) so all earlier [RESULT]s are bit-for-bit unchanged.",
        ],
    }
    out_json = logs / "ambiguity_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[ab] -> {out_png}")
    print(f"[ab] -> {out_json}")
    print(json.dumps(summary["headline"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
