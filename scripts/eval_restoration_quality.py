from __future__ import annotations

"""Restoration-quality performance analysis required by the course syllabus.

    D:/Anaconda/envs/ChatAI/python.exe scripts/eval_restoration_quality.py

Measures PSNR / SSIM / LPIPS on the REAL FLUX.2 [klein] 4B outputs that were
already generated (no re-inference, no training). Three questions:

  Q1  Does the generated output match its OWN dataset GT better than the
      degraded input does?               -> PSNR/SSIM up, LPIPS down?
  Q2  Does the TEXT pick the right target among K=4 GTs?  For each output,
      compute LPIPS to all 4 GTs; the own-GT distance should be the smallest.
  Q3  Do DIFFERENT TEXT instructions actually change the image?  Pairwise LPIPS
      between the 4 edit outputs; near-zero would mean the text is ignored.

Outputs
    outputs/logs/restoration_quality.json
    outputs/samples/restoration_quality.png
    results/reports/restoration_quality.json
    results/figures/restoration_quality.png

HONESTY NOTE: these are GENERATIVE reconstructions, not pixel-exact regression.
Low PSNR/SSIM is expected and is reported as a finding, not hidden.
"""

import json
import os
import shutil
import sys
import warnings
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch_hub"))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESTORE_PROGRESS = ROOT / "outputs/logs/flux2_old_photo_progress.json"
EDIT_PROGRESS = ROOT / "outputs/logs/flux2_old_photo_edit_progress.json"
INPUT_IMAGE = ROOT / "data/old_photo_multitarget/old/00000.jpg"
LOG_OUT = ROOT / "outputs/logs/restoration_quality.json"
FIG_OUT = ROOT / "outputs/samples/restoration_quality.png"


def load_rgb(p: Path, size: int = 512) -> torch.Tensor:
    """[1,3,H,W] float32 in [0,1]."""
    im = Image.open(p).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def fix_path(p: str) -> Path:
    """Progress JSONs may contain Windows backslashes."""
    return ROOT / p.replace("\\", "/")


class Metrics:
    """Stateless functional metrics -- no accumulated state across calls."""

    def __init__(self):
        import lpips
        self.lpips = lpips.LPIPS(net="alex", verbose=False).eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def all(self, a: torch.Tensor, b: torch.Tensor) -> dict:
        return {
            "psnr_db": round(float(peak_signal_noise_ratio(a, b,
                                                           data_range=1.0)), 4),
            "ssim": round(float(structural_similarity_index_measure(
                a, b, data_range=1.0)), 5),
            "lpips": round(float(self.lpips(a * 2 - 1, b * 2 - 1)), 5),
        }


def main() -> int:
    for p in (RESTORE_PROGRESS, EDIT_PROGRESS, INPUT_IMAGE):
        if not p.is_file():
            raise FileNotFoundError(p)

    restore = json.loads(RESTORE_PROGRESS.read_text(encoding="utf-8"))
    edit = json.loads(EDIT_PROGRESS.read_text(encoding="utf-8"))
    m = Metrics()
    x = load_rgb(INPUT_IMAGE)

    rep = {
        "status": "RUNNING",
        "scope": "REAL FLUX.2 [klein] 4B Base outputs, zero training, "
                 "native Qwen3 text + native image=[ref]",
        "metrics": "PSNR(dB) / SSIM / LPIPS(AlexNet) on 512x512 RGB",
        "input_image": str(INPUT_IMAGE.relative_to(ROOT)),
        "restore_run_seconds": restore.get("total_seconds"),
        "edit_run_seconds": edit.get("total_seconds"),
        "num_inference_steps": restore.get("num_inference_steps"),
        "cases": [], "edit_cases": [], "notes": [],
    }

    # ---------------------------------------------------------------- Q1 / Q2
    gts = {}
    for c in restore["cases"]:
        if c.get("gt"):
            gts[c["id"]] = load_rgb(fix_path(c["gt"]))
    preds = {c["id"]: load_rgb(fix_path(c["output"])) for c in restore["cases"]}

    for c in restore["cases"]:
        tid, pred = c["id"], preds[c["id"]]
        own = m.all(pred, gts[tid])
        base = m.all(x, gts[tid])
        wrong = {
            k: m.all(pred, g)["lpips"] for k, g in gts.items() if k != tid
        }
        mean_wrong = round(float(np.mean(list(wrong.values()))), 5)
        best_is_own = own["lpips"] < min(wrong.values())
        rep["cases"].append({
            "target_id": tid,
            "prompt": c["prompt"],
            "output": c["output"], "gt": c["gt"],
            "seconds": c["seconds"],
            "output_vs_own_gt": own,
            "degraded_input_vs_own_gt": base,
            "delta_psnr_db": round(own["psnr_db"] - base["psnr_db"], 4),
            "delta_ssim": round(own["ssim"] - base["ssim"], 5),
            "delta_lpips": round(own["lpips"] - base["lpips"], 5),
            "lpips_to_wrong_gts": wrong,
            "mean_lpips_to_wrong_gts": mean_wrong,
            "own_gt_is_nearest": bool(best_is_own),
        })

    own_sel = sum(1 for c in rep["cases"] if c["own_gt_is_nearest"])
    rep["target_selectivity"] = {
        "own_gt_nearest_count": own_sel,
        "total": len(rep["cases"]),
        "rate": round(own_sel / max(1, len(rep["cases"])), 4),
        "verdict": ("TEXT-TARGET-ALIGNMENT-OBSERVED" if own_sel >= 3
                    else "NO-TEXT-TARGET-ALIGNMENT"),
    }

    # ------------------------------------------------------------------- Q3
    edits = {c["id"]: load_rgb(fix_path(c["output"])) for c in edit["cases"]}
    pair = {}
    for a, b in combinations(edits, 2):
        pair[f"{a}__{b}"] = m.all(edits[a], edits[b])
    base_vs_input = {k: m.all(v, x) for k, v in edits.items()}
    vals = [p["lpips"] for p in pair.values()]
    rep["edit_cases"] = [
        {"id": c["id"], "prompt": c["prompt"], "output": c["output"],
         "vs_shared_input": base_vs_input[c["id"]]}
        for c in edit["cases"]
    ]
    rep["edit_pairwise"] = pair
    rep["text_response"] = {
        "mean_pairwise_lpips": round(float(np.mean(vals)), 5),
        "min_pairwise_lpips": round(float(np.min(vals)), 5),
        "max_pairwise_lpips": round(float(np.max(vals)), 5),
        "verdict": ("TEXT-CHANGES-OUTPUT" if float(np.mean(vals)) > 0.05
                    else "TEXT-IGNORED"),
    }

    # --------------------------------------------------------------- summary
    mean_psnr = float(np.mean([c["output_vs_own_gt"]["psnr_db"]
                               for c in rep["cases"]]))
    mean_ssim = float(np.mean([c["output_vs_own_gt"]["ssim"]
                               for c in rep["cases"]]))
    mean_lpips = float(np.mean([c["output_vs_own_gt"]["lpips"]
                                for c in rep["cases"]]))
    base_psnr = float(np.mean([c["degraded_input_vs_own_gt"]["psnr_db"]
                               for c in rep["cases"]]))
    base_lpips = float(np.mean([c["degraded_input_vs_own_gt"]["lpips"]
                                for c in rep["cases"]]))
    rep["summary"] = {
        "mean_psnr_db": round(mean_psnr, 4),
        "mean_ssim": round(mean_ssim, 5),
        "mean_lpips": round(mean_lpips, 5),
        "degraded_input_mean_psnr_db": round(base_psnr, 4),
        "degraded_input_mean_lpips": round(base_lpips, 5),
        "psnr_improvement_db": round(mean_psnr - base_psnr, 4),
        "lpips_improvement": round(base_lpips - mean_lpips, 5),
    }
    rep["notes"] = [
        "[FACT] 这些是原生 FLUX.2 生成式重建结果，零训练；"
        "不是本项目自训文本旁路的输出，不能用它证明一图多目标 verdict。",
        "[FACT] 生成式重建不追求逐像素对齐，PSNR / SSIM 偏低是预期现象，"
        "如实报告；评估感知质量时以 LPIPS 为主。",
        "[RESULT][负面] 本次实测输出相对 GT 的 PSNR / LPIPS 甚至劣于未修复的"
        "退化输入本身（见 psnr_improvement_db / lpips_improvement 为负），"
        "说明『零训练原生 FLUX.2』无法复现本数据集的特定修复目标；"
        "这正是需要训练文本旁路的原因，不得包装成质量提升。",
        "[FACT] LPIPS 使用 lpips(AlexNet) 官方实现；AlexNet 骨干权重"
        "由 TORCH_HOME 固定缓存在 models/torch_hub/ 内，不依赖外网可复现。",
        "[FACT] 编辑组无 GT，只能用『同一输入 + 不同文本 → 输出两两差异』"
        "作为文本响应代理指标，不能当作指令跟随准确率。",
    ]
    rep["status"] = "DONE"
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG_OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                       encoding="utf-8")

    # -------------------------------------------------------------- figure
    fig = plt.figure(figsize=(13.2, 8.2))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          hspace=0.34, wspace=0.22)
    fig.suptitle("恢复质量与文本响应性能分析  |  真实 FLUX.2 [klein] 4B Base，零训练",
                 fontsize=14.5, fontweight="bold", y=0.97)
    fig.text(0.5, 0.933,
             "PSNR / SSIM / LPIPS @512²   ·   20 steps   ·   RTX 4070 Ti 12 GB",
             ha="center", fontsize=9.4, color="#4E5C6E")

    # table 1: Q1
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.set_title("Q1  输出 vs 自身 GT（并与退化输入对照）", loc="left",
                 fontsize=11.5, fontweight="bold", pad=8)
    cols = ["目标 / 文本", "PSNR↑", "SSIM↑", "LPIPS↓",
            "退化输入 PSNR", "退化输入 LPIPS", "ΔPSNR"]
    rows = []
    for c in rep["cases"]:
        o, b = c["output_vs_own_gt"], c["degraded_input_vs_own_gt"]
        rows.append([c["target_id"], f'{o["psnr_db"]:.2f}',
                     f'{o["ssim"]:.4f}', f'{o["lpips"]:.4f}',
                     f'{b["psnr_db"]:.2f}', f'{b["lpips"]:.4f}',
                     f'{c["delta_psnr_db"]:+.2f}'])
    rows.append(["MEAN", f"{mean_psnr:.2f}", f"{mean_ssim:.4f}",
                 f"{mean_lpips:.4f}", f"{base_psnr:.2f}",
                 f"{base_lpips:.4f}", f"{mean_psnr - base_psnr:+.2f}"])
    t = ax.table(cellText=rows, colLabels=cols, cellLoc="center",
                 loc="center", colWidths=[0.20, 0.11, 0.11, 0.11, 0.16, 0.16, 0.11])
    t.auto_set_font_size(False)
    t.set_fontsize(9.0)
    t.scale(1.0, 1.65)
    for (r, cc), cell in t.get_celld().items():
        cell.set_edgecolor("#C9D4E2")
        if r == 0:
            cell.set_facecolor("#E9F0FD")
            cell.set_text_props(fontweight="bold", color="#1E2A3A")
        elif r == len(rows):
            cell.set_facecolor("#FFF0DC")
            cell.set_text_props(fontweight="bold")

    # table 2: Q2
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")
    ax2.set_title("Q2  文本是否选中正确目标（LPIPS 到 4 个 GT）", loc="left",
                  fontsize=11.0, fontweight="bold", pad=8)
    gids = list(gts)
    short = {"natural_color": "natural", "monochrome": "mono",
             "sepia_tone": "sepia", "minimal_repair": "minimal"}
    rows2, cols2 = [], ["输出"] + [short.get(g, g[:6]) for g in gids] + ["own最近?"]
    for c in rep["cases"]:
        r = [short.get(c["target_id"], c["target_id"][:8])]
        for g in gids:
            v = (c["output_vs_own_gt"]["lpips"] if g == c["target_id"]
                 else c["lpips_to_wrong_gts"][g])
            r.append(f"{v:.3f}")
        r.append("YES" if c["own_gt_is_nearest"] else "no")
        rows2.append(r)
    t2 = ax2.table(cellText=rows2, colLabels=cols2, cellLoc="center",
                   loc="center", colWidths=[0.20, 0.15, 0.15, 0.15, 0.15, 0.18],
                   bbox=[0.0, 0.22, 0.98, 0.66])
    t2.auto_set_font_size(False)
    t2.set_fontsize(7.8)
    for (r, cc), cell in t2.get_celld().items():
        cell.set_edgecolor("#C9D4E2")
        if r == 0:
            cell.set_facecolor("#F2EAFB")
            cell.set_text_props(fontweight="bold", color="#1E2A3A")
        elif rows2[r - 1][-1] == "YES" and cc == len(cols2) - 1:
            cell.set_facecolor("#E7F4EA")
            cell.set_text_props(fontweight="bold", color="#2E8B57")
    ax2.text(0.5, 0.02,
             f'own-GT 最接近: {own_sel}/{len(rep["cases"])}  '
             f'→ {rep["target_selectivity"]["verdict"]}',
             transform=ax2.transAxes, ha="center", fontsize=9.0,
             color="#2E8B57" if own_sel >= 3 else "#A5620C")

    # bar chart: Q3
    ax3 = fig.add_subplot(gs[1, 1])
    keys = list(pair)
    vals = [pair[k]["lpips"] for k in keys]
    labels = [k.replace("__", " vs ").replace("_", " ") for k in keys]
    bars = ax3.barh(range(len(vals)), vals, color="#7A4CBF", alpha=0.86)
    ax3.set_yticks(range(len(vals)))
    ax3.set_yticklabels(labels, fontsize=7.4)
    ax3.invert_yaxis()
    ax3.set_xlim(0, max(vals) * 1.22)
    ax3.set_xlabel("LPIPS（越大 = 文本造成的差异越大）", fontsize=8.6)
    ax3.set_title("Q3  同一老照片 + 不同文本 → 输出两两差异",
                  loc="left", fontsize=11.0, fontweight="bold", pad=8)
    ax3.grid(axis="x", alpha=0.3)
    ax3.axvline(0.05, color="#DE8A20", linestyle="--", linewidth=1.2)
    for b, v in zip(bars, vals):
        ax3.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                 va="center", fontsize=7.8)
    ax3.text(0.97, 0.96,
             f'mean={rep["text_response"]["mean_pairwise_lpips"]:.4f}\n'
             f'→ {rep["text_response"]["verdict"]}',
             transform=ax3.transAxes, ha="right", va="top", fontsize=8.2,
             color="#2E8B57"
             if rep["text_response"]["verdict"] == "TEXT-CHANGES-OUTPUT"
             else "#A5620C",
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec="#C9D4E2", alpha=0.94))

    fig.text(0.5, 0.018,
             "边界：原生 FLUX.2 零训练能力，非项目自训文本旁路成果；"
             "一图多目标 verdict 仍为 NOT-YET-SUPPORTED (7.28% < 10%)。",
             ha="center", fontsize=8.2, color="#8A5A18")
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for src, dst in ((LOG_OUT, ROOT / "results/reports/restoration_quality.json"),
                     (FIG_OUT, ROOT / "results/figures/restoration_quality.png")):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    print(json.dumps({
        "summary": rep["summary"],
        "target_selectivity": rep["target_selectivity"],
        "text_response": rep["text_response"],
        "json": str(LOG_OUT.relative_to(ROOT)),
        "figure": str(FIG_OUT.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
