from __future__ import annotations

"""Draw the SYSTEM ARCHITECTURE figure required by the official course syllabus.

    D:/Anaconda/envs/ChatAI/python.exe scripts/draw_system_architecture.py

Outputs
    results/figures/system_architecture.png   (report / slide ready, 200 dpi)
    outputs/samples/system_architecture.png  (working copy)

The figure is a FACTUAL diagram of what actually exists in this repository,
not an aspirational design: frozen Qwen3-4B + frozen FLUX.2 [klein] 4B DiT/VAE
are the two cooperating models; the ONLY trainable parameters are the added
text bypass. See docs/系统架构图.md for the editable Mermaid source.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.utils import ROOT

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# palette: light theme, dark text (report / PPT ready)
BLUE = ("#E9F0FD", "#3C6FD6")     # frozen model
ORANGE = ("#FFF0DC", "#DE8A20")   # trainable
GREEN = ("#E7F4EA", "#2E8B57")    # data
PURPLE = ("#F2EAFB", "#7A4CBF")   # evaluation
GREY = ("#F1F3F6", "#5B6B7C")     # application / output
INK = "#1E2A3A"
MUTED = "#4E5C6E"


def box(ax, x, y, w, h, title, sub="", c=BLUE, fs=10.3, sfs=8.2, lw=1.3,
        style="round,pad=0.02,rounding_size=0.09"):
    fc, ec = c
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                       edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    if sub:
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=INK, zorder=4)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=sfs, color=MUTED, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=INK, zorder=4)
    return p


def band(ax, y, h, label, c="#FAFBFD"):
    p = FancyBboxPatch((0.35, y), 15.30, h,
                       boxstyle="round,pad=0.02,rounding_size=0.10",
                       linewidth=1.0, edgecolor="#D6DEE8", facecolor=c, zorder=1)
    ax.add_patch(p)
    ax.text(0.52, y + h - 0.20, label, ha="left", va="center", fontsize=9.2,
            fontweight="bold", color="#6A7A8C", zorder=2)


def arrow(ax, x1, y1, x2, y2, label="", color="#5B6B7C", style="-|>",
          rad=0.0, ls="-", fs=8.0, lx=None, ly=None):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=12, linewidth=1.35, color=color,
                        linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=5)
    ax.add_patch(a)
    if label:
        px = lx if lx is not None else (x1 + x2) / 2
        py = ly if ly is not None else (y1 + y2) / 2
        ax.text(px, py, label, ha="center", va="center", fontsize=fs,
                color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.16", fc="white",
                          ec="none", alpha=0.92))


def main() -> int:
    fig, ax = plt.subplots(figsize=(16.0, 10.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10.15)
    ax.axis("off")

    ax.text(8.0, 9.93, "系统架构：多模态参考人脸图像恢复",
            ha="center", va="center", fontsize=19, fontweight="bold", color=INK)
    ax.text(8.0, 9.57,
            "双模型协同 · 冻结主干 · 唯一可训练文本旁路   |   RTX 4070 Ti 12 GB   |   2026-11-17",
            ha="center", va="center", fontsize=9.6, color=MUTED)

    # ---------------------------------------------------------------- layer 5
    band(ax, 8.02, 1.20, "应用层  APPLICATION")
    box(ax, 0.90, 8.18, 4.55, 0.78, "Web 交互 Demo（零新增依赖）",
        "上传老照片 + 文本 → 修复 / 编辑图", GREY)
    box(ax, 5.72, 8.18, 4.55, 0.78, "命令行复现脚本集",
        "训练 / 推理 / 消融 / 出图 / 评测", GREY)
    box(ax, 10.55, 8.18, 4.55, 0.78, "技术报告素材与私人仓库",
        "架构图 · 指标表 · 版本化代码", GREY)

    # ---------------------------------------------------------------- layer 4
    band(ax, 6.50, 1.24, "评测层  EVALUATION  (可证伪，不包装结论)")
    box(ax, 0.90, 6.64, 3.45, 0.82, "因果性  d_text",
        "文本 ON / OFF 输出差", PURPLE)
    box(ax, 4.54, 6.64, 3.45, 0.82, "一图多目标 penalty/gap",
        "严格 A/B：collapse_targets", PURPLE)
    box(ax, 8.18, 6.64, 3.45, 0.82, "样式选择性",
        "own vs wrong text loss", PURPLE)
    box(ax, 11.82, 6.64, 3.28, 0.82, "质量与效率",
        "PSNR / SSIM / LPIPS · 耗时 · 显存", PURPLE)

    # ---------------------------------------------------------------- layer 3
    band(ax, 4.98, 1.24, "训练 / 推理层  TRAIN & INFER")
    box(ax, 0.90, 5.12, 6.85, 0.82, "训练链：flow matching velocity MSE",
        "x_t=(1-t)x0+t·x1,  t~U(0.9,1.0),  loss=MSE(v_pred, x1-x0)", GREY)
    box(ax, 8.10, 5.12, 7.00, 0.82, "推理链：原生 FLUX.2 pipeline（零训练）",
        "image=[老照片] + Qwen3 prompt → 修复 / 内容编辑图", GREY)

    # ---------------------------------------------------------------- layer 2
    band(ax, 2.10, 2.62, "模型层  TWO COOPERATING MODELS + TRAINABLE TEXT BYPASS")

    # dashed highlight around the ONLY trainable column
    ax.add_patch(FancyBboxPatch((5.22, 2.34), 5.20, 2.06,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=1.7, edgecolor=ORANGE[1],
                                facecolor="#FFFAF1", linestyle="--", zorder=2))
    ax.text(15.45, 4.52, "橙色虚线框 = 唯一可训练（梯度只回传这里）",
            ha="right", va="center", fontsize=8.2, fontweight="bold",
            color="#A5620C", zorder=6)

    # top row = the actual model chain
    box(ax, 0.90, 3.40, 4.05, 0.72, "模型 ① Qwen3-4B 文本编码 [冻结]",
        "36 层 · hidden 2560 · 8.05 GB", BLUE, fs=9.7)
    box(ax, 5.36, 3.40, 4.92, 0.72, "可训练文本旁路 TextProjector",
        "7680 → 3072 · 末层零初始化 identity-at-init", ORANGE, fs=9.7)
    box(ax, 10.62, 3.40, 4.48, 0.72, "模型 ② FLUX.2 [klein] 4B DiT [冻结]",
        "5 double + 20 single · inner_dim 3072", BLUE, fs=9.7)

    # bottom row = supporting facts
    box(ax, 0.90, 2.42, 4.05, 0.72, "离线文本特征缓存",
        "prompt_embeds [1,512,7680] · 原子写入 / 自愈", BLUE, fs=9.5)
    box(ax, 5.36, 2.42, 4.92, 0.72, "5 × 解耦交叉注意力 + forward hook",
        "Q←img, K/V←text, o_proj=0 → 注入 double block img 流", ORANGE, fs=9.3)
    box(ax, 10.62, 2.42, 4.48, 0.72, "AutoencoderKLFlux2 VAE [冻结]",
        "32ch latent · patch 2×2 → in_channels 128", BLUE, fs=9.5)

    # horizontal model chain
    arrow(ax, 4.95, 3.76, 5.36, 3.76, "文本特征", color="#3C6FD6", fs=7.6,
          ly=3.98)
    arrow(ax, 10.28, 3.76, 10.62, 3.76, "additive bias", color="#DE8A20",
          fs=7.6, ly=3.98)
    # vertical detail links
    for cx in (2.925, 7.82, 12.86):
        arrow(ax, cx, 3.40, cx, 3.14, color="#9AA8B8", style="-|>", fs=7)

    ax.text(8.0, 2.18,
            "冻结边界：主干 3,959,590,691 参数 requires_grad=False；"
            "参考图 latent 与原生 prompt_embeds 均不参与梯度",
            ha="center", va="center", fontsize=7.9, color="#8A5A18", zorder=6)

    # ---------------------------------------------------------------- layer 1
    band(ax, 0.42, 1.42, "数据层  DATA (全部落在项目根目录)")
    box(ax, 0.90, 0.58, 3.45, 0.82, "合成 toy 数据",
        "64×64 · 64 对 · 6 种退化", GREEN)
    box(ax, 4.54, 0.58, 3.45, 0.82, "老照片一图多目标",
        "800 组 × K=4 · 463 MB", GREEN)
    box(ax, 8.18, 0.58, 3.45, 0.82, "官方 IP2P 500 MB 分片",
        "1,195 行 · 512² JPEG", GREEN)
    box(ax, 11.82, 0.58, 3.28, 0.82, "真实 FLUX.2 输出",
        "修复 4 张 + 内容编辑 4 张", GREEN)

    # ------------------------------------------------------------- inter-layer
    arrow(ax, 8.0, 1.84, 8.0, 2.10, "数据与 GT", color="#2E8B57",
          lx=8.0, ly=1.98, fs=7.8)
    arrow(ax, 8.0, 4.72, 8.0, 4.98, "冻结权重 + 唯一可训练分支",
          color="#3C6FD6", lx=8.0, ly=4.86, fs=7.8)
    arrow(ax, 8.0, 6.22, 8.0, 6.50, "输出 / loss / 消融", color="#7A4CBF",
          lx=8.0, ly=6.36, fs=7.8)
    arrow(ax, 8.0, 7.74, 8.0, 8.02, "指标与图", color="#5B6B7C",
          lx=8.0, ly=7.88, fs=7.8)

    # gradient note is now part of the model-layer header (see above)

    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    work = ROOT / "outputs" / "samples"
    work.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "system_architecture.png", dpi=200,
                bbox_inches="tight", facecolor="white", pad_inches=0.12)
    fig.savefig(work / "system_architecture.png", dpi=200,
                bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print(f"[arch] -> {out_dir / 'system_architecture.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
