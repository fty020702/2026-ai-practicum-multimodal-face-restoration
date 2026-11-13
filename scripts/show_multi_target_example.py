from __future__ import annotations

"""Visualise ONE shared degraded image with its K different clean targets.

    # DIV2K-derived A-class dataset (default, backwards compatible)
    D:/Anaconda/envs/ChatAI/python.exe scripts/show_multi_target_example.py

    # generated OLD-PHOTO A-class dataset (2026-11-13)
    D:/Anaconda/envs/ChatAI/python.exe scripts/show_multi_target_example.py \
        --manifest data/old_photo_multitarget/train.jsonl \
        --out old_photo_multitarget_example.png \
        --json old_photo_multitarget_example.json \
        --title "ONE aged old photo -> FOUR different high-res restorations"

Outputs (defaults)
    outputs/samples/multi_target_example.png
    outputs/logs/multi_target_example.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.utils import ROOT


def wrap(text: str, width: int = 27) -> str:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/multi_target/train.jsonl")
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--out", default="multi_target_example.png")
    ap.add_argument("--json", default="multi_target_example.json")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    mpath = ROOT / a.manifest
    recs = [
        json.loads(l)
        for l in mpath.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    if not recs:
        raise RuntimeError(f"empty manifest: {mpath}; run the build script first")
    rec = recs[a.group]
    K = len(rec["targets"])
    title = a.title or (
        "ONE degraded image  ->  FOUR different clean targets\n"
        "the ONLY information that selects a target is the text"
    )

    outdir = ROOT / "outputs" / "samples"
    logdir = ROOT / "outputs" / "logs"
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(17, 8.4))
    gs = fig.add_gridspec(2, 1 + K, height_ratios=[1.0, 0.62],
                          hspace=0.12, wspace=0.24)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(Image.open(ROOT / rec["degraded"]).convert("RGB"))
    ax.set_title("SHARED INPUT x\n(identical for all 4 targets)",
                 fontsize=11, fontweight="bold", color="#b23b3b")
    ax.axis("off")

    report = {
        "manifest": a.manifest,
        "group_index": a.group,
        "group_id": rec["group_id"],
        "degraded": rec["degraded"],
        "shared_defects": rec["defects"],
        "caption": rec["caption"],
        "official_edit_prompt": rec.get("official_edit_prompt"),
        "targets": [],
    }
    for k, t in enumerate(rec["targets"]):
        im = Image.open(ROOT / t["clean"]).convert("RGB")
        ai = fig.add_subplot(gs[0, k + 1])   # do NOT shadow argparse's `a`
        ai.imshow(im)
        ai.set_title(f"y{k}: {t['id']}", fontsize=11, fontweight="bold")
        ai.axis("off")

        b = fig.add_subplot(gs[1, k + 1])
        b.axis("off")
        b.text(0.0, 1.0, "TEXT (prompt 0):\n" + wrap(t["prompts"][0], 27),
               va="top", ha="left", fontsize=8.2, family="monospace",
               transform=b.transAxes,
               bbox=dict(boxstyle="round,pad=0.35", fc="#f2f5fa", ec="#9fb3c8"))
        report["targets"].append({
            "id": t["id"], "clean": t["clean"],
            "description": t["description"],
            "n_prompt_variants": len(t["prompts"]),
            "prompt_0": t["prompts"][0],
            "prompts": t["prompts"],
        })

    out_png = outdir / a.out
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    out_json = logdir / a.json
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[example] group={rec['group_id']} K={K} -> {out_png}")
    print(f"[example] -> {out_json}")
    for t in rec["targets"]:
        print(f"  {t['id']:<16} {t['prompts'][0][:95]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
