from __future__ import annotations

"""Build the A-class dataset: ONE shared degraded image -> K different GTs.

Input  : data/old_photo/hr/*.png   (REAL DIV2K 256x256 crops, already on disk)
Output : data/multi_target/
             degraded/NNNNN.png          ONE shared degraded input x
             targets/NNNNN_<id>.png      K=4 different clean targets y_k
             train.jsonl                 groups, each with K (clean, prompts)
             dataset_info.json           incl. measured inter-target distances
         outputs/logs/multi_target_build_progress.json   (pollable)

    D:/Anaconda/envs/ChatAI/python.exe scripts/build_multi_target_data.py
    D:/Anaconda/envs/ChatAI/python.exe scripts/build_multi_target_data.py --pairs 24

No network access. If the old_photo GT crops are missing, run
scripts/build_old_photo_data.py first. This script will NEVER fabricate
synthetic images as a silent fallback.

See docs/一图多目标数据集调研.md for the paper/dataset comparison.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from src.data.multi_target_old_photo import (
    SHARED_DEGRADATION_PLAN,
    TARGET_IDS,
    TARGET_SPECS,
    build_prompt_variants,
    make_shared_degraded,
    make_target,
)
from src.utils import ROOT, load_config, save_json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def progress(stage: str, done: int, total: int, msg: str = "") -> None:
    print(f"[{stage} {done}/{total}] {msg}", flush=True)
    save_json({
        "stage": stage, "done": done, "total": total, "msg": msg,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }, ROOT / "outputs" / "logs" / "multi_target_build_progress.json")


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)))


def build(args) -> int:
    cfg = load_config(override=args.override)
    seed = int(cfg["project"]["seed"])
    base = ROOT / "data" / "multi_target"
    d_dir, t_dir = base / "degraded", base / "targets"
    d_dir.mkdir(parents=True, exist_ok=True)
    t_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = ROOT / "data" / "old_photo" / "hr"
    gt_files = sorted(
        p for p in gt_dir.glob("*")
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    if not gt_files:
        raise RuntimeError(
            f"[multi_target] no real GT crops in {gt_dir}. Run "
            "scripts/build_old_photo_data.py first. No synthetic fallback."
        )
    if args.pairs:
        gt_files = gt_files[: args.pairs]

    cap_path = ROOT / "data" / "old_photo" / "captions.json"
    captions = json.loads(cap_path.read_text(encoding="utf-8")) if cap_path.is_file() else []
    if not captions:
        captions = ["a historical photograph with natural scene detail"]

    lines, group_stats = [], []
    t0 = time.time()
    for i, gp in enumerate(gt_files):
        rng = np.random.default_rng(seed + i * 104729)
        clean = np.asarray(Image.open(gp).convert("RGB"), dtype=np.uint8)
        caption = captions[(i * 7 + 3) % len(captions)]

        # ONE shared degraded input for ALL targets of this image
        degraded = make_shared_degraded(clean, rng)
        stem = f"{i:05d}"
        Image.fromarray(degraded).save(d_dir / f"{stem}.png")

        targets, imgs = [], {}
        for tid in TARGET_IDS:
            y = make_target(clean, tid, rng, strong=args.strong)
            rel = f"data/multi_target/targets/{stem}_{tid}.png"
            Image.fromarray(y).save(base / "targets" / f"{stem}_{tid}.png")
            imgs[tid] = y
            targets.append({
                "id": tid,
                "clean": rel,
                "description": TARGET_SPECS[tid]["description"],
                "prompts": build_prompt_variants(tid, caption, rng, n=3),
            })

        # measured, not assumed: are the K targets actually different?
        pair_rms = [
            rms(imgs[a], imgs[b])
            for ai, a in enumerate(TARGET_IDS) for b in TARGET_IDS[ai + 1:]
        ]
        psnr = 20 * np.log10(255.0 / max(1e-6, float(np.mean(pair_rms))))
        group_stats.append({
            "group_id": i, "gt_source": gp.name,
            "inter_target_rms_min": round(min(pair_rms), 3),
            "inter_target_rms_mean": round(float(np.mean(pair_rms)), 3),
            "inter_target_rms_max": round(max(pair_rms), 3),
            "approx_psnr_between_targets_db": round(float(psnr), 2),
        })

        lines.append({
            "group_id": i,
            "degraded": f"data/multi_target/degraded/{stem}.png",
            "caption": caption,
            "defects": list(SHARED_DEGRADATION_PLAN),
            "targets": targets,
        })
        progress("GROUPS", i + 1, len(gt_files),
                 f"{stem} rms_mean={np.mean(pair_rms):.1f} "
                 f"({time.time() - t0:.1f}s)")

    (base / "train.jsonl").write_text(
        "".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines),
        encoding="utf-8",
    )

    n_prompts = sum(len(t["prompts"]) for l in lines for t in l["targets"])
    rms_means = [g["inter_target_rms_mean"] for g in group_stats]
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "task": "text-disambiguated old photo restoration",
        "design": "A-class: ONE shared degraded input -> K DIFFERENT clean "
                  "targets, each selected ONLY by its text",
        "why": "single-target data makes text redundant under MSE; with K "
               "targets the no-text MSE floor is Var_k(y_k) > 0, so plain "
               "velocity MSE already forces the model to use the text",
        "paper_benchmarks": [
            "InstructPix2Pix, CVPR 2023 (one source + instruction -> target)",
            "RFIR / TransRFIR, ESWA 2025 (partial / negative restoration prompts)",
            "PRISM (compound degradations, partial & negative prompts)",
            "InstructIR, ECCV 2024 (randomly sampled instructions)",
            "Wan et al., CVPR 2020 (structured / unstructured old-photo defects)",
            "CIN25 / Text2Colors-PAT (colourisation is ill-posed, multi-solution)",
        ],
        "gt_source": {
            "path": "data/old_photo/hr/*.png",
            "origin": "mAiello00/DIV2K (DIV2K_train_HR), real 2K photographs",
            "n_groups": len(lines),
        },
        "caption_source": "data/old_photo/captions.json (real human captions)",
        "n_groups": len(lines),
        "target_strength": "strong" if args.strong else "mild (round 1)",
        "K_targets_per_group": len(TARGET_IDS),
        "n_training_samples": len(lines) * len(TARGET_IDS),
        "n_prompt_variants_total": n_prompts,
        "target_ids": list(TARGET_IDS),
        "shared_degradation_plan": list(SHARED_DEGRADATION_PLAN),
        "inter_target_distance": {
            "unit": "RGB uint8 RMS over pixels (mean over all target pairs)",
            "mean_over_groups": round(float(np.mean(rms_means)), 3),
            "min_over_groups": round(float(np.min(rms_means)), 3),
            "max_over_groups": round(float(np.max(rms_means)), 3),
            "note": ">0 is REQUIRED; if 0 the text has nothing to disambiguate",
        },
        "anti_confound": [
            "the degraded image is IDENTICAL for all K targets",
            "target style phrase is in the FIRST words of every prompt",
            "3 distinct paraphrases per target (toy hash encoder sees 8 tokens)",
            "captions and constraints are sampled independently of target id",
            "the shared x always contains a colour cast, grain AND scratches, "
            "so no target style is inferable from the image",
        ],
        "limitations": [
            "targets are deterministic tone/effort transforms of a real GT, "
            "not real alternate photographs; this validates disambiguation, "
            "not restoration quality or real user preference distribution",
            "toy HashTextEncoder is not semantic; language understanding "
            "claims require the Qwen3-4B offline cache path",
            "real historical photos still have NO clean GT (unpaired eval only)",
        ],
        "per_group": group_stats,
    }
    (base / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[build] DONE groups={len(lines)} K={len(TARGET_IDS)} "
          f"samples={len(lines) * len(TARGET_IDS)} prompts={n_prompts} "
          f"inter_target_rms={np.mean(rms_means):.2f}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"[build] -> {base}", flush=True)
    progress("DONE", len(lines), len(lines), "multi-target dataset built")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=0,
                    help="cap number of source groups (0=all)")
    ap.add_argument("--strong", action="store_true",
                    help="amplify inter-target tone differences (round 2)")
    ap.add_argument("--override", default=None)
    return build(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
