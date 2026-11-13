from __future__ import annotations

"""Build the dataset the user asked for:

    OLD PHOTO (generated)  +  TEXT  ->  DIFFERENT HIGH-RES RESTORATIONS

Why self-built (2026-11-13): no public dataset has this combination.
  * Wan et al. CVPR 2020 / RealOld (Pik-Fix WACV 2023) / GGT-100K:
    ONE (degraded, clean) pair per image, no text-selected variants.
  * CIN25 / Text2Colors: multiple colourisations, but no text and no damage.
  * InstructPix2Pix / UltraEdit / SEED-Data-Edit: multiple instructions,
    but the input is an edit source, not an old photograph.
  * RFIR: partial / negative prompts, but rain/haze/low-light/snow, full data
    not officially released.
Real historical photos have NO pixel-aligned clean GT, so the only workable
route is: clean HQ image -> paper-style physical ageing -> old-photo input.
The user explicitly allowed generated images.

Construction (all deterministic from the seed):
    GT pool : official 499.83 MB InstructPix2Pix shard `edited_image`
              (512x512 JPEG, SD+Prompt-to-Prompt generated)
    x       : make_shared_degraded(GT) -- ONE old-photo input per group
              (fading + sepia + film_grain + scratches + dust_spots + vignette)
              shared by ALL K targets, so no target style is inferable from x
    y_k     : K=4 DIFFERENT high-res restoration targets via make_target()
              natural_color / monochrome / sepia_tone / minimal_repair
    text_k  : 3 distinct paraphrases per target, style phrase FIRST,
              + real `original_prompt` caption + independent constraint
              (caption/constraint are independent of the target id)

    => Var_k(y_k) > 0 ; only the text says which restoration is wanted.

    D:/Anaconda/envs/ChatAI/python.exe scripts/build_old_photo_multitarget.py --groups 300

Outputs
    data/old_photo_multitarget/old/          old-photo inputs x
    data/old_photo_multitarget/clean/        K high-res targets per group
    data/old_photo_multitarget/train.jsonl   MultiTargetDataset manifest
    data/old_photo_multitarget/dataset_info.json
    outputs/logs/old_photo_multitarget_progress.json

[HONESTY] x is a SYNTHETICALLY AGED modern/generated photograph, not a
digitized historical print; y1..y3 are deterministic tone transforms of the
GT, not independently restored photographs. This validates text-selected
restoration, not restoration quality or real user preference.
"""

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow.parquet as pq
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


def decode(item) -> np.ndarray:
    b = item.get("bytes") if isinstance(item, dict) else item
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"),
                    dtype=np.uint8, copy=True)


def save_jpg(a: np.ndarray, p: Path, quality: int = 93) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(p, "JPEG", quality=quality)


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(
        (a.astype(np.float32) - b.astype(np.float32)) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--quality", type=int, default=93)
    a = ap.parse_args()

    cfg = load_config(override=ROOT / "configs" / "ip2p_500m.yaml")
    parquet = ROOT / cfg["data"]["ip2p_parquet"]
    if not parquet.is_file():
        raise RuntimeError(f"missing 500MB shard: {parquet}")
    pf = pq.ParquetFile(parquet)

    base = ROOT / "data" / "old_photo_multitarget"
    odir, cdir = base / "old", base / "clean"
    odir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)

    lines, rms_list = [], []
    cols = ["edited_image", "original_prompt", "edit_prompt"]
    gid = 0
    for batch in pf.iter_batches(batch_size=8, columns=cols):
        ys = batch.column("edited_image").to_pylist()
        ops = batch.column("original_prompt").to_pylist()
        eps = batch.column("edit_prompt").to_pylist()
        for y, op, ep in zip(ys, ops, eps):
            if gid >= a.groups:
                break
            rng = np.random.default_rng(a.seed + gid * 104729)
            try:
                gt = decode(y)
            except Exception:                            # noqa: BLE001
                continue
            # ONE shared OLD-PHOTO input for all K targets
            x = make_shared_degraded(gt, rng)
            stem = f"{gid:05d}"
            xrel = f"data/old_photo_multitarget/old/{stem}.jpg"
            save_jpg(x, ROOT / xrel, a.quality)

            cap = " ".join(((op or "").strip() or
                            "a historical photograph").split()[:22])
            imgs, targets = {}, []
            for tid in TARGET_IDS:
                yk = make_target(gt, tid, rng, strong=True)
                rel = f"data/old_photo_multitarget/clean/{stem}_{tid}.jpg"
                save_jpg(yk, ROOT / rel, a.quality)
                imgs[tid] = yk
                targets.append({
                    "id": tid,
                    "clean": rel,
                    "description": TARGET_SPECS[tid]["description"],
                    "prompts": build_prompt_variants(tid, cap, rng, n=3),
                })

            pair = [rms(imgs[u], imgs[v])
                    for i, u in enumerate(TARGET_IDS)
                    for v in TARGET_IDS[i + 1:]]
            rms_list.append(float(np.mean(pair)))
            lines.append({
                "group_id": gid,
                "degraded": xrel,
                "caption": cap,
                "official_edit_prompt": (ep or "").strip(),
                "defects": list(SHARED_DEGRADATION_PLAN),
                "targets": targets,
            })
            gid += 1
            if gid % 25 == 0:
                print(f"[build] {gid}/{a.groups} groups  "
                      f"rms={np.mean(pair):.1f}", flush=True)
                save_json({"stage": "build", "done": gid,
                           "total": a.groups,
                           "last_rms": round(float(np.mean(pair)), 2)},
                          ROOT / "outputs" / "logs" /
                          "old_photo_multitarget_progress.json")
        if gid >= a.groups:
            break

    (base / "train.jsonl").write_text("".join(
        json.dumps(l, ensure_ascii=False) + "\n" for l in lines),
        encoding="utf-8")

    sizes = [p.stat().st_size for p in base.rglob("*.jpg")]
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "task": "OLD PHOTO (generated) + TEXT -> DIFFERENT HIGH-RES RESTORATIONS",
        "design": "ONE shared synthetically-aged old-photo input -> K=4 "
                  "different high-res restoration targets; text is the ONLY "
                  "disambiguating signal",
        "why_self_built": "no public dataset combines old-photo damage, "
                          "text instructions and multiple aligned HQ targets; "
                          "real historical photos have no clean GT",
        "benchmarks": [
            "Wan et al., CVPR 2020 (structured/unstructured old-photo defects)",
            "RealOld / Pik-Fix, WACV 2023 (200 real old photos, 1 GT each)",
            "GGT-100K, 2026 (generative GT, Old Photo 6,105 pairs, 1 GT each)",
            "CIN25 / Text2Colors-PAT (colourisation is ill-posed)",
            "InstructPix2Pix, CVPR 2023 (instruction -> target triplet)",
            "RFIR, ESWA 2025 (partial / negative restoration prompts)",
        ],
        "gt_pool": {
            "source": "timbrooks/instructpix2pix-clip-filtered shard 0",
            "path": str(parquet.relative_to(ROOT)),
            "bytes": parquet.stat().st_size,
            "rows_available": pf.metadata.num_rows,
            "column_used": "edited_image (512x512 JPEG)",
            "note": "SD + Prompt-to-Prompt generated, NOT real photographs",
        },
        "old_photo_input": {
            "function": "make_shared_degraded() from "
                        "src/data/multi_target_old_photo.py",
            "defects": list(SHARED_DEGRADATION_PLAN),
            "shared_across_targets": True,
        },
        "n_groups": len(lines),
        "K_targets_per_group": len(TARGET_IDS),
        "n_samples": len(lines) * len(TARGET_IDS),
        "n_prompt_variants": sum(len(t["prompts"]) for l in lines
                                 for t in l["targets"]),
        "target_ids": list(TARGET_IDS),
        "image_size": list(gt.shape[:2]),
        "inter_target_rms_mean": round(float(np.mean(rms_list)), 3),
        "inter_target_rms_min": round(float(np.min(rms_list)), 3),
        "jpg_bytes_total": int(sum(sizes)),
        "jpg_count": len(sizes),
        "anti_confound": [
            "the OLD-PHOTO input is IDENTICAL for all K targets",
            "the shared x always has a colour cast, grain AND scratches, so "
            "the desired restoration style cannot be read from the image",
            "style phrase is in the FIRST words of every prompt",
            "3 distinct paraphrases per target",
            "caption (official original_prompt) and constraints are "
            "independent of the target id",
        ],
        "honesty": [
            "x is a synthetically aged generated photograph, not a digitized "
            "historical print",
            "y0 is the official GT; y1..y3 are deterministic tone transforms "
            "of y0, not independent expert restorations",
            "validates text-selected restoration, NOT restoration quality, "
            "language understanding or real user preference",
        ],
    }
    (base / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build] DONE groups={len(lines)} K={len(TARGET_IDS)} "
          f"samples={info['n_samples']} rms={info['inter_target_rms_mean']} "
          f"jpg={info['jpg_bytes_total'] / 1e6:.1f}MB -> {base}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
