from __future__ import annotations

"""Build an A-CLASS view from the official 500 MB InstructPix2Pix shard.

Motivation (2026-11-13): the official shard is a valid triplet format and
trains fine, but scripts/diagnose_ip2p_text.py measured NO text-specific
response (wrong_penalty_relative = 0.017%). scripts/inspect_ip2p_shard.py
explains why: a_class_native=false, 1,195 distinct inputs, 1 target each.
So we reuse the VERIFIED A-class construction from §8.2, but with the
OFFICIAL 512x512 image pairs as the GT pool instead of 24 DIV2K crops:

    x   = official original_image (SHARED for all K targets)
    y0  = official edited_image  (natural_color, unmodified)
    y1..y3 = deterministic plausible styles of y0
             (monochrome / sepia_tone / minimal_repair)
    text_k = a distinct style paraphrase (FIRST words, hash-encoder visible)
             + the REAL official edit_prompt as content tail

    => Var_k(y_k) > 0 ; only the text says which target is wanted.

    D:/Anaconda/envs/ChatAI/python.exe scripts/build_ip2p_multitarget.py --groups 200

Outputs
    data/ip2p_500m/multitarget/inputs/*.jpg
    data/ip2p_500m/multitarget/targets/*.jpg
    data/ip2p_500m/multitarget/train.jsonl
    data/ip2p_500m/multitarget/dataset_info.json

[HONESTY] y0 is a real official edited_image (SD/P2P generated, not a real
photograph); y1..y3 are deterministic tone transforms of it, NOT independently
collected photographs. This validates text disambiguation, not restoration
quality or real user preference.
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
    TARGET_IDS,
    TARGET_SPECS,
    build_prompt_variants,
    make_target,
)
from src.utils import ROOT, load_config, save_json


def decode(item) -> np.ndarray:
    b = item.get("bytes") if isinstance(item, dict) else item
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"),
                    dtype=np.uint8, copy=True)


def save_jpg(a: np.ndarray, p: Path, quality: int = 92) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(p, "JPEG", quality=quality)


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(
        (a.astype(np.float32) - b.astype(np.float32)) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261107)
    a = ap.parse_args()

    cfg = load_config(override=ROOT / "configs" / "ip2p_500m.yaml")
    parquet = ROOT / cfg["data"]["ip2p_parquet"]
    if not parquet.is_file():
        raise RuntimeError(f"missing shard: {parquet}")
    pf = pq.ParquetFile(parquet)

    base = ROOT / "data" / "ip2p_500m" / "multitarget"
    inp, tgt = base / "inputs", base / "targets"
    inp.mkdir(parents=True, exist_ok=True)
    tgt.mkdir(parents=True, exist_ok=True)

    lines, stats = [], []
    cols = ["original_image", "edited_image", "edit_prompt", "original_prompt"]
    gid = 0
    for batch in pf.iter_batches(batch_size=16, columns=cols):
        xs = batch.column("original_image").to_pylist()
        ys = batch.column("edited_image").to_pylist()
        eps = batch.column("edit_prompt").to_pylist()
        ops = batch.column("original_prompt").to_pylist()
        for x, y, ep, op in zip(xs, ys, eps, ops):
            if gid >= a.groups:
                break
            rng = np.random.default_rng(a.seed + gid * 104729)
            try:
                xim, yim = decode(x), decode(y)
            except Exception:                            # noqa: BLE001
                continue
            stem = f"{gid:05d}"
            xrel = f"data/ip2p_500m/multitarget/inputs/{stem}.jpg"
            save_jpg(xim, ROOT / xrel)

            imgs, targets = {}, []
            caption = ((ep or "").strip() or "make the requested edit")
            for tid in TARGET_IDS:
                yk = make_target(yim, tid, rng, strong=True)
                rel = f"data/ip2p_500m/multitarget/targets/{stem}_{tid}.jpg"
                save_jpg(yk, ROOT / rel)
                imgs[tid] = yk
                targets.append({
                    "id": tid,
                    "clean": rel,
                    "description": TARGET_SPECS[tid]["description"],
                    "prompts": build_prompt_variants(tid, caption, rng, n=3),
                })
            pair = [rms(imgs[u], imgs[v])
                    for i, u in enumerate(TARGET_IDS)
                    for v in TARGET_IDS[i + 1:]]
            stats.append(float(np.mean(pair)))
            lines.append({
                "group_id": gid,
                "degraded": xrel,
                "caption": caption,
                "original_prompt": (op or "").strip()[:180],
                "defects": ["official_ip2p_input"],
                "targets": targets,
            })
            gid += 1
            if gid % 25 == 0:
                print(f"[build] {gid}/{a.groups} groups, "
                      f"last inter-target rms={pair and np.mean(pair):.1f}",
                      flush=True)
                save_json({"stage": "build", "done": gid, "total": a.groups},
                          ROOT / "outputs" / "logs" /
                          "ip2p_multitarget_progress.json")
        if gid >= a.groups:
            break

    (base / "train.jsonl").write_text("".join(
        json.dumps(l, ensure_ascii=False) + "\n" for l in lines),
        encoding="utf-8")

    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "design": "A-class: ONE official input image -> K DIFFERENT targets, "
                  "text is the only disambiguating signal",
        "source_shard": str(parquet.relative_to(ROOT)),
        "source_shard_bytes": parquet.stat().st_size,
        "source_rows_available": pf.metadata.num_rows,
        "n_groups": len(lines),
        "K_targets_per_group": len(TARGET_IDS),
        "n_samples": len(lines) * len(TARGET_IDS),
        "n_prompt_variants": sum(len(t["prompts"]) for l in lines
                                 for t in l["targets"]),
        "target_ids": list(TARGET_IDS),
        "inter_target_rms_mean": round(float(np.mean(stats)), 3),
        "inter_target_rms_min": round(float(np.min(stats)), 3),
        "anti_confound": [
            "degraded input is IDENTICAL for all K targets",
            "style phrase comes FIRST (toy hash encoder reads 8 tokens)",
            "3 distinct paraphrases per target",
            "the REAL official edit_prompt is used as a target-independent tail",
        ],
        "honesty": [
            "official images are SD + Prompt-to-Prompt generated, NOT real photos",
            "y0 is the official edited_image; y1..y3 are deterministic tone "
            "transforms of y0, not independently captured photographs",
            "validates text disambiguation, not restoration quality",
        ],
    }
    (base / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build] DONE groups={len(lines)} K={len(TARGET_IDS)} "
          f"samples={info['n_samples']} rms={info['inter_target_rms_mean']} "
          f"-> {base}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
