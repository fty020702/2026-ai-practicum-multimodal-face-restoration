from __future__ import annotations

"""Materialise the paired dataset to disk.

    D:/Anaconda/envs/ChatAI/python.exe scripts/prepare_data.py --num 64

Writes data/clean/*.png, data/degraded/*.png and data/pairs.jsonl.

To use REAL images instead of the procedural ones: put your high-quality
images in data/clean/ BEFORE running. RestorationDataset then reads them
automatically and applies synthetic degradations on top.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from src.data.dataset import make_dataset
from src.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=None)
    ap.add_argument("--override", default=None)
    args = ap.parse_args()

    cfg = load_config(override=args.override)
    ds = make_dataset(cfg, num_pairs=args.num, root=cfg["_root"])
    clean_dir = Path(cfg["_root"]) / "data" / "clean"
    deg_dir = Path(cfg["_root"]) / "data" / "degraded"
    clean_dir.mkdir(parents=True, exist_ok=True)
    deg_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for i in range(len(ds)):
        b = ds[i]
        c = (b["clean"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        d = (b["degraded"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        cn, dn = f"{i:05d}.png", f"{i:05d}.png"
        Image.fromarray(c).save(clean_dir / cn)
        Image.fromarray(d).save(deg_dir / dn)
        lines.append({"clean": f"data/clean/{cn}", "degraded": f"data/degraded/{dn}",
                      "task": b["task"], "prompt": b["prompt"]})

    out = Path(cfg["_root"]) / "data" / "pairs.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"[data] source={ds.source} pairs={len(lines)}")
    print(f"[data] -> {clean_dir}  {deg_dir}  {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
