from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import apply_degradation
from .prompts import CONTENT_HINTS, build_prompt

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False


def _real_clean_images(root: Path, size: int):
    files = [
        p for p in sorted((root / "data" / "clean").glob("*"))
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    ]
    if not files or not _HAS_PIL:
        return None
    out = []
    for p in files:
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        out.append(np.array(im))
    return out


def make_synthetic_clean(size: int, rng) -> np.ndarray:
    """Deterministic structured image: gradient + texture + geometric shapes.

    Much closer to a real image than i.i.d. noise, so PSNR/SSIM comparisons
    are meaningful and the restoration problem is actually learnable.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.stack(
        [40 + 120 * xx, 60 + 90 * yy, 90 + 80 * (1 - xx) * yy], axis=-1
    ).astype(np.float32)

    f = int(rng.integers(6, 18))
    tex = (np.sin(xx * f * 2 * np.pi) * np.cos(yy * f * 2 * np.pi) * 18)[..., None]
    img = img + tex

    for _ in range(3):
        x0, y0 = rng.integers(0, int(size * 0.7), 2)
        w, h = rng.integers(max(4, size // 8), max(5, size // 3), 2)
        c = rng.uniform(60, 230, 3).astype(np.float32)
        img[y0:y0 + h, x0:x0 + w] = c

    cx, cy = rng.uniform(0.2, 0.8, 2) * size
    r = rng.uniform(0.08, 0.22) * size
    mask = (xx * size - cx) ** 2 + (yy * size - cy) ** 2 < r ** 2
    img[mask] = img[mask] * 0.35 + np.array([200, 90, 70], np.float32) * 0.65

    img += rng.normal(0, 3, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


class RestorationDataset(Dataset):
    """Paired (clean, degraded) data with task-level + content-level prompts.

    If data/clean/*.png|jpg exists, real images are used; otherwise the
    dataset is generated procedurally and is fully reproducible from the seed.
    """

    def __init__(self, cfg, num_pairs: int | None = None, split: str = "train", root=None):
        d = cfg["data"]
        self.size = int(d["image_size"])
        self.tasks = list(d["degradations"])
        root = Path(root) if root else Path(cfg["_root"])
        n = int(num_pairs or d["num_pairs"])
        seed = int(cfg["project"]["seed"]) + (0 if split == "train" else 777)
        rng = np.random.default_rng(seed)

        real = _real_clean_images(root, self.size)
        self.source = "real" if real else "synthetic"
        self.samples = []
        for i in range(n):
            clean = real[i % len(real)].copy() if real else make_synthetic_clean(self.size, rng)
            task = self.tasks[i % len(self.tasks)]
            degraded = apply_degradation(clean, task, rng)
            content = CONTENT_HINTS[i % len(CONTENT_HINTS)]
            self.samples.append((clean, degraded, task, build_prompt(task, content)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clean, degraded, task, prompt = self.samples[idx]
        return {
            "clean": torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0,
            "degraded": torch.from_numpy(degraded).permute(2, 0, 1).float() / 255.0,
            "task": task,
            "prompt": prompt,
        }


def make_dataset(cfg, num_pairs: int | None = None, split: str = "train", root=None):
    """Dataset factory dispatched by `data.source`.

    source == "old_photo"     -> OldPhotoDataset (REAL photographs + paper-style
        old-photo degradations, de-confounded prompts). One target per image.
    source == "multi_target"  -> MultiTargetDataset (ONE shared degraded image
        -> K DIFFERENT clean targets, each selected by its own text). This is
        the only source where the text is mathematically NECESSARY.
    source == "ip2p"          -> IP2PShardDataset (OFFICIAL InstructPix2Pix
        triplet: original_image -> degraded, edited_image -> clean,
        edit_prompt -> prompt). One ~500 MB parquet shard, no synthetic
        degradation. See docs/一图多目标数据集调研.md for the honest note
        that this shard is NOT A-class (one target per input).
    anything else / absent    -> the original RestorationDataset, VERBATIM.
        configs/default.yaml has no `data.source`, so every existing toy
        [RESULT] (smoke test / ablation) is bit-for-bit unchanged.
    """
    src = cfg.get("data", {}).get("source")
    if src == "old_photo":
        from .old_photo_dataset import OldPhotoDataset
        return OldPhotoDataset(cfg, num_pairs=num_pairs, split=split, root=root)
    if src == "multi_target":
        from .multi_target_dataset import MultiTargetDataset
        return MultiTargetDataset(cfg, num_pairs=num_pairs, split=split, root=root)
    if src == "ip2p":
        from .ip2p_dataset import IP2PShardDataset
        return IP2PShardDataset(cfg, num_pairs=num_pairs, split=split, root=root)
    return RestorationDataset(cfg, num_pairs=num_pairs, split=split, root=root)
