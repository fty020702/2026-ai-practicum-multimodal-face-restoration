from __future__ import annotations

"""Dataset: ONE degraded image -> K distinct clean targets + K distinct texts.

Batch contract is IDENTICAL to RestorationDataset / OldPhotoDataset, so
train.py, inference.py, evaluate.py need no changes beyond `data.source`:

    clean, degraded : float32 [3, H, W] in [0, 1]
    task, prompt    : str
    target_id       : str   (extra; which of the K targets this sample wants)
    group_id        : int   (extra; index of the shared degraded image)

The important difference from OldPhotoDataset:
    OldPhotoDataset : same x, same y, K different texts  -> text is redundant
    this dataset    : same x, K different y, K texts     -> text is NECESSARY

`get_group(i)` exposes all K (clean, prompt, target_id) triples of one image
for the ambiguity diagnostic in scripts/diagnose_ambiguity.py.
`all_prompts()` returns the exact finite set of prompts the dataset can emit,
so scripts/precompute_prompts.py can offline-encode every one of them.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .multi_target_old_photo import SHARED_DEGRADATION_PLAN

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False


class MultiTargetDataset(Dataset):
    """A-class disambiguation dataset built from REAL DIV2K photographs."""

    def __init__(self, cfg, num_pairs: int | None = None, split: str = "train", root=None):
        if not _HAS_PIL:
            raise RuntimeError("Pillow is required for MultiTargetDataset")

        d = cfg["data"]
        self.size = int(d["image_size"])
        self.root = Path(root) if root else Path(cfg["_root"])
        self.split = split
        self.seed = int(cfg["project"]["seed"]) + (0 if split == "train" else 777)

        manifest_rel = d.get(
            "multi_target_manifest", "data/multi_target/train.jsonl"
        )
        mpath = self.root / manifest_rel
        if not mpath.is_file():
            raise FileNotFoundError(
                f"[multi_target] manifest not found: {mpath}\n"
                "  Run first: D:/Anaconda/envs/ChatAI/python.exe "
                "scripts/build_multi_target_data.py\n"
                "  This dataset will NOT silently fall back to synthetic toy "
                "images or to a single-target dataset."
            )

        self.records = [
            json.loads(line)
            for line in mpath.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if num_pairs:
            self.records = self.records[: int(num_pairs)]
        if not self.records:
            raise RuntimeError(f"[multi_target] empty manifest: {mpath}")

        self.k = len(self.records[0]["targets"])
        if any(len(r["targets"]) != self.k for r in self.records):
            raise RuntimeError("[multi_target] every group must have the same K")
        self.target_ids = [t["id"] for t in self.records[0]["targets"]]

        # STRICT CONTROL (2026-11-12). `collapse_targets: true` makes all K
        # targets of a group point at the SAME clean image while keeping all
        # K different texts. Same images, same texts, same degradation, same
        # model, same steps -- the ONLY variable is whether the GT changes
        # with the text. This is the clean A/B that isolates the user's
        # hypothesis instead of comparing two unrelated datasets.
        self.collapse = bool(d.get("collapse_targets", False))
        if self.collapse:
            for i, rec in enumerate(self.records):
                same = rec["targets"][0]["clean"]
                for t in rec["targets"]:
                    t["clean"] = same
        self.source = ("single_target_control" if self.collapse
                       else "multi_target_real")

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.records) * self.k

    def _load(self, rel: str) -> torch.Tensor:
        p = self.root / rel
        if not p.is_file():
            raise FileNotFoundError(f"[multi_target] image not found: {p}")
        im = Image.open(p).convert("RGB")
        if self.size and (im.width != self.size or im.height != self.size):
            im = im.resize((self.size, self.size), Image.BILINEAR)
        a = np.array(im, dtype=np.uint8, copy=True)
        return torch.from_numpy(a).permute(2, 0, 1).float() / 255.0

    def _pick_prompt(self, rec: dict, k: int, rng) -> str:
        prompts = rec["targets"][k].get("prompts") or []
        if not prompts:
            raise RuntimeError(f"[multi_target] group {rec.get('group_id')} "
                               f"target {k} has no prompts")
        return prompts[int(rng.integers(0, len(prompts)))]

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> dict:
        i, k = divmod(idx, self.k)
        rec = self.records[i]
        # local RNG => reproducible regardless of DataLoader shuffle / workers
        rng = np.random.default_rng(self.seed + i * 7919 + k * 104729)
        tgt = rec["targets"][k]
        return {
            "clean": self._load(tgt["clean"]),
            "degraded": self._load(rec["degraded"]),
            "task": tgt["id"],
            "prompt": self._pick_prompt(rec, k, rng),
            "target_id": tgt["id"],
            "group_id": int(rec.get("group_id", i)),
            "defects": ",".join(SHARED_DEGRADATION_PLAN),
        }

    # ------------------------------------------------------------------ #
    def get_group(self, i: int) -> list[dict]:
        """All K targets of ONE shared degraded image (for the diagnostic).

        Uses prompts[0] of each target: a fixed, comparable text per target.
        """
        rec = self.records[i]
        degraded = self._load(rec["degraded"])
        return [
            {
                "clean": self._load(t["clean"]),
                "degraded": degraded,
                "prompt": (t.get("prompts") or [""])[0],
                "target_id": t["id"],
                "group_id": int(rec.get("group_id", i)),
            }
            for t in rec["targets"]
        ]

    def all_prompts(self) -> list[str]:
        """The exact finite prompt set this dataset can ever emit."""
        out = set()
        for rec in self.records:
            for t in rec["targets"]:
                out.update(t.get("prompts") or [])
        return sorted(out)
