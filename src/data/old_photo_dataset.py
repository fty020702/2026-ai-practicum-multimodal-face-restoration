from __future__ import annotations

"""Real-image, paired old-photo restoration dataset.

Differences from the old synthetic RestorationDataset
-----------------------------------------------------
1. `clean` is a REAL photograph (DIV2K HR), not make_synthetic_clean().
2. `degraded` is produced by the paper-style old-photo pipeline in
   src/data/old_photo.py (structured + unstructured defects), not by six
   hand-picked operators.
3. ANTI-CONFOUND: the old data bound task and prompt by `i % 6`, so the model
   could infer the "text" from the image alone and any d_text ablation was
   meaningless. Here:
     - each image gets a random COMBINATION of defects,
     - the task prompt is a noisy subset of those defects (+ a random
       distractor instruction that was NOT applied),
     - each image is emitted `prompts_per_image` times with DIFFERENT text
       (different instruction subset, different real caption, different
       user constraint),
     - content captions come from a REAL caption pool and are decoupled from
       the degradation type.
   => changing the text changes the output only if the new text branch works.

4. Real old photographs have NO clean GT. They are kept in a SEPARATE
   eval manifest (data/old_photo/eval_real.jsonl) for no-reference IQA
   (NIQE / BRISQUE / CLIP-IQA) and user study, never as (clean, degraded)
   training pairs. This follows Wan et al. CVPR 2020 and LL-Bench.

Batch contract is IDENTICAL to RestorationDataset so train.py / evaluate.py
need no changes beyond the dataset dispatch:
    clean, degraded : float32 [3, H, W] in [0, 1]
    task, prompt    : str
    defects         : str (extra, comma separated)
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .old_photo import DEFECT_PHRASES, USER_CONSTRAINTS

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False


class OldPhotoDataset(Dataset):
    """Paired real-photo dataset from a JSONL manifest + de-confounded prompts."""

    def __init__(self, cfg, num_pairs: int | None = None, split: str = "train", root=None):
        if not _HAS_PIL:
            raise RuntimeError("Pillow is required for OldPhotoDataset")

        d = cfg["data"]
        self.size = int(d["image_size"])
        self.root = Path(root) if root else Path(cfg["_root"])
        self.k = max(1, int(d.get("prompts_per_image", 3)))
        self.split = split
        self.seed = int(cfg["project"]["seed"]) + (0 if split == "train" else 777)

        manifest_rel = d.get("old_photo_manifest", "data/old_photo/train.jsonl")
        if split != "train" and d.get("old_photo_eval_paired"):
            manifest_rel = d["old_photo_eval_paired"]
        mpath = self.root / manifest_rel
        if not mpath.is_file():
            raise FileNotFoundError(
                f"[old_photo] manifest not found: {mpath}\n"
                "  Run first:  D:/Anaconda/envs/ChatAI/python.exe "
                "scripts/build_old_photo_data.py\n"
                "  (or put your own real photos in data/raw/ and use --offline)\n"
                "  This dataset will NOT silently fall back to synthetic toy images."
            )

        self.records = [
            json.loads(line)
            for line in mpath.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if num_pairs:
            self.records = self.records[: int(num_pairs)]
        if not self.records:
            raise RuntimeError(f"[old_photo] empty manifest: {mpath}")

        cap_path = self.root / d.get("old_photo_captions", "data/old_photo/captions.json")
        self.captions: list[str] = []
        if cap_path.is_file():
            raw = json.loads(cap_path.read_text(encoding="utf-8"))
            self.captions = [c.strip() for c in raw if isinstance(c, str) and c.strip()]

        self.source = "old_photo_real"

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.records) * self.k

    def _load(self, rel: str) -> np.ndarray:
        p = self.root / rel
        if not p.is_file():
            raise FileNotFoundError(f"[old_photo] image not found: {p}")
        im = Image.open(p).convert("RGB")
        if self.size and (im.width != self.size or im.height != self.size):
            im = im.resize((self.size, self.size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)

    @staticmethod
    def _to_tensor(a: np.ndarray) -> torch.Tensor:
        # PIL -> np.asarray may return a read-only view; copy before handing it
        # to torch to avoid the "non-writable tensor" undefined-behaviour warning.
        return torch.from_numpy(np.array(a, copy=True)).permute(2, 0, 1).float() / 255.0

    # ------------------------------------------------------------------ #
    def _build_prompt(self, rec: dict, rng, variant: int) -> tuple[str, str]:
        """Task-level instruction (noisy) + content-level user text (decoupled)."""
        defects = rec.get("defects", []) or []
        phrases = [DEFECT_PHRASES[d] for d in defects if d in DEFECT_PHRASES]

        # (a) drop one real instruction -> text is NOT a deterministic function
        #     of the image's degradation
        if len(phrases) > 1 and rng.random() < 0.60:
            phrases.pop(int(rng.integers(0, len(phrases))))
        # (b) inject a distractor instruction that was NOT applied
        unused = [p for k, p in DEFECT_PHRASES.items() if k not in defects]
        if unused and rng.random() < 0.45:
            phrases.append(unused[int(rng.integers(0, len(unused)))])
        rng.shuffle(phrases)
        task = "restore this old photo: " + ", ".join(phrases[:3]) if phrases \
            else "restore this old photo"

        # (c) REAL caption from the caption pool; variant>0 swaps it for another
        #     caption of the SAME image -> controlled text-only change
        cap = rec.get("caption")
        if variant > 0 and self.captions and rng.random() < 0.70:
            cap = self.captions[int(rng.integers(0, len(self.captions)))]
        # (d) the USER-ADDED semantic constraint (the actual research subject)
        constraint = USER_CONSTRAINTS[int(rng.integers(0, len(USER_CONSTRAINTS)))]

        parts = [p for p in (cap, constraint) if p]
        prompt = task + ". " + "; ".join(parts) if parts else task
        return task, prompt

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> dict:
        i, variant = divmod(idx, self.k)
        rec = self.records[i]
        # per-(image, variant) local RNG => reproducible and independent of
        # DataLoader shuffle order / number of workers
        rng = np.random.default_rng(self.seed + i * 7919 + variant * 104729)

        clean = self._load(rec["clean"])
        degraded = self._load(rec["degraded"])
        task, prompt = self._build_prompt(rec, rng, variant)
        return {
            "clean": self._to_tensor(clean),
            "degraded": self._to_tensor(degraded),
            "task": task,
            "prompt": prompt,
            "defects": ",".join(rec.get("defects", []) or []),
        }
