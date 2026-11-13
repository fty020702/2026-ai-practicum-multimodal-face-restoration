from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(base=None, override=None):
    """Load configs/default.yaml, optionally deep-merged with an override file."""
    p = Path(base) if base else ROOT / "configs" / "default.yaml"
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if override:
        with open(override, "r", encoding="utf-8") as f:
            deep_update(cfg, yaml.safe_load(f))
    cfg["_config_path"] = str(p)
    cfg["_override_path"] = str(override) if override else None
    cfg["_root"] = str(ROOT)
    return cfg


def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def ensure_dir(p) -> Path:
    q = Path(p)
    q.mkdir(parents=True, exist_ok=True)
    return q


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def count_params(module, trainable_only: bool = True) -> int:
    return sum(
        p.numel()
        for p in module.parameters()
        if (p.requires_grad or not trainable_only)
    )
