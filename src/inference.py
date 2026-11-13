from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from src.condition import TextConditionModule, freeze_backbone
from src.backends.toy import ToyMMDiT
from src.data.dataset import make_dataset
from src.data.text_features import HashTextEncoder, OfflineTextCache
from src.utils import ROOT, load_config, resolve_device


@torch.no_grad()
def sample(
    backbone,
    cond,
    cache,
    degraded: torch.Tensor,
    prompt: str,
    cfg,
    device,
    text_enabled: bool = True,
    steps: int | None = None,
    noise: torch.Tensor | None = None,
):
    """Euler integration of the flow ODE from t=1 (noise) to t=0 (image).

    Passing an explicit `noise` is essential for controlled comparisons:
    otherwise differences between two prompts could come from different
    initial noise instead of from the text.
    """
    steps = int(steps or cfg["train"]["num_timesteps"])
    B = degraded.shape[0]
    ref = backbone.ref_tokens_from_image(degraded)

    cond.set_enabled(text_enabled)
    if text_enabled:
        cond.set_text(cache.get([prompt] * B).to(device))
    else:
        cond.clear_text()
    ref = cond.fuse_condition(ref)

    shape = backbone.image_to_tokens(degraded).shape
    x = noise if noise is not None else torch.randn(shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), 1.0 - i * dt, device=device)
        v = backbone(x, t, ref)
        x = x - dt * v
    return backbone.tokens_to_image(x).clamp(0, 1)


def load_checkpoint(cfg, device, ckpt=None, mode=None):
    if cfg.get("backend", {}).get("name") == "flux2_klein":
        from src.system import build_flux2_system
        return build_flux2_system(cfg, device, ckpt=ckpt, mode=mode)

    if mode:
        cfg["condition"]["mode"] = mode
    # CRITICAL for controlled ablations: every case must start from the SAME
    # frozen backbone. Without this seed each case gets a different random
    # trunk and the comparison is meaningless.
    torch.manual_seed(cfg["project"]["seed"])
    backbone = ToyMMDiT(cfg).to(device)
    freeze_backbone(backbone)
    dim = cfg["backbone"]["dim"]
    cond = TextConditionModule(cfg, [dim] * cfg["backbone"]["num_blocks"]).to(device)
    cond.attach(backbone.hook_targets)
    encoder = HashTextEncoder(
        cfg["condition"]["text_in_dim"], cfg["backbone"]["num_text_tokens"]
    )
    cache = OfflineTextCache(ROOT, encoder)

    p = Path(ckpt) if ckpt else None
    if p and p.exists():
        state = torch.load(p, map_location=device, weights_only=False)
        if state.get("mode") != cfg["condition"]["mode"]:
            raise ValueError(
                f"checkpoint mode {state.get('mode')} != config mode {cfg['condition']['mode']}"
            )
        cond.load_state_dict(state["cond_state"])
        print(f"[infer] loaded {p}")
    else:
        print("[infer] no checkpoint -> zero-initialised (identity) text branch")
    return backbone, cond, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--override", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--prompt", default="restore and denoise this image, keep the brick texture")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(override=args.override)
    device = resolve_device(args.device or cfg["train"]["device"])
    backbone, cond, cache = load_checkpoint(cfg, device, args.ckpt)
    ds = make_dataset(cfg, root=cfg["_root"])
    degraded = ds[args.index]["degraded"].unsqueeze(0).to(device)

    gen = torch.Generator(device=device).manual_seed(20261107)
    shape = backbone.image_to_tokens(degraded).shape
    noise = torch.randn(shape, generator=gen, device=device)

    out = sample(
        backbone, cond, cache, degraded, args.prompt, cfg, device,
        text_enabled=not args.no_text, noise=noise,
    )[0]
    arr = (out.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    dst = Path(args.out) if args.out else ROOT / "outputs" / "samples" / "inference.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(dst)
    print(f"[infer] saved {dst}")


if __name__ == "__main__":
    main()
