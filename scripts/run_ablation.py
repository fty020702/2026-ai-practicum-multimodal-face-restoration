from __future__ import annotations

"""Four-way ablation. Produces outputs/logs/ablation.json and a comparison PNG.

  A  no_text           frozen backbone, reference-image branch only
  B  text_zero_init    frozen backbone + UNTRAINED (zero-init) text branch
  C  projector_only    trained, single injection point ("just a projector")
  D  bypass            trained, per-block zero-init decoupled cross-attention

A vs B isolates the zero-initialisation invariant (should be identical).
C/D vs A shows what training the text branch buys you.
text_response_test is the falsification check: if different prompts give the
same output, the text input is decoration, not control.

Run: D:/Anaconda/envs/ChatAI/python.exe scripts/run_ablation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.dataset import make_dataset
from src.evaluate import evaluate_checkpoint, save_comparison_grid
from src.inference import load_checkpoint, sample
from src.utils import ROOT, load_config, resolve_device, save_json

CASES = ("A_no_text", "B_text_zero_init", "C_projector_only", "D_bypass")


def build(cfg, device, mode, ckpt):
    return load_checkpoint(cfg, device, ckpt=ckpt, mode=mode)


def first_image(cfg, device, backbone, cond, cache, ds, text_enabled, prompt=None):
    b = ds[0]
    degraded = b["degraded"].unsqueeze(0).to(device)
    gen = torch.Generator(device=device).manual_seed(1000)
    shape = backbone.image_to_tokens(degraded).shape
    noise = torch.randn(shape, generator=gen, device=device)
    return sample(
        backbone, cond, cache, degraded, prompt or b["prompt"], cfg, device,
        text_enabled=text_enabled, noise=noise,
    )


def main():
    cfg = load_config()
    device = resolve_device(cfg["train"]["device"])
    # The config's 0.01 threshold is calibrated for a REAL pretrained backbone.
    # On this tiny frozen-random toy trunk, use a toy-scale threshold and say
    # so explicitly in the report.
    cfg["eval"]["response_threshold"] = 1e-5
    ds = make_dataset(cfg, root=cfg["_root"])
    ck = ROOT / "outputs" / "checkpoints"

    plan = {
        "A_no_text": ("projector_only", None, False),
        "B_text_zero_init": ("projector_only", None, True),
        "C_projector_only": ("projector_only", ck / "projector_only" / "cond_step000200.pt", True),
        "D_bypass": ("projector_plus_bypass", ck / "bypass" / "cond_step000200.pt", True),
    }

    report, tiles, labels = {}, [], []
    for name in CASES:
        mode, ckpt, text_on = plan[name]
        if ckpt is not None and not ckpt.exists():
            print(f"[abl] SKIP {name}: missing {ckpt} (run train.py first)")
            continue
        backbone, cond, cache = build(cfg, device, mode, ckpt)
        res = evaluate_checkpoint(
            cfg, device, backbone, cond, cache, ds,
            num_samples=8, text_enabled=text_on, use_lpips=False,
        )
        res["mode"] = mode
        res["checkpoint"] = str(ckpt) if ckpt else None
        res["text_enabled"] = text_on
        report[name] = res
        print(f"[abl] {name:20s} psnr={res['mean_psnr']:.3f} ssim={res['mean_ssim']:.4f} "
              f"d_text={res['mean_d_text']:.6f} text_effect={res['text_effect_detected']}")

        img = first_image(cfg, device, backbone, cond, cache, ds, text_on)
        tiles.append(img)
        labels.append(name)
        cond.detach()

    # reference tiles
    b = ds[0]
    tiles.insert(0, b["degraded"].unsqueeze(0))
    labels.insert(0, "INPUT_degraded")
    tiles.insert(1, b["clean"].unsqueeze(0))
    labels.insert(1, "GT_clean")

    png = ROOT / "outputs" / "samples" / "ablation.png"
    save_comparison_grid(tiles, labels, png)
    out = ROOT / "outputs" / "logs" / "ablation.json"
    save_json({"device": str(device), "note": "ToyMMDiT, 8 samples, same noise per case",
               "cases": report, "grid": str(png)}, out)
    print(f"[abl] grid -> {png}")
    print(f"[abl] report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
