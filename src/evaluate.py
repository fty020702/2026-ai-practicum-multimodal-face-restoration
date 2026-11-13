from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.inference import sample

# Lazy: constructing lpips.LPIPS(net="alex") tries to download pretrained
# weights on first use. Keep import-time free so the smoke test never blocks.
_LPIPS = None
_LPIPS_TRIED = False


def _get_lpips():
    global _LPIPS, _LPIPS_TRIED
    if _LPIPS_TRIED:
        return _LPIPS
    _LPIPS_TRIED = True
    try:
        import lpips as _lpips_pkg
        m = _lpips_pkg.LPIPS(net="alex")
        m.eval()
        _LPIPS = m
    except Exception:  # pragma: no cover
        _LPIPS = None
    return _LPIPS


# --------------------------------------------------------------------------- #
# Fidelity metrics
# --------------------------------------------------------------------------- #
def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = float(((pred - gt) ** 2).mean().item())
    return float("inf") if mse <= 0 else 10 * math.log10(1.0 / mse)


def ssim(pred: torch.Tensor, gt: torch.Tensor, win: int = 8) -> float:
    """Simplified block-wise SSIM on grayscale (no gaussian window)."""
    a = pred[0].permute(1, 2, 0).mean(2).cpu().numpy()
    b = gt[0].permute(1, 2, 0).mean(2).cpu().numpy()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    h, w = a.shape
    vals = []
    for i in range(0, max(1, h - win + 1), win):
        for j in range(0, max(1, w - win + 1), win):
            x, y = a[i:i + win, j:j + win], b[i:i + win, j:j + win]
            if x.size == 0:
                continue
            mx, my = x.mean(), y.mean()
            vx, vy = x.var(), y.var()
            cov = ((x - mx) * (y - my)).mean()
            vals.append(
                ((2 * mx * my + C1) * (2 * cov + C2))
                / ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))
            )
    return float(np.mean(vals)) if vals else 0.0


def lpips_score(pred: torch.Tensor, gt: torch.Tensor):
    model = _get_lpips()
    if model is None:
        return None
    try:
        with torch.no_grad():
            return float(model(pred * 2 - 1, gt * 2 - 1).mean().item())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# The falsification test.
# If a trained text branch produces (almost) the SAME output for two different
# prompts, then the projector has merely learned a constant bias and the text
# input is doing nothing. That is a FAIL, regardless of how good PSNR looks.
# --------------------------------------------------------------------------- #
def text_response_test(backbone, cond, cache, degraded, prompt_pair, cfg, device):
    gen = torch.Generator(device=device).manual_seed(20261107)
    shape = backbone.image_to_tokens(degraded).shape
    noise = torch.randn(shape, generator=gen, device=device)

    out_a = sample(backbone, cond, cache, degraded, prompt_pair[0], cfg, device,
                   text_enabled=True, noise=noise)
    out_b = sample(backbone, cond, cache, degraded, prompt_pair[1], cfg, device,
                   text_enabled=True, noise=noise)
    out_none = sample(backbone, cond, cache, degraded, "", cfg, device,
                      text_enabled=False, noise=noise)

    d_text = float(((out_a - out_b) ** 2).mean().sqrt().item())
    d_base = float(((out_a - out_none) ** 2).mean().sqrt().item())
    thr = float(cfg["eval"]["response_threshold"])
    return {
        "prompt_a": prompt_pair[0],
        "prompt_b": prompt_pair[1],
        "d_text": round(d_text, 6),
        "d_base": round(d_base, 6),
        "ratio": round(d_text / (d_base + 1e-8), 4),
        "threshold_abs": thr,
        "passed": d_text > thr,
        "note": "same initial noise for all three runs; differences can only come from text",
    }


def evaluate_checkpoint(cfg, device, backbone, cond, cache, dataset,
                        num_samples=8, text_enabled=True, use_lpips=False):
    """Fidelity + text responsiveness for one trained (or untrained) system.

    use_lpips=False by default: constructing the LPIPS model downloads weights
    on first use, which would stall offline runs. Enable it explicitly when the
    weights are already cached.
    """
    rows, responses = [], []
    n = min(num_samples, len(dataset))
    for i in range(n):
        b = dataset[i]
        degraded = b["degraded"].unsqueeze(0).to(device)
        gt = b["clean"].unsqueeze(0).to(device)
        gen = torch.Generator(device=device).manual_seed(1000 + i)
        shape = backbone.image_to_tokens(degraded).shape
        noise = torch.randn(shape, generator=gen, device=device)

        pred = sample(backbone, cond, cache, degraded, b["prompt"], cfg, device,
                      text_enabled=text_enabled, noise=noise)
        rec = {"index": i, "task": b["task"], "psnr": round(psnr(pred, gt), 4),
               "ssim": round(ssim(pred, gt), 4)}
        lp = lpips_score(pred, gt) if use_lpips else None
        if lp is not None:
            rec["lpips"] = round(lp, 4)
        rows.append(rec)

    for pair in cfg["eval"]["text_response_pairs"]:
        b = dataset[0]
        degraded = b["degraded"].unsqueeze(0).to(device)
        responses.append(text_response_test(backbone, cond, cache, degraded, pair, cfg, device))

    return {
        "per_sample": rows,
        "mean_psnr": round(float(np.mean([r["psnr"] for r in rows])), 4),
        "mean_ssim": round(float(np.mean([r["ssim"] for r in rows])), 4),
        "text_response": responses,
        "mean_d_text": round(float(np.mean([r["d_text"] for r in responses])), 6),
        "text_effect_detected": any(r["passed"] for r in responses),
    }


def save_comparison_grid(images: list, labels: list, path):
    """images: list of [3,H,W] tensors in [0,1]."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for im in images:
        arr = (im[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
        tiles.append(Image.fromarray(arr))
    w, h = tiles[0].size
    canvas = Image.new("RGB", (w * len(tiles), h + 24), (255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for k, (tile, lab) in enumerate(zip(tiles, labels)):
        canvas.paste(tile, (k * w, 24))
        draw.text((k * w + 4, 5), lab, fill=(0, 0, 0))
    canvas.save(dst)
    return dst
