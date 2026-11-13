from __future__ import annotations

"""Paper-style old-photo degradation pipeline.

WHY SYNTHETIC DEGRADATION ON REAL PHOTOS (not procedural toy images)
--------------------------------------------------------------------
Real historical photographs almost never have a pixel-aligned clean ground
truth. Every paper in this area therefore builds *paired supervision* by
taking a real, modern, high-quality photograph and simulating the physical
degradation of aged prints, then uses real (unpaired) old photos only for
domain adaptation / no-reference evaluation / user studies.

Benchmarked against
-------------------
* Wan et al., "Bringing Old Photos Back to Life", CVPR 2020 (oral)
    - synthetic pairs from Pascal VOC, 256x256 random crops
    - STRUCTURED defects: scratches, dust spots, holes/tears, paper texture
      (collected scratch/paper textures, elastic deformation, layer addition /
      lighten-only / screen blending, feathered holes exposing paper)
    - UNSTRUCTURED defects: film grain noise, blur, fading
    - 5,718 collected real old photos; 783 annotated scratches
    - test: DIV2K synthetic degradations + real old-photo test set
    - metrics: PSNR / SSIM / LPIPS / FID
* InstructIR, ECCV 2024
    - text INSTRUCTIONS describe the degradation; >10k GPT-4 prompts over
      7 tasks, randomly sampled at training time (not one-to-one with labels)
* Time-Travel Rephotography (ACM TOG 2020) / LL-Bench
    - real historical photos have NO GT -> NIQE / BRISQUE / CLIP-IQA,
      no-reference IQA + user study

This module implements the degradation half of that recipe. The clean GT is a
REAL photograph (DIV2K HR), not make_synthetic_clean(). The pairing, prompts
and evaluation split live in old_photo_dataset.py / build_old_photo_data.py.
"""

import numpy as np

from .degradation import _blur, _jpeg, _resize

# ---------------------------------------------------------------------------
# Defect taxonomy (mirrors the paper's structured / unstructured split)
# ---------------------------------------------------------------------------
STRUCTURED = ("scratches", "dust_spots", "holes", "paper_texture")
UNSTRUCTURED = (
    "film_grain", "blur", "fading", "sepia", "jpeg", "vignette", "resolution_loss"
)
ALL_DEFECTS = STRUCTURED + UNSTRUCTURED

# Each defect -> a natural-language restoration instruction. Used to build
# TASK-level prompts. Deliberately phrased like InstructIR instructions.
DEFECT_PHRASES = {
    "scratches": "remove scratches and hairline cracks",
    "dust_spots": "remove dust spots and white specks",
    "holes": "fill in the torn and missing areas",
    "paper_texture": "suppress the paper and fiber texture",
    "film_grain": "remove film grain noise",
    "blur": "sharpen the soft out-of-focus details",
    "fading": "restore the faded contrast and tonal range",
    "sepia": "correct the yellowed color cast",
    "jpeg": "remove compression blocking artifacts",
    "vignette": "even out the darkened corners",
    "resolution_loss": "recover fine detail lost to low resolution",
}

# The USER-ADDED text branch (the thing this whole project is about): semantic
# constraints that are NOT derivable from the degradation type.
USER_CONSTRAINTS = (
    "preserve facial identity and do not change the face",
    "keep the original sepia tone",
    "do not over-smooth fabric and fine textures",
    "keep the background architecture intact",
    "preserve any handwritten text on the sign",
    "avoid hallucinating details that were not in the original",
    "keep skin tones natural",
    "preserve the grain of the original print",
    "keep the foliage natural",
    "preserve the expression of the portrait",
)


# ---------------------------------------------------------------------------
# Low-level helpers (pure numpy; no cv2 dependency for the structured defects)
# ---------------------------------------------------------------------------
def _clip_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def _lowfreq_noise(h: int, w: int, rng, cell: int = 12) -> np.ndarray:
    """Upscaled smooth gaussian field, normalized to zero mean / unit std."""
    sh, sw = max(2, h // cell), max(2, w // cell)
    small = rng.normal(0, 1, (sh, sw)).astype(np.float32)
    ky, kx = (h + sh - 1) // sh, (w + sw - 1) // sw
    big = np.kron(small, np.ones((ky, kx), np.float32))[:h, :w]
    big = (
        np.roll(big, 1, 0) + np.roll(big, -1, 0)
        + np.roll(big, 1, 1) + np.roll(big, -1, 1) + 4.0 * big
    ) / 8.0
    return (big - big.mean()) / (big.std() + 1e-6)


def _draw_polyline(mask: np.ndarray, pts, width: int = 0) -> None:
    h, w = mask.shape
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for x, y in zip(np.linspace(x0, x1, n), np.linspace(y0, y1, n)):
            xi, yi = int(round(x)), int(round(y))
            for dy in range(-width, width + 1):
                for dx in range(-width, width + 1):
                    xx, yy = xi + dx, yi + dy
                    if 0 <= xx < w and 0 <= yy < h:
                        mask[yy, xx] = 1.0


# ---------------------------------------------------------------------------
# STRUCTURED defects (Wan et al.: scratches, dust, holes, paper texture)
# ---------------------------------------------------------------------------
def scratches(img: np.ndarray, rng, n: int | None = None,
              strength: float | None = None) -> np.ndarray:
    """Thin bright/dark hairline cracks, screen/lighten-only blended.

    Paper: collected scratch textures + elastic deformation + layer addition,
    lighten-only and screen blending. We synthesize the strokes directly.
    """
    out = img.astype(np.float32).copy()
    h, w = out.shape[:2]
    n = int(n if n is not None else rng.integers(3, 14))
    strength = float(strength if strength is not None else rng.uniform(0.35, 0.8))

    for _ in range(n):
        x0, y0 = float(rng.uniform(0, w)), float(rng.uniform(0, h))
        # mostly near-vertical (old prints are often cracked along the grain)
        ang = float(rng.uniform(-0.55, 0.55)) + (np.pi / 2 if rng.random() < 0.65 else 0.0)
        length = float(rng.uniform(0.15, 0.85) * max(h, w))
        pts = []
        for k in range(14):
            s = k / 13.0
            jitter = rng.normal(0, max(0.8, length * 0.006))
            pts.append((
                x0 + np.cos(ang) * length * s + jitter,
                y0 + np.sin(ang) * length * s + jitter,
            ))
        mask = np.zeros((h, w), np.float32)
        _draw_polyline(mask, pts, width=0)
        # feather the stroke a little
        soft = (mask + np.roll(mask, 1, 0) + np.roll(mask, -1, 0)
                + np.roll(mask, 1, 1) + np.roll(mask, -1, 1)) / 5.0
        m = soft > 0
        if not m.any():
            continue
        # scratches on a faded print are usually BRIGHT (exposed paper base),
        # occasionally dark (fold shadow)
        val = float(rng.uniform(175, 250) if rng.random() < 0.8 else rng.uniform(15, 70))
        a = strength * np.clip(soft[m], 0, 1)[..., None]
        out[m] = out[m] * (1.0 - a) + val * a
    return _clip_u8(out)


def dust_spots(img: np.ndarray, rng, n: int | None = None) -> np.ndarray:
    """Small bright/dark specks scattered over the print."""
    out = img.astype(np.float32).copy()
    h, w = out.shape[:2]
    n = int(n if n is not None else rng.integers(10, 55))
    for _ in range(n):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = float(rng.uniform(0.5, 2.8))
        x1, x2 = max(0, int(cx - r - 1)), min(w, int(cx + r + 2))
        y1, y2 = max(0, int(cy - r - 1)), min(h, int(cy + r + 2))
        if x2 <= x1 or y2 <= y1:
            continue
        yy, xx = np.mgrid[y1:y2, x1:x2]
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        val = float(rng.uniform(185, 252) if rng.random() < 0.72 else rng.uniform(5, 55))
        sub = out[y1:y2, x1:x2]
        sub[m] = sub[m] * 0.12 + val * 0.88
    return _clip_u8(out)


def holes(img: np.ndarray, rng, n: int | None = None) -> np.ndarray:
    """Irregular feathered holes exposing the paper base underneath.

    Paper: "feathered holes to expose the paper texture". The exposed area is
    filled with a warm paper colour + low-frequency fiber noise.
    """
    out = img.astype(np.float32).copy()
    h, w = out.shape[:2]
    n = int(n if n is not None else rng.integers(1, 4))
    paper = np.array([
        rng.uniform(198, 228), rng.uniform(186, 216), rng.uniform(158, 196)
    ], dtype=np.float32)
    fiber = _lowfreq_noise(h, w, rng, cell=8)[..., None] * float(rng.uniform(3, 9))
    yy, xx = np.mgrid[0:h, 0:w]

    for _ in range(n):
        cx, cy = float(rng.uniform(0, w)), float(rng.uniform(0, h))
        rx = float(rng.uniform(0.025, 0.11) * w)
        ry = float(rng.uniform(0.025, 0.11) * h)
        ang = float(rng.uniform(0, 2 * np.pi))
        d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        d = d + 0.30 * np.sin(xx * 0.09 + ang) * np.cos(yy * 0.07 - ang)
        a = np.clip((1.12 - d) * 2.6, 0.0, 1.0)[..., None]
        if a.max() <= 0:
            continue
        exposed = paper[None, None, :] + fiber
        out = out * (1.0 - a) + exposed * a
    return _clip_u8(out)


def paper_texture(img: np.ndarray, rng, strength: float | None = None) -> np.ndarray:
    """Low-frequency paper fiber + slight warm cast over the whole print."""
    out = img.astype(np.float32)
    h, w = out.shape[:2]
    strength = float(strength if strength is not None else rng.uniform(0.05, 0.16))
    tex = _lowfreq_noise(h, w, rng, cell=10)
    out = out + tex[..., None] * float(rng.uniform(6, 16))
    warm = np.array([8.0, 3.0, -7.0], np.float32)[None, None, :]
    out = out * (1.0 - strength) + (out + warm) * strength
    return _clip_u8(out)


# ---------------------------------------------------------------------------
# UNSTRUCTURED defects (Wan et al.: film grain, blur, fading; + common aging)
# ---------------------------------------------------------------------------
def film_grain(img: np.ndarray, rng, sigma: float | None = None) -> np.ndarray:
    """Signal-dependent grain: strongest in midtones, like real film."""
    sigma = float(sigma if sigma is not None else rng.uniform(6, 22))
    out = img.astype(np.float32)
    lum = out.mean(axis=2, keepdims=True) / 255.0
    weight = 0.45 + 0.9 * np.sin(np.pi * np.clip(lum, 0, 1))
    n = rng.normal(0, sigma, out.shape).astype(np.float32)
    # coarse clumping so it is not pure pixel i.i.d. noise
    clump = _lowfreq_noise(out.shape[0], out.shape[1], rng, cell=3)[..., None]
    return _clip_u8(out + n * weight + clump * sigma * 0.35)


def blur(img: np.ndarray, rng, k: int | None = None) -> np.ndarray:
    k = int(k if k is not None else rng.choice([3, 5, 7])) | 1
    return _blur(img, k)


def fading(img: np.ndarray, rng, amount: float | None = None) -> np.ndarray:
    """Contrast loss + lifted blacks + washed highlights (faded print)."""
    a = float(amount if amount is not None else rng.uniform(0.3, 0.75))
    out = img.astype(np.float32)
    out = (out - 128.0) * (1.0 - 0.55 * a) + 128.0
    out = out + a * float(rng.uniform(18, 52))          # lifted blacks
    out = out * (1.0 - 0.08 * a) + 255.0 * (0.08 * a)  # washed highlights
    return _clip_u8(out)


def sepia(img: np.ndarray, rng, strength: float | None = None) -> np.ndarray:
    """Yellow/brown color cast typical of aged silver-gelatin prints."""
    s = float(strength if strength is not None else rng.uniform(0.25, 0.7))
    out = img.astype(np.float32)
    m = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ], dtype=np.float32)
    target = out @ m.T
    cast = np.array([10.0, 4.0, -10.0], np.float32)[None, None, :]
    return _clip_u8(out * (1.0 - s) + (target + cast) * s)


def jpeg(img: np.ndarray, rng, q: int | None = None) -> np.ndarray:
    return _jpeg(img, int(q if q is not None else rng.integers(12, 38)))


def vignette(img: np.ndarray, rng, strength: float | None = None) -> np.ndarray:
    """Darkened corners from old lens / uneven paper exposure."""
    s = float(strength if strength is not None else rng.uniform(0.15, 0.45))
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    a = np.clip((r - 0.55) / 0.75, 0, 1) * s
    return _clip_u8(img.astype(np.float32) * (1.0 - a[..., None]))


def resolution_loss(img: np.ndarray, rng, scale: float | None = None) -> np.ndarray:
    scale = float(scale if scale is not None else rng.choice([2, 3, 4]))
    h, w = img.shape[:2]
    small = _resize(img, max(1, int(w / scale)), max(1, int(h / scale)))
    return _resize(small, w, h)


_FUNCS = {
    "scratches": scratches,
    "dust_spots": dust_spots,
    "holes": holes,
    "paper_texture": paper_texture,
    "film_grain": film_grain,
    "blur": blur,
    "fading": fading,
    "sepia": sepia,
    "jpeg": jpeg,
    "vignette": vignette,
    "resolution_loss": resolution_loss,
}


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
def sample_defect_plan(rng, n_min: int = 2, n_max: int = 5) -> list[str]:
    """Random COMBINATION of defects, NOT one degradation == one task label.

    This is the anti-confound fix: the old toy dataset bound `task` and
    `prompt` by `i % 6`, which makes the text-response ablation meaningless
    (the model can infer the task from the image alone). Here every image gets
    a random mix, always >= 1 unstructured defect, with structured defects
    layered on top at random.
    """
    n = int(rng.integers(n_min, n_max + 1))
    un, st = list(UNSTRUCTURED), list(STRUCTURED)
    rng.shuffle(un)
    rng.shuffle(st)
    plan = [un[0]]
    pool = un[1:] + st
    rng.shuffle(pool)
    plan += pool[: max(0, n - 1)]
    rng.shuffle(plan)
    return plan


def degrade_old_photo(
    img: np.ndarray,
    rng,
    plan: list[str] | None = None,
    size: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Apply a random old-photo degradation mix.

    Args:
        img:  uint8 HWC RGB real clean photograph (the GT).
        rng:  numpy Generator (deterministic per sample).
        plan: optional explicit defect list; otherwise sampled.
        size: optional square resize before degradation (paper uses 256 crops).

    Returns:
        (degraded_uint8, applied_defect_names)
    """
    if size is not None and (img.shape[0] != size or img.shape[1] != size):
        img = _resize(img, size, size)
    plan = list(plan) if plan is not None else sample_defect_plan(rng)
    out = np.asarray(img, dtype=np.uint8)
    applied: list[str] = []
    for name in plan:
        if name not in _FUNCS:
            raise ValueError(f"unknown old-photo defect {name}; choose from {ALL_DEFECTS}")
        out = _FUNCS[name](out, rng)
        applied.append(name)
    return _clip_u8(out), applied
