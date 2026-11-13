from __future__ import annotations

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False

try:
    from PIL import Image
    import io as _io
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False


DEGRADATIONS = (
    "gaussian_noise",
    "gaussian_blur",
    "jpeg",
    "downsample_up",
    "rain",
    "haze",
)


def _blur(img: np.ndarray, k: int) -> np.ndarray:
    if _HAS_CV2:
        return cv2.GaussianBlur(img, (k, k), 0)
    # separable box blur fallback
    out = img.astype(np.float32)
    for _ in range(3):
        out = (np.roll(out, 1, 1) + out + np.roll(out, -1, 1)) / 3.0
        out = (np.roll(out, 1, 0) + out + np.roll(out, -1, 0)) / 3.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _jpeg(img: np.ndarray, q: int) -> np.ndarray:
    if _HAS_CV2:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if _HAS_PIL:
        bio = _io.BytesIO()
        Image.fromarray(img).save(bio, format="JPEG", quality=q)
        bio.seek(0)
        return np.array(Image.open(bio).convert("RGB"))
    return img


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if _HAS_CV2:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR)
    if _HAS_PIL:
        return np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))
    return img


def gaussian_noise(img, rng, sigma=None):
    sigma = float(sigma if sigma is not None else rng.uniform(10, 45))
    n = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def gaussian_blur(img, rng, k=None):
    k = int(k if k is not None else rng.choice([3, 5, 7]))
    return _blur(img, k | 1)


def jpeg(img, rng, q=None):
    return _jpeg(img, int(q if q is not None else rng.integers(10, 35)))


def downsample_up(img, rng, scale=None):
    scale = float(scale if scale is not None else rng.choice([2, 3, 4]))
    h, w = img.shape[:2]
    small = _resize(img, max(1, int(w / scale)), max(1, int(h / scale)))
    return _resize(small, w, h)


def rain(img, rng, strength=None):
    strength = float(strength if strength is not None else rng.uniform(0.15, 0.4))
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    n_lines = int(rng.integers(60, 180))
    for _ in range(n_lines):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        length = int(rng.integers(8, 25))
        x2 = min(w - 1, x + length // 3)
        y2 = min(h - 1, y + length)
        if _HAS_CV2:
            cv2.line(out, (x, y), (x2, y2), (230, 230, 230), 1)
        else:
            steps = max(1, y2 - y)
            for s in range(steps):
                yy = y + s
                xx = x + (x2 - x) * s // steps
                if 0 <= yy < h and 0 <= xx < w:
                    out[yy, xx] = out[yy, xx] * (1 - strength) + 230 * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def haze(img, rng, beta=None):
    beta = float(beta if beta is not None else rng.uniform(0.25, 0.6))
    h, w = img.shape[:2]
    yy = np.linspace(0, 1, h)[:, None]          # [h,1]
    t = np.exp(-beta * (1.0 - yy))              # [h,1]
    t = np.repeat(t, w, axis=1)[..., None]      # [h,w,1] -> broadcasts over 3 channels
    A = np.array([210, 220, 225], dtype=np.float32).reshape(1, 1, 3)
    out = img.astype(np.float32) * t + A * (1 - t)
    out += rng.normal(0, 2, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


_FUNCS = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "jpeg": jpeg,
    "downsample_up": downsample_up,
    "rain": rain,
    "haze": haze,
}


def apply_degradation(img: np.ndarray, kind: str, rng) -> np.ndarray:
    if kind not in _FUNCS:
        raise ValueError(f"unknown degradation {kind}; choose from {DEGRADATIONS}")
    return _FUNCS[kind](img, rng)
