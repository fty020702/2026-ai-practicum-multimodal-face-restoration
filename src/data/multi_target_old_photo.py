from __future__ import annotations

"""One degraded input -> MULTIPLE different clean targets, disambiguated by text.

Why this module exists
----------------------
`OldPhotoDataset` emits `prompts_per_image` DIFFERENT texts for the same
(degraded, clean) pair. That is NOT enough: if the target image is identical,
the Bayes-optimal predictor under MSE is the same for every text, so ignoring
the text is still optimal. That is exactly what
scripts/diagnose_text_usage.py measured on the trained 100-step FLUX.2
checkpoint (wrong text changed the loss by 1 bf16 ULP / 1.08%).

Here the SAME degraded image x is paired with K DIFFERENT clean targets
y_0 ... y_{K-1}, and the k-th text is the ONLY information that says which
target is wanted:

    no-text optimum     ȳ = mean_k y_k
    no-text MSE floor   Var_k(y_k)  > 0
    with-text MSE floor 0
    ambiguity gap       Var_k(y_k)

So a plain velocity-MSE loss already penalises a text-ignoring model. No
contrastive / alignment loss is required. See docs/一图多目标数据集调研.md.

The ambiguity is chosen to be REAL and OLD-PHOTO-SPECIFIC: given a faded,
scratched, grainy, sepia-cast print you cannot know whether the user wants
full natural colour, a black-and-white restoration, a vintage sepia look, or
a conservative scratch-only cleanup. That preference exists only in the text.
(This mirrors the documented ill-posedness of colourisation: CIN25,
Text2Colors/PAT, and the multi-version outputs of commercial repair tools.)

Design constraints
------------------
* Targets are DETERMINISTIC transforms of a REAL photograph (DIV2K HR), not
  synthetic images and not diffusion-generated phantoms. Pixel differences
  between targets are large and measurable.
* The degradation applied to x is SHARED across all K targets and always
  contains a colour cast, grain and structured damage, so no target style can
  be inferred from x alone.
* Prompts put the target style in the FIRST words. The toy path uses a hash
  encoder with an 8-token window; if every prompt began with the same
  preamble, all text features would collide and the experiment would be void.
* captions / constraints are sampled INDEPENDENTLY of the target id.

Facts vs claims
---------------
[FACT] degradation operators come from src/data/old_photo.py (Wan et al.
       CVPR 2020 structured/unstructured taxonomy).
[INFERENCE] the four tone/effort variants are plausible restoration outputs,
       NOT real alternate photographs. This dataset validates text
       disambiguation, not restoration quality or real user preference.
"""

import numpy as np

from .old_photo import (
    dust_spots,
    fading,
    film_grain,
    scratches,
    sepia,
    vignette,
)

# ---------------------------------------------------------------------------
# The SHARED input degradation. Deliberately mixes a colour cast, an
# unstructured grain/contrast defect and structured physical damage, so the
# desired output style is genuinely unobservable from x.
# ---------------------------------------------------------------------------
SHARED_DEGRADATION_PLAN = (
    "fading",
    "sepia",
    "film_grain",
    "scratches",
    "dust_spots",
    "vignette",
)

# ---------------------------------------------------------------------------
# K target styles. Every style has several paraphrases so the model cannot
# win by memorising one magic keyword. Each paraphrase MUST start with a
# distinct style-bearing phrase (toy hash encoder reads only 8 tokens).
# ---------------------------------------------------------------------------
TARGET_IDS = ("natural_color", "monochrome", "sepia_tone", "minimal_repair")

TARGET_SPECS = {
    "natural_color": {
        "description": "full restoration to the original natural colour",
        "prompts": (
            "restore in full natural color: remove scratches dust and grain, "
            "recover the original colors",
            "reconstruct the original natural colors: remove every aging defect",
            "full color restoration: remove scratches dust and grain, correct "
            "the yellowed color cast",
            "bring back the true colors: clean all damage and neutralise the cast",
        ),
    },
    "monochrome": {
        "description": "full restoration but keep a black-and-white look",
        "prompts": (
            "restore in black and white: remove all aging defects but keep the "
            "monochrome look",
            "keep it black and white: remove scratches dust and grain, no color",
            "monochrome restoration: clean every defect while preserving the "
            "grayscale appearance",
            "black and white only: repair the print and keep the grayscale tone",
        ),
    },
    "sepia_tone": {
        "description": "full restoration but preserve the vintage warm brown cast",
        "prompts": (
            "keep the original sepia tone: remove scratches dust and grain, "
            "preserve the warm brown cast",
            "restore with sepia tone: clean all defects but keep the vintage "
            "brown color",
            "vintage sepia restoration: remove aging damage, retain the warm tone",
            "sepia finish: repair the damage and keep the old brown tint",
        ),
    },
    "minimal_repair": {
        "description": "conservative cleanup: only physical damage removed",
        "prompts": (
            "minimal repair only: remove scratches and dust, keep the aged film "
            "grain and color cast",
            "conservative cleanup: only fix cracks and spots, preserve grain "
            "and fading",
            "remove only the physical damage: leave the original grain and "
            "faded tone intact",
            "light touch restoration: erase cracks but retain the aged texture",
        ),
    },
}

# Content-level additions, deliberately INDEPENDENT of target id. In the toy
# path these are mostly outside the 8-token window; on the real Qwen3 path they
# add genuine linguistic variety without leaking the target identity.
CONTENT_HINTS = (
    "preserve facial identity",
    "keep the background architecture intact",
    "avoid hallucinating details",
    "do not over-smooth fabric and fine textures",
    "keep skin tones natural",
    "preserve any handwritten text",
    "keep the foliage natural",
    "preserve the expression of the portrait",
)


# ---------------------------------------------------------------------------
# Target image construction. All transforms are deterministic given `rng`,
# and all operate on the SAME real clean crop.
# ---------------------------------------------------------------------------
def _grayscale(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    lum = x @ np.array([0.299, 0.587, 0.114], np.float32)
    return np.clip(np.repeat(lum[..., None], 3, axis=2), 0, 255).astype(np.uint8)


def make_target(clean: np.ndarray, target_id: str, rng,
                strong: bool = False) -> np.ndarray:
    """Build ONE clean target from the real GT, by user-intent style.

    `strong=True` amplifies the inter-target pixel differences. Round 1
    (2026-11-12) used mild tone shifts: Var_k(y_k) ~ 0.0014 in token space,
    which is ~700x smaller than the x1 noise variance and gives the optimiser
    a very weak style signal. Round 2 uses stronger but still perfectly
    plausible restoration styles to test whether the bottleneck is data
    signal strength or the frozen random toy trunk itself.
    """
    y = np.asarray(clean, dtype=np.uint8)
    if target_id == "natural_color":
        return y.copy()
    if target_id == "monochrome":
        g = _grayscale(y).astype(np.float32)
        if strong:  # high-contrast silver-gelatin print look
            g = np.clip((g - 128.0) * 1.32 + 128.0, 0, 255)
        return g.astype(np.uint8)
    if target_id == "sepia_tone":
        # fixed strength -> the target itself is not a random "degradation"
        return sepia(y, rng, strength=0.95 if strong else 0.55)
    if target_id == "minimal_repair":
        # physically clean, but the user explicitly wants the aged look kept
        out = sepia(y, rng, strength=0.85 if strong else 0.40)
        out = fading(out, rng, amount=0.80 if strong else 0.35)
        return film_grain(out, rng, sigma=20.0 if strong else 7.0)
    raise ValueError(
        f"unknown target id {target_id!r}; choose from {TARGET_IDS}"
    )


def make_shared_degraded(clean: np.ndarray, rng) -> np.ndarray:
    """The ONE degraded input x shared by all K targets of this image."""
    out = np.asarray(clean, dtype=np.uint8)
    out = fading(out, rng, amount=0.55)
    out = sepia(out, rng, strength=0.60)
    out = film_grain(out, rng, sigma=12.0)
    out = scratches(out, rng, n=9, strength=0.6)
    out = dust_spots(out, rng, n=34)
    return vignette(out, rng, strength=0.30)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_target_prompt(target_id: str, caption: str, rng) -> str:
    """Style phrase comes FIRST; caption/constraint are target-independent."""
    if target_id not in TARGET_SPECS:
        raise ValueError(f"unknown target id {target_id!r}")
    style = TARGET_SPECS[target_id]["prompts"][
        int(rng.integers(0, len(TARGET_SPECS[target_id]["prompts"])))
    ]
    constraint = CONTENT_HINTS[int(rng.integers(0, len(CONTENT_HINTS)))]
    cap = (caption or "").strip()
    # keep the prompt compact; the real Qwen3 path pads to 512 tokens anyway
    if cap:
        cap = " ".join(cap.split()[:22])
    tail = "; ".join(p for p in (cap, constraint) if p)
    return style + (". " + tail if tail else "")


def build_prompt_variants(
    target_id: str, caption: str, rng, n: int = 3
) -> list[str]:
    """n DISTINCT paraphrases of the SAME target intent.

    Distinctness matters: the toy hash encoder reads only 8 tokens, so if all
    variants shared the first phrase they would produce identical features and
    the dataset would silently collapse to one text per target. We explicitly
    take n different style phrases, then attach independently sampled
    caption/constraint tails.
    """
    if target_id not in TARGET_SPECS:
        raise ValueError(f"unknown target id {target_id!r}")
    styles = list(TARGET_SPECS[target_id]["prompts"])
    rng.shuffle(styles)
    styles = styles[: max(1, min(n, len(styles)))]
    cap = " ".join((caption or "").split()[:22]).strip()
    out = []
    for s in styles:
        constraint = CONTENT_HINTS[int(rng.integers(0, len(CONTENT_HINTS)))]
        tail = "; ".join(p for p in (cap, constraint) if p)
        out.append(s + (". " + tail if tail else ""))
    return out


def all_prompt_variants() -> list[str]:
    """Every possible style paraphrase, for offline Qwen3 pre-encoding."""
    return [p for tid in TARGET_IDS for p in TARGET_SPECS[tid]["prompts"]]
