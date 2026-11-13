from __future__ import annotations

# Task-level prompt: tells the model WHAT restoration operation to perform.
# Content-level prompt: user-supplied semantic constraint (the thing this
# project is actually adding on top of an existing reference-image branch).
TASK_PROMPTS = {
    "gaussian_noise": "restore and denoise this image",
    "gaussian_blur": "restore and deblur this image, recover sharp edges",
    "jpeg": "remove jpeg compression artifacts from this image",
    "downsample_up": "super resolve this image, recover fine texture",
    "rain": "restore and derain this image, remove rain streaks",
    "haze": "restore and dehaze this image, recover contrast",
}

CONTENT_HINTS = (
    "keep the brick texture",
    "preserve facial detail",
    "keep the foliage natural",
    "preserve the text on the sign",
    "keep the sky smooth",
    "preserve fine fabric detail",
)


def build_prompt(task: str, content: str | None = None) -> str:
    base = TASK_PROMPTS.get(task, "restore this image")
    return f"{base}, {content}" if content else base


def task_from_prompt(prompt: str) -> str:
    for k in TASK_PROMPTS:
        if k.replace("_", " ") in prompt.lower() or TASK_PROMPTS[k] in prompt:
            return k
    return "unknown"
