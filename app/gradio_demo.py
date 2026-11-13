from __future__ import annotations

"""[UNVERIFIED] Minimal interactive demo (gradio is NOT installed in ChatAI).

    pip install gradio
    D:/Anaconda/envs/ChatAI/python.exe app/gradio_demo.py

Uses the locally verified ToyMMDiT + trained text branch. It demonstrates the
interaction pattern (reference image + user text -> restored image), NOT
production-quality restoration. Swap build_system() for Flux2KleinBackend once
the real env is ready.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from src.inference import load_checkpoint, sample
from src.utils import ROOT, load_config, resolve_device

try:
    import gradio as gr
except ImportError:
    raise SystemExit(
        "gradio is not installed in this env. Run: pip install gradio\n"
        "(the rest of the project works without it)"
    )

CKPT = ROOT / "outputs" / "checkpoints" / "bypass" / "cond_step000200.pt"


def build():
    cfg = load_config()
    device = resolve_device(cfg["train"]["device"])
    cfg["condition"]["mode"] = "projector_plus_bypass"
    return cfg, device, load_checkpoint(cfg, device, ckpt=CKPT)


CFG, DEVICE, (BACKBONE, COND, CACHE) = build()


@torch.no_grad()
def run(image, prompt: str, use_text: bool):
    if image is None:
        return None
    size = CFG["data"]["image_size"]
    arr = np.array(Image.fromarray(image).convert("RGB").resize((size, size)))
    degraded = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    degraded = degraded.to(DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(20261107)
    shape = BACKBONE.image_to_tokens(degraded).shape
    noise = torch.randn(shape, generator=gen, device=DEVICE)
    out = sample(BACKBONE, COND, CACHE, degraded, prompt, CFG, DEVICE,
                 text_enabled=use_text, noise=noise)[0]
    a = (out.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(a).resize((size * 4, size * 4))


def main():
    demo = gr.Interface(
        fn=run,
        inputs=[
            gr.Image(label="Degraded / reference image"),
            gr.Textbox(label="User text", value="restore and denoise this image, keep the brick texture"),
            gr.Checkbox(label="Enable text branch", value=True),
        ],
        outputs=gr.Image(label="Output"),
        title="Text-guided image restoration (toy-verified architecture)",
        description=(
            "Toggle 'Enable text branch' to compare the frozen baseline with the "
            "trained text-conditioned output. ToyMMDiT = architecture demo only."
        ),
    )
    demo.launch()


if __name__ == "__main__":
    main()
