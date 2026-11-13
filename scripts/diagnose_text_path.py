from __future__ import annotations

"""Diagnose whether a text signal actually survives the forward path.

Run: D:/Anaconda/envs/ChatAI/python.exe scripts/diagnose_text_path.py

Prints, for each condition mode, the magnitude of the difference between two
different prompts at every stage: hash features -> projector output -> fused
reference condition -> single backbone forward -> full Euler sample (raw and
clamped). This distinguishes an implementation bug from a genuine
"the text branch learned a constant bias" result.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.backends.toy import ToyMMDiT
from src.condition import TextConditionModule, freeze_backbone
from src.data.dataset import RestorationDataset
from src.data.text_features import HashTextEncoder, OfflineTextCache
from src.smoke_test import make_smoke_cfg
from src.train import compute_loss
from src.utils import ROOT, resolve_device


def d(a, b):
    return float((a - b).abs().max().item()), float(((a - b) ** 2).mean().sqrt().item())


def run(mode, device, steps=60):
    cfg = make_smoke_cfg(mode, device)
    torch.manual_seed(cfg["project"]["seed"])
    backbone = ToyMMDiT(cfg).to(device)
    freeze_backbone(backbone)
    dim = cfg["backbone"]["dim"]
    cond = TextConditionModule(cfg, [dim] * cfg["backbone"]["num_blocks"]).to(device)
    cond.attach(backbone.hook_targets)
    enc = HashTextEncoder(cfg["condition"]["text_in_dim"], cfg["backbone"]["num_text_tokens"])
    cache = OfflineTextCache(ROOT, enc)
    ds = RestorationDataset(cfg, root=cfg["_root"])
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in ds[0].items()}

    opt = torch.optim.AdamW([p for p in cond.parameters() if p.requires_grad], lr=1e-3)
    torch.manual_seed(cfg["project"]["seed"] + 1)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        compute_loss(backbone, cond, cache, batch, device, True).backward()
        opt.step()

    pa, pb = cfg["eval"]["text_response_pairs"][0]
    ea, eb = cache.get([pa]).to(device), cache.get([pb]).to(device)
    degraded = batch["degraded"].to(device)

    print(f"\n=== {mode} ===")
    print(f"  hash feature diff (max,rms)      : {d(ea, eb)}")
    with torch.no_grad():
        ca, cb = cond.projector(ea), cond.projector(eb)
        print(f"  projector output diff            : {d(ca, cb)}")
        cond.set_enabled(True); cond.set_text(ea); ra = cond.fuse_condition(backbone.ref_tokens_from_image(degraded))
        cond.set_text(eb); rb = cond.fuse_condition(backbone.ref_tokens_from_image(degraded))
        print(f"  fused reference cond diff        : {d(ra, rb)}")

        ref0 = backbone.ref_tokens_from_image(degraded)
        x_t = torch.randn_like(backbone.image_to_tokens(degraded))
        t = torch.tensor([0.5], device=device)
        cond.set_text(ea); oa = backbone(x_t, t, cond.fuse_condition(ref0))
        cond.set_text(eb); ob = backbone(x_t, t, cond.fuse_condition(ref0))
        print(f"  single backbone forward diff     : {d(oa, ob)}")

        # full Euler, RAW (no clamp) with identical initial noise
        g = torch.Generator(device=device).manual_seed(7)
        noise = torch.randn(x_t.shape, generator=g, device=device)
        steps_e = cfg["train"]["num_timesteps"]
        outs = {}
        for tag, emb in (("a", ea), ("b", eb)):
            cond.set_text(emb)
            ref = cond.fuse_condition(ref0)
            x = noise
            dt = 1.0 / steps_e
            for i in range(steps_e):
                tt = torch.full((1,), 1.0 - i * dt, device=device)
                x = x - dt * backbone(x, tt, ref)
            outs[tag] = backbone.tokens_to_image(x)
        raw = outs["a"], outs["b"]
        clamped = raw[0].clamp(0, 1), raw[1].clamp(0, 1)
        print(f"  raw sample diff                  : {d(*raw)}")
        print(f"  clamped sample diff              : {d(*clamped)}")
        sat = float(((clamped[0] <= 0) | (clamped[0] >= 1)).float().mean().item())
        print(f"  saturated fraction (a)           : {sat:.4f}")
    cond.detach()


if __name__ == "__main__":
    dev = resolve_device("auto")
    print(f"[diag] device={dev}")
    for m in ("projector_only", "projector_plus_bypass"):
        run(m, dev)
