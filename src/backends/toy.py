from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Attention(nn.Module):
    """Q from x, K/V from kv (or x itself for self-attention)."""

    def __init__(self, dim: int, heads: int = 4, kv_dim: int | None = None):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        assert dim % heads == 0
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim or dim, dim)
        self.v = nn.Linear(kv_dim or dim, dim)
        self.o = nn.Linear(dim, dim)
        self._n = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, kv: torch.Tensor | None = None):
        kv = x if kv is None else kv
        B, N, _ = x.shape
        M = kv.shape[1]
        h = self.heads
        q = self.q(x).view(B, N, h, self.head_dim).transpose(1, 2)
        k = self.k(kv).view(B, M, h, self.head_dim).transpose(1, 2)
        v = self.v(kv).view(B, M, h, self.head_dim).transpose(1, 2)
        a = torch.softmax((q @ k.transpose(-1, -2)) / self._n, dim=-1)
        out = (a @ v).transpose(1, 2).contiguous().view(B, N, -1)
        return self.o(out)


class ToyJointBlock(nn.Module):
    """One MMDiT-style double-stream block.

    img stream  = noisy latent tokens
    ref stream  = reference-image tokens, occupying the ORIGINAL text/condition
                  branch (this is the FDI / Qwen2VL-Flux setup from the chat)

    The cross-attention on `ref` is INSIDE the frozen trunk. The new user-text
    condition is NOT here; it is injected by TextConditionModule via hooks on
    this block's output. That separation is the whole point of the design.
    """

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = Attention(dim, heads)  # Q=img, KV=ref  (FROZEN branch)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, img, ref, t_emb):
        img = img + self.self_attn(self.norm1(img + t_emb.unsqueeze(1)))
        img = img + self.cross_attn(self.norm2(img), ref)
        img = img + self.ffn(self.norm3(img))
        return img, ref  # tuple -> hooks act on output[0]


class ToyMMDiT(nn.Module):
    """Small, fully runnable stand-in for FLUX.2 [klein] 4B Base.

    Uses a flow-matching objective on patchified pixels instead of VAE latents,
    so it needs no VAE and no downloaded weights. Its purpose is to verify the
    ARCHITECTURE (freezing, zero-init injection, gradient isolation, text
    responsiveness), not to reproduce restoration quality.
    """

    def __init__(self, cfg):
        super().__init__()
        b, d = cfg["backbone"], cfg["data"]
        self.dim = b["dim"]
        self.patch_size = d["patch_size"]
        self.image_size = d["image_size"]
        self.num_ref_tokens = b["num_ref_tokens"]

        self.patch_embed = nn.Conv2d(
            3, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.unpatch = nn.ConvTranspose2d(
            self.dim, 3, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.ref_proj = nn.Linear(self.dim, self.dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim)
        )
        self.blocks = nn.ModuleList(
            [ToyJointBlock(self.dim, b["heads"]) for _ in range(b["num_blocks"])]
        )
        self.out_norm = nn.LayerNorm(self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)

    # ------------------------------------------------------------------ #
    def image_to_tokens(self, img: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] -> [B,N,D]"""
        h = self.patch_embed(img)
        return h.flatten(2).transpose(1, 2)

    def tokens_to_image(self, tok: torch.Tensor, size: int | None = None) -> torch.Tensor:
        """[B,N,D] -> [B,3,H,W]"""
        size = size or self.image_size
        s = size // self.patch_size
        h = tok.transpose(1, 2).reshape(tok.shape[0], self.dim, s, s)
        return self.unpatch(h)

    def ref_tokens_from_image(self, img: torch.Tensor) -> torch.Tensor:
        """Downsample the degraded image so it yields num_ref_tokens tokens."""
        side = int(math.sqrt(self.num_ref_tokens)) * self.patch_size
        small = F.interpolate(img, size=(side, side), mode="area")
        return self.ref_proj(self.image_to_tokens(small))

    @property
    def hook_targets(self) -> Sequence[nn.Module]:
        return list(self.blocks)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        ref: torch.Tensor,
        img_hint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict flow velocity v = x1 - x0.

        x_t : [B,N,D] interpolated latent tokens
        t   : [B] timestep in [0,1]
        ref : [B,M,D] FROZEN reference-image condition tokens (already fused
              with projected text by the caller in projector_only mode)
        """
        t_emb = self.time_mlp(timestep_embedding(t, self.dim))
        img = x_t + t_emb.unsqueeze(1)
        for blk in self.blocks:
            img, ref = blk(img, ref, t_emb)
        return self.out_proj(self.out_norm(img))
