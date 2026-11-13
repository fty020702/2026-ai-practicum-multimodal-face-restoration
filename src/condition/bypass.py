from __future__ import annotations

import math

import torch
from torch import nn


class DecoupledTextCrossAttention(nn.Module):
    """One per-block decoupled cross-attention path for the NEW text branch.

    Mirrors IP-Adapter's mechanism: hidden states supply Q, the projected user
    text supplies K/V on a separate path, and the result is merged back through
    a ZERO-INITIALISED output projection. All parameters live in the trainable
    text branch; the frozen backbone only supplies activations (no grad).

    Because o_proj starts at zero, output == 0 exactly at init, yet
    dL/d(o_proj.weight) is non-zero on the very first backward pass, so the
    path opens up gradually during training instead of shock-loading the model.
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        heads: int = 4,
        attn_dim: int | None = None,
        scale: float = 1.0,
    ):
        super().__init__()
        inner = attn_dim or hidden_dim
        assert inner % heads == 0, "attn_dim must be divisible by heads"
        self.heads = heads
        self.inner = inner
        self.scale_out = scale
        self.head_dim = inner // heads
        self._norm = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, inner)
        self.k_proj = nn.Linear(cond_dim, inner)
        self.v_proj = nn.Linear(cond_dim, inner)
        self.o_proj = nn.Linear(inner, hidden_dim)

        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)

    def forward(self, hidden: torch.Tensor, text_cond: torch.Tensor) -> torch.Tensor:
        """hidden [B, N, H], text_cond [B, L, C] -> delta [B, N, H]."""
        B, N, _ = hidden.shape
        L = text_cond.shape[1]
        h = self.heads

        q = self.q_proj(hidden).view(B, N, h, self.head_dim).transpose(1, 2)
        k = self.k_proj(text_cond).view(B, L, h, self.head_dim).transpose(1, 2)
        v = self.v_proj(text_cond).view(B, L, h, self.head_dim).transpose(1, 2)

        attn = torch.softmax((q @ k.transpose(-1, -2)) / self._norm, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, self.inner)
        return self.o_proj(out) * self.scale_out
