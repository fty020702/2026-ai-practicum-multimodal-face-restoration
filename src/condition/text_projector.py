from __future__ import annotations

import torch
from torch import nn


class TextProjector(nn.Module):
    """Map frozen text-encoder outputs into the backbone condition space.

    The final linear is zero-initialised, so at step 0 the projector emits
    exactly zero. This is the single most important property of this design:
    the train-from-scratch text branch cannot perturb the frozen backbone
    before it has learned anything, which makes the ablation interpretable.

    Input : [B, L, in_dim]   (frozen text encoder output, no grad)
    Output: [B, L, out_dim]  (trainable condition tokens)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        zero_init_last: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        layers, d = [], in_dim
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(d, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ]
            d = hidden_dim
        last = nn.Linear(d, out_dim)
        layers.append(last)

        self.net = nn.Sequential(*layers)
        self.last = last
        self.zero_init_last = zero_init_last
        if zero_init_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def is_identity_at_init(self) -> bool:
        return (
            self.zero_init_last
            and torch.count_nonzero(self.last.weight).item() == 0
            and torch.count_nonzero(self.last.bias).item() == 0
        )
