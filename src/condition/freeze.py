from __future__ import annotations

import torch
from torch import nn


def freeze_backbone(module: nn.Module) -> nn.Module:
    """Freeze parameters and stop running-stat updates in the frozen trunk."""
    for p in module.parameters():
        p.requires_grad_(False)
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout)):
            m.eval()
    return module


def trainable_named_params(*modules):
    seen = set()
    for m in modules:
        for n, p in m.named_parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                yield n, p


def check_gradient_isolation(backbone: nn.Module, cond_module: nn.Module, loss: torch.Tensor):
    """Run backward and verify the core invariant of the whole proposal.

    Returns (leaked, trained):
      leaked  - list of FROZEN backbone params that received non-zero grad
      trained - list of text-branch params that received non-zero grad
    A correct implementation must give leaked == [] and trained != [].
    """
    loss.backward()
    leaked = [
        n for n, p in backbone.named_parameters()
        if p.grad is not None and torch.count_nonzero(p.grad).item() > 0
    ]
    trained = [
        n for n, p in cond_module.named_parameters()
        if p.grad is not None and torch.count_nonzero(p.grad).item() > 0
    ]
    return leaked, trained
