from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .bypass import DecoupledTextCrossAttention
from .text_projector import TextProjector


class TextConditionModule(nn.Module):
    """The ONLY trainable component in the system.

    The setup from the shared chat: a pretrained model whose *text* branch is
    already occupied by reference-image features (FDI / Qwen2VL-Flux style).
    We keep the diffusion trunk AND that reference-image branch frozen, and add
    a new user-text branch on top.

    Two operating modes:

      projector_only
          The projected text is pooled into a single zero-initialised bias that
          is ADDED to the frozen reference-image condition. Single injection
          point, minimal capacity. This is literally "just train a projector"
          from the chat. (We add rather than concatenate on purpose; see
          fuse_condition.)

      projector_plus_bypass  (recommended)
          The projected text tokens feed one zero-initialised decoupled
          cross-attention bypass per transformer block (IP-Adapter style).
          Same number of trainable params order, but per-layer control.
    """

    def __init__(self, cfg, block_dims: Sequence[int]):
        super().__init__()
        c = cfg["condition"]
        self.mode = c["mode"]
        if self.mode not in ("projector_only", "projector_plus_bypass"):
            raise ValueError(f"unknown condition.mode: {self.mode}")

        self.projector = TextProjector(
            in_dim=c["text_in_dim"],
            hidden_dim=c["hidden_dim"],
            out_dim=c["cond_dim"],
            num_layers=c["projector_layers"],
            dropout=c["projector_dropout"],
            zero_init_last=c["zero_init_last"],
        )
        # projector_only injection: pool text -> one bias vector over ref tokens.
        #
        # DO NOT zero-initialise this layer's WEIGHT. The projector's last layer
        # is already zero-initialised, so its output (the input here) is exactly
        # 0 at init. If this weight were also 0 we would get a gradient deadlock:
        #   dL/d(W_ref)  = grad_output @ pooled = 0
        #   d(bias)/d(pooled) = W_ref^T          = 0
        # so neither layer could ever open up and the text branch would collapse
        # to a constant bias. Random W + zero bias keeps the initial output
        # bit-identical (0) while leaving the gradient path open.
        # ref_bias_out_dim is the feature width of the tensor that
        # fuse_condition() adds to. On the toy backbone all dims collapse to
        # one value, so the default (= hidden_dim) keeps the verified toy path
        # bit-identical. On FLUX.2 the reference-image tokens are packed VAE
        # latents of width `in_channels` (128), while the per-block bypass
        # works in `inner_dim` (3072 on FLUX.2 klein 4B, measured); apply_runtime_config() therefore sets
        # ref_bias_out_dim = transformer.config.in_channels separately.
        self.ref_bias_proj = nn.Linear(
            c["cond_dim"], c.get("ref_bias_out_dim", c["hidden_dim"])
        )
        nn.init.xavier_uniform_(self.ref_bias_proj.weight)
        nn.init.zeros_(self.ref_bias_proj.bias)
        # In projector_only mode the per-block bypasses are not used at all;
        # do not even create them, otherwise they sit in the optimiser with
        # grad=None and inflate the trainable parameter count.
        self.bypasses = nn.ModuleList(
            [
                DecoupledTextCrossAttention(
                    hidden_dim=dim,
                    cond_dim=c["cond_dim"],
                    heads=c["bypass_heads"],
                    scale=c["bypass_scale"],
                )
                for dim in block_dims
            ]
            if self.mode == "projector_plus_bypass"
            else []
        )
        self.handles: list = []
        self._cond = None
        self._enabled = True

    # ------------------------------------------------------------------ #
    # Text features are produced OUTSIDE this module by a frozen encoder.
    # Passing them in explicitly is what makes offline text caching possible
    # (critical for 12 GB cards: Qwen3-4B never has to enter VRAM).
    # ------------------------------------------------------------------ #
    def set_text(self, raw_text_emb: torch.Tensor | None):
        if raw_text_emb is None or not self._enabled:
            self._cond = None
            return None
        # FLUX.2 runs the frozen trunk in bf16, so the trainable text branch is
        # bf16 too. OfflineTextCache returns float32; cast to the module's own
        # parameter dtype here. On the toy path (all float32) this is a no-op,
        # so the verified [RESULT] numbers stay bit-identical.
        ref = next(self.projector.parameters())
        raw = raw_text_emb.to(device=ref.device, dtype=ref.dtype)
        self._cond = self.projector(raw)
        return self._cond

    def clear_text(self):
        self._cond = None

    def set_enabled(self, flag: bool):
        self._enabled = bool(flag)
        if not flag:
            self.clear_text()

    def fuse_condition(self, cond_ref: torch.Tensor) -> torch.Tensor:
        """Used by projector_only: inject projected text as one pooled bias.

        ADD, not CONCAT. Appending extra keys changes the softmax
        normalisation inside the FROZEN cross-attention even when the added
        vectors are exactly zero, which would silently break the
        identity-at-init guarantee. A zero-initialised additive bias does not.
        """
        if self.mode == "projector_only" and self._cond is not None:
            pooled = self._cond.mean(dim=1, keepdim=True)   # [B,1,C]
            return cond_ref + self.ref_bias_proj(pooled)     # [B,1,H]
        return cond_ref

    # ------------------------------------------------------------------ #
    # Hook-based per-block injection. Hooks are registered on frozen modules;
    # only the bypass parameters inside this module carry gradients.
    # ------------------------------------------------------------------ #
    def attach(self, blocks: Sequence[nn.Module], img_index: int = 0):
        """Register per-block bypass hooks.

        img_index selects WHICH element of a tuple output is the image stream.
        ToyMMDiT returns (img, ref) -> 0.
        FLUX.2 double blocks return (encoder_hidden_states, hidden_states)
        = (txt, img) -> 1. Verified against diffusers 0.37.1 source.
        Non-tuple outputs are treated as a single (possibly concatenated)
        stream and are modified in place.
        """
        self.detach()
        if self.mode != "projector_plus_bypass":
            return self
        for i, blk in enumerate(blocks):
            self.handles.append(blk.register_forward_hook(self._make_hook(i, img_index)))
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        return self

    def _make_hook(self, idx: int, img_index: int = 0):
        def hook(module, inputs, output):
            if not self._enabled or self._cond is None:
                return output
            if isinstance(output, tuple):
                img = output[img_index]
                new_img = img + self.bypasses[idx](img, self._cond)
                return tuple(
                    new_img if k == img_index else v for k, v in enumerate(output)
                )
            return output + self.bypasses[idx](output, self._cond)

        return hook

    def forward(self, hidden: torch.Tensor, text_cond: torch.Tensor) -> torch.Tensor:
        """Standalone path (no hooks) for unit tests."""
        if len(self.bypasses) == 0:
            return hidden
        return hidden + self.bypasses[0](hidden, text_cond)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def identity_check(self) -> bool:
        proj_ok = self.projector.is_identity_at_init()
        # ref_bias_proj.weight is intentionally NOT zero; identity comes from
        # projector output == 0 and ref_bias_proj.bias == 0.
        bias_ok = torch.count_nonzero(self.ref_bias_proj.bias).item() == 0
        bypass_ok = all(
            torch.count_nonzero(b.o_proj.weight).item() == 0
            and torch.count_nonzero(b.o_proj.bias).item() == 0
            for b in self.bypasses
        )
        return proj_ok and bias_ok and bypass_ok
