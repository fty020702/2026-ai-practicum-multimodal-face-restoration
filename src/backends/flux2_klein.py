from __future__ import annotations

"""FLUX.2 [klein] 4B Base adapter.

STATUS
  Structure / API contract : [FACT] - verified 2026-11-07 against the
                             INSTALLED diffusers 0.37.1 source (no weights
                             needed). Reproduce with scripts/check_flux2_contract.py
  Real BF16 weights + dims : [FACT] - 2026-11-07, 21 files / 15.98 GB on disk;
                             scripts/inspect_backbone.py measured inner_dim=3072,
                             joint_attention_dim=7680, in_channels=128,
                             5 double + 20 single blocks, (txt, img) output.
  End-to-end training      : [UNVERIFIED] - needs >= 9 GB free VRAM or a CPU
                             forward-only contract verification.

[CORRECTION 2026-11-07] The dedicated FP8 repo is NOT a loadable pipeline.
  black-forest-labs/FLUX.2-klein-base-4b-fp8 contains ONE 4.09 GB checkpoint
  (flux-2-klein-base-4b-fp8.safetensors) + 3 jpgs + README/LICENSE. It has
  NO model_index.json, NO scheduler/, NO transformer/config.json, NO
  text_encoder/, NO vae/, NO tokenizer/. Its keys are BFL/ComfyUI NATIVE
  (double_blocks.0.img_attn.qkv.weight_scale / input_scale,
  single_blocks.0.linear1, float8_e4m3fn + per-layer scales); Flux2KleinPipeline
  inherits DiffusionPipeline and has NEITHER from_single_file NOR FP8 support.
  => from_pretrained(<fp8 repo>) CANNOT work.
  load() therefore reads ONLY the local BF16 full pipeline folder
  models/FLUX.2-klein-4B-Base/, written by scripts/download_models.py.

Verified from diffusers 0.37.1 source (NOT guessed):

  * Flux2KleinPipeline.__init__(scheduler=FlowMatchEulerDiscreteScheduler,
        vae=AutoencoderKLFlux2, text_encoder=Qwen3ForCausalLM,
        tokenizer=Qwen2TokenizerFast, transformer=Flux2Transformer2DModel,
        is_distilled: bool = False). tokenizer_max_length = 512.
    `is_distilled=False` is exactly the Base / distilled distinction: train on
    FLUX.2-klein-4B-Base (50 steps), never on the 4-step distilled checkpoint.
  * Flux2Transformer2DModel has BOTH `transformer_blocks` (double stream) and
    `single_transformer_blocks`; self.inner_dim = num_attention_heads *
    attention_head_dim.
  * Flux2TransformerBlock.forward -> (encoder_hidden_states, hidden_states),
    i.e. **(txt, img)**. The image stream is output[1], so bypass hooks MUST
    use img_index=1. Getting this backwards would inject text into the text
    stream and silently do nothing to the generated image.
  * In transformer.forward, single blocks receive
    torch.cat([encoder_hidden_states, hidden_states], dim=1) and return ONE
    tensor by default (split_hidden_states=False). The same bypass must NOT be
    blindly attached there. This adapter therefore targets ONLY the 5 double
    blocks (measured from the real weights) - enough injection points without
    the concatenation ambiguity.
  * _get_qwen3_prompt_embeds: apply_chat_template(enable_thinking=False),
    text_encoder(output_hidden_states=True), stack hidden_states layers
    (9, 18, 27) -> [B,3,S,H] -> reshape [B,S,3*H]. The frozen text feature
    dim is therefore `transformer.config.joint_attention_dim` (= 3 * Qwen3
    hidden_size), NOT Qwen3's hidden_size.
  * __call__ accepts `prompt_embeds` directly -> offline text caching works.

Training/denoising forward contract (verified by
scripts/check_flux2_train_contract.py, 18/18 PASS):

  * Reference images `image=[...]` are VAE-encoded, patchified, BN-normalised
    and CONCATENATED INTO THE IMAGE STREAM (hidden_states), each with a
    distinct t-coordinate id. They are NOT in the text stream. The original
    chat's premise ("reference image occupies the text branch") therefore does
    NOT hold literally for FLUX.2 [klein]; encoder_hidden_states is the
    Qwen3 prompt_embeds TEXT stream and is free. We keep it FROZEN + CONSTANT
    and route user text ONLY through our trainable branch.
  * encoder_hidden_states=prompt_embeds, timestep=t/1000 (pipeline) then
    *1000 (transformer) -> net sigma in [0,1000]; our flow-matching t is in
    [0,1] so we pass t directly.
  * noise_pred = output[:, : latents.size(1)]  (reference tokens dropped).
  * THREE dims must not be conflated: in_channels (packed VAE latents, ~128)
    / inner_dim (block hidden) / joint_attention_dim (Qwen3 prompt embeds).

Hardware facts (see docs/方案设计.md §6): Klein 4B DiT ~7.75 GB BF16 +
Qwen3-4B encoder ~8.05 GB > 12 GB VRAM. The official recipe is
pipe.enable_model_cpu_offload(); its interaction with our register_forward_hook
injection is [UNVERIFIED]. Using the FP8 single file would additionally require
a reverse-quantisation, a BFL->diffusers key remap and an FP8 linear layer,
which this project does NOT provide.
"""

import inspect
from pathlib import Path

import torch
from torch import nn

from src.condition import TextConditionModule, freeze_backbone


def cfg_fp8_repo(cfg) -> None:
    """DEPRECATED - always returns None.

    [FACT 2026-11-07] The FP8 repo is a single ComfyUI-native scaled-FP8
    checkpoint, NOT a diffusers pipeline. It must NEVER be passed to
    Flux2KleinPipeline.from_pretrained(). Kept as a no-op so older configs and
    imports do not crash. See the module docstring for the full evidence.
    """
    return None


class Flux2KleinBackend:
    # Verified: double blocks return (txt, img).
    IMG_INDEX = 1
    # Single blocks are deliberately excluded (concatenated stream).
    BLOCK_QUERY = ("transformer_blocks",)

    def __init__(self, cfg, device=None):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        b = cfg["backend"]
        self.model_id = b["model_id"]
        self.dtype = b["dtype"]
        self.pipe = None
        self.transformer = None
        self.text_encoder = None
        self.tokenizer = None
        self.block_modules: list[nn.Module] = []
        self.block_dims: list[int] = []

    # ------------------------------------------------------------------ #
    def load(self):
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError as e:
            raise RuntimeError(
                "Flux2KleinPipeline requires diffusers >= 0.37. "
                "ChatAI env now has 0.37.1 (see docs/环境升级.md)."
            ) from e

        # [FACT 2026-11-07] ONLY the BF16 Base repo is a loadable diffusers
        # pipeline. The FP8 repo is a single ComfyUI-native checkpoint and
        # CANNOT be passed to from_pretrained (see module docstring).
        # scripts/download_models.py writes that full pipeline here.
        local = Path(self.cfg["_root"]) / "models" / "FLUX.2-klein-4B-Base"
        required = (
            "model_index.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.json",
            "vae/config.json",
            "tokenizer/tokenizer.json",
        )
        missing = [r for r in required if not (local / r).is_file()]
        if missing:
            raise RuntimeError(
                f"FLUX.2 BF16 full pipeline incomplete at {local};\n"
                f"  missing: {missing}\n"
                "  Run scripts/download_models.py (NOT --fp8-singlefile).\n"
                "  Progress: outputs/logs/download_state.json / "
                "outputs/logs/download_progress.json"
            )

        # [FACT] Official recipe: torch_dtype=torch.bfloat16. `dtype` in the
        # config is only a weight-variant label, NEVER a torch.float8 cast.
        self.pipe = Flux2KleinPipeline.from_pretrained(
            str(local), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )

        b = self.cfg["backend"]
        # Default: keep the WHOLE pipeline on CPU. On a 12 GB 4070 Ti the
        # BF16 DiT (7.75 GB) + Qwen3-4B (8.05 GB) + VAE cannot fit, and
        # pipe.to(cuda) would OOM. CPU is enough for runtime-dim inspection,
        # offline prompt precompute and the forward-contract check.
        if b.get("offload_text_encoder", False):
            # [UNVERIFIED] accelerate offload hooks vs our forward-hook
            # injection; opt-in only, reconcile with real weights first.
            self.pipe.enable_model_cpu_offload()
        elif b.get("load_transformer_to_device", False):
            if self.device.type == "cuda" and torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                if free / 1e9 < 9.0:
                    raise RuntimeError(
                        f"only {free/1e9:.1f} GB VRAM free; the BF16 DiT alone "
                        "needs ~7.8 GB plus activations. Close other GPU "
                        "processes, or run the CPU-side contract verification "
                        "with --device cpu. The FP8 repo is not a workaround "
                        "(not loadable by diffusers)."
                    )
            self.pipe.transformer.to(self.device)

        self.transformer = self.pipe.transformer
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        freeze_backbone(self.transformer)
        freeze_backbone(self.text_encoder)
        self._discover_blocks()
        return self

    def _discover_blocks(self):
        for name in self.BLOCK_QUERY:
            obj = getattr(self.transformer, name, None)
            if isinstance(obj, (nn.ModuleList, list, tuple)) and len(obj) > 0:
                self.block_modules = list(obj)
                break
        if not self.block_modules:
            raise RuntimeError(
                "No double-stream block list found. "
                "Run scripts/check_flux2_contract.py and inspect_backbone.py."
            )
        self.block_dims = [self.inner_dim] * len(self.block_modules)

    @property
    def inner_dim(self) -> int:
        v = getattr(self.transformer, "inner_dim", None)
        if v is not None:
            return int(v)
        for m in self.transformer.modules():
            if isinstance(m, nn.Linear):
                return max(m.in_features, m.out_features)
        raise RuntimeError("cannot infer inner_dim from transformer")

    @property
    def joint_attention_dim(self) -> int:
        v = getattr(getattr(self.transformer, "config", None), "joint_attention_dim", None)
        if v is not None:
            return int(v)
        # fallback: 3 stacked Qwen3 hidden layers (verified in pipeline source)
        h = getattr(getattr(self.text_encoder, "config", None), "hidden_size", None)
        if h is not None:
            return 3 * int(h)
        raise RuntimeError("cannot infer joint_attention_dim")

    @property
    def in_channels(self) -> int:
        """Width of a PACKED VAE latent token (= x_embedder input width).

        Verified from diffusers 0.37.1 source:
          pipeline: num_channels_latents = transformer.config.in_channels // 4
          transformer: self.x_embedder = nn.Linear(in_channels, inner_dim)
          _pack_latents: [B, C, H, W] -> [B, H*W, C]  with C = in_channels
        The projector_only additive bias lives in THIS space, not inner_dim.
        """
        v = getattr(getattr(self.transformer, "config", None), "in_channels", None)
        if v is not None:
            return int(v)
        lin = getattr(self.transformer, "x_embedder", None)
        if isinstance(lin, nn.Linear):
            return int(lin.in_features)
        raise RuntimeError("cannot infer in_channels from transformer.x_embedder")

    def apply_runtime_config(self, cfg):
        """Overwrite the guessed dims in configs/flux2_klein.yaml with real ones.

        Must be called AFTER load() and BEFORE building TextConditionModule.

        FLUX.2 has THREE distinct feature widths (verified by
        scripts/check_flux2_train_contract.py); the toy backbone has only one:
          joint_attention_dim -> frozen Qwen3 prompt_embeds   (text_in_dim)
          inner_dim           -> double-block hidden stream   (bypass q/o)
          in_channels         -> packed VAE latent tokens    (ref bias target)
        """
        d, j, c = self.inner_dim, self.joint_attention_dim, self.in_channels
        cfg["condition"]["text_in_dim"] = j
        cfg["condition"]["hidden_dim"] = d
        cfg["condition"]["cond_dim"] = d
        cfg["condition"]["ref_bias_out_dim"] = c
        cfg["condition"]["bypass_heads"] = self._safe_heads(
            cfg["condition"].get("bypass_heads", 8), d
        )
        cfg["_runtime_dims"] = {
            "inner_dim": d, "joint_attention_dim": j, "in_channels": c,
        }
        return cfg

    @staticmethod
    def _safe_heads(wanted: int, dim: int) -> int:
        """Largest divisor of dim not exceeding the configured head count."""
        for h in range(int(wanted), 0, -1):
            if dim % h == 0:
                return h
        return 1

    @property
    def hook_targets(self):
        return self.block_modules

    # ------------------------------------------------------------------ #
    def encode_texts(self, prompts):
        """Frozen text-encoder outputs, shaped [B, S, joint_attention_dim].

        Uses the pipeline's own static method so the chat template, the
        (9,18,27) layer stacking and the reshape are EXACTLY the ones the
        pretrained DiT expects. Run offline on CPU; the result is cached.
        """
        from diffusers import Flux2KleinPipeline
        with torch.no_grad():
            return Flux2KleinPipeline._get_qwen3_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=list(prompts),
                dtype=torch.bfloat16,
            )

    def encode_free_text(self, prompt: str):
        """Inference-time path for arbitrary user text.

        On a 12 GB card: encode with the encoder offloaded/4-bit, then
        empty_cache() before the DiT denoising loop. ComfyUI does the same for
        FLUX.2. [UNVERIFIED] - needs real weights to time and verify.
        """
        emb = self.encode_texts([prompt])
        torch.cuda.empty_cache()
        return emb

    def describe(self):
        return {
            "model_id": self.model_id,
            "dtype": self.dtype,
            "num_double_blocks": len(self.block_modules),
            "inner_dim": self.inner_dim if self.transformer else None,
            "joint_attention_dim": self.joint_attention_dim if self.transformer else None,
            "img_index": self.IMG_INDEX,
            "block_types": [type(b).__name__ for b in self.block_modules[:2]],
            "text_encoder_type": type(self.text_encoder).__name__ if self.text_encoder else None,
            "status": "structure FACT / weights+training UNVERIFIED",
        }


def build_flux2_system(cfg, device):
    """Load backend, reconcile dims, then attach the ONLY trainable module.

    [UNVERIFIED] end-to-end: requires weights on disk and free VRAM. The call
    order here is what makes the hook land on the image stream.
    """
    backend = Flux2KleinBackend(cfg, device).load()
    backend.apply_runtime_config(cfg)
    # [FACT] The frozen FLUX.2 trunk runs in bf16 (official recipe), so the
    # forward hook receives bf16 hidden states. The bypass q/k/v/o linears
    # MUST be bf16 as well, otherwise `img + bypass(img, text_cond)` fails
    # with a mixed-dtype matmul error. TextConditionModule.set_text() casts
    # the (float32) offline cache entries to this dtype.
    cond = TextConditionModule(cfg, backend.block_dims).to(
        device=device, dtype=torch.bfloat16
    )
    cond.attach(backend.hook_targets, img_index=backend.IMG_INDEX)
    return backend, cond
