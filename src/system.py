from __future__ import annotations

"""Backend dispatch: the SAME (backbone, cond, cache) triple for toy and FLUX.2.

WHY THIS FILE EXISTS
  train.py / inference.py / evaluate.py were originally hard-wired to
  ToyMMDiT. They all speak one tiny interface:

      backbone.image_to_tokens(img)   [B,3,H,W] -> [B,N,D]
      backbone.tokens_to_image(tok)  [B,N,D]   -> [B,3,H,W] in [0,1]
      backbone.ref_tokens_from_image(img)      -> [B,M,D]
      backbone(x_t, t, ref)                   -> velocity [B,N,D]
      backbone.hook_targets                   -> modules to hook
      cond.fuse_condition(ref), cond.set_text(...), cond.set_enabled(...)

  Rather than rewrite all callers, `Flux2BackboneAdapter` IMPLEMENTS that same
  interface on top of Flux2KleinPipeline. So compute_loss(), sample(),
  evaluate_checkpoint() and run_ablation.py work unchanged for both backends.

STATUS
  Structure / forward contract : [FACT]  (scripts/check_flux2_contract.py +
                                  scripts/check_flux2_train_contract.py)
  Real BF16 weights on disk + runtime dims : [FACT] 2026-11-07
    (scripts/download_models.py -> scripts/inspect_backbone.py)
  VAE latent round-trip (T0) : [RESULT] 2026-11-07 OK
    -> scripts/test_latent_roundtrip.py. FIXED the 32-vs-128 BN-order bug:
       denormalise in the 128-channel PATCHIFIED space BEFORE unpatchify.
       The float32 latent inverse is EXACT (max abs err 0.0), bf16 is bounded.
  End-to-end forward / training : see outputs/logs/verify_real_forward.json
    -> scripts/verify_real_forward.py (CPU); preflight_flux2() REFUSES a CUDA
       build without >= 9 GB free VRAM, so this code can never silently run
       in a broken/half-loaded state.

  The toy path is NOT touched by this module. train.py and inference.py keep
  their original toy builders verbatim and only branch here when
  cfg["backend"]["name"] == "flux2_klein".

Grounding (diffusers 0.37.1 source, see scripts/check_flux2_train_contract.py):
  * reference image -> VAE encoded, patchified, BN-normalised, CONCATENATED
    INTO hidden_states (image stream), with distinct t-coordinate img_ids.
    It is NOT in encoder_hidden_states (text stream).
  * encoder_hidden_states = prompt_embeds (Qwen3, frozen + CONSTANT here).
  * transformer.forward(hidden_states, encoder_hidden_states, timestep,
        img_ids, txt_ids, guidance=None, return_dict=False)[0]
  * transformer multiplies timestep by 1000 internally -> pass sigma t in [0,1].
  * only output[:, :N_generated] is used as velocity.
  * in_channels (packed latents) / inner_dim (block hidden) / joint_attention_dim
    (text) are THREE different widths. The projector_only bias targets
    in_channels; the per-block bypass targets inner_dim.
"""

import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.condition import TextConditionModule, freeze_backbone
from src.data.text_features import OfflineTextCache
from src.utils import ROOT

# A FIXED, content-free preamble. It is identical for every sample and every
# prompt, so any d_text measured by the falsification test can only come from
# our NEW trainable branch, never from the native prompt input.
NEUTRAL_PROMPT = "restore this image"

WEIGHTS_SUBDIR = "FLUX.2-klein-4B-Base"
TEXT_CACHE_NAMESPACE = "flux2_klein_qwen3"


# --------------------------------------------------------------------------- #
# Preflight: fail loudly instead of running unverified code by accident.
# --------------------------------------------------------------------------- #
def preflight_flux2(cfg, device=None, require_weights: bool = True):
    """Return a status dict; raise RuntimeError with actionable steps if blocked."""
    rep = {"backend": "flux2_klein", "checks": {}, "blockers": []}
    try:
        import diffusers
        from diffusers import Flux2KleinPipeline  # noqa: F401
        rep["checks"]["diffusers"] = diffusers.__version__
        rep["checks"]["pipeline_import"] = True
    except Exception as e:
        rep["blockers"].append(f"Flux2KleinPipeline import failed: {e}")

    # Only a CUDA target needs VRAM. A CPU run is a legitimate way to verify
    # runtime dims / prompt encoding / forward contract without a 4B DiT in VRAM.
    dev_str = str(device) if device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    rep["checks"]["target_device"] = dev_str
    if "cuda" in dev_str and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        rep["checks"]["vram_total_GB"] = round(total / 1e9, 2)
        rep["checks"]["vram_free_GB"] = round(free / 1e9, 2)
        if free / 1e9 < 9.0:
            rep["blockers"].append(
                f"only {free/1e9:.1f} GB VRAM free; the BF16 DiT alone is 7.75 GB. "
                "Options: close other GPU processes, run the CPU-side contract "
                "verification with --device cpu, or [UNVERIFIED] implement FP8 "
                "dequant. The FP8 repo is NOT loadable by diffusers as-is."
            )

    wdir = Path(cfg.get("_root", str(ROOT))) / "models" / WEIGHTS_SUBDIR
    required = (
        "model_index.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "vae/config.json",
        "tokenizer/tokenizer.json",
    )
    missing = [r for r in required if not (wdir / r).is_file()]
    rep["checks"]["weights_dir"] = str(wdir)
    rep["checks"]["weights_present"] = not missing
    rep["checks"]["weights_missing"] = missing
    if require_weights and missing:
        rep["blockers"].append(
            f"FLUX.2 BF16 full pipeline incomplete at {wdir} -> run "
            "scripts/download_models.py (default, NOT --fp8-singlefile); "
            f"missing {missing[:4]}"
        )

    if rep["blockers"]:
        msg = (
            "FLUX.2 [klein] backend is [UNVERIFIED] and cannot be built yet:\n  - "
            + "\n  - ".join(rep["blockers"])
            + "\n\nVerified without weights: scripts/check_flux2_contract.py and "
            "scripts/check_flux2_train_contract.py (both all PASS).\n"
            "Then: scripts/inspect_backbone.py -> scripts/verify_real_backend.py "
            "-> scripts/precompute_prompts.py -> train.py --override configs/flux2_klein.yaml"
        )
        raise RuntimeError(msg)
    rep["verdict"] = "READY (still [UNVERIFIED] until real weights are loaded)"
    return rep


# --------------------------------------------------------------------------- #
# Frozen Qwen3 text features, plugged into the existing OfflineTextCache.
# --------------------------------------------------------------------------- #
class Qwen3TextEncoderAdapter:
    """Adapts Flux2KleinBackend.encode_texts() to the encoder interface that
    OfflineTextCache expects: .encode(list[str]) -> [B,L,D] and .dim/.num_tokens.

    Uses Flux2KleinPipeline._get_qwen3_prompt_embeds (verified [FACT]), so the
    chat template, the (9,18,27) layer stacking and the reshape are EXACTLY
    the ones the pretrained DiT was trained with. NOT HashTextEncoder.
    """

    def __init__(self, backend, max_sequence_length: int = 512):
        self.backend = backend
        self.dim = backend.joint_attention_dim
        self.num_tokens = max_sequence_length

    def encode(self, texts):
        te = getattr(self.backend, "text_encoder", None)
        if te is None:
            raise RuntimeError(
                "Qwen3 text encoder was released to save VRAM but a prompt is "
                "missing from the offline cache. Run scripts/precompute_prompts.py "
                "before training (the encoder must never enter VRAM during training)."
            )
        return self.backend.encode_texts(texts).detach().cpu().float()


def make_flux2_text_cache(cfg, backend) -> OfflineTextCache:
    return OfflineTextCache(
        ROOT, Qwen3TextEncoderAdapter(backend), namespace=TEXT_CACHE_NAMESPACE
    )


# --------------------------------------------------------------------------- #
# VAE latent packing / unpacking, transcribed VERBATIM from
# diffusers 0.37.1 pipelines/flux2/pipeline_flux2_klein.py.
#
# Kept as module-level PURE functions (taking `pipe`) so the latent round-trip
# can be unit-tested by loading ONLY the 0.17 GB AutoencoderKLFlux2, instead
# of the whole 16 GB BF16 pipeline. See scripts/test_latent_roundtrip.py.
#
# [FACT] AutoencoderKLFlux2 config: latent_channels=32, patch_size=[2,2].
# Its BN running stats are therefore 128-wide: they live in the PATCHIFIED
# latent space (32 * 2 * 2 = 128), NOT in the 32-channel VAE latent space.
#   _encode_vae_image (L459-471): vae.encode -> _patchify_latents
#                                  -> (x - bn_mean) / bn_std  (128-wide)
#   __call__ decode  (L898-909): x * bn_std + bn_mean       (128-wide)
#                                  -> _unpatchify_latents -> vae.decode
# The inverse MUST denormalise BEFORE unpatchify. Getting this order wrong is
# exactly the T0 failure seen on 2026-11-07.
# --------------------------------------------------------------------------- #
def packed_side(n: int) -> int:
    """Side of the square token grid; FLUX.2 packs a 2x2 patch per token."""
    s = int(math.isqrt(n))
    if s * s != n:
        raise ValueError(
            f"token count {n} is not a square; packed FLUX.2 latents must "
            "be a square grid (H/16 * W/16)."
        )
    return s


def encode_image_to_packed(pipe, img: torch.Tensor, act_dtype):
    """[B,3,H,W] in [0,1] -> (packed [B,N,128], 4D patchified [B,128,s,s]).

    Mirrors L459-471 (_encode_vae_image) + L388-396 (_pack_latents):
      x in [-1,1] -> vae.encode -> [B,32,h,w] -> _patchify_latents
      -> [B,128,h/2,w/2] -> BN NORMALISE (128-wide) -> _pack_latents
      -> [B,N,128].
    """
    vae = pipe.vae
    x = (img * 2.0 - 1.0).to(device=vae.device, dtype=vae.dtype)
    # VAE downsamples 8x and _patchify_latents needs H/8, W/8 divisible by 2
    # -> image side must be a multiple of 16 (L487-488: 2*(h//(scale*2))).
    h, w = x.shape[-2:]
    mult = 16
    nh, nw = max(mult, round(h / mult) * mult), max(mult, round(w / mult) * mult)
    if (nh, nw) != (h, w):
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)

    lat4d = pipe._encode_vae_image(x, generator=None)   # [B,128,nh/16,nw/16]
    packed = pipe._pack_latents(lat4d)                   # [B,N,128]
    return packed.to(act_dtype), lat4d


def decode_packed_to_image(pipe, tok: torch.Tensor) -> torch.Tensor:
    """[B,N,128] -> [B,3,H,W] in [0,1]. EXACT inverse of encode_image_to_packed.

    Mirrors L898-909. ORDER IS CRITICAL [FACT]:
      1) reshape packed tokens -> [B,128,s,s] (PATCHIFIED + BN-normalised space)
      2) x * bn_std + bn_mean  (128-wide stats, applied HERE)
      3) _unpatchify_latents  -> [B,32,2s,2s]
      4) vae.decode           -> [B,3,16s,16s]; [-1,1] -> [0,1]

    The previous implementation denormalised AFTER unpatchify, i.e. tried to
    add a 128-wide BN stat to a 32-channel tensor, raising
      RuntimeError: The size of tensor a (32) must match the size of tensor b
      (128) at non-singleton dimension 1
    which was the T0 failure of scripts/verify_real_forward.py on 2026-11-07.
    """
    B, N, C = tok.shape
    s = packed_side(N)
    vae = pipe.vae
    lat4d = tok.transpose(1, 2).reshape(B, C, s, s).to(vae.dtype)
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(lat4d.device, lat4d.dtype)
    bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(lat4d.device, lat4d.dtype)
    lat4d = lat4d * bn_std + bn_mean                  # 128-channel space
    lat = pipe._unpatchify_latents(lat4d)             # [B,32,2s,2s]
    img = vae.decode(lat, return_dict=False)[0]
    return ((img + 1.0) * 0.5).clamp(0, 1).float()


# --------------------------------------------------------------------------- #
# The adapter.
# --------------------------------------------------------------------------- #
class Flux2BackboneAdapter(nn.Module):
    """FLUX.2 [klein] speaking the ToyMMDiT interface.

    [UNVERIFIED] end-to-end: latent round-trip, dtype/offload reconciliation
    and gradient flow through hooks under checkpointing need real weights.
    All shape logic below is transcribed from diffusers 0.37.1 source.
    """

    def __init__(self, backend, cfg):
        super().__init__()
        self.backend = backend
        self.pipe = backend.pipe
        # Register as submodules so backbone.parameters() / freeze_backbone()
        # see the real frozen trunk (and train.py's param counts stay honest).
        self.transformer = backend.transformer
        self.vae = self.pipe.vae
        self.resolution = int(cfg["backend"].get("resolution", 512))
        self.act_dtype = torch.bfloat16

        # Frozen, CONSTANT Qwen3 text stream. Computed once at build time.
        # Prefer the offline cache (precompute_prompts.py explicitly includes
        # NEUTRAL_PROMPT) to avoid one 8 GB Qwen3-4B CPU forward per build.
        emb = self._neutral_prompt_embeds(backend)
        self.register_buffer("prompt_embeds", emb, persistent=False)
        self.register_buffer(
            "txt_ids", self.pipe._prepare_text_ids(emb).long(), persistent=False
        )
        if cfg["backend"].get("gradient_checkpointing", True):
            # Forward hooks + reentrant checkpointing interact subtly and can
            # starve the bypass params of gradients. Verify with real weights
            # before turning this back on; correctness first, memory second.
            try:
                self.transformer.gradient_checkpointing = False
                if hasattr(self.transformer, "_set_gradient_checkpointing"):
                    self.transformer._set_gradient_checkpointing(value=False)
            except Exception:
                pass

    def _neutral_prompt_embeds(self, backend):
        """Return [1,S,joint_attention_dim] bf16 for NEUTRAL_PROMPT.

        scripts/precompute_prompts.py::collect_prompts() explicitly adds
        NEUTRAL_PROMPT to the offline cache, and that cache entry was produced
        by the SAME Flux2KleinPipeline._get_qwen3_prompt_embeds used below.
        Reading it saves one full Qwen3-4B CPU forward (~8 GB weights, minutes
        of compute and a large RAM peak) on every build. Falls back to a real
        encode when the cache is absent, so this never silently changes the
        semantics of the constant text stream.
        """
        p = (ROOT / "data" / "text_cache" / TEXT_CACHE_NAMESPACE
             / f"{OfflineTextCache._key(NEUTRAL_PROMPT)}.pt")
        if p.is_file():
            e = torch.load(p, weights_only=True)
            if e.ndim == 2:                       # [S,D] -> [1,S,D]
                e = e.unsqueeze(0)
            return e.detach().to(self.act_dtype)
        return backend.encode_texts([NEUTRAL_PROMPT]).detach().to(self.act_dtype)

    # -- helpers: now thin wrappers around the module-level pure functions --- #
    @torch.no_grad()
    def _encode_image(self, img: torch.Tensor):
        return encode_image_to_packed(self.pipe, img, self.act_dtype)

    @staticmethod
    def _side(n: int) -> int:
        return packed_side(n)

    def _img_ids(self, n_gen: int, n_ref: int, B: int, C: int, device):
        """Generated tokens get t=0; reference tokens get t=10 (scale=10).

        Mirrors pipeline lines 840 / _prepare_latent_ids / _prepare_image_ids.
        """
        sg, sr = self._side(n_gen), self._side(n_ref)
        dummy_gen = torch.zeros(B, C, sg, sg)
        dummy_ref = torch.zeros(1, C, sr, sr)
        gen_ids = self.pipe._prepare_latent_ids(dummy_gen)          # [B,N,4]
        ref_ids = self.pipe._prepare_image_ids([dummy_ref])          # [1,M,4]
        ref_ids = ref_ids.repeat(B, 1, 1)
        return torch.cat([gen_ids, ref_ids], dim=1).to(device)

    # -- the ToyMMDiT interface ------------------------------------------- #
    @torch.no_grad()
    def image_to_tokens(self, img: torch.Tensor) -> torch.Tensor:
        return self._encode_image(img)[0]

    @torch.no_grad()
    def ref_tokens_from_image(self, img: torch.Tensor) -> torch.Tensor:
        return self._encode_image(img)[0]

    @torch.no_grad()
    def tokens_to_image(self, tok: torch.Tensor) -> torch.Tensor:
        return decode_packed_to_image(self.pipe, tok)

    @property
    def hook_targets(self):
        return self.backend.hook_targets

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, ref: torch.Tensor):
        """Predict flow velocity v = x1 - x0 for the GENERATED tokens.

        x_t : [B,N,C] interpolated packed latents (sigma-space)
        t   : [B] sigma in [0,1] (transformer multiplies by 1000 internally)
        ref : [B,M,C] frozen reference-image packed latents (already fused
              with the projected text bias in projector_only mode)
        """
        B, N, C = x_t.shape
        M = ref.shape[1]
        device = x_t.device
        hidden = torch.cat([x_t, ref], dim=1).to(self.act_dtype)
        img_ids = self._img_ids(N, M, B, C, device)

        emb = self.prompt_embeds.to(device=device, dtype=self.act_dtype)
        if emb.shape[0] != B:
            emb = emb.expand(B, -1, -1)
        txt_ids = self.txt_ids.to(device)
        if txt_ids.shape[0] != B:
            txt_ids = txt_ids.expand(B, -1, -1)

        out = self.transformer(
            hidden_states=hidden,
            encoder_hidden_states=emb,
            timestep=t.to(device=device, dtype=self.act_dtype),
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            return_dict=False,
        )[0]
        # Pipeline does exactly this: reference-image tokens are dropped.
        return out[:, :N]


# --------------------------------------------------------------------------- #
# Builder consumed by train.py / inference.py.
# --------------------------------------------------------------------------- #
def _load_cond_checkpoint(cond, ckpt, cfg, device):
    p = Path(ckpt) if ckpt else None
    if p and p.exists():
        state = torch.load(p, map_location=device, weights_only=False)
        if state.get("mode") != cfg["condition"]["mode"]:
            raise ValueError(
                f"checkpoint mode {state.get('mode')} != config mode "
                f"{cfg['condition']['mode']}"
            )
        cond.load_state_dict(state["cond_state"])
        print(f"[system] loaded cond checkpoint {p}")
    elif ckpt:
        print(f"[system] WARNING: checkpoint not found: {ckpt} (using zero-init)")
    else:
        print("[system] no checkpoint -> zero-initialised (identity) text branch")
    return cond


def _release_text_encoder(backend):
    """Free the ~8 GB Qwen3-4B encoder after the constant preamble is cached."""
    try:
        backend.text_encoder = None
        backend.tokenizer = None
        if getattr(backend, "pipe", None) is not None:
            backend.pipe.text_encoder = None
            backend.pipe.tokenizer = None
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_flux2_system(cfg, device, ckpt=None, mode=None):
    """Return (backbone, cond, cache) for the real FLUX.2 [klein] 4B Base path.

    [UNVERIFIED] until the real BF16 FULL PIPELINE is on disk and a GPU with
    >= 9 GB free VRAM is available. CPU-only contract verification is
    supported via --device cpu (the 4B DiT then stays in system RAM).
    """
    preflight_flux2(cfg, device)
    if mode:
        cfg["condition"]["mode"] = mode

    # Imported lazily: importing diffusers at module import time would slow
    # down (and could break) the fully-verified toy path.
    from src.backends.flux2_klein import build_flux2_system as _build_backend

    backend, cond = _build_backend(cfg, device)      # loads + apply_runtime_config + attach(img_index=1)
    cache = make_flux2_text_cache(cfg, backend)       # needs the encoder: do it BEFORE release

    backbone = Flux2BackboneAdapter(backend, cfg).to(device)
    freeze_backbone(backbone)

    _load_cond_checkpoint(cond, ckpt, cfg, device)
    if cfg["backend"].get("release_text_encoder", True):
        _release_text_encoder(backend)

    print(f"[system] backend=flux2_klein runtime_dims={cfg.get('_runtime_dims')}")
    print(f"[system] frozen params={sum(p.numel() for p in backbone.parameters()):,} "
          f"trainable text params={cond.trainable_parameter_count():,} "
          f"identity_at_init={cond.identity_check()}")
    return backbone, cond, cache
