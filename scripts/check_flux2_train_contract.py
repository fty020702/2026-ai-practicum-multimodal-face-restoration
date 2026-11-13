from __future__ import annotations

"""[FACT] Verify the FLUX.2 *training / denoising forward* contract.

This is the companion to scripts/check_flux2_contract.py. That script checked
the STRUCTURE (module names, hook index, Qwen3 embedding construction). This
one checks HOW a reference image and a text prompt actually reach the frozen
transformer during a denoising step, straight from the INSTALLED
diffusers 0.37.1 source. No weights are loaded.

WHY THIS MATTERS
  The whole project premise (from the shared DeepSeek chat) is "the pretrained
  model's TEXT input branch is already occupied by reference-image features,
  so we add user text on a NEW branch". On FLUX.2 [klein] that premise does NOT
  hold literally:

    reference image `image=[...]`  -> VAE-encoded -> CONCATENATED INTO THE
                                     IMAGE STREAM (hidden_states), with its own
                                     t-coordinate ids. NOT into the text stream.
    encoder_hidden_states           -> prompt_embeds (Qwen3 text stream). It is
                                     the TEXT branch and is *free*.

  So the honest mapping is: keep the Qwen3 text stream FROZEN AND CONSTANT
  (a neutral preamble), keep the reference-image tokens frozen in the image
  stream, and route the NEW user text ONLY through our trainable branch. That
  also makes the falsification test meaningful: any d_text > 0 must come from
  the new branch, not from a native prompt input.

  A second, equally important finding: FLUX.2 has THREE distinct feature dims,
  while the toy backbone collapses them all into one `dim`:
    in_channels          (packed VAE latent token dim, default 128)
    inner_dim            (double/single block hidden dim = heads * head_dim)
    joint_attention_dim  (Qwen3 prompt_embeds dim = 3 * Qwen3 hidden_size)
  The projector_only additive bias must therefore target `in_channels`, while
  the per-block bypass must target `inner_dim`. See docs/方案设计.md §12.

    D:/Anaconda/envs/ChatAI/python.exe scripts/check_flux2_train_contract.py

Writes outputs/logs/flux2_train_contract.json. Exit 0 only if all checks pass.
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT, save_json


def _src(obj) -> str:
    try:
        return inspect.getsource(obj)
    except Exception:
        return ""


def main():
    rep = {"status": "FACT", "checks": {}, "facts": {}, "design_implications": []}
    try:
        import diffusers
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
        from diffusers.models.transformers.transformer_flux2 import (
            Flux2TransformerBlock, Flux2SingleTransformerBlock,
        )
    except Exception as e:
        rep["error"] = f"import failed: {e}"
        save_json(rep, ROOT / "outputs" / "logs" / "flux2_train_contract.json")
        print(rep)
        return 1

    rep["facts"]["diffusers_version"] = diffusers.__version__
    call_src = _src(Flux2KleinPipeline.__call__)
    tr_fwd = _src(Flux2Transformer2DModel.forward)
    tr_init = _src(Flux2Transformer2DModel.__init__)
    dbl_fwd = _src(Flux2TransformerBlock.forward)
    vae_src = _src(Flux2KleinPipeline._encode_vae_image)

    # ---- signatures (recorded verbatim, not guessed) -------------------- #
    rep["facts"]["transformer_forward_signature"] = str(
        inspect.signature(Flux2Transformer2DModel.forward)
    )
    rep["facts"]["double_block_forward_signature"] = str(
        inspect.signature(Flux2TransformerBlock.forward)
    )
    rep["facts"]["pipeline_call_signature"] = str(
        inspect.signature(Flux2KleinPipeline.__call__)
    )

    # ---- pipeline.__call__ facts --------------------------------------- #
    rep["checks"]["ref_image_concatenated_into_image_stream"] = (
        "torch.cat([latents, image_latents], dim=1)" in call_src
    )
    rep["checks"]["ref_image_ids_concatenated"] = (
        "torch.cat([latent_ids, image_latent_ids], dim=1)" in call_src
    )
    rep["checks"]["encoder_hidden_states_is_prompt_embeds"] = (
        "encoder_hidden_states=prompt_embeds" in call_src
    )
    rep["checks"]["noise_pred_sliced_to_generated_tokens"] = (
        "noise_pred[:, : latents.size(1)" in call_src
    )
    rep["checks"]["pipeline_timestep_divided_by_1000"] = (
        "timestep=timestep / 1000" in call_src
    )
    rep["checks"]["num_channels_latents_is_in_channels_div_4"] = (
        "num_channels_latents = self.transformer.config.in_channels // 4" in call_src
    )
    rep["checks"]["guidance_none_for_klein_base"] = (
        "guidance=None" in call_src
    )
    rep["checks"]["cfg_uses_negative_prompt_embeds"] = (
        "negative_prompt_embeds" in call_src
    )
    rep["checks"]["vae_bn_normalisation_on_ref_image"] = (
        "image_latents = (image_latents - latents_bn_mean) / latents_bn_std" in vae_src
    )
    rep["checks"]["latents_bn_denormalised_before_decode"] = (
        "latents = latents * latents_bn_std + latents_bn_mean" in call_src
    )

    # ---- transformer.forward facts -------------------------------------- #
    rep["checks"]["timestep_multiplied_by_1000_internally"] = (
        "timestep = timestep.to(hidden_states.dtype) * 1000" in tr_fwd
    )
    rep["checks"]["x_embedder_projects_in_channels_to_inner_dim"] = (
        "self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)" in tr_init
    )
    rep["checks"]["context_embedder_projects_joint_dim_to_inner_dim"] = (
        "self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)"
        in tr_init
    )
    rep["checks"]["x_embedder_applied_to_hidden_states"] = (
        "hidden_states = self.x_embedder(hidden_states)" in tr_fwd
    )
    rep["checks"]["single_blocks_consume_concat_txt_img"] = (
        "hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)" in tr_fwd
    )
    rep["checks"]["text_tokens_removed_after_single_blocks"] = (
        "hidden_states = hidden_states[:, num_txt_tokens:, ...]" in tr_fwd
    )
    rep["checks"]["double_block_returns_txt_first"] = (
        "return encoder_hidden_states, hidden_states" in dbl_fwd
    )
    rep["checks"]["return_dict_false_gives_tuple"] = (
        "return (output,)" in tr_fwd
    )

    # ---- dimension facts (informational) -------------------------------- #
    rep["facts"]["in_channels_default"] = 128
    rep["facts"]["inner_dim_formula"] = "num_attention_heads * attention_head_dim"
    rep["facts"]["joint_attention_dim_default"] = 15360  # FLUX.2 dev / Mistral; Klein-4B uses 3*Qwen3 hidden
    rep["facts"]["qwen3_prompt_embeds_dim_note"] = (
        "Klein 4B: joint_attention_dim = 3 * Qwen3 hidden_size (layers 9,18,27 stacked)"
    )

    # ---- design implications (the correction to the original premise) --- #
    rep["design_implications"] = [
        "CORRECTION: in FLUX.2 [klein] the reference image is NOT in the text "
        "branch. It is VAE-encoded, patchified, BN-normalised and concatenated "
        "into hidden_states (the IMAGE stream) with a distinct t-coordinate id. "
        "encoder_hidden_states is the Qwen3 prompt_embeds TEXT stream and is free.",
        "Therefore on FLUX.2 we keep prompt_embeds FROZEN AND CONSTANT (a neutral "
        "preamble) and route user text ONLY through the new trainable branch; "
        "otherwise d_text could not be attributed to the new branch.",
        "THREE dims must not be conflated: in_channels (packed latents, ~128) for "
        "the projector_only additive bias; inner_dim for per-block bypass "
        "hooks; joint_attention_dim (~7680 on Klein 4B) for the frozen text "
        "features. The toy backbone collapses all three into one dim.",
        "Bypass hooks on transformer_blocks modify output[1] (img stream), which "
        "includes BOTH generated tokens and reference-image tokens. Only the "
        "first latents.size(1) outputs are used as velocity, matching the "
        "pipeline's noise_pred slice.",
        "timestep passed to transformer must be in [0,1] (sigma-like); the "
        "transformer multiplies it by 1000 internally. Our flow-matching t is "
        "already in [0,1], so pass t directly.",
        "Gradient checkpointing + forward hooks interact subtly; verify grad "
        "flow to the bypass params with real weights before enabling it.",
        "ALL of the above is source-level [FACT]. Weight loading, latent "
        "round-trip correctness and end-to-end training remain [UNVERIFIED].",
    ]

    bools = {k: v for k, v in rep["checks"].items() if isinstance(v, bool)}
    rep["all_passed"] = all(bools.values())
    rep["n_checks"] = len(bools)
    out = ROOT / "outputs" / "logs" / "flux2_train_contract.json"
    save_json(rep, out)
    for k, v in bools.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n[train-contract] all_passed={rep['all_passed']} ({rep['n_checks']} checks) -> {out}")
    for s in rep["design_implications"]:
        print(f"  * {s[:110]}")
    return 0 if rep["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
