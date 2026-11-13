from __future__ import annotations

"""[RESULT] Causal test: does USER TEXT change the REAL frozen FLUX.2 trunk?

WHY THIS EXISTS
  scripts/verify_real_forward.py (real 4B BF16 weights, CPU) passed
  T0/T1/T2/T3 but reported T4 d_text = 0.0 after ONE training step, so its
  status was FAILED. scripts/verify_text_warmup.py then showed on a tiny
  float32 stub that this is a ZERO-INIT WARM-UP ARTEFACT, not an architecture
  defect:

    step 1: only o_proj.weight/bias get gradients (2 params on the stub,
            10 params = 5 double blocks x 2 on the real run). projector(raw)
            is still exactly 0 -> K/V are prompt-independent -> d_text = 0.
    step 2: o_proj.weight != 0 opens the gradient path into k/v and the
            projector -> from step 3 forward, d_text > 0.

  Re-running 3 full CPU training steps on the 4B trunk would cost ~75 minutes
  (one backward alone measured 1368.2 s). This script instead asks the CAUSAL
  question directly, at a cost of 4 forward passes and NO backward:

    * load the REAL BF16 full pipeline (weights stay on disk/mmap)
    * keep the frozen NEUTRAL_PROMPT text stream constant
    * feed two DIFFERENT user prompts through the trainable text branch
    * measure d_text = RMS(out_a - out_b) on the real transformer output

  HONESTY / SCOPE  (read this before quoting the result)
    * The trunk weights are REAL (black-forest-labs/FLUX.2-klein-base-4B,
      21 files / 15.98 GB, downloaded 2026-11-07).
    * The "activated" text branch is WEIGHT-PERTURBED, NOT TRAINED: we replace
      the two zero-initialised tensors (projector.last.weight and each
      bypass.o_proj.weight) with small deterministic random values to emulate
      the post-warm-up state. It proves the text pathway is causally CONNECTED
      to the frozen trunk. It is NOT a claim about converged training or about
      restoration quality.
    * A trained-weight d_text still requires multi-step GPU training
      (>= 9 GB free VRAM on the 4070 Ti); see configs/flux2_klein.yaml.
    * The frozen constant NEUTRAL_PROMPT stream can be truncated with
      --text-tokens for CPU speed; the trainable text branch under test is
      length-independent. This is recorded, never hidden.

    D:/Anaconda/envs/ChatAI/python.exe scripts/verify_real_text_effect.py

Writes outputs/logs/real_text_effect.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.utils import ROOT, load_config, save_json, set_seed


def log(msg):
    print(f"[text-effect] {msg}", flush=True)


def rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.float() - b.float()) ** 2).mean().sqrt().item())


def maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def activate_cond(cond, scale: float, seed: int) -> dict:
    """Deterministically replace the two ZERO-INITIALISED tensors with small
    non-zero values, emulating the state AFTER the warm-up steps.

    HONESTY: these are PERTURBED weights, NOT trained weights. Only the two
    tensors that are zero at init are touched; all randomly-initialised
    tensors (k_proj/v_proj/q_proj/ref_bias_proj) are left exactly as built.
    Biases stay zero so the difference between two prompts comes ONLY from
    the text condition, not from a constant offset.
    """
    g = torch.Generator().manual_seed(seed)
    touched, n = [], 0
    with torch.no_grad():
        w = cond.projector.last.weight
        w.copy_(torch.randn(w.shape, generator=g, dtype=w.dtype, device=w.device) * scale)
        cond.projector.last.bias.zero_()
        touched.append(f"projector.last.weight {tuple(w.shape)}")
        n += w.numel()

        for i, b in enumerate(cond.bypasses):
            w = b.o_proj.weight
            w.copy_(torch.randn(w.shape, generator=g, dtype=w.dtype, device=w.device) * scale)
            b.o_proj.bias.zero_()
            touched.append(f"bypasses[{i}].o_proj.weight {tuple(w.shape)}")
            n += w.numel()
    return {"scale": scale, "seed": seed, "n_weights": n, "tensors": touched,
            "kind": "WEIGHT-PERTURBED (NOT trained)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-tokens", type=int, default=32,
                    help="CPU speed knob: truncate the FROZEN constant "
                         "NEUTRAL_PROMPT text stream to N tokens (0 = full 512).")
    ap.add_argument("--scale", type=float, default=0.02,
                    help="std of the deterministic perturbation applied to the "
                         "two zero-init tensors to emulate post-warm-up weights.")
    args = ap.parse_args()

    cfg = load_config(override=ROOT / "configs" / "flux2_klein.yaml")
    set_seed(cfg["project"]["seed"])
    device = torch.device("cpu")
    out = ROOT / "outputs" / "logs" / "real_text_effect.json"
    rep = {
        "status": "UNVERIFIED",
        "device": "cpu",
        "what": "causal test: user text -> real frozen FLUX.2 [klein] 4B trunk output",
        "honesty_note": (
            "REAL trunk weights; text branch is WEIGHT-PERTURBED, NOT TRAINED. "
            "This proves the text PATHWAY is causally connected to the frozen "
            "trunk, NOT converged training and NOT restoration quality."
        ),
        "weights": "black-forest-labs/FLUX.2-klein-base-4B (BF16 full pipeline, local)",
        "text_tokens": args.text_tokens if args.text_tokens > 0 else "full (512)",
        "checks": {},
        "timings_s": {},
    }

    # ---- hard preflight + REAL pipeline load (no backward needed) ----
    from src.system import build_flux2_system, preflight_flux2
    pre = preflight_flux2(cfg, device)
    rep["checks"]["preflight"] = pre["verdict"]
    log(f"preflight={pre['verdict']}")
    t0 = time.time()
    try:
        backbone, cond, cache = build_flux2_system(cfg, device)
    except Exception as e:
        rep["status"] = "FAILED"
        rep["error"] = f"build_flux2_system failed: {type(e).__name__}: {e}"
        save_json(rep, out)
        log(rep["error"])
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 1
    rep["timings_s"]["load_and_build"] = round(time.time() - t0, 1)
    rep["checks"]["runtime_dims"] = cfg.get("_runtime_dims")
    rep["checks"]["frozen_params"] = sum(p.numel() for p in backbone.parameters())
    rep["checks"]["trainable_params"] = cond.trainable_parameter_count()
    log(f"loaded in {rep['timings_s']['load_and_build']}s; "
        f"frozen={rep['checks']['frozen_params']:,} "
        f"trainable={rep['checks']['trainable_params']:,}")

    if args.text_tokens > 0:
        n = int(args.text_tokens)
        backbone.prompt_embeds = backbone.prompt_embeds[:, :n].contiguous()
        backbone.txt_ids = backbone.txt_ids[:, :n].contiguous()
        rep["checks"]["text_truncation_note"] = (
            "frozen constant NEUTRAL_PROMPT stream shortened for CPU speed; "
            "the trainable text branch under test is length-independent"
        )

    # ---- deterministic tiny batch, same recipe as verify_real_forward.py ----
    from src.data.dataset import make_dataset
    ds = make_dataset(cfg, root=cfg["_root"])
    batch = ds[0]
    clean = batch["clean"].unsqueeze(0)
    degraded = batch["degraded"].unsqueeze(0)
    pair = cfg["eval"]["text_response_pairs"][0]
    rep["checks"]["pair"] = list(pair)
    with torch.no_grad():
        x0 = backbone.image_to_tokens(clean)
        ref = backbone.ref_tokens_from_image(degraded)
    g = torch.Generator().manual_seed(cfg["project"]["seed"])
    x1 = torch.randn(x0.shape, generator=g, dtype=x0.dtype)
    t = torch.tensor([0.37], dtype=x0.dtype)
    x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    rep["checks"]["latent_tokens"] = list(x0.shape)
    log(f"latent={list(x0.shape)} pair={pair}")

    emb_a = cache.get([pair[0]]).to(device)
    emb_b = cache.get([pair[1]]).to(device)
    rep["checks"]["text_emb_shapes"] = [list(emb_a.shape), list(emb_b.shape)]

    def fwd(prompt_emb):
        with torch.no_grad():
            cond.set_text(prompt_emb)
            return backbone(x_t, t, cond.fuse_condition(ref)).clone()

    # ---- Stage A: ZERO-INIT control (reproduces the real run's T4=0) ----
    ta = time.time()
    log("stage A: zero-init control, 2 real transformer forwards ...")
    out_a0, out_b0 = fwd(emb_a), fwd(emb_b)
    rep["timings_s"]["zero_init_2_forwards"] = round(time.time() - ta, 1)
    rep["checks"]["zero_init"] = {
        "d_text_rms": rms(out_a0, out_b0),
        "d_text_maxabs": maxabs(out_a0, out_b0),
        "identity_check": bool(cond.identity_check()),
        "expected": "0.0 (projector output is exactly 0, so K/V are prompt-independent)",
    }
    log(f"stage A d_text={rep['checks']['zero_init']['d_text_rms']} "
        f"identity={rep['checks']['zero_init']['identity_check']} "
        f"({rep['timings_s']['zero_init_2_forwards']}s)")

    # ---- Stage B: cheap BRANCH-LEVEL causality (no trunk forward) ----
    with torch.no_grad():
        cond.set_text(emb_a)
        c_a = cond._cond.clone()
        cond.set_text(emb_b)
        c_b = cond._cond.clone()
    dummy = torch.randn(1, 32, c_a.shape[-1], dtype=c_a.dtype, device=c_a.device)
    with torch.no_grad():
        delta_a = cond.bypasses[0](dummy, c_a)
        delta_b = cond.bypasses[0](dummy, c_b)
    rep["checks"]["branch_level_zero_init"] = {
        "d_cond_rms": rms(c_a, c_b),
        "d_bypass_delta_rms": rms(delta_a, delta_b),
    }

    # ---- Stage C: ACTIVATED text branch, 2 real transformer forwards ----
    act = activate_cond(cond, args.scale, cfg["project"]["seed"])
    rep["checks"]["activation"] = act
    log(f"activated cond: {act['n_weights']:,} weights, scale={act['scale']}")

    with torch.no_grad():
        cond.set_text(emb_a)
        c_a = cond._cond.clone()
        cond.set_text(emb_b)
        c_b = cond._cond.clone()
        delta_a = cond.bypasses[0](dummy, c_a)
        delta_b = cond.bypasses[0](dummy, c_b)
    rep["checks"]["branch_level_activated"] = {
        "d_cond_rms": rms(c_a, c_b),
        "d_bypass_delta_rms": rms(delta_a, delta_b),
    }
    log(f"branch-level d_cond={rms(c_a, c_b):.6f} "
        f"d_bypass_delta={rms(delta_a, delta_b):.6f}")

    tc = time.time()
    log("stage C: activated cond, 2 real transformer forwards ...")
    out_a1, out_b1 = fwd(emb_a), fwd(emb_b)
    rep["timings_s"]["activated_2_forwards"] = round(time.time() - tc, 1)
    d_text = rms(out_a1, out_b1)
    rep["checks"]["activated"] = {
        "d_text_rms": d_text,
        "d_text_maxabs": maxabs(out_a1, out_b1),
        "output_rms": float(out_a1.float().square().mean().sqrt().item()),
        "relative_d_text": d_text / max(
            float(out_a1.float().square().mean().sqrt().item()), 1e-9),
        "text_effect": d_text > 1e-6,
    }
    log(f"stage C d_text={d_text} text_effect={d_text > 1e-6} "
        f"({rep['timings_s']['activated_2_forwards']}s)")

    # ---- verdict ----
    zi = rep["checks"]["zero_init"]
    ac = rep["checks"]["activated"]
    ok = (
        zi["identity_check"]
        and zi["d_text_rms"] == 0.0
        and ac["text_effect"]
        and rep["checks"]["branch_level_activated"]["d_bypass_delta_rms"] > 0.0
    )
    rep["ok"] = bool(ok)
    rep["status"] = "OK" if ok else "FAILED"
    rep["conclusion"] = {
        "text_pathway_causally_connected_on_real_weights": bool(ac["text_effect"]),
        "T4_zero_explained": (
            "verify_real_forward.py measured d_text AFTER ONE optimizer step; "
            "at that point projector(raw)==0 (zero-init warm-up), so the two "
            "prompts are bit-identical. scripts/verify_text_warmup.py shows the "
            "projector opens from step 2/3; this script shows that once the "
            "text branch is non-zero, the SAME real 4B trunk DOES respond to "
            "user text."
        ),
        "still_unverified": (
            "d_text from genuinely TRAINED weights, and any restoration-quality "
            "claim (PSNR/SSIM/LPIPS/user study), need multi-step training with "
            ">= 9 GB free VRAM on the 4070 Ti."
        ),
    }
    save_json(rep, out)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    log(f"status={rep['status']} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
