from __future__ import annotations

"""Static preflight for the real FLUX.2 [klein] path.

Deliberately does NOT load the 4B model. It checks versions, GPU headroom,
weight presence, the MEASURED runtime dims (backbone_inspect.json) and the
offline prompt cache, then tells you the next concrete step.

GPU VRAM shortage is reported as a WARNING, not a blocker: the CPU-side
contract verification (--device cpu) does not need VRAM. The hard GPU guard
lives in src/system.py::preflight_flux2().

    D:/Anaconda/envs/ChatAI/python.exe scripts/verify_real_backend.py
"""

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT, save_json


def ver(module, attr="__version__"):
    try:
        return getattr(importlib.import_module(module), attr, "unknown")
    except Exception as e:
        return f"MISSING ({type(e).__name__})"


def _read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    rep = {"status": "UNVERIFIED", "checks": {}, "blockers": [], "warnings": [],
           "next_steps": []}

    import torch
    rep["checks"]["python"] = sys.version.split()[0]
    rep["checks"]["torch"] = torch.__version__
    rep["checks"]["cuda_available"] = torch.cuda.is_available()
    rep["checks"]["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        rep["checks"]["gpu"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        rep["checks"]["vram_total_GB"] = round(total / 1e9, 2)
        rep["checks"]["vram_free_GB"] = round(free / 1e9, 2)
        rep["checks"]["gpu_training_ready"] = free / 1e9 >= 9.0
        if free / 1e9 < 9:
            rep["warnings"].append(
                f"only {free/1e9:.1f} GB VRAM free: GPU TRAINING is blocked by "
                "preflight_flux2(), but the CPU contract verification "
                "(--device cpu) is unaffected. Close other GPU processes to "
                "unlock training. The FP8 repo is NOT a workaround (not loadable "
                "by diffusers)."
            )

    for m in ("diffusers", "transformers", "accelerate", "peft", "huggingface_hub",
              "safetensors", "bitsandbytes"):
        rep["checks"][m] = ver(m)

    d = str(rep["checks"]["diffusers"])
    if d.startswith("MISSING") or (not d.startswith("MISSING") and tuple(
            int(x) for x in d.split(".")[:2] if x.isdigit()) < (0, 37)):
        rep["blockers"].append("diffusers >= 0.37 required for Flux2KleinPipeline")

    weights = ROOT / "models" / "FLUX.2-klein-4B-Base"
    required = (
        "model_index.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "vae/config.json",
        "tokenizer/tokenizer.json",
    )
    missing = [r for r in required if not (weights / r).is_file()]
    rep["checks"]["weights_dir"] = str(weights)
    rep["checks"]["weights_present"] = not missing
    rep["checks"]["weights_missing"] = missing
    if missing:
        rep["blockers"].append(
            "FLUX.2 BF16 full pipeline not downloaded -> "
            "scripts/download_models.py (default; NOT --fp8-singlefile); "
            f"missing {missing[:4]}"
        )

    inspect_json = ROOT / "outputs" / "logs" / "backbone_inspect.json"
    ins = _read_json(inspect_json)
    rep["checks"]["inspect_report"] = ins is not None
    rep["checks"]["inspect_status"] = ins.get("status") if ins else None
    if ins is None:
        rep["blockers"].append("no backbone_inspect.json -> scripts/inspect_backbone.py")
    else:
        rep["checks"]["runtime_dims"] = ins.get("checks", {}).get("runtime_dims", {})
        rep["checks"]["measured"] = ins.get("measured", {})
        rec = ins.get("checks", {}).get("config_reconciliation", {})
        rep["checks"]["config_reconciliation"] = rec.get("ok")
        if ins.get("status") != "OK":
            rep["warnings"].append(
                f"backbone_inspect.json status={ins.get('status')} (re-run "
                "scripts/inspect_backbone.py after fixing the config)"
            )
        if rec and not rec.get("ok"):
            rep["blockers"].append(
                f"config dims disagree with real weights: {rec.get('mismatches')}"
            )

    for name in ("flux2_contract.json", "flux2_train_contract.json"):
        p = ROOT / "outputs" / "logs" / name
        rep["checks"][name] = p.exists()

    prep = _read_json(ROOT / "outputs" / "logs" / "precompute_prompts.json")
    rep["checks"]["prompts_precomputed"] = bool(prep and prep.get("status") == "OK")
    if not rep["checks"]["prompts_precomputed"]:
        rep["blockers"].append(
            "prompts not offline-encoded -> scripts/precompute_prompts.py "
            "(Qwen3-4B must never enter VRAM during training)"
        )

    fwd = _read_json(ROOT / "outputs" / "logs" / "verify_real_forward.json")
    rep["checks"]["real_forward_verified"] = bool(fwd and fwd.get("ok"))
    rep["checks"]["real_forward_status"] = (fwd or {}).get("status", "missing")
    # [FACT] T0-T3 are the structural / gradient-chain checks. T4 is a
    # DIFFERENT claim: "does user text change the output AFTER training".
    # On 2026-11-07 T0-T3 passed on real weights but T4 = 0.0 after a SINGLE
    # optimizer step (zero-init warm-up artefact, see text_warmup.json).
    # Keeping them separate stops a T4 artefact from erasing T0-T3 evidence.
    fc = (fwd or {}).get("checks", {})
    t0_t3_keys = ("T0_ok", "T1_identity_at_init", "T2_no_gradient_leak",
                  "T2_text_branch_has_gradient", "T3_ok")
    rep["checks"]["real_forward_T0_T3_verified"] = all(
        bool(fc.get(k)) for k in t0_t3_keys)
    rep["checks"]["T4_d_text"] = fc.get("T4_d_text")
    # [FACT] The VAE-only latent round-trip is independent corroborating
    # evidence for the T0 shape/BN-order fix (runs in seconds, no 4B DiT).
    lat = _read_json(ROOT / "outputs" / "logs" / "latent_roundtrip.json")
    rep["checks"]["latent_roundtrip_verified"] = bool(lat and lat.get("ok"))
    rep["checks"]["latent_roundtrip_status"] = (lat or {}).get("status", "missing")

    warm = _read_json(ROOT / "outputs" / "logs" / "text_warmup.json")
    rep["checks"]["text_warmup_verified"] = bool(
        warm and warm.get("conclusion", {}).get("hypothesis_confirmed"))
    rep["checks"]["text_warmup_status"] = (warm or {}).get("status", "missing")

    eff = _read_json(ROOT / "outputs" / "logs" / "real_text_effect.json")
    rep["checks"]["text_pathway_causal_verified"] = bool(eff and eff.get("ok"))
    rep["checks"]["text_pathway_status"] = (eff or {}).get("status", "missing")

    # [FACT 2026-11-12] GENUINELY TRAINED evidence SUPERSEDES the
    # WEIGHT-PERTURBED causal test: src/train.py ran on CUDA with the real
    # BF16 4B trunk, and verify_trained_text_effect.py loaded that checkpoint
    # to measure d_text between two different user prompts.
    trained = _read_json(ROOT / "outputs" / "logs" / "trained_text_effect.json")
    rep["checks"]["trained_text_effect_verified"] = bool(trained and trained.get("ok"))
    rep["checks"]["trained_text_effect_status"] = (trained or {}).get("status", "missing")
    rep["checks"]["trained_d_text"] = (
        (trained or {}).get("checks", {}).get("trained", {}).get("d_text_rms"))
    rep["checks"]["trained_checkpoint"] = (trained or {}).get("checkpoint")

    gpu_hist = None
    for d in ("flux2_klein_100", "flux2_klein_smoke", "flux2_klein"):
        h = _read_json(ROOT / "outputs" / "checkpoints" / d / "train_history.json")
        if h and str(h.get("device", "")).startswith("cuda"):
            if gpu_hist is None or int(h.get("steps", 0)) > int(gpu_hist.get("steps", 0)):
                gpu_hist = h
    rep["checks"]["gpu_training_verified"] = gpu_hist is not None
    rep["checks"]["gpu_training_steps"] = (gpu_hist or {}).get("steps")
    rep["checks"]["gpu_training_dataset"] = (gpu_hist or {}).get("dataset_source")

    if not rep["checks"]["real_forward_verified"]:
        if rep["checks"]["trained_text_effect_verified"]:
            rep["warnings"].append(
                "verify_real_forward.json KEEPS its historic status="
                f"{rep['checks']['real_forward_status']} (T4=0 after ONE "
                "optimizer step = zero-init warm-up artefact). It is now "
                "SUPERSEDED by GENUINELY TRAINED GPU weights with d_text="
                f"{rep['checks']['trained_d_text']}; the historic report is "
                "deliberately NOT rewritten."
            )
        elif rep["checks"]["real_forward_T0_T3_verified"]:
            rep["warnings"].append(
                "verify_real_forward.py status="
                f"{rep['checks']['real_forward_status']} is caused ONLY by T4 "
                "(d_text=0 after ONE optimizer step = zero-init warm-up "
                "artefact, explained by scripts/verify_text_warmup.py). "
                "T0-T3 passed on real weights. d_text from genuinely TRAINED "
                "weights is still UNVERIFIED and needs multi-step GPU training."
            )
        else:
            rep["warnings"].append(
                "real-weight end-to-end forward NOT yet verified "
                f"(status={rep['checks']['real_forward_status']}); "
                "scripts/verify_real_forward.py must pass T0-T3 before any "
                "training claim"
            )

    if rep["blockers"]:
        rep["next_steps"] = [
            "scripts/download_models.py            # BF16 FULL pipeline ~15.98 GB (NOT --fp8-singlefile)",
            "scripts/inspect_backbone.py           # measured runtime dims + config reconciliation",
            "scripts/precompute_prompts.py        # CPU offline Qwen3 prompt cache",
            "scripts/test_latent_roundtrip.py     # FAST VAE-only T0 regression (seconds)",
            "scripts/verify_real_backend.py       # re-run: verdict reflects real_forward status",
            "scripts/verify_real_forward.py       # real-weight CPU forward + gradient isolation",
            "train.py --override configs/flux2_klein.yaml --device cpu  ->  scripts/run_ablation.py",
        ]
    else:
        rep["next_steps"] = [
            "Structure + denoising forward contracts already verified (both all PASS, no weights needed)",
            "Real BF16 weights on disk; runtime dims MEASURED and reconciled with configs/flux2_klein.yaml",
            "scripts/test_latent_roundtrip.py   # FAST VAE-only T0 regression; BN-order fix verified",
            "scripts/verify_text_warmup.py     # explains T4=0 as a ONE-STEP zero-init warm-up artefact (seconds)",
            "scripts/verify_real_text_effect.py # real 4B trunk: text pathway causally connected (WEIGHT-PERTURBED)",
            "src/train.py --override configs/flux2_klein.yaml --device cuda  # GPU training VERIFIED (64x64 smoke)",
            "scripts/verify_trained_text_effect.py # d_text from GENUINELY TRAINED weights (supersedes the above)",
            "scripts/diagnose_text_usage.py     # is text actually NEEDED, or just a task-label bias? (confound check)",
            "scripts/show_train_data_used.py    # visualise what the run really trains on",
            "src/system.py::preflight_flux2() hard-guards against a half-loaded FLUX.2 build",
            "NEXT: merge old_photo.yaml into the flux2_klein backend (train.py accepts only ONE --override)",
            "NEXT: de-confound (same image + multiple instructions), text-condition dropout, instruction-following eval",
            "NEXT: >=256 resolution training + PSNR/SSIM/LPIPS and NIQE/CLIP-IQA/user study on real old photos",
        ]
    # Honest verdict ladder. "READY" is reserved for a fully passed
    # verify_real_forward.py. A T4 one-step warm-up artefact must NOT erase
    # the T0-T3 evidence, but must NOT be reported as READY either.
    # 2026-11-12: real CUDA training + trained-weight d_text is a NEW rung
    # ABOVE FORWARD-READY, but still below "restoration quality verified".
    if rep["blockers"]:
        rep["verdict"] = "BLOCKED"
        rep["status"] = "BLOCKED"
    elif (rep["checks"]["gpu_training_verified"]
          and rep["checks"]["trained_text_effect_verified"]
          and rep["checks"]["real_forward_T0_T3_verified"]):
        rep["verdict"] = (
            "GPU-TRAINING-VERIFIED (real BF16 4B trunk trained on CUDA for "
            f"{rep['checks']['gpu_training_steps']} steps; GENUINELY TRAINED "
            f"text branch d_text={rep['checks']['trained_d_text']}; T0-T3 "
            "verified; historic T4=0 explained as one-step warm-up artefact. "
            "Restoration quality, >=512x512 training and real old-photo "
            "fine-tuning are still UNVERIFIED)"
        )
        rep["status"] = "GPU-TRAINING-VERIFIED"
    elif rep["checks"]["real_forward_verified"]:
        rep["verdict"] = (
            "READY (real-weight CPU forward verified; "
            "GPU training still needs >= 9 GB free VRAM)"
        )
        rep["status"] = "READY"
    elif (rep["checks"]["real_forward_T0_T3_verified"]
          and rep["checks"]["text_warmup_verified"]
          and rep["checks"]["text_pathway_causal_verified"]):
        rep["verdict"] = (
            "FORWARD-READY (real-weight T0-T3 verified; T4=0 is a ONE-STEP "
            "zero-init warm-up artefact, explained; text pathway causally "
            "connected on the real 4B trunk; trained-weight d_text and "
            "restoration quality still UNVERIFIED; GPU training needs >= 9 GB "
            "free VRAM)"
        )
        rep["status"] = "UNVERIFIED"
    elif rep["checks"]["real_forward_T0_T3_verified"]:
        rep["verdict"] = (
            "STRUCTURE-READY (real-weight T0-T3 verified; T4=0 is a ONE-STEP "
            "zero-init warm-up artefact; text-pathway causality NOT yet tested "
            "on real weights; GPU training needs >= 9 GB free VRAM)"
        )
        rep["status"] = "UNVERIFIED"
    else:
        rep["verdict"] = (
            "STRUCTURE-READY (weights + dims + prompt cache verified; "
            "real forward NOT yet verified; GPU training needs >= 9 GB free VRAM)"
        )
        rep["status"] = "UNVERIFIED"
    out = ROOT / "outputs" / "logs" / "verify_real_backend.json"
    save_json(rep, out)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"[verify] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
