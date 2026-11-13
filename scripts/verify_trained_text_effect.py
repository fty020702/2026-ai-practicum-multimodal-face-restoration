from __future__ import annotations

"""[RESULT] d_text measured on GENUINELY TRAINED text-branch weights.

This is the step that upgrades the project's verdict beyond FORWARD-READY.
Until now the only evidence that user text changes the real frozen FLUX.2
[klein] 4B trunk came from scripts/verify_real_text_effect.py, whose text
branch was WEIGHT-PERTURBED, NOT TRAINED. Here we load a REAL checkpoint
written by src/train.py --override configs/flux2_klein.yaml --device cuda
and ask the same causal question:

    two different user prompts -> two different trunk outputs?

    D:/Anaconda/envs/ChatAI/python.exe scripts/verify_trained_text_effect.py
    D:/Anaconda/envs/ChatAI/python.exe scripts/verify_trained_text_effect.py \
        --checkpoint outputs/checkpoints/flux2_klein_smoke/cond_step000020.pt

HONESTY
  * Real BF16 4B trunk weights + REAL trained cond weights.
  * This measures the TEXT RESPONSE of the trained branch, NOT restoration
    quality (no PSNR/SSIM/LPIPS) and NOT language understanding.
  * Running on cuda needs >= 9 GB free VRAM; the training process must have
    exited first. Use --device cpu as a (much slower) fallback.

Writes outputs/logs/trained_text_effect.json
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
    print(f"[trained-effect] {msg}", flush=True)


def rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.float() - b.float()) ** 2).mean().sqrt().item())


def maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def find_checkpoint(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        return p if p.is_file() else None
    for d in ("flux2_klein_smoke", "flux2_klein"):
        base = ROOT / "outputs" / "checkpoints" / d
        if not base.is_dir():
            continue
        ckpts = sorted(base.glob("cond_step*.pt"))
        if ckpts:
            return ckpts[-1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="cond checkpoint .pt; default = latest step in the "
                         "smoke / real checkpoint dirs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text-tokens", type=int, default=0,
                    help="CPU speed knob: truncate the FROZEN constant "
                         "NEUTRAL_PROMPT stream to N tokens (0 = full 512). "
                         "Not needed on GPU; recorded if used.")
    ap.add_argument("--out", default="trained_text_effect.json",
                    help="report filename under outputs/logs/")
    args = ap.parse_args()

    cfg = load_config(override=ROOT / "configs" / "flux2_klein.yaml")
    set_seed(cfg["project"]["seed"])
    device = torch.device(args.device if torch.cuda.is_available() or
                          args.device == "cpu" else "cpu")
    out = ROOT / "outputs" / "logs" / args.out
    rep = {
        "status": "UNVERIFIED",
        "device": str(device),
        "what": "d_text from GENUINELY TRAINED text-branch weights on the real "
                "FLUX.2 [klein] 4B frozen trunk",
        "honesty_note": (
            "REAL trunk weights + REAL trained cond weights. Measures text "
            "response only; NOT restoration quality and NOT language "
            "understanding."
        ),
        "checks": {},
        "timings_s": {},
    }

    ckpt = find_checkpoint(args.checkpoint)
    if ckpt is None:
        rep["status"] = "FAILED"
        rep["error"] = "no cond checkpoint found; run src/train.py first"
        save_json(rep, out)
        log(rep["error"])
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 1
    rep["checkpoint"] = str(ckpt.relative_to(ROOT))
    log(f"checkpoint={rep['checkpoint']}")

    from src.system import build_flux2_system, preflight_flux2
    pre = preflight_flux2(cfg, device)
    rep["checks"]["preflight"] = pre["verdict"]
    t0 = time.time()
    try:
        backbone, cond, cache = build_flux2_system(cfg, device, ckpt=str(ckpt))
    except Exception as e:
        rep["status"] = "FAILED"
        rep["error"] = f"build_flux2_system failed: {type(e).__name__}: {e}"
        save_json(rep, out)
        log(rep["error"])
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 1
    rep["timings_s"]["load_and_build"] = round(time.time() - t0, 1)
    rep["checks"]["identity_check_after_training"] = bool(cond.identity_check())
    rep["checks"]["runtime_dims"] = cfg.get("_runtime_dims")
    rep["checks"]["frozen_params"] = sum(p.numel() for p in backbone.parameters())
    rep["checks"]["trainable_params"] = cond.trainable_parameter_count()
    log(f"loaded in {rep['timings_s']['load_and_build']}s; "
        f"identity_after_training={rep['checks']['identity_check_after_training']}")

    if args.text_tokens > 0:
        n = int(args.text_tokens)
        backbone.prompt_embeds = backbone.prompt_embeds[:, :n].contiguous()
        backbone.txt_ids = backbone.txt_ids[:, :n].contiguous()
        rep["checks"]["text_tokens_truncated_to"] = n
        rep["checks"]["text_truncation_note"] = (
            "frozen constant NEUTRAL_PROMPT stream shortened; the trainable "
            "text branch under test is length-independent"
        )

    # Same deterministic tiny batch recipe as verify_real_forward.py / train.py.
    from src.data.dataset import make_dataset
    ds = make_dataset(cfg, root=cfg["_root"])
    batch = ds[0]
    clean = batch["clean"].unsqueeze(0).to(device)
    degraded = batch["degraded"].unsqueeze(0).to(device)
    pair = cfg["eval"]["text_response_pairs"][0]
    rep["checks"]["pair"] = list(pair)
    g = torch.Generator().manual_seed(cfg["project"]["seed"])
    with torch.no_grad():
        x0 = backbone.image_to_tokens(clean)
        ref = backbone.ref_tokens_from_image(degraded)
    # A CPU-side Generator cannot directly fill a CUDA tensor
    # ("Expected a 'cuda' device type for generator but found 'cpu'").
    # Generate on CPU then move, so the seed stays device-independent.
    x1 = torch.randn(x0.shape, generator=g, dtype=x0.dtype).to(x0.device)
    t = torch.tensor([0.37], dtype=x0.dtype, device=x0.device)
    x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    rep["checks"]["latent_tokens"] = list(x0.shape)
    log(f"latent={list(x0.shape)} pair={pair}")

    emb_a = cache.get([pair[0]]).to(device)
    emb_b = cache.get([pair[1]]).to(device)

    def fwd(prompt_emb):
        with torch.no_grad():
            cond.set_enabled(True)
            cond.set_text(prompt_emb)
            return backbone(x_t, t, cond.fuse_condition(ref)).clone()

    tt = time.time()
    out_a = fwd(emb_a)
    out_b = fwd(emb_b)
    with torch.no_grad():
        cond.set_enabled(False)
        cond.clear_text()
        out_off = backbone(x_t, t, ref).clone()
    rep["timings_s"]["three_forwards"] = round(time.time() - tt, 1)

    d_text = rms(out_a, out_b)
    out_rms = float(out_a.float().square().mean().sqrt().item())
    rep["checks"]["trained"] = {
        "d_text_rms": d_text,
        "d_text_maxabs": maxabs(out_a, out_b),
        "output_rms": out_rms,
        "relative_d_text": d_text / max(out_rms, 1e-9),
        "d_text_onoff_rms": rms(out_a, out_off),
        "text_effect": d_text > 1e-6,
    }
    log(f"d_text_rms={d_text} relative={d_text / max(out_rms, 1e-9):.6f} "
        f"d_onoff={rep['checks']['trained']['d_text_onoff_rms']} "
        f"({rep['timings_s']['three_forwards']}s)")

    # Corroboration: previous weight-perturbed result and zero-init baseline.
    prev = ROOT / "outputs" / "logs" / "real_text_effect.json"
    if prev.is_file():
        d = json.loads(prev.read_text(encoding="utf-8"))
        rep["checks"]["comparison"] = {
            "zero_init_d_text": d.get("checks", {}).get("zero_init", {}).get("d_text_rms"),
            "weight_perturbed_d_text": d.get("checks", {}).get("activated", {}).get("d_text_rms"),
            "genuinely_trained_d_text": d_text,
        }

    ok = bool(rep["checks"]["trained"]["text_effect"]
              and not rep["checks"]["identity_check_after_training"])
    rep["ok"] = ok
    rep["status"] = "OK" if ok else "FAILED"
    rep["conclusion"] = {
        "trained_weights_text_effect": bool(rep["checks"]["trained"]["text_effect"]),
        "upgrades": (
            "Replaces the WEIGHT-PERTURBED evidence with GENUINELY TRAINED "
            "weights. T0-T3 (real weights) + warm-up explanation + trained "
            "d_text together justify upgrading the verdict."
            if ok else
            "Checkpoint did not open the text branch yet: d_text is still 0 "
            "(likely too few steps) or the branch is still zero-initialised. "
            "Run more training steps."
        ),
        "still_unverified": (
            "Restoration quality (PSNR/SSIM/LPIPS on paired synthetic data; "
            "NIQE/CLIP-IQA/user study on real old photos without GT), formal "
            "512x512 training, and gradient-checkpointing/offload compatibility."
        ),
    }
    save_json(rep, out)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    log(f"status={rep['status']} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
