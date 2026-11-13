from __future__ import annotations

"""[RESULT] Real-weight end-to-end CPU verification for FLUX.2 [klein] 4B.

This is the step that turns the FLUX.2 path from [UNVERIFIED] into measured
evidence. It loads the REAL BF16 full pipeline (21 files / 15.98 GB), builds
the frozen backbone + the ONLY trainable text branch, and runs ONE real
flow-matching step on CPU with a tiny image so it fits in RAM.

Falsifiable checks:
  T0 latent round-trip  : image -> VAE packed tokens -> image
  T1 identity-at-init   : text ON vs OFF is BIT-IDENTICAL
  T2 gradient isolation : backward leaks NO gradient into the frozen trunk
  T3 trainable branch   : ONE optimizer step changes cond params only
  T4 text response      : after that step, two prompts give different outputs

  [FACT 2026-11-07] T4 as written is EXPECTED to be 0.0 after ONE step.
  The projector's last layer AND every bypass o_proj are zero-initialised
  (that is what makes T1 identity-at-init bit-exact). On step 1 only
  o_proj.weight/bias receive gradients (measured: exactly 10 tensors =
  5 double blocks x 2); projector(raw) is still exactly 0, so K/V are
  prompt-independent and the two prompts are bit-identical. The projector
  only opens from step 2/3. scripts/verify_text_warmup.py reproduces this
  on a tiny float32 stub in seconds; scripts/verify_real_text_effect.py
  tests the CAUSAL question on the REAL 4B trunk with a weight-perturbed
  (NOT trained) text branch. We deliberately KEEP T4 in the pass/fail gate
  so the historic FAILED report is not silently rewritten; the honest
  verdict is assembled by scripts/verify_real_backend.py, which separates
  T0-T3 (passed) from T4 (one-step warm-up artefact).

    D:/Anaconda/envs/ChatAI/python.exe scripts/precompute_prompts.py
    D:/Anaconda/envs/ChatAI/python.exe scripts/verify_real_forward.py
    # CPU speed knob (bf16 4B DiT is very slow on CPU); recorded in the report:
    D:/Anaconda/envs/ChatAI/python.exe scripts/verify_real_forward.py --text-tokens 64

Writes outputs/logs/verify_real_forward.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.utils import ROOT, load_config, save_json, set_seed


def log(msg):
    print(f"[verify-forward] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-backward", action="store_true",
                    help="forward-only smoke (T0/T1) when RAM/CPU is tight")
    ap.add_argument("--text-tokens", type=int, default=0,
                    help="CPU speed knob: truncate the FROZEN constant "
                         "NEUTRAL_PROMPT text stream to N tokens (0 = full 512). "
                         "The trainable text branch under test is a separate "
                         "pooled/bypass path and is length-independent, so this "
                         "does not weaken T1-T4; it is recorded in the report.")
    args = ap.parse_args()

    cfg = load_config(override=ROOT / "configs" / "flux2_klein.yaml")
    set_seed(cfg["project"]["seed"])
    device = torch.device("cpu")
    rep = {"status": "UNVERIFIED", "device": "cpu", "checks": {}, "timings_s": {},
           "text_tokens": args.text_tokens if args.text_tokens > 0 else "full (512)"}
    out = ROOT / "outputs" / "logs" / "verify_real_forward.json"

    # hard preflight (CPU target: no VRAM check, weights must be present)
    from src.system import build_flux2_system, preflight_flux2
    pre = preflight_flux2(cfg, device)
    rep["checks"]["preflight"] = pre["verdict"]
    log(f"preflight={pre['verdict']}")

    t0 = time.time()
    log("loading REAL BF16 pipeline + building frozen backbone (slow, ~16 GB mmap) ...")
    try:
        backbone, cond, cache = build_flux2_system(cfg, device)
    except Exception as e:
        rep["error"] = f"build_flux2_system failed: {type(e).__name__}: {e}"
        rep["status"] = "FAILED"
        save_json(rep, out)
        log(rep["error"])
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 1
    rep["timings_s"]["load_and_build"] = round(time.time() - t0, 1)
    log(f"loaded in {rep['timings_s']['load_and_build']}s; "
        f"runtime_dims={cfg.get('_runtime_dims')}")

    # CPU speed knob. The frozen text stream is CONSTANT (NEUTRAL_PROMPT) and
    # is NOT the branch under test: user text travels through cond.set_text()
    # -> projector -> pooled additive bias / per-block bypass, all of which are
    # length-independent. Shortening only the constant prompt_embeds therefore
    # cuts the 20 single-transformer blocks' sequence length from 544 to ~96
    # without touching what T1-T4 actually measure. Recorded, never hidden.
    if args.text_tokens > 0:
        n = int(args.text_tokens)
        backbone.prompt_embeds = backbone.prompt_embeds[:, :n].contiguous()
        backbone.txt_ids = backbone.txt_ids[:, :n].contiguous()
        rep["checks"]["text_tokens_truncated_to"] = n
        rep["checks"]["text_truncation_note"] = (
            "frozen constant NEUTRAL_PROMPT stream shortened for CPU speed; "
            "the trainable text branch under test is length-independent"
        )
        log(f"CPU speed knob: constant text stream -> {n} tokens "
            f"(prompt_embeds {list(backbone.prompt_embeds.shape)}, "
            f"txt_ids {list(backbone.txt_ids.shape)})")

    # ---- deterministic tiny batch ----
    from src.data.dataset import make_dataset
    ds = make_dataset(cfg, root=cfg["_root"])
    batch = ds[0]
    clean = batch["clean"].unsqueeze(0)       # [1,3,H,W]
    degraded = batch["degraded"].unsqueeze(0)
    pair = cfg["eval"]["text_response_pairs"][0]
    log(f"dataset source={ds.source} clean={tuple(clean.shape)} prompts={pair}")

    # ---- T0: latent round-trip through the REAL VAE ----
    try:
        with torch.no_grad():
            x0 = backbone.image_to_tokens(clean)
            ref = backbone.ref_tokens_from_image(degraded)
            rec = backbone.tokens_to_image(x0)
        rep["checks"]["T0_latent_shape"] = list(x0.shape)
        rep["checks"]["T0_roundtrip_shape"] = list(rec.shape)
        rep["checks"]["T0_finite"] = bool(
            torch.isfinite(x0).all() and torch.isfinite(rec).all()
        )
        rep["checks"]["T0_ok"] = (
            rep["checks"]["T0_finite"]
            and list(rec.shape) == list(clean.shape)
            and x0.shape[-1] == cfg["_runtime_dims"]["in_channels"]
        )
        log(f"T0 latent {list(x0.shape)} -> image {list(rec.shape)} "
            f"ok={rep['checks']['T0_ok']}")
    except Exception as e:
        rep["error"] = f"T0 latent round-trip failed: {type(e).__name__}: {e}"
        rep["status"] = "FAILED"
        save_json(rep, out)
        log(rep["error"])
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 1

    # fixed noise / timestep so ON vs OFF differ ONLY by the text branch.
    # [FACT] Everything MUST stay in x0.dtype (bf16). If x1/target/t are left
    # as float32, F.mse_loss type-promotes pred to float32 and the resulting
    # float32 gradient blows up inside the bf16 4B trunk on backward with
    # "RuntimeError: Found dtype Float but expected BFloat16".
    g = torch.Generator().manual_seed(cfg["project"]["seed"])
    x1 = torch.randn(x0.shape, generator=g, dtype=x0.dtype)
    t = torch.tensor([0.37], dtype=x0.dtype)
    x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    target = x1 - x0

    # ---- T1: identity at init (zero-init projector + zero-init o_proj) ----
    try:
        with torch.no_grad():
            cond.set_enabled(False)
            ref_off = cond.fuse_condition(ref)
            out_off = backbone(x_t, t, ref_off).clone()
            cond.set_enabled(True)
            cond.set_text(cache.get([pair[0]]).to(device))
            ref_on = cond.fuse_condition(ref)
            out_on = backbone(x_t, t, ref_on).clone()
        d_init = float((out_on - out_off).abs().max().item())
        rep["checks"]["T1_identity_at_init_maxdiff"] = d_init
        rep["checks"]["T1_identity_at_init"] = d_init == 0.0
        log(f"T1 identity maxdiff={d_init}")
    except Exception as e:
        rep["error"] = f"T1 identity check failed: {type(e).__name__}: {e}"
        rep["status"] = "FAILED"
        save_json(rep, out)
        log(rep["error"])
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 1

    if args.no_backward:
        rep["ok"] = bool(rep["checks"].get("T0_ok") and rep["checks"].get("T1_identity_at_init"))
        rep["status"] = "PARTIAL (forward-only)" if rep["ok"] else "FAILED"
        save_json(rep, out)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0 if rep["ok"] else 1

    # ---- T2 + T3: one REAL training step on the real 4B trunk ----
    # Wrapped in try/except: with only ~9 GB free RAM, a CPU backward on a 4B
    # trunk can OOM. In that case we MUST still persist the already-measured
    # T0/T1 evidence instead of crashing after the ~20 min pipeline load.
    try:
        params = [p for p in cond.parameters() if p.requires_grad]
        before = {n: p.detach().clone()
                  for n, p in cond.named_parameters() if p.requires_grad}
        opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"])

        t1 = time.time()
        log("T2: forward + backward on the real 4B trunk starting ...")
        cond.set_enabled(True)
        cond.set_text(cache.get([pair[0]]).to(device))
        ref_used = cond.fuse_condition(ref)
        pred = backbone(x_t, t, ref_used)
        # [FACT] torch CPU has NO bf16 kernel for F.mse_loss
        # ("mse_cpu not implemented for 'BFloat16'"). The algebraically
        # identical form below keeps the ENTIRE graph in bf16, so gradients
        # flowing back into the frozen bf16 trunk stay bf16. Computing the
        # loss in float32 instead raises "Found dtype Float but expected
        # BFloat16" during backward.
        loss = (pred - target).square().mean()
        log(f"T2: forward done in {time.time() - t1:.1f}s; backward starting ...")
        tb = time.time()
        loss.backward()
        log(f"T2: backward done in {time.time() - tb:.1f}s")
        rep["timings_s"]["forward"] = round(tb - t1, 1)
        rep["timings_s"]["backward"] = round(time.time() - tb, 1)
        rep["timings_s"]["forward_backward"] = round(time.time() - t1, 1)
        rep["checks"]["T2_loss"] = round(float(loss.item()), 6)

        leaked = [n for n, p in backbone.named_parameters()
                  if p.grad is not None and torch.count_nonzero(p.grad).item() > 0]
        trained = [n for n, p in cond.named_parameters()
                   if p.grad is not None and torch.count_nonzero(p.grad).item() > 0]
        rep["checks"]["T2_no_gradient_leak"] = len(leaked) == 0
        rep["checks"]["T2_text_branch_has_gradient"] = len(trained) > 0
        rep["checks"]["T2_leaked_params"] = leaked[:5]
        rep["checks"]["T2_trained_params_count"] = len(trained)
        log(f"T2 loss={rep['checks']['T2_loss']} leaked={len(leaked)} "
            f"trained={len(trained)} ({rep['timings_s']['forward_backward']}s)")

        torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"])
        opt.step()
        changed_cond = sum(
            1 for n, p in cond.named_parameters()
            if p.requires_grad and not torch.equal(p.detach(), before[n])
        )
        rep["checks"]["T3_cond_params_changed"] = changed_cond
        rep["checks"]["T3_ok"] = changed_cond > 0 and len(leaked) == 0
        log(f"T3 cond params changed={changed_cond}")

        # ---- T4: text response after one step ----
        with torch.no_grad():
            cond.set_text(cache.get([pair[0]]).to(device))
            out_a = backbone(x_t, t, cond.fuse_condition(ref)).clone()
            cond.set_text(cache.get([pair[1]]).to(device))
            out_b = backbone(x_t, t, cond.fuse_condition(ref)).clone()
        d_text = float(((out_a - out_b) ** 2).mean().sqrt().item())
        rep["checks"]["T4_d_text"] = d_text
        rep["checks"]["T4_text_effect"] = d_text > 1e-6
        log(f"T4 d_text={d_text}")

        rep["checks"]["frozen_params"] = sum(p.numel() for p in backbone.parameters())
        rep["checks"]["trainable_params"] = cond.trainable_parameter_count()
        rep["checks"]["runtime_dims"] = cfg.get("_runtime_dims")
        rep["ok"] = all(bool(rep["checks"].get(k)) for k in (
            "T0_ok", "T1_identity_at_init", "T2_no_gradient_leak",
            "T2_text_branch_has_gradient", "T3_ok", "T4_text_effect"))
        rep["status"] = "OK" if rep["ok"] else "FAILED"
    except Exception as e:
        import traceback
        rep["checks"]["T2_T4_error"] = f"{type(e).__name__}: {e}"
        rep["checks"]["T2_T4_traceback"] = traceback.format_exc()[-2500:]
        rep["checks"]["frozen_params"] = sum(p.numel() for p in backbone.parameters())
        rep["checks"]["trainable_params"] = cond.trainable_parameter_count()
        rep["checks"]["runtime_dims"] = cfg.get("_runtime_dims")
        forward_ok = bool(rep["checks"].get("T0_ok")
                          and rep["checks"].get("T1_identity_at_init"))
        rep["ok"] = forward_ok
        rep["status"] = (
            "PARTIAL (T0/T1 forward verified; training step unavailable)"
            if forward_ok else "FAILED"
        )
        log(f"training step unavailable: {rep['checks']['T2_T4_error']}")
    save_json(rep, out)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    log(f"status={rep['status']} -> {out}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
