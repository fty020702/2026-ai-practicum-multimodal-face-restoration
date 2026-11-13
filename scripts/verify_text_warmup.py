from __future__ import annotations

"""[RESULT] Diagnose why T4 d_text=0 after a SINGLE training step.

WHY THIS EXISTS
  On 2026-11-07 the real-weight CPU verification
  (scripts/verify_real_forward.py) passed T0/T1/T2/T3 but reported T4
  d_text=0.0. Two candidate explanations were considered:

    (A) bf16 ULP (~0.0078 at |x|=1) swallows the one-step text bias (~lr=5e-5)
    (B) the text branch has not "opened" after ONE step because of the
        zero-initialisation gradient path.

  Source inspection of src/condition/module.py settles it as (B):
    * In `projector_plus_bypass` mode `fuse_condition()` returns cond_ref
      UNCHANGED; user text travels ONLY through the per-block bypass hooks.
    * Both the projector's last layer AND every bypass `o_proj` are
      zero-initialised (needed for identity-at-init, T1 maxdiff=0.0).
    * Step 1: dL/d(o_proj.weight) is non-zero, but
      dL/d(k_proj, v_proj) = grad @ o_proj.weight = 0. So ONLY o_proj
      updates. The real run measured EXACTLY 10 trained tensors
      = 5 double blocks x {o_proj.weight, o_proj.bias}.
    * Therefore `projector(raw)` is still exactly 0 after step 1, so K/V are
      prompt-independent and the trunk output is bit-identical for any two
      prompts -> d_text = 0.0.
    * From step 2 onward o_proj.weight != 0, so gradients finally reach k/v
      and the projector, and the text signal opens up.

  This script reproduces that on a TINY float32 stub (NO 16 GB pipeline, NO
  4B DiT): seconds instead of 35 minutes. It is a STUB [RESULT-on-stub], not
  a claim about the real trunk; it only demonstrates the gradient dynamics
  that explain the real run's T4.

    D:/Anaconda/envs/ChatAI/python.exe scripts/verify_text_warmup.py

Writes outputs/logs/text_warmup.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from src.condition.module import TextConditionModule
from src.utils import ROOT, save_json


class _StubBlock(nn.Module):
    """Mimics FLUX.2 double blocks returning (txt, img); img_index = 1."""

    def forward(self, x):
        return x, x


def _cfg():
    return {"condition": {
        "mode": "projector_plus_bypass",
        "text_in_dim": 32,
        "hidden_dim": 64,
        "cond_dim": 64,
        "ref_bias_out_dim": 64,
        "projector_layers": 2,
        "projector_dropout": 0.0,
        "zero_init_last": True,
        "bypass_heads": 4,
        "bypass_scale": 1.0,
    }}


def run(steps: int = 6, lr: float = 5e-5):
    torch.manual_seed(20261107)
    cond = TextConditionModule(_cfg(), block_dims=[64])
    blk = _StubBlock()
    cond.attach([blk], img_index=1)
    opt = torch.optim.AdamW(
        [p for p in cond.parameters() if p.requires_grad], lr=lr
    )

    emb_a = torch.randn(1, 8, 32)
    emb_b = torch.randn(1, 8, 32)
    x = torch.randn(1, 12, 64)
    target = torch.randn(1, 12, 64)

    rec = []
    for step in range(1, steps + 1):
        cond.set_text(emb_a)
        out = blk(x)[1]                       # hook injects bypass delta
        loss = (out - target).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            n_grad = sum(
                1 for p in cond.parameters()
                if p.grad is not None and torch.count_nonzero(p.grad).item() > 0
            )
        opt.step()

        with torch.no_grad():
            cond.set_text(emb_a)
            c_a, d_a = cond._cond.clone(), blk(x)[1] - x
            cond.set_text(emb_b)
            c_b, d_b = cond._cond.clone(), blk(x)[1] - x
            d_cond = float(((c_a - c_b) ** 2).mean().sqrt().item())
            d_delta = float(((d_a - d_b) ** 2).mean().sqrt().item())
            proj_out_norm = float(c_a.abs().max().item())
        rec.append({
            "step": step,
            "params_with_nonzero_grad": n_grad,
            "projector_out_max_abs": round(proj_out_norm, 8),
            "d_text_condition": round(d_cond, 8),
            "d_bypass_delta": round(d_delta, 8),
        })
        print(f"[warmup] step={step} params_with_grad={n_grad} "
              f"proj_out={proj_out_norm:.8f} d_cond={d_cond:.8f} "
              f"d_delta={d_delta:.8f}", flush=True)
    return rec


def main():
    rec = run()
    step1, last = rec[0], rec[-1]
    rep = {
        "status": "OK",
        "what": "float32 STUB reproduction of the zero-init warm-up dynamics",
        "not_a_claim": "This is a tiny stub, NOT the real 4B trunk. It only "
                       "explains the gradient dynamics behind the real run's T4.",
        "measured_real_run": {
            "T2_trained_params_count": 10,
            "interpretation": "5 double blocks x {o_proj.weight, o_proj.bias}; "
                              "projector and k/v/q have NO gradient on step 1",
            "T3_cond_params_changed": 10,
            "T4_d_text": 0.0,
        },
        "stub_steps": rec,
        "conclusion": {
            "step1_text_effect": step1["d_bypass_delta"] > 0,
            "after_multi_step_text_effect": last["d_bypass_delta"] > 0,
            "hypothesis_confirmed": (
                step1["d_bypass_delta"] == 0.0
                and last["d_bypass_delta"] > 0.0
            ),
            "explanation": (
                "T4=0 on the real CPU run is a ONE-STEP warm-up artefact of "
                "zero-initialised o_proj, NOT an architecture defect and NOT "
                "just bf16 rounding. Multi-step training (or GPU training) "
                "opens the projector and makes d_text non-zero. The toy "
                "float32 smoke_test already shows d_text 0.0 -> 0.003216 / "
                "0.000563 after training."
            ),
        },
    }
    out = ROOT / "outputs" / "logs" / "text_warmup.json"
    save_json(rep, out)
    print(json.dumps(rep["conclusion"], ensure_ascii=False, indent=2))
    print(f"[warmup] -> {out}")
    return 0 if rep["conclusion"]["hypothesis_confirmed"] else 1


if __name__ == "__main__":
    sys.exit(main())
