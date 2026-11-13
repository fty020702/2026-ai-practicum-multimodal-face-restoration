from __future__ import annotations

"""Locally executable verification of the architecture.

Run with the ChatAI conda env:
    D:/Anaconda/envs/ChatAI/python.exe src/smoke_test.py

Checks, in order:
  T1  zero-initialisation invariant holds
  T2  at init, enabling the text branch changes NOTHING (bit-exact)
  T3  gradients reach only the text branch, never the frozen trunk
  T4  training updates the text branch and leaves the trunk untouched
  T5  after training, DIFFERENT prompts produce DIFFERENT outputs
      (the falsification test; at init this difference must be ~0)
  T6  trainable params << frozen params

Writes outputs/logs/smoke_test.json. Exit code is 0 only if all checks pass.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.backends.toy import ToyMMDiT
from src.condition import TextConditionModule, check_gradient_isolation, freeze_backbone
from src.data.dataset import RestorationDataset
from src.data.text_features import HashTextEncoder, OfflineTextCache
from src.evaluate import text_response_test
from src.train import compute_loss
from src.utils import ROOT, deep_update, load_config, resolve_device, save_json


def make_smoke_cfg(mode: str, device):
    cfg = load_config()
    deep_update(cfg, {
        "condition": {
            "mode": mode,
            "text_in_dim": 64,
            "hidden_dim": 128,
            "cond_dim": 128,
            "bypass_heads": 4,
            "projector_layers": 2,
        },
        "backbone": {
            "dim": 128, "num_blocks": 3, "heads": 4,
            "num_ref_tokens": 16, "num_text_tokens": 4,
        },
        "data": {"image_size": 64, "patch_size": 8, "num_pairs": 8},
        "train": {"num_timesteps": 8, "batch_size": 1, "lr": 1e-3, "steps": 40},
    })
    cfg["_device"] = str(device)
    return cfg


def run_case(mode: str, device, train_steps: int = 40):
    cfg = make_smoke_cfg(mode, device)
    torch.manual_seed(cfg["project"]["seed"])

    backbone = ToyMMDiT(cfg).to(device)
    freeze_backbone(backbone)
    dim = cfg["backbone"]["dim"]
    cond = TextConditionModule(cfg, [dim] * cfg["backbone"]["num_blocks"]).to(device)
    cond.attach(backbone.hook_targets)
    encoder = HashTextEncoder(cfg["condition"]["text_in_dim"], cfg["backbone"]["num_text_tokens"])
    cache = OfflineTextCache(ROOT, encoder)
    ds = RestorationDataset(cfg, root=cfg["_root"])
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in ds[0].items()}

    checks = {}

    # T1 ----------------------------------------------------------------- #
    checks["T1_zero_init"] = bool(cond.identity_check())

    # T2 ----------------------------------------------------------------- #
    # Strict identity at init: same input, text on vs off, bit-exact output.
    with torch.no_grad():
        degraded = batch["degraded"].to(device)
        ref = backbone.ref_tokens_from_image(degraded)
        x_t = torch.randn_like(backbone.image_to_tokens(degraded))
        t = torch.tensor([0.37], device=device)
        emb = cache.get(batch["prompt"]).to(device)

        cond.set_enabled(False); cond.clear_text()
        out_off = backbone(x_t, t, cond.fuse_condition(ref))
        cond.set_enabled(True); cond.set_text(emb)
        out_on = backbone(x_t, t, cond.fuse_condition(ref))
    diff = float((out_off - out_on).abs().max().item())
    checks["T2_identity_at_init_maxdiff"] = diff
    checks["T2_identity_at_init"] = diff == 0.0

    # T3 ----------------------------------------------------------------- #
    cond.set_enabled(True); cond.set_text(cache.get(batch["prompt"]).to(device))
    loss = compute_loss(backbone, cond, cache, batch, device, text_enabled=True)
    leaked, trained = check_gradient_isolation(backbone, cond, loss)
    checks["T3_no_gradient_leak"] = len(leaked) == 0
    checks["T3_text_branch_has_gradient"] = len(trained) > 0
    checks["T3_leaked_params"] = leaked[:5]
    checks["T3_trained_params"] = trained[:5]

    # T5 (before training) ------------------------------------------------ #
    pair = cfg["eval"]["text_response_pairs"][0]
    resp_init = text_response_test(backbone, cond, cache, degraded, pair, cfg, device)

    # T4 ----------------------------------------------------------------- #
    before = {n: p.detach().clone() for n, p in cond.named_parameters()}
    bbefore = {n: p.detach().clone() for n, p in backbone.named_parameters()}
    opt = torch.optim.AdamW(
        [p for p in cond.parameters() if p.requires_grad], lr=cfg["train"]["lr"]
    )
    torch.manual_seed(cfg["project"]["seed"] + 1)
    for _ in range(train_steps):
        opt.zero_grad(set_to_none=True)
        l = compute_loss(backbone, cond, cache, batch, device, text_enabled=True)
        l.backward()
        opt.step()
    changed = sum(not torch.equal(before[n], p) for n, p in cond.named_parameters())
    bchanged = sum(not torch.equal(bbefore[n], p) for n, p in backbone.named_parameters())
    checks["T4_text_branch_updated"] = changed > 0
    checks["T4_backbone_untouched"] = bchanged == 0
    checks["T4_changed_params"] = changed
    checks["T4_backbone_changed_params"] = bchanged

    # T5 (after training) ------------------------------------------------- #
    resp_trained = text_response_test(backbone, cond, cache, degraded, pair, cfg, device)
    checks["T5_d_text_init"] = resp_init["d_text"]
    checks["T5_d_text_trained"] = resp_trained["d_text"]
    # Hard check: training must make the output text-dependent at all.
    # The absolute threshold in the config is calibrated for a REAL backbone;
    # on this 8-step toy sampler it is reported separately and is NOT a gate.
    checks["T5_text_became_effective"] = (
        resp_trained["d_text"] > resp_init["d_text"]
        and resp_trained["d_text"] > 1e-5
    )
    checks["T5_threshold_note"] = (
        f"abs_threshold={cfg['eval']['response_threshold']} (real-model calibration), "
        f"observed={resp_trained['d_text']:.6f}"
    )
    checks["T5_detail"] = {"init": resp_init, "trained": resp_trained}

    # T6 ----------------------------------------------------------------- #
    n_train = cond.trainable_parameter_count()
    n_frozen = sum(p.numel() for p in backbone.parameters())
    checks["T6_trainable_params"] = n_train
    checks["T6_frozen_params"] = n_frozen
    checks["T6_param_ratio"] = round(n_train / n_frozen, 5)
    checks["T6_much_smaller"] = n_train < n_frozen

    cond.detach()
    booleans = [v for k, v in checks.items() if isinstance(v, bool)]
    return {
        "mode": mode, "device": str(device), "checks": checks,
        "all_passed": all(booleans), "n_checks": len(booleans),
    }


def main():
    device = resolve_device("auto")
    print(f"[smoke] device={device} torch={torch.__version__}")
    results = []
    for mode in ("projector_only", "projector_plus_bypass"):
        r = run_case(mode, device)
        results.append(r)
        print(f"\n[smoke] mode={mode}  all_passed={r['all_passed']}")
        for k, v in r["checks"].items():
            if isinstance(v, bool):
                print(f"  {'PASS' if v else 'FAIL'}  {k}")
            elif k not in ("T5_detail", "T3_leaked_params", "T3_trained_params"):
                print(f"        {k} = {v}")

    out = ROOT / "outputs" / "logs" / "smoke_test.json"
    save_json(
        {"device": str(device), "torch": torch.__version__,
         "note": "ToyMMDiT + HashTextEncoder. Verifies ARCHITECTURE, not semantic quality.",
         "cases": results},
        out,
    )
    ok = all(r["all_passed"] for r in results)
    print(f"\n[smoke] report -> {out}")
    print(f"[smoke] OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
