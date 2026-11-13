from __future__ import annotations

"""Falsification probe for the 500 MB official triplet dataset.

Question: does the OFFICIAL InstructPix2Pix edit_prompt actually get USED by
the training objective? Unlike the old synthetic toy data (degradation type
<-> prompt 6-to-6), these edit prompts are open-ended ("Turn it into a photo",
"make her a scientist") and genuinely absent from the image. But the shard is
NOT one-input-many-targets (inspect_ip2p_shard.py: a_class_native=false), so
there is no per-input ambiguity floor. This script measures what actually
happens after training instead of assuming.

Probe: hold x0 / x1 / t / ref FIXED for each sample and vary ONLY the text.
    loss_own   : the sample's real edit_prompt
    loss_wrong : a DIFFERENT sample's real edit_prompt
    loss_off   : text branch disabled
    d_text_wrong / d_text_onoff : output-space distances

    D:/Anaconda/envs/ChatAI/python.exe scripts/diagnose_ip2p_text.py \
        --checkpoint outputs/checkpoints/ip2p_500m/cond_step001200.pt

Writes outputs/logs/ip2p_text_diagnosis.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.utils import ROOT, load_config, save_json, set_seed


def rms(a, b):
    return float(((a.float() - b.float()) ** 2).mean().sqrt().item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--t", type=float, default=0.95)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="ip2p_text_diagnosis.json")
    a = ap.parse_args()

    cfg = load_config(override=ROOT / "configs" / "ip2p_500m.yaml")
    set_seed(cfg["project"]["seed"])
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = ROOT / "outputs" / "logs" / a.out

    from src.data.dataset import make_dataset
    from src.train import build_system
    ds = make_dataset(cfg, root=ROOT)
    backbone, cond, cache = build_system(cfg, device)

    ck = a.checkpoint
    if ck is None:
        cks = sorted((ROOT / "outputs" / "checkpoints" / "ip2p_500m")
                     .glob("cond_step*.pt"))
        if not cks:
            rep = {"status": "FAILED", "error": "no trained checkpoint yet"}
            save_json(rep, out)
            return 1
        ck = cks[-1]
    p = Path(ck)
    if not p.is_absolute():
        p = ROOT / p
    state = torch.load(p, map_location="cpu", weights_only=False)
    cond.load_state_dict(state["cond_state"], strict=True)

    rep = {
        "status": "UNVERIFIED",
        "checkpoint": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
        "dataset": "timbrooks/instructpix2pix-clip-filtered shard 0 (499.83 MB)",
        "rows_used": len(ds),
        "distinct_prompts": len(ds.all_prompts()),
        "a_class_native": False,
        "timestep_t": a.t,
        "method": "hold x0/x1/t/ref fixed, vary ONLY the text",
        "device": str(device),
    }

    n = min(a.samples, len(ds))
    t = torch.tensor([a.t], dtype=torch.float32, device=device)
    rows = []
    with torch.no_grad():
        for i in range(n):
            b = ds[i]
            other = ds[(i + 1) % len(ds)]
            x0 = backbone.image_to_tokens(b["clean"].unsqueeze(0).to(device))
            ref = backbone.ref_tokens_from_image(
                b["degraded"].unsqueeze(0).to(device))
            g = torch.Generator().manual_seed(
                int(cfg["project"]["seed"]) + i * 7919)
            x1 = torch.randn(x0.shape, generator=g, dtype=x0.dtype).to(device)
            x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
            target = x1 - x0

            def run(text):
                if text is None:
                    cond.set_enabled(False)
                    cond.clear_text()
                    ref_u = ref
                else:
                    cond.set_enabled(True)
                    cond.set_text(cache.get([text]).to(device))
                    ref_u = cond.fuse_condition(ref)
                pred = backbone(x_t, t, ref_u)
                return pred.clone(), float(
                    (pred - target).square().mean().item())

            po, lo = run(b["prompt"])
            pw, lw = run(other["prompt"])
            pf, lf = run(None)
            rows.append({
                "i": i, "own_prompt": b["prompt"],
                "wrong_prompt": other["prompt"],
                "loss_own": lo, "loss_wrong": lw, "loss_off": lf,
                "wrong_penalty": lw - lo,
                "d_text_wrong": rms(po, pw),
                "d_text_onoff": rms(po, pf),
            })

    m = {
        "loss_own": sum(r["loss_own"] for r in rows) / n,
        "loss_wrong": sum(r["loss_wrong"] for r in rows) / n,
        "loss_off": sum(r["loss_off"] for r in rows) / n,
        "wrong_penalty": sum(r["wrong_penalty"] for r in rows) / n,
        "d_text_wrong": sum(r["d_text_wrong"] for r in rows) / n,
        "d_text_onoff": sum(r["d_text_onoff"] for r in rows) / n,
    }
    m["wrong_penalty_relative"] = m["wrong_penalty"] / max(m["loss_own"], 1e-9)
    m["style_selectivity"] = m["d_text_wrong"] / max(m["d_text_onoff"], 1e-12)
    m["text_needed_by_loss"] = m["wrong_penalty_relative"] >= 0.01
    rep["means"] = {k: (round(v, 8) if isinstance(v, float) else v)
                    for k, v in m.items()}
    rep["per_sample"] = [
        {k: (round(v, 8) if isinstance(v, float) else v)
         for k, v in r.items()} for r in rows
    ]
    rep["verdict"] = (
        "TEXT-SPECIFIC-RESPONSE: wrong edit_prompt measurably increases loss"
        if m["text_needed_by_loss"] else
        "NO TEXT-SPECIFIC RESPONSE: wrong edit_prompt barely changes loss; "
        "consistent with a single-target-per-input dataset (no ambiguity "
        "floor) plus a hash text encoder that cannot generalise semantics"
    )
    rep["honesty"] = [
        "[FACT] 1,195 distinct inputs, 1 target each -> no per-input "
        "Var_k(y_k) floor. This is NOT the A-class design in §8.2.",
        "[FACT] toy trunk is randomly initialised and frozen; HashTextEncoder "
        "is not semantic.",
        "[INFERENCE] even a positive result here would show text-feature "
        "sensitivity, not language understanding or restoration quality.",
    ]
    rep["status"] = "OK"
    save_json(rep, out)
    print(json.dumps({"means": rep["means"], "verdict": rep["verdict"]},
                     ensure_ascii=False, indent=2))
    print(f"[ip2p-diag] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
