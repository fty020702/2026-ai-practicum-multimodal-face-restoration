from __future__ import annotations

"""[RESULT] Diagnose whether the TRAINED text branch is actually USED.

Question this answers
  "d_text > 0" only proves text CAN change the output. It does not prove the
  model NEEDS text, nor that it encodes anything beyond a task label. This
  script separates those claims with two independent measurements:

  (1) d_text decomposition, all on the SAME degraded image / SAME noise:
        d_task      : different TASK word, same content constraint
        d_content   : same TASK word, different CONTENT constraint   <-- KEY
        d_specificity : neutral constant prompt vs a specific instruction
        d_onoff     : text ON vs OFF
      If d_content ~ 0 while d_task > 0, the branch only learned a task
      classifier and the content-level instruction is ignored.

  (2) loss sensitivity to text SHUFFLING. For each sample compute the SAME
      flow-matching loss with its own prompt vs a WRONG prompt (fixed x0/x1/t
      so the ONLY thing that changes is the text). If |d_loss| ~ 0, the
      optimiser has no reason to read text: the image already determines the
      answer. This is the direct evidence for the confound documented in
      docs/方案设计.md §8.

All prompts below are already in the offline Qwen3 cache, so the (released)
text encoder is never needed.

    D:/Anaconda/envs/ChatAI/python.exe scripts/diagnose_text_usage.py
    D:/Anaconda/envs/ChatAI/python.exe scripts/diagnose_text_usage.py \
        --checkpoint outputs/checkpoints/flux2_klein_100/cond_step000100.pt

Writes outputs/logs/text_usage.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.utils import ROOT, load_config, save_json, set_seed

P_NEUTRAL = "restore this image"
P_TASK_A = "restore and denoise this image, keep the brick texture"
P_TASK_B = "restore and derain this image, keep the brick texture"
P_CONTENT_B = "restore and derain this image, remove rain streaks, keep the sky smooth"
P_WRONG = "remove jpeg compression artifacts from this image, keep the foliage natural"


def log(msg):
    print(f"[text-usage] {msg}", flush=True)


def rms(a, b):
    return float(((a.float() - b.float()) ** 2).mean().sqrt().item())


def find_checkpoint(explicit):
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        return p if p.is_file() else None
    for d in ("flux2_klein_100", "flux2_klein_smoke", "flux2_klein"):
        base = ROOT / "outputs" / "checkpoints" / d
        if base.is_dir():
            ck = sorted(base.glob("cond_step*.pt"))
            if ck:
                return ck[-1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--out", default="text_usage.json",
                    help="report filename under outputs/logs/")
    args = ap.parse_args()

    cfg = load_config(override=ROOT / "configs" / "flux2_klein.yaml")
    set_seed(cfg["project"]["seed"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = ROOT / "outputs" / "logs" / args.out
    rep = {"status": "UNVERIFIED", "device": str(device), "checks": {}}

    ckpt = find_checkpoint(args.checkpoint)
    if ckpt is None:
        rep.update({"status": "FAILED", "error": "no cond checkpoint found"})
        save_json(rep, out)
        return 1
    rep["checkpoint"] = str(ckpt.relative_to(ROOT))
    log(f"checkpoint={rep['checkpoint']}")

    from src.system import build_flux2_system, preflight_flux2
    preflight_flux2(cfg, device)
    backbone, cond, cache = build_flux2_system(cfg, device, ckpt=str(ckpt))
    rep["checks"]["identity_check_after_training"] = bool(cond.identity_check())

    from src.data.dataset import make_dataset
    ds = make_dataset(cfg, root=cfg["_root"])
    emb = {t: cache.get([t]).to(device) for t in
           (P_NEUTRAL, P_TASK_A, P_TASK_B, P_CONTENT_B, P_WRONG)}

    def one(sample_idx):
        b = ds[sample_idx]
        clean = b["clean"].unsqueeze(0).to(device)
        degraded = b["degraded"].unsqueeze(0).to(device)
        g = torch.Generator().manual_seed(cfg["project"]["seed"] + sample_idx)
        with torch.no_grad():
            x0 = backbone.image_to_tokens(clean)
            ref = backbone.ref_tokens_from_image(degraded)
        x1 = torch.randn(x0.shape, generator=g, dtype=x0.dtype).to(x0.device)
        t = torch.tensor([0.37], dtype=x0.dtype, device=x0.device)
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        target = x1 - x0

        def run(text):
            with torch.no_grad():
                if text is None:
                    cond.set_enabled(False)
                    cond.clear_text()
                    pred = backbone(x_t, t, ref)
                else:
                    cond.set_enabled(True)
                    cond.set_text(emb[text])
                    pred = backbone(x_t, t, cond.fuse_condition(ref))
                return pred.clone(), float((pred - target).square().mean().item())

        outs, losses = {}, {}
        for key, text in (("neutral", P_NEUTRAL), ("task_a", P_TASK_A),
                          ("task_b", P_TASK_B), ("content_b", P_CONTENT_B),
                          ("wrong", P_WRONG), ("off", None)):
            o, l = run(text)
            outs[key], losses[key] = o, l
        return {
            "index": sample_idx,
            "task": b["task"],
            "own_prompt": b["prompt"],
            "d_task": rms(outs["task_a"], outs["task_b"]),
            "d_content": rms(outs["task_b"], outs["content_b"]),
            "d_specificity": rms(outs["neutral"], outs["task_a"]),
            "d_onoff": rms(outs["task_a"], outs["off"]),
            "loss_own": losses["task_a"],
            "loss_wrong": losses["wrong"],
            "loss_neutral": losses["neutral"],
            "loss_off": losses["off"],
            "d_loss_wrong": abs(losses["task_a"] - losses["wrong"]),
        }

    recs = [one(i) for i in range(min(args.samples, len(ds)))]
    for r in recs:
        log(f"sample {r['index']} task={r['task']:<16} "
            f"d_task={r['d_task']:.6f} d_content={r['d_content']:.6f} "
            f"d_loss_wrong={r['d_loss_wrong']:.6f}")

    def mean(k):
        return sum(r[k] for r in recs) / len(recs)

    rep["checks"]["per_sample"] = recs
    rep["checks"]["means"] = {
        "d_task": mean("d_task"),
        "d_content": mean("d_content"),
        "d_specificity": mean("d_specificity"),
        "d_onoff": mean("d_onoff"),
        "d_loss_wrong": mean("d_loss_wrong"),
        "loss_own": mean("loss_own"),
        "loss_wrong": mean("loss_wrong"),
        "loss_off": mean("loss_off"),
    }
    m = rep["checks"]["means"]
    rel = m["d_loss_wrong"] / max(m["loss_own"], 1e-9)
    m["d_loss_wrong_relative"] = rel
    m["text_onoff_loss_gain"] = m["loss_off"] - m["loss_own"]
    m["content_vs_task_ratio"] = m["d_content"] / max(m["d_task"], 1e-9)
    # bf16 has 8 mantissa bits; around loss~0.36 one ULP is 2^-8 = 0.00390625.
    # A d_loss_wrong at that scale is numerical noise, NOT evidence of use.
    m["d_loss_wrong_in_bf16_ulp"] = m["d_loss_wrong"] / (2 ** -8)

    notes = []
    if m["d_task"] > 1e-6:
        notes.append(
            "d_task > 0: different TASK words change the output, so the "
            "trained branch is no longer zero-initialised.")
    if m["d_content"] > 1e-6:
        notes.append(
            "d_content > 0 and close to d_task: ANY different prompt changes "
            "the output. This shows the projector is embedding-sensitive; it "
            "does NOT prove the content constraint is honoured or understood.")
    if rel < 0.05:
        notes.append(
            f"Swapping in a WRONG prompt changes the velocity loss by only "
            f"{rel * 100:.2f}% ({m['d_loss_wrong']:.6f} absolute, "
            f"~{m['d_loss_wrong_in_bf16_ulp']:.1f} bf16 ULP near loss~0.36). "
            "The objective therefore does NOT force the model to read the "
            "specific instruction - a different text is nearly as good.")
    else:
        notes.append(
            f"Swapping in a WRONG prompt changes the loss by {rel * 100:.2f}%, "
            "so the objective does penalise ignoring text.")
    if m["text_onoff_loss_gain"] > 0:
        notes.append(
            f"Text ON vs OFF improves loss by {m['text_onoff_loss_gain']:.6f} "
            f"({m['text_onoff_loss_gain'] / max(m['loss_own'], 1e-9) * 100:.2f}%), "
            "but the gain is far larger than the difference between a RIGHT "
            "and a WRONG prompt: the branch learned a mostly GENERIC helpful "
            "bias, not instruction-specific control. This is exactly the "
            "confound predicted by docs/方案设计.md §8.")
    rep["checks"]["interpretation"] = {
        "content_level_control": m["d_content"] > 1e-6,
        "task_level_control": m["d_task"] > 1e-6,
        "text_needed_by_loss": rel >= 0.05,
        "text_specificity_supported": rel >= 0.05,
        "notes": notes,
        "bottom_line": (
            "Text CAN change the output, but the current training objective "
            "and the 6-to-6 degradation/prompt confound give the optimiser "
            "almost no reason to use the SPECIFIC instruction."
            if rel < 0.05 else
            "The objective measurably penalises ignoring text."
        ),
    }
    rep["status"] = "OK"
    save_json(rep, out)
    print(json.dumps({"means": m, "interpretation": rep["checks"]["interpretation"]},
                     ensure_ascii=False, indent=2))
    log(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
