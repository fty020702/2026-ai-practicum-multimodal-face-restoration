from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.backends.toy import ToyMMDiT
from src.condition import TextConditionModule, freeze_backbone
from src.data.dataset import make_dataset
from src.data.text_features import HashTextEncoder, OfflineTextCache
from src.utils import (
    ROOT, count_params, ensure_dir, load_config, resolve_device, save_json, set_seed,
)


def build_system(cfg, device):
    """Freeze everything, then attach the single trainable text branch.

    Dispatch: the toy builder below is kept VERBATIM (all [RESULT] numbers
    depend on it). The real FLUX.2 [klein] path lives in src/system.py and is
    only reached when backend.name == "flux2_klein".
    """
    if cfg.get("backend", {}).get("name") == "flux2_klein":
        from src.system import build_flux2_system
        return build_flux2_system(cfg, device)

    backbone = ToyMMDiT(cfg).to(device)
    freeze_backbone(backbone)

    dim = cfg["backbone"]["dim"]
    block_dims = [dim] * cfg["backbone"]["num_blocks"]
    cond = TextConditionModule(cfg, block_dims).to(device)
    cond.attach(backbone.hook_targets)

    encoder = HashTextEncoder(
        dim=cfg["condition"]["text_in_dim"],
        num_tokens=cfg["backbone"]["num_text_tokens"],
    )
    cache = OfflineTextCache(ROOT, encoder)
    return backbone, cond, cache


def compute_loss(backbone, cond, cache, batch, device, text_enabled=True,
                 t_range=None):
    """One flow-matching training step (velocity prediction).

    t_range (2026-11-12, optional): (t_min, t_max). Default (0, 1) is a NO-OP
    so every existing [RESULT] stays bit-for-bit identical. Restricting t to
    the HIGH-NOISE end (e.g. (0.8, 1.0)) is needed for the one-image-many-
    targets experiment: when t is small, x_t = (1-t)x0 + t*x1 leaks x0 into
    the model input and the model can identify the target WITHOUT reading the
    text. At t -> 1, x_t ~= x1 is pure noise and the text becomes the only
    source of target identity. See docs/一图多目标数据集调研.md.
    """
    clean = batch["clean"].to(device)
    degraded = batch["degraded"].to(device)
    prompts = batch["prompt"]

    with torch.no_grad():
        x0 = backbone.image_to_tokens(clean)
        ref = backbone.ref_tokens_from_image(degraded)

    x1 = torch.randn_like(x0)
    B = x0.shape[0]
    t = torch.rand(B, device=device)
    if t_range is not None and tuple(t_range) != (0.0, 1.0):
        lo, hi = float(t_range[0]), float(t_range[1])
        t = lo + (hi - lo) * t
    x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    target = x1 - x0

    cond.set_enabled(text_enabled)
    if text_enabled:
        cond.set_text(cache.get(prompts).to(device))
    else:
        cond.clear_text()

    ref_used = cond.fuse_condition(ref)
    pred = backbone(x_t, t, ref_used)
    return F.mse_loss(pred, target)


def train(cfg, args):
    set_seed(cfg["project"]["seed"])
    device = resolve_device(args.device or cfg["train"]["device"])
    steps = int(args.steps or cfg["train"]["steps"])
    out_dir = ensure_dir(Path(cfg["_root"]) / cfg["train"]["out_dir"])
    if args.mode:
        cfg["condition"]["mode"] = args.mode

    backbone, cond, cache = build_system(cfg, device)
    ds = make_dataset(cfg, root=cfg["_root"])
    loader = DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True
    )

    opt = torch.optim.AdamW(
        [p for p in cond.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    print(f"[train] device={device} source={ds.source} pairs={len(ds)} mode={cfg['condition']['mode']}")
    print(f"[train] frozen backbone params={count_params(backbone, False):,}")
    print(f"[train] trainable text-branch params={cond.trainable_parameter_count():,}")
    print(f"[train] identity_at_init={cond.identity_check()}")

    t_range = (
        float(cfg["train"].get("t_min", 0.0)),
        float(cfg["train"].get("t_max", 1.0)),
    )
    if t_range != (0.0, 1.0):
        print(f"[train] HIGH-NOISE t sampling: t in [{t_range[0]}, {t_range[1]}] "
              "(text must identify the target; x_t carries almost no x0)")

    history, it, t0 = [], iter(loader), time.time()
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        opt.zero_grad(set_to_none=True)
        loss = compute_loss(backbone, cond, cache, batch, device,
                            text_enabled=True, t_range=t_range)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in cond.parameters() if p.requires_grad], cfg["train"]["grad_clip"]
        )
        opt.step()

        if step % cfg["train"]["log_every"] == 0 or step == 1:
            rec = {"step": step, "loss": round(float(loss.item()), 6),
                   "elapsed_s": round(time.time() - t0, 1)}
            history.append(rec)
            print(f"[train] step {step:5d}/{steps}  loss={rec['loss']:.6f}  {rec['elapsed_s']}s")
        if step % cfg["train"]["ckpt_every"] == 0 or step == steps:
            torch.save(
                {"step": step, "cond_state": cond.state_dict(),
                 "mode": cfg["condition"]["mode"], "config": cfg},
                out_dir / f"cond_step{step:06d}.pt",
            )

    save_json(
        {"mode": cfg["condition"]["mode"], "steps": steps, "device": str(device),
         "dataset_source": ds.source, "history": history},
        out_dir / "train_history.json",
    )
    print(f"[train] done. checkpoints -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--override", default=None, help="path to an override yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--mode", choices=["projector_only", "projector_plus_bypass"], default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(override=args.override)
    if args.out_dir:
        cfg["train"]["out_dir"] = args.out_dir
    train(cfg, args)


if __name__ == "__main__":
    main()
