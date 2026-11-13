from __future__ import annotations

"""[RESULT] Quantify whether the TEXT is mathematically NECESSARY.

The user's proposed fix (2026-11-12):
    "text should carry information the image does not have;
     one low-quality image + several different high-quality targets,
     each paired with a different text"

This script turns that into falsifiable numbers, on the TOY trunk (fast) or
on any cond checkpoint trained with the same `condition` config.

Data-level gap (no training needed)
-----------------------------------
For one shared degraded x with K different clean targets y_k:

    ȳ          = mean_k image_to_tokens(y_k)
    gap        = mean_k || image_to_tokens(y_k) − ȳ ||²   = Var_k(y_k)

Why this is a real MSE floor and not hand-waving
-----------------------------------------------
At t → 1, x_t = (1−t)x0 + t x1 ≈ x1 (pure noise), so the model CANNOT read
x0 back out of its input. It must output  v ≈ x1 − E[x0 | ref, text]  and
    loss = || x0_k − E[x0 | ref, text] ||²
With no text, E[x0] = ȳ and loss = Var_k(y_k) > 0. With the correct text,
E[x0] = x0_k and loss → 0. So a plain velocity-MSE objective already punishes
a text-ignoring model. (At small t the flow-matching x_t does carry partial x0
information; therefore all probes below default to t = 0.999. This caveat is
recorded in the report, see `x_t_leakage_note`.)

Trained probe
-------------
For each (group, target k) hold x0 / x1 / t / ref FIXED and vary ONLY the text:

    loss_own    : the prompt of target k
    loss_wrong  : the prompt of a DIFFERENT target of the SAME group
    loss_off    : text branch disabled

`wrong_text_penalty = loss_wrong − loss_own` should be ~0 on single-target
data (measured earlier: 1 bf16 ULP / 1.08%) and clearly > 0 on this dataset.

    D:/Anaconda/envs/ChatAI/python.exe scripts/diagnose_ambiguity.py
    D:/Anaconda/envs/ChatAI/python.exe scripts/diagnose_ambiguity.py \
        --checkpoint outputs/checkpoints/old_photo/cond_step000300.pt \
        --checkpoint outputs/checkpoints/multi_target/cond_step000400.pt \
        --out ambiguity_compare.json

Writes outputs/logs/ambiguity.json (or --out).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.utils import ROOT, load_config, save_json, set_seed


def log(m: str) -> None:
    print(f"[ambiguity] {m}", flush=True)


def rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.float() - b.float()) ** 2).mean().sqrt().item())


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--override", default="configs/multi_target.yaml")
    ap.add_argument("--checkpoint", action="append", default=[],
                    help="cond checkpoint(s) to probe; repeatable")
    ap.add_argument("--groups", type=int, default=6)
    ap.add_argument("--t", type=float, default=0.999,
                    help="flow-matching timestep; 0.999 = pure-noise end")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="ambiguity.json")
    args = ap.parse_args()

    ov = resolve(args.override)
    cfg = load_config(override=ov)
    set_seed(cfg["project"]["seed"])
    device = torch.device(args.device)
    out = ROOT / "outputs" / "logs" / args.out

    from src.data.dataset import make_dataset
    from src.train import build_system

    ds = make_dataset(cfg, root=cfg["_root"])
    if not hasattr(ds, "get_group"):
        raise RuntimeError(
            "[ambiguity] this diagnostic requires MultiTargetDataset "
            "(data.source: multi_target). Use configs/multi_target.yaml."
        )
    backbone, cond, cache = build_system(cfg, device)
    K = ds.k
    n_groups = min(args.groups, len(ds.records))
    t_val = torch.tensor([float(args.t)], dtype=torch.float32, device=device)

    rep = {
        "status": "UNVERIFIED",
        "override": str(ov.relative_to(ROOT)) if ov.is_relative_to(ROOT) else str(ov),
        "device": str(device),
        "dataset_source": ds.source,
        "n_groups_probed": n_groups,
        "K_targets": K,
        "target_ids": ds.target_ids,
        "timestep_t": float(args.t),
        "x_t_leakage_note": (
            "At small t, x_t = (1-t)x0 + t x1 leaks x0 into the model input, "
            "which weakens the no-text MSE floor. All probes use t=0.999 "
            "(x_t ~= x1 pure noise) so the gap is a valid floor. Training "
            "itself samples t uniformly in [0,1]."
        ),
        "data_gap": {},
        "text_features": {},
        "checkpoints": {},
    }

    # ---------------------------------------------------------------- #
    # (1) DATA-LEVEL ambiguity gap: no training needed.
    # ---------------------------------------------------------------- #
    floor_vals, pairwise = [], []
    txt_gaps = []
    for i in range(n_groups):
        grp = ds.get_group(i)
        with torch.no_grad():
            x0s = [backbone.image_to_tokens(g["clean"].unsqueeze(0).to(device))
                   for g in grp]
        stack = torch.stack(x0s, dim=0)                    # [K, 1, N, D]
        mean = stack.mean(0)
        floor_vals.append(float((stack - mean).square().mean().item()))
        for a in range(K):
            for b in range(a + 1, K):
                pairwise.append(rms(x0s[a], x0s[b]))

        with torch.no_grad():
            embs = cache.get([g["prompt"] for g in grp])   # [K, L, D]
        for a in range(K):
            for b in range(a + 1, K):
                txt_gaps.append(rms(embs[a], embs[b]))

    rep["data_gap"] = {
        "definition": "mean_k || image_to_tokens(y_k) - mean_k(...) ||^2 "
                      "= Var_k(y_k) in token space",
        "no_text_mse_floor_mean": round(sum(floor_vals) / len(floor_vals), 8),
        "no_text_mse_floor_min": round(min(floor_vals), 8),
        "no_text_mse_floor_max": round(max(floor_vals), 8),
        "pairwise_target_rms_mean": round(sum(pairwise) / len(pairwise), 8),
        "positive": min(floor_vals) > 0,
    }
    rep["text_features"] = {
        "pairwise_prompt_rms_mean": round(sum(txt_gaps) / len(txt_gaps), 8),
        "pairwise_prompt_rms_min": round(min(txt_gaps), 8),
        "distinguishable": min(txt_gaps) > 1e-8,
        "encoder_note": "toy HashTextEncoder reads the FIRST 8 tokens; if "
                        "all prompts shared a preamble this would be 0",
    }
    log(f"Var_k(y_k) = {rep['data_gap']['no_text_mse_floor_mean']:.6f} "
        f"(>0 => text is necessary); "
        f"prompt feature gap = {rep['text_features']['pairwise_prompt_rms_mean']:.6f}")

    # ---------------------------------------------------------------- #
    # (2) TRAINED probes: hold everything fixed, vary ONLY the text.
    # ---------------------------------------------------------------- #
    # Precompute the fixed probes once, shared by all checkpoints.
    probes = []
    for i in range(n_groups):
        grp = ds.get_group(i)
        with torch.no_grad():
            ref = backbone.ref_tokens_from_image(
                grp[0]["degraded"].unsqueeze(0).to(device)
            )
        for k in range(K):
            with torch.no_grad():
                x0 = backbone.image_to_tokens(
                    grp[k]["clean"].unsqueeze(0).to(device)
                )
            g = torch.Generator().manual_seed(
                int(cfg["project"]["seed"]) + i * 7919 + k * 104729
            )
            x1 = torch.randn(x0.shape, generator=g, dtype=x0.dtype).to(device)
            x_t = (1 - t_val[:, None, None]) * x0 + t_val[:, None, None] * x1
            target = x1 - x0
            probes.append({
                "group": i, "k": k, "target_id": grp[k]["target_id"],
                "x_t": x_t, "t": t_val, "ref": ref, "target": target,
                "own": grp[k]["prompt"],
                "wrong": grp[(k + 1) % K]["prompt"],
            })
    with torch.no_grad():
        emb_cache = {
            p: cache.get([p]).to(device)
            for pr in probes for p in (pr["own"], pr["wrong"])
        }

    ckpts = args.checkpoint if args.checkpoint else ["<identity-at-init>"]
    for ck in ckpts:
        if ck == "<identity-at-init>":
            cond.load_state_dict(cond.state_dict())   # already zero-init
            label = "identity_at_init (untrained)"
        else:
            p = resolve(ck)
            if not p.is_file():
                log(f"SKIP missing checkpoint {p}")
                continue
            state = torch.load(p, map_location="cpu", weights_only=False)
            cond.load_state_dict(state["cond_state"], strict=True)
            label = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        log(f"probing {label} ...")

        rows = []
        with torch.no_grad():
            for pr in probes:
                def run(text):
                    if text is None:
                        cond.set_enabled(False)
                        cond.clear_text()
                        ref_u = pr["ref"]
                    else:
                        cond.set_enabled(True)
                        cond.set_text(emb_cache[text])
                        ref_u = cond.fuse_condition(pr["ref"])
                    pred = backbone(pr["x_t"], pr["t"], ref_u)
                    return pred.clone(), float(
                        (pred - pr["target"]).square().mean().item())

                po, lo = run(pr["own"])
                pw, lw = run(pr["wrong"])
                pf, lf = run(None)
                # d_text_wrong isolates the bottleneck: if the text branch
                # emits nearly the SAME delta for all 4 target styles, no
                # amount of data-level ambiguity can produce a penalty.
                rows.append({
                    "group": pr["group"], "target_id": pr["target_id"],
                    "loss_own": lo, "loss_wrong": lw, "loss_off": lf,
                    "wrong_penalty": lw - lo,
                    "d_text_wrong": rms(po, pw),
                    "d_text_onoff": rms(po, pf),
                })
        n = len(rows)
        m = {
            "loss_own": sum(r["loss_own"] for r in rows) / n,
            "loss_wrong": sum(r["loss_wrong"] for r in rows) / n,
            "loss_off": sum(r["loss_off"] for r in rows) / n,
            "wrong_penalty": sum(r["wrong_penalty"] for r in rows) / n,
            "d_text_wrong": sum(r["d_text_wrong"] for r in rows) / n,
            "d_text_onoff": sum(r["d_text_onoff"] for r in rows) / n,
        }
        m["style_selectivity"] = m["d_text_wrong"] / max(m["d_text_onoff"], 1e-12)
        m["wrong_penalty_relative"] = m["wrong_penalty"] / max(m["loss_own"], 1e-9)
        # IMPORTANT (2026-11-12): the toy trunk is RANDOMLY INITIALISED and
        # FROZEN, so most of the velocity loss is irreducible noise (~1.4).
        # The theoretical gap is only ~0.0013. Judging by a 5% relative-to-
        # total-loss threshold (used for the pretrained FLUX.2 trunk) would
        # falsely reject a perfectly working text branch. The sound yardstick
        # is the penalty as a FRACTION OF THE DATA-LEVEL GAP.
        gap = rep["data_gap"]["no_text_mse_floor_mean"]
        m["penalty_to_gap_ratio"] = m["wrong_penalty"] / max(gap, 1e-12)
        m["text_onoff_gain"] = m["loss_off"] - m["loss_own"]
        m["text_needed_by_loss"] = m["penalty_to_gap_ratio"] >= 0.10
        rep["checkpoints"][label] = {
            "means": {k: round(v, 8) if isinstance(v, float) else v
                      for k, v in m.items()},
            "per_probe": [
                {k: round(v, 8) if isinstance(v, float) else v
                 for k, v in r.items()} for r in rows
            ],
        }
        log(f"  loss_own={m['loss_own']:.6f} loss_wrong={m['loss_wrong']:.6f} "
            f"loss_off={m['loss_off']:.6f} "
            f"wrong_penalty={m['wrong_penalty']:.6f} "
            f"({m['wrong_penalty_relative'] * 100:.2f}%)")

    # ---------------------------------------------------------------- #
    # (3) verdict
    # ---------------------------------------------------------------- #
    trained = {k: v["means"] for k, v in rep["checkpoints"].items()
               if "identity_at_init" not in k}
    if trained:
        best_gap = max(v["penalty_to_gap_ratio"] for v in trained.values())
        rep["verdict"] = (
            "TEXT-NECESSITY-SUPPORTED: swapping in the wrong target's text "
            f"increases the loss by {best_gap * 100:.1f}% of the theoretical "
            "no-text MSE floor" if best_gap >= 0.10 else
            "NOT-YET-SUPPORTED: wrong text still barely changes the loss; "
            "train longer / bias training t toward 1 / check that the "
            "checkpoint was trained on THIS multi-target dataset"
        )
        rep["primary_metric"] = "penalty_to_gap_ratio (>=0.10 = supported)"
    else:
        rep["verdict"] = (
            "DATA-LEVEL-ONLY: gap > 0 proves the dataset makes text necessary "
            "in principle; no trained checkpoint was probed."
        )
    rep["status"] = "OK"
    save_json(rep, out)
    print(json.dumps({
        "data_gap": rep["data_gap"],
        "text_features": rep["text_features"],
        "means": {k: v["means"] for k, v in rep["checkpoints"].items()},
        "verdict": rep["verdict"],
    }, ensure_ascii=False, indent=2))
    log(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
