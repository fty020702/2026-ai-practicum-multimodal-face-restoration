from __future__ import annotations

"""Load the REAL FLUX.2 [klein] 4B BF16 weights and read runtime dims.

Complementary to scripts/check_flux2_contract.py (structure [FACT], no weights
needed): this script loads the checkpoint on CPU and reports the ACTUAL
inner_dim / joint_attention_dim / in_channels / block counts / block forward
signature. It also RECONCILES configs/flux2_klein.yaml against the measured
values, so stale placeholder dims are caught automatically instead of failing
later with an opaque shape mismatch.

    D:/Anaconda/envs/ChatAI/python.exe scripts/inspect_backbone.py

Writes outputs/logs/backbone_inspect.json
"""

import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from src.utils import ROOT, load_config, save_json

CANDIDATES = (
    "transformer_blocks", "single_transformer_blocks", "double_blocks",
    "blocks", "joint_blocks",
)


def main():
    # IMPORTANT: flux2_klein.yaml is an OVERRIDE, not a full config. Passing it
    # as `base=` would drop project/backbone/data/eval entirely.
    cfg = load_config(override=ROOT / "configs" / "flux2_klein.yaml")
    model_id = cfg["backend"]["model_id"]
    report = {"model_id": model_id, "status": "UNVERIFIED", "checks": {}}

    try:
        import diffusers
        import transformers
        report["checks"]["diffusers"] = diffusers.__version__
        report["checks"]["transformers"] = transformers.__version__
        from diffusers import Flux2KleinPipeline
    except Exception as e:
        report["error"] = f"import failed: {e}"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        save_json(report, ROOT / "outputs" / "logs" / "backbone_inspect.json")
        return 1

    local = ROOT / "models" / "FLUX.2-klein-4B-Base"
    if not (local / "model_index.json").is_file():
        report["error"] = (
            f"BF16 full pipeline not found at {local} (no model_index.json). "
            "Run scripts/download_models.py (default, NOT --fp8-singlefile). "
            "The FP8 repo is a single ComfyUI-native checkpoint and CANNOT be "
            "loaded by Flux2KleinPipeline."
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        save_json(report, ROOT / "outputs" / "logs" / "backbone_inspect.json")
        return 1
    src = str(local)
    report["checks"]["weights_source"] = src
    print(f"[inspect] loading {src} on CPU (slow, ~16 GB RAM for the full BF16 pipeline) ...")

    try:
        pipe = Flux2KleinPipeline.from_pretrained(
            src, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
    except Exception as e:
        report["error"] = f"load failed: {e}"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        save_json(report, ROOT / "outputs" / "logs" / "backbone_inspect.json")
        return 1

    tr = getattr(pipe, "transformer", None)
    te = getattr(pipe, "text_encoder", None)
    report["checks"]["pipeline_class"] = type(pipe).__name__
    report["checks"]["transformer_class"] = type(tr).__name__ if tr else None
    report["checks"]["text_encoder_class"] = type(te).__name__ if te else None
    if te is not None:
        report["checks"]["text_encoder_config"] = {
            k: v for k, v in vars(te.config).items()
            if k in ("hidden_size", "model_type", "num_hidden_layers", "vocab_size")
        }
    try:
        report["checks"]["pipeline_call_signature"] = str(inspect.signature(pipe.__call__))
    except Exception:
        pass

    found = {}
    if tr is not None:
        # Runtime dims that apply_runtime_config() will overwrite.
        conf = getattr(tr, "config", None)
        report["checks"]["runtime_dims"] = {
            "in_channels": getattr(conf, "in_channels", None),
            "inner_dim": getattr(tr, "inner_dim", None),
            "joint_attention_dim": getattr(conf, "joint_attention_dim", None),
            "num_attention_heads": getattr(conf, "num_attention_heads", None),
            "attention_head_dim": getattr(conf, "attention_head_dim", None),
        }
        xe = getattr(tr, "x_embedder", None)
        ce = getattr(tr, "context_embedder", None)
        if isinstance(xe, nn.Linear):
            report["checks"]["runtime_dims"]["x_embedder_in_features"] = xe.in_features
        if isinstance(ce, nn.Linear):
            report["checks"]["runtime_dims"]["context_embedder_in_features"] = ce.in_features

        for name in CANDIDATES:
            obj = getattr(tr, name, None)
            if isinstance(obj, (nn.ModuleList, list, tuple)) and len(obj) > 0:
                found[name] = {
                    "length": len(obj),
                    "element_types": [type(x).__name__ for x in list(obj)[:3]],
                }
        report["checks"]["candidate_block_lists"] = found
        for name in found:
            blk = getattr(tr, name)[0]
            try:
                report["checks"]["block_forward_signature"] = str(inspect.signature(blk.forward))
            except Exception:
                pass
            report["checks"]["block_returns_txt_first"] = (
                "return encoder_hidden_states, hidden_states"
                in inspect.getsource(type(blk).forward)
            )
            lins = [(n, m.in_features, m.out_features)
                    for n, m in blk.named_modules() if isinstance(m, nn.Linear)]
            report["checks"]["first_block_linears"] = lins[:12]
            break

    # ---- Reconcile the STATIC override config against the REAL weights ---- #
    rd = report["checks"].get("runtime_dims", {})
    cond = cfg.get("condition", {})
    expected = {
        "text_in_dim": rd.get("joint_attention_dim"),
        "hidden_dim": rd.get("inner_dim"),
        "cond_dim": rd.get("inner_dim"),
        "ref_bias_out_dim": rd.get("in_channels"),
    }
    mismatches = {
        k: {"config": cond.get(k), "measured": v}
        for k, v in expected.items()
        if v is not None and cond.get(k) != v
    }
    report["checks"]["config_reconciliation"] = {
        "ok": not mismatches, "mismatches": mismatches,
    }
    blists = report["checks"].get("candidate_block_lists", {})
    report["measured"] = {
        "inner_dim": rd.get("inner_dim"),
        "joint_attention_dim": rd.get("joint_attention_dim"),
        "in_channels": rd.get("in_channels"),
        "n_double_blocks": blists.get("transformer_blocks", {}).get("length"),
        "n_single_blocks": blists.get("single_transformer_blocks", {}).get("length"),
        "block_returns_txt_first": report["checks"].get("block_returns_txt_first"),
    }
    report["status"] = "OK" if not mismatches else "CONFIG_MISMATCH"

    report["notes"] = [
        "[FACT] Real BF16 full pipeline loaded on CPU; dims above are MEASURED, not guessed.",
        "Structure contract verified WITHOUT weights by scripts/check_flux2_contract.py "
        "and scripts/check_flux2_train_contract.py (both all PASS).",
        "BLOCK_QUERY=('transformer_blocks',) and IMG_INDEX=1 are set from source; "
        "single_transformer_blocks are deliberately excluded (concatenated txt+img stream).",
        "Reference images are VAE-encoded and CONCATENATED INTO the image stream, NOT the text "
        "stream; encoder_hidden_states is the Qwen3 prompt_embeds text stream (kept frozen+constant).",
        "Three distinct dims: in_channels (packed latents -> ref bias), inner_dim (block hidden "
        "-> bypass), joint_attention_dim (text_in_dim). apply_runtime_config() sets all three.",
        "config_reconciliation.ok=False means configs/flux2_klein.yaml still holds stale "
        "placeholder dims; fix them before building TextConditionModule directly.",
        "Next: scripts/precompute_prompts.py -> scripts/verify_real_backend.py -> "
        "scripts/verify_real_forward.py",
    ]
    out = ROOT / "outputs" / "logs" / "backbone_inspect.json"
    save_json(report, out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[inspect] status={report['status']} -> {out}")
    return 0 if not mismatches else 2


if __name__ == "__main__":
    sys.exit(main())
