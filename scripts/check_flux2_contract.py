from __future__ import annotations

"""[RESULT] Verify the FLUX.2 adapter contract WITHOUT downloading weights.

Runs against the INSTALLED diffusers source, plus a simulated (txt, img) block
that proves the bypass hook lands on the image stream when img_index=1.

    D:/Anaconda/envs/ChatAI/python.exe scripts/check_flux2_contract.py

Writes outputs/logs/flux2_contract.json.
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from src.condition import TextConditionModule
from src.utils import ROOT, save_json


class FakeDoubleBlock(nn.Module):
    """Mimics Flux2TransformerBlock: returns (txt, img), NOT (img, txt)."""

    def forward(self, x):
        return torch.full_like(x, 7.0), torch.full_like(x, 1.0)


def hook_test(img_index):
    cfg = {"condition": {
        "mode": "projector_plus_bypass", "text_in_dim": 16, "hidden_dim": 16,
        "cond_dim": 16, "projector_layers": 1, "projector_dropout": 0.0,
        "zero_init_last": False, "bypass_heads": 2, "bypass_scale": 1.0,
    }}
    cond = TextConditionModule(cfg, [16])
    with torch.no_grad():
        for b in cond.bypasses:
            b.o_proj.weight.fill_(0.05)   # pretend the bypass is trained
    blk = FakeDoubleBlock()
    cond.attach([blk], img_index=img_index)
    cond.set_text(torch.randn(1, 4, 16))
    out = blk(torch.ones(1, 6, 16))
    txt, img = out[0], out[1]
    cond.detach()
    return {
        "img_index": img_index,
        "txt_changed": bool((txt - 7.0).abs().max() > 1e-6),
        "img_changed": bool((img - 1.0).abs().max() > 1e-6),
        "img_delta": round(float((img - 1.0).abs().max().item()), 6),
    }


def main():
    rep = {"status": "RESULT", "checks": {}, "notes": []}
    try:
        import diffusers
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
        from diffusers.models.transformers.transformer_flux2 import (
            Flux2TransformerBlock, Flux2SingleTransformerBlock,
        )
    except Exception as e:
        rep["error"] = f"import failed: {e}"
        save_json(rep, ROOT / "outputs" / "logs" / "flux2_contract.json")
        print(rep)
        return 1

    rep["checks"]["diffusers_version"] = diffusers.__version__
    init_src = inspect.getsource(Flux2KleinPipeline.__init__)
    rep["checks"]["text_encoder_is_qwen3"] = "Qwen3ForCausalLM" in init_src
    rep["checks"]["has_is_distilled_flag"] = "is_distilled" in init_src
    rep["checks"]["call_accepts_prompt_embeds"] = (
        "prompt_embeds" in str(inspect.signature(Flux2KleinPipeline.__call__))
    )
    rep["checks"]["tokenizer_max_length"] = (
        "512" in init_src
    )

    tr_src = inspect.getsource(Flux2Transformer2DModel.__init__)
    rep["checks"]["has_transformer_blocks"] = "self.transformer_blocks" in tr_src
    rep["checks"]["has_single_transformer_blocks"] = "self.single_transformer_blocks" in tr_src
    rep["checks"]["inner_dim_formula"] = "num_attention_heads * attention_head_dim" in tr_src

    dbl_src = inspect.getsource(Flux2TransformerBlock.forward)
    rep["checks"]["double_block_returns_txt_first"] = (
        "return encoder_hidden_states, hidden_states" in dbl_src
    )
    sgl_src = inspect.getsource(Flux2SingleTransformerBlock.forward)
    rep["checks"]["single_block_has_split_flag"] = "split_hidden_states" in sgl_src

    q_src = inspect.getsource(Flux2KleinPipeline._get_qwen3_prompt_embeds)
    rep["checks"]["qwen_layers_9_18_27"] = "(9, 18, 27)" in q_src
    rep["checks"]["embeds_are_3x_hidden"] = "num_channels * hidden_dim" in q_src

    # Simulated hook: with img_index=1 the IMAGE stream must change and the
    # TEXT stream must stay untouched. With 0 (the toy default) it is reversed.
    r1 = hook_test(1)
    r0 = hook_test(0)
    rep["checks"]["hook_img_index_1"] = r1
    rep["checks"]["hook_img_index_0_toy"] = r0
    rep["checks"]["hook_contract_ok"] = (
        r1["img_changed"] and not r1["txt_changed"]
        and r0["txt_changed"] and not r0["img_changed"]
    )

    ok = all(v for k, v in rep["checks"].items()
             if isinstance(v, bool))
    rep["all_passed"] = ok
    rep["notes"] = [
        "img_index=1 is required for FLUX.2 double blocks (txt, img).",
        "Single blocks are excluded: they consume a concatenated txt+img stream.",
        "This test uses source inspection + a simulated block; it does NOT load weights.",
    ]
    out = ROOT / "outputs" / "logs" / "flux2_contract.json"
    save_json(rep, out)
    for k, v in rep["checks"].items():
        print(f"  {'PASS' if v is True else 'FAIL' if v is False else '     '} {k}: {v}")
    print(f"[contract] all_passed={ok} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
