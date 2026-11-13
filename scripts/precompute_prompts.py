from __future__ import annotations

"""[UNVERIFIED] Offline-encode ALL prompts with the frozen Qwen3-4B encoder.

On a 12 GB 4070 Ti this is a HARD REQUIREMENT, not an optimisation: the
Qwen3-4B encoder (~8 GB BF16) must never enter VRAM during training. This
script loads ONLY the text encoder + tokenizer (not the 4B DiT), runs on CPU,
and fills data/text_cache/flux2_klein_qwen3/ so train.py never calls the
encoder at all.

    D:/Anaconda/envs/ChatAI/python.exe scripts/precompute_prompts.py

Writes outputs/logs/precompute_prompts.json.

[UNVERIFIED]: requires the BF16 FULL PIPELINE on disk (it contains
text_encoder/ and tokenizer/). Run scripts/download_models.py (default,
NOT --fp8-singlefile), then scripts/inspect_backbone.py first. The FP8 repo
has NO text_encoder/ and cannot serve this script. The embedding construction
itself IS verified [FACT] (it delegates to
Flux2KleinPipeline._get_qwen3_prompt_embeds, see
scripts/check_flux2_contract.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.dataset import make_dataset
from src.data.text_features import OfflineTextCache
from src.system import TEXT_CACHE_NAMESPACE, WEIGHTS_SUBDIR
from src.utils import ROOT, load_config, save_json


class _PrecomputeEncoder:
    """Minimal encoder object for OfflineTextCache. .encode(list[str]) -> [B,L,D].

    MEMORY FIX (2026-11-07). The old version passed ALL prompts in ONE call.
    `_get_qwen3_prompt_embeds` pads every prompt to max_sequence_length (512)
    and Qwen3-4B has 36 layers, so the CPU activation peak exceeded the free
    RAM. The process was then killed by the OS with NO traceback: the verify
    log stops right after "Loading checkpoint shards" and the cache dir stayed
    empty. We now encode in small batches under torch.inference_mode(); main()
    additionally writes ONE prompt at a time so no large stack is ever built.
    """

    def __init__(self, text_encoder, tokenizer, dtype=torch.bfloat16,
                 batch_size: int = 1, max_sequence_length: int = 512):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.batch_size = max(1, int(batch_size))       # keep activation peak tiny
        self.max_sequence_length = int(max_sequence_length)
        self.dim = None
        self.num_tokens = None

    def encode(self, texts):
        from diffusers import Flux2KleinPipeline
        texts = list(texts)
        chunks = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                e = Flux2KleinPipeline._get_qwen3_prompt_embeds(
                    text_encoder=self.text_encoder,
                    tokenizer=self.tokenizer,
                    prompt=texts[i:i + self.batch_size],
                    dtype=self.dtype,
                    device=torch.device("cpu"),
                    max_sequence_length=self.max_sequence_length,
                )
                chunks.append(e.detach().to(device="cpu", dtype=torch.float32))
                del e
        return torch.cat(chunks, dim=0)


def _free_ram_gb():
    """Windows: available physical RAM in GB, or None if unavailable."""
    try:
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys / 1e9
    except Exception:  # noqa: BLE001  (non-Windows / ctypes unavailable)
        return None


def collect_prompts(cfg):
    ds = make_dataset(cfg, root=cfg["_root"])
    # MultiTargetDataset emits a randomly chosen paraphrase per __getitem__,
    # so iterating it cannot guarantee full coverage. It exposes the exact
    # finite prompt set; use that when available (2026-11-12).
    prompts = set(ds.all_prompts()) if hasattr(ds, "all_prompts") else set()
    prompts.update(b["prompt"] for b in ds)
    for pair in cfg["eval"]["text_response_pairs"]:
        prompts.update(pair)
    # train.py / inference.py may also be given ad-hoc prompts; include a few
    # neutral ones so the constant-preamble path is covered too.
    from src.system import NEUTRAL_PROMPT
    prompts.add(NEUTRAL_PROMPT)
    return sorted(prompts)


def main():
    # Default stays VERBATIM for the existing flux2_klein verification chain.
    # Pass --override configs/multi_target_flux2.yaml to offline-encode the
    # one-image-many-targets prompt set instead.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--override", default="configs/flux2_klein.yaml")
    a = ap.parse_args()
    ov = Path(a.override)
    cfg = load_config(override=ov if ov.is_absolute() else ROOT / ov)
    local = ROOT / "models" / WEIGHTS_SUBDIR
    rep = {"status": "UNVERIFIED", "weights_dir": str(local),
           "n_prompts": 0, "override": str(ov)}
    needed = ("text_encoder/config.json", "tokenizer/tokenizer.json")
    missing = [n for n in needed if not (local / n).is_file()]
    if missing:
        rep["error"] = (
            f"BF16 full pipeline incomplete at {local}; missing {missing}; "
            "run scripts/download_models.py (default, NOT --fp8-singlefile) first"
        )
        print(rep)
        save_json(rep, ROOT / "outputs" / "logs" / "precompute_prompts.json")
        return 1

    # FAST PATH (2026-11-07). The completed verification chain proved this
    # script actually SUCCEEDED (10 prompts -> [10,512,7680] cache, ~3 h on
    # CPU). Re-running it under low free RAM must NOT overwrite that OK report
    # with an "insufficient free RAM" failure, and must NOT reload the 8 GB
    # Qwen3-4B encoder just to discover every file is already on disk.
    prompts = collect_prompts(cfg)
    cache_dir = ROOT / "data" / "text_cache" / TEXT_CACHE_NAMESPACE
    missing_prompts = [
        t for t in prompts
        if not (cache_dir / f"{OfflineTextCache._key(t)}.pt").is_file()
    ]
    if not missing_prompts:
        first = torch.load(
            cache_dir / f"{OfflineTextCache._key(prompts[0])}.pt", weights_only=True
        )
        rep.update({
            "n_prompts": len(prompts),
            "n_written": 0,
            "n_cached": len(prompts),
            "emb_shape": list(first.shape),
            "cache_dir": str(cache_dir),
            "status": "OK",
            "note": "cache already complete; skipped Qwen3-4B load and RAM precheck",
            "next": "train.py --override configs/flux2_klein.yaml (encoder is never loaded)",
        })
        out = ROOT / "outputs" / "logs" / "precompute_prompts.json"
        save_json(rep, out)
        print(f"[precompute] cache complete ({len(prompts)} prompts) -> {cache_dir} "
              f"shape={list(first.shape)}; skipped encoder load")
        print(f"[precompute] -> {out}")
        return 0

    # Qwen3-4B BF16 weights alone are ~8 GB. Refuse to start when there is not
    # enough free RAM: otherwise Windows kills the process with NO traceback
    # and the cache silently stays empty (the exact failure seen 2026-11-07).
    MIN_FREE_GB = 10.0
    free = _free_ram_gb()
    rep["free_ram_GB"] = round(free, 2) if free is not None else "unknown"
    if free is not None and free < MIN_FREE_GB:
        rep["error"] = (
            f"insufficient free RAM: {free:.2f} GB available, need >= {MIN_FREE_GB:.0f} GB "
            "(Qwen3-4B BF16 ~8 GB + activations). Close other memory-heavy "
            "processes / reboot, then re-run this script."
        )
        rep["fix"] = [
            "close browser / IDE / other Python processes, or reboot",
            "re-run: scripts/precompute_prompts.py",
            "then: scripts/verify_real_backend.py (should say READY)",
        ]
        print(rep)
        save_json(rep, ROOT / "outputs" / "logs" / "precompute_prompts.json")
        return 2

    try:
        from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
        te = Qwen3ForCausalLM.from_pretrained(
            str(local), subfolder="text_encoder", torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()
        tok = Qwen2TokenizerFast.from_pretrained(str(local), subfolder="tokenizer")
    except Exception as e:
        rep["error"] = f"failed to load text encoder / tokenizer: {e}"
        print(rep)
        save_json(rep, ROOT / "outputs" / "logs" / "precompute_prompts.json")
        return 1

    prompts = collect_prompts(cfg)
    enc = _PrecomputeEncoder(te, tok, batch_size=1)
    cache = OfflineTextCache(ROOT, enc, namespace=TEXT_CACHE_NAMESPACE)

    # ONE prompt at a time: never materialise a [N,512,7680] stack in RAM.
    # Uses the same namespace + SHA key as OfflineTextCache.get(), so train.py
    # hits the cache without ever loading Qwen3-4B.
    written = skipped = 0
    first_shape = None
    for t in prompts:
        p = cache.dir / f"{cache._key(t)}.pt"
        if p.is_file():
            skipped += 1
            if first_shape is None:
                first_shape = list(torch.load(p, weights_only=True).shape)
            continue
        e = enc.encode([t])
        torch.save(e[0], p)
        if first_shape is None:
            first_shape = list(e[0].shape)
        written += 1
        del e
    rep.update({
        "n_prompts": len(prompts),
        "n_written": written,
        "n_cached": skipped,
        "emb_shape": first_shape,
        "cache_dir": str(cache.dir),
        "status": "OK",
        "next": "train.py --override configs/flux2_klein.yaml (encoder will be released, never in VRAM)",
    })
    del te, enc
    out = ROOT / "outputs" / "logs" / "precompute_prompts.json"
    save_json(rep, out)
    print(f"[precompute] {len(prompts)} prompts -> {cache.dir} shape={first_shape} "
          f"(written={written} cached={skipped})")
    print(f"[precompute] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
