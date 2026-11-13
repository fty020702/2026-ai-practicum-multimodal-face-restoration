from __future__ import annotations

"""[RESULT] FAST VAE-only regression for the FLUX.2 latent round-trip (T0).

WHY THIS EXISTS
  scripts/verify_real_forward.py loads the WHOLE 16 GB BF16 pipeline and took
  ~20 min on CPU. On 2026-11-07 it failed at T0 with
      RuntimeError: The size of tensor a (32) must match the size of tensor b
      (128) at non-singleton dimension 1
  because tokens_to_image() applied the 128-wide BN denormalisation AFTER
  unpatchify (which had already collapsed the width to 32). Re-running the
  full chain merely to check a shape fix wastes 20 minutes, so this script
  loads ONLY AutoencoderKLFlux2 (0.17 GB) and exercises the EXACT SAME
  module-level helpers used by the real adapter
  (src/system.py::encode_image_to_packed / decode_packed_to_image) through a
  tiny pipeline shim. It finishes in seconds.

  It also contains a NEGATIVE CONTROL: the old buggy order must still raise
  the 32-vs-128 error. If it does not, this regression test has gone toothless.

    D:/Anaconda/envs/ChatAI/python.exe scripts/test_latent_roundtrip.py
    D:/Anaconda/envs/ChatAI/python.exe scripts/test_latent_roundtrip.py --sizes 64 128 256

Writes outputs/logs/latent_roundtrip.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.utils import ROOT, save_json


class _PipelineShim:
    """Expose ONLY the VAE methods that the module-level latent helpers call.

    Flux2KleinPipeline._encode_vae_image is a plain instance method; binding it
    here is sufficient because it only touches self.vae (bn / config / encode).
    The other three are staticmethods in diffusers 0.37.1.
    """

    def __init__(self, vae):
        self.vae = vae


def _test_image(size: int, seed: int) -> torch.Tensor:
    """Deterministic [1,3,size,size] float32 image in [0,1]."""
    g = torch.Generator().manual_seed(seed)
    yy = torch.linspace(0, 1, size)[None, :, None]
    xx = torch.linspace(0, 1, size)[None, None, :]
    base = (0.35 + 0.30 * yy + 0.20 * xx).expand(3, size, size).clone()
    return (base + 0.12 * torch.randn((3, size, size), generator=g)).clamp(0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128],
                    help="square image sides; non-multiples of 16 are resized")
    ap.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"],
                    help="act_dtype for the packed latents (bf16 = real training path)")
    args = ap.parse_args()

    from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
    from src.system import encode_image_to_packed, decode_packed_to_image

    # Bind the real pipeline methods (class methods only, NO weights loaded).
    _PipelineShim._encode_vae_image = Flux2KleinPipeline._encode_vae_image
    _PipelineShim._patchify_latents = staticmethod(Flux2KleinPipeline._patchify_latents)
    _PipelineShim._unpatchify_latents = staticmethod(Flux2KleinPipeline._unpatchify_latents)
    _PipelineShim._pack_latents = staticmethod(Flux2KleinPipeline._pack_latents)

    vae_dir = ROOT / "models" / "FLUX.2-klein-4B-Base" / "vae"
    rep = {
        "status": "UNVERIFIED",
        "device": "cpu",
        "vae": str(vae_dir),
        "helpers": "src/system.py::encode_image_to_packed / decode_packed_to_image",
        "checks": {},
        "cases": [],
    }
    out = ROOT / "outputs" / "logs" / "latent_roundtrip.json"
    if not (vae_dir / "config.json").is_file():
        rep["error"] = f"VAE weights missing at {vae_dir}; run scripts/download_models.py first"
        save_json(rep, out)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 1

    t0 = time.time()
    vae = AutoencoderKLFlux2.from_pretrained(
        str(vae_dir), torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    shim = _PipelineShim(vae)
    rep["timings_s"] = {"vae_load": round(time.time() - t0, 1)}
    rep["checks"]["bn_running_mean_width"] = int(vae.bn.running_mean.numel())
    rep["checks"]["vae_latent_channels"] = int(vae.config.latent_channels)
    rep["checks"]["vae_patch_size"] = list(vae.config.patch_size)
    print(f"[latent] VAE loaded in {rep['timings_s']['vae_load']}s; "
          f"bn_width={rep['checks']['bn_running_mean_width']} "
          f"latent_channels={rep['checks']['vae_latent_channels']} "
          f"patch_size={rep['checks']['vae_patch_size']}", flush=True)

    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(torch.float32)
    bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(torch.float32)

    all_ok = True
    for size in args.sizes:
        img = _test_image(size, seed=20261107 + size)
        t1 = time.time()
        with torch.no_grad():
            # GROUND TRUTH: exact inverse of _encode_vae_image with NO packing
            # and NO dtype quantisation. This isolates the VAE's OWN
            # reconstruction error from our latent helpers.
            x = img * 2 - 1
            norm4d = shim._encode_vae_image(x, generator=None)   # [B,128,s,s]
            raw = shim._unpatchify_latents(norm4d * bn_std + bn_mean)
            rec_ref = ((vae.decode(raw, return_dict=False)[0] + 1) * 0.5).clamp(0, 1).float()
            ref_mae = float((rec_ref - img).abs().mean().item())

            results = []
            for dname in args.dtypes:
                act_dtype = getattr(torch, dname)
                packed, lat4d = encode_image_to_packed(shim, img, act_dtype)
                rec = decode_packed_to_image(shim, packed)
                mae = float((rec - img).abs().mean().item())

                # Strongest correctness check: the latent-space inverse must be
                # EXACT (denormalise in the 128-wide space, THEN unpatchify).
                B, N, C = packed.shape
                s = int(N ** 0.5)
                lat = packed.transpose(1, 2).reshape(B, C, s, s).to(torch.float32)
                inv = shim._unpatchify_latents(lat * bn_std + bn_mean)
                inv_err = float((inv - raw).abs().max().item())

                r = {
                    "dtype": dname,
                    "packed_shape": list(packed.shape),
                    "lat4d_shape": list(lat4d.shape),
                    "reconstructed_shape": list(rec.shape),
                    "finite": bool(torch.isfinite(packed).all() and torch.isfinite(rec).all()),
                    "in_range_0_1": bool((rec >= 0).all() and (rec <= 1).all()),
                    "mae": round(mae, 6),
                    "max_abs_err": round(float((rec - img).abs().max().item()), 6),
                    "reference_mae": round(ref_mae, 6),
                    "delta_vs_reference": round(mae - ref_mae, 6),
                    "latent_inverse_max_abs_err": inv_err,
                    "shape_ok": list(rec.shape) == list(img.shape),
                }
                # The absolute MAE (~0.107) is DOMINATED by the VAE's own
                # reconstruction error on our synthetic noisy test image; it is
                # NOT a measure of our helpers. What must hold is that our
                # helpers reproduce that reference to within bf16 quantisation.
                # float32 -> the latent inverse must be EXACT (0.0).
                # bfloat16 -> quantisation error is amplified by bn_std (~1.77)
                #             when denormalising, so allow a bounded residual.
                inv_tol = 1e-5 if dname == "float32" else 0.05
                r["latent_inverse_tolerance"] = inv_tol
                r["ok"] = bool(
                    r["finite"] and r["in_range_0_1"] and r["shape_ok"]
                    and abs(r["delta_vs_reference"]) < 0.002
                    and r["latent_inverse_max_abs_err"] < inv_tol
                )
                results.append(r)
                all_ok = all_ok and r["ok"]
                print(f"[latent] size={size} dtype={dname:<9} packed={r['packed_shape']} "
                      f"rec={r['reconstructed_shape']} mae={r['mae']} "
                      f"ref={r['reference_mae']} delta={r['delta_vs_reference']:+.6f} "
                      f"inv_err={inv_err:.9f} ok={r['ok']}", flush=True)

        rep["cases"].append({
            "input_size": size,
            "reference_mae": round(ref_mae, 6),
            "seconds": round(time.time() - t1, 1),
            "dtypes": results,
        })

    # ---- NEGATIVE CONTROL: the old buggy order must still fail ------------ #
    try:
        with torch.no_grad():
            B, N, C = packed.shape
            s = int(N ** 0.5)
            bad = packed.transpose(1, 2).reshape(B, C, s, s).to(vae.dtype)
            bad = shim._unpatchify_latents(bad)          # [B,32,2s,2s]
            bm = vae.bn.running_mean.view(1, -1, 1, 1)  # [1,128,1,1]
            bad = bad * bm + bm                          # 32 vs 128 -> must raise
        rep["checks"]["negative_control_caught"] = False
        rep["checks"]["negative_control_note"] = (
            "old bug was NOT reproduced; the regression test may be toothless"
        )
    except RuntimeError as e:
        rep["checks"]["negative_control_caught"] = "must match" in str(e)
        rep["checks"]["negative_control_error"] = str(e)

    rep["checks"]["bn_width_is_128"] = rep["checks"]["bn_running_mean_width"] == 128
    rep["ok"] = bool(all_ok
                     and rep["checks"]["bn_width_is_128"]
                     and rep["checks"]["negative_control_caught"])
    rep["status"] = "OK" if rep["ok"] else "FAILED"
    rep["timings_s"]["total"] = round(time.time() - t0, 1)
    save_json(rep, out)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"[latent] status={rep['status']} -> {out}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
