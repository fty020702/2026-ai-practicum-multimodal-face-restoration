from __future__ import annotations

"""Reproducible HTTP-level verification of app/demo_server.py.

    # terminal 1
    D:/Anaconda/envs/ChatAI/python.exe app/demo_server.py --port 7860
    # terminal 2
    D:/Anaconda/envs/ChatAI/python.exe scripts/verify_demo_http.py
    D:/Anaconda/envs/ChatAI/python.exe scripts/verify_demo_http.py --generate

Without --generate the 17 GB model is NEVER loaded (fast, GPU-free).
With --generate it runs one real low-cost generation (8 steps / 384 px) and
saves the returned PNG, proving the Demo is actually usable end to end.

Output: outputs/logs/demo_http_verify.json
"""

import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

BASE = "http://127.0.0.1:7860"
OUT = ROOT / "outputs/logs/demo_http_verify.json"
SAVE_IMG = ROOT / "outputs/samples/demo_http_generated.png"


def req(path, payload=None, timeout=30):
    url = BASE + path
    if payload is None:
        r = urllib.request.Request(url, method="GET")
    else:
        r = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return f.status, json.loads(f.read().decode("utf-8"))


def get_err(path, payload, timeout=30):
    r = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode("utf-8"),
        method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return f.status, json.loads(f.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true",
                    help="also run one real FLUX.2 generation (loads 17 GB)")
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args()
    BASE = a.base

    rep = {"base": BASE, "checks": [], "boundary": [], "status": "RUNNING"}
    try:
        code, _ = urllib.request.urlopen(BASE + "/", timeout=10).status, None
        rep["checks"].append({"name": "GET /", "http": code, "pass": code == 200})
        code, s = req("/api/status")
        rep["status_info"] = {
            "weights_present": s.get("weights_present"),
            "pipeline_state": s.get("pipeline_state"),
            "cuda_available": s.get("cuda_available"),
            "gpu_free_mib": s.get("gpu_free_mib"),
            "presets": len(s.get("presets", [])),
        }
        rep["checks"].append({"name": "GET /api/status", "http": code,
                              "pass": code == 200 and s.get("weights_present")})
        code, ex = req("/api/examples")
        rep["checks"].append({"name": "GET /api/examples", "http": code,
                              "pass": code == 200 and len(ex.get("items", [])) > 0})
        img = (ex["items"][0]["url"] if ex.get("items") else "")

        # a valid RGB image + all boundary cases
        cases = [
            ("合法输入", {"image": img, "prompt": "restore this old photo"}),
            ("空图", {"image": "", "prompt": "restore"}),
            ("空文本回退", {"image": img, "prompt": ""}),
            ("超长文本截断", {"image": img, "prompt": "repair " * 300}),
            ("损坏图像", {"image": "data:image/png;base64,bm90X2FuX2ltYWdl",
                          "prompt": "restore"}),
            ("越界参数裁剪", {"image": img, "prompt": "restore",
                             "steps": 999, "size": 4096,
                             "guidance": 99, "seed": -1}),
            ("灰度输入", {"image": _gray_url(), "prompt": "restore"}),
        ]
        for name, payload in cases:
            code, j = get_err("/api/validate", payload)
            rec = {"case": name, "http": code, "ok": j.get("ok"),
                   "errors": j.get("errors", []),
                   "warnings": j.get("warnings", []),
                   "normalized": j.get("normalized")}
            rep["boundary"].append(rec)
        rep["checks"].append({
            "name": "POST /api/validate (7 cases)",
            "pass": (rep["boundary"][0]["ok"] is True
                     and rep["boundary"][1]["ok"] is False
                     and rep["boundary"][4]["ok"] is False
                     and len(rep["boundary"][2]["warnings"]) > 0
                     and len(rep["boundary"][5]["warnings"]) > 0),
        })

        if a.generate:
            payload = {"image": img,
                       "prompt": "Restore this old photograph into a clean "
                                 "natural-color photograph, remove scratches "
                                 "and film grain.",
                       "steps": 8, "size": 384, "guidance": 4.0,
                       "seed": 20261117}
            code, j = get_err("/api/generate", payload, timeout=600)
            ok = code == 200 and j.get("ok") and j.get("image", "").startswith("data:image")
            rep["generation"] = {"http": code, "ok": j.get("ok"),
                                 "seconds": j.get("seconds"),
                                 "warnings": j.get("warnings", []),
                                 "used_guidance": j.get("used_guidance"),
                                 "error": j.get("error") or j.get("errors")}
            if ok:
                raw = base64.b64decode(j["image"].split(",", 1)[1])
                SAVE_IMG.parent.mkdir(parents=True, exist_ok=True)
                SAVE_IMG.write_bytes(raw)
                im = Image.open(io.BytesIO(raw))
                rep["generation"]["saved"] = str(SAVE_IMG.relative_to(ROOT))
                rep["generation"]["size"] = list(im.size)
            rep["checks"].append({"name": "POST /api/generate (real FLUX.2)",
                                  "pass": bool(ok)})

        rep["all_pass"] = all(c["pass"] for c in rep["checks"])
        rep["status"] = "PASS" if rep["all_pass"] else "FAIL"
    except Exception as e:                                         # noqa: BLE001
        rep["status"] = "ERROR"
        rep["error"] = f"{type(e).__name__}: {e}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["status"] == "PASS" else 1


def _gray_url() -> str:
    buf = io.BytesIO()
    Image.new("L", (96, 96), 128).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


if __name__ == "__main__":
    sys.exit(main())
