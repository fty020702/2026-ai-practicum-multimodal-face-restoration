from __future__ import annotations

"""Zero-dependency interactive Web Demo for the course project.

    D:/Anaconda/envs/ChatAI/python.exe app/demo_server.py --port 7860
    D:/Anaconda/envs/ChatAI/python.exe app/demo_server.py --selftest

Uses ONLY the Python standard library for the web layer (http.server), plus
torch / diffusers / PIL which are ALREADY in the ChatAI env. Deliberately does
NOT install gradio: gradio 4.x upgrades pydantic/rich/typer and breaks the
pre-existing fschat / spacy / streamlit / weasel in this shared env.

What it demonstrates (course requirement: "Demo 达到基本可用，可展示鲁棒性 /
边界情况"):
  * upload an old photo + type a text instruction -> real FLUX.2 [klein] 4B
    Base output (native Qwen3 text + native image=[ref], ZERO training)
  * preset buttons for 4 restoration styles and 4 content-edit instructions
  * a "鲁棒性 / 边界情况" panel that exercises validation without loading the
    17 GB model: empty image, empty prompt, over-long prompt, corrupt image,
    out-of-range parameters, grayscale input

API
  GET  /                 static HTML/JS UI
  GET  /api/status       weights / pipeline / GPU state
  GET  /api/examples     preset old-photo thumbnails (base64)
  POST /api/validate     validation + normalization ONLY, no model
  POST /api/generate     validation + real FLUX.2 generation

HONESTY: this is the NATIVE frozen model's capability, NOT the project's
self-trained text bypass. It must not be used to claim TEXT-NECESSITY-SUPPORTED.
"""

import argparse
import base64
import gc
import io
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch_hub"))

from PIL import Image  # noqa: E402

WEIGHTS = ROOT / "models" / "FLUX.2-klein-4B-Base"
EXAMPLE_DIR = ROOT / "data" / "old_photo_multitarget" / "old"
NEUTRAL_RESTORE_PROMPT = (
    "Restore this old photograph: remove all scratches, dust spots and film "
    "grain, repair the faded tones. Keep the original background, the person "
    "and the face EXACTLY as they are."
)

PRESETS = [
    {"group": "修复风格", "id": "natural_color",
     "label": "自然色修复",
     "prompt": "Restore this old photograph into a natural-color, clean, "
               "sharp modern photograph. Remove scratches, dust and grain."},
    {"group": "修复风格", "id": "monochrome",
     "label": "黑白修复",
     "prompt": "Restore this old photograph as a clean, high-contrast "
               "BLACK AND WHITE monochrome photograph. Remove all damage."},
    {"group": "修复风格", "id": "sepia_tone",
     "label": "怀旧棕褐",
     "prompt": "Restore this old photograph with a warm SEPIA brown tone, "
               "preserving a vintage look while removing all scratches."},
    {"group": "修复风格", "id": "minimal_repair",
     "label": "保守修复",
     "prompt": "Conservatively repair this old photograph: remove only the "
               "scratches and dust, keep all original tones and texture."},
    {"group": "内容编辑", "id": "blue_sky",
     "label": "背景改蓝天",
     "prompt": "Restore this old photograph: remove all scratches, dust spots "
               "and film grain, and CHANGE THE BACKGROUND to a clear bright "
               "blue sky with a few white clouds. Keep the person unchanged."},
    {"group": "内容编辑", "id": "beard",
     "label": "人脸加胡子",
     "prompt": "Restore this old photograph: remove all scratches, dust spots "
               "and film grain, and ADD A THICK DARK BEARD AND MOUSTACHE to "
               "the person's face. Keep the original background unchanged."},
    {"group": "内容编辑", "id": "sky_and_beard",
     "label": "蓝天 + 胡子",
     "prompt": "Restore this old photograph: remove all scratches, CHANGE THE "
               "BACKGROUND to a clear bright blue sky with white clouds, and "
               "ADD A THICK DARK BEARD AND MOUSTACHE to the person's face."},
    {"group": "内容编辑", "id": "baseline",
     "label": "仅修复（对照）",
     "prompt": NEUTRAL_RESTORE_PROMPT},
]

# --------------------------------------------------------------------------- state
_LOCK = threading.Lock()
_PIPE = None
_PIPE_STATE = "NOT_LOADED"       # NOT_LOADED | LOADING | READY | FAILED
_PIPE_ERROR = ""
_LAST = {"seconds": None, "prompt": "", "warnings": [], "at": None}


def _clamp(v, lo, hi, default):
    try:
        x = type(default)(v)
    except (TypeError, ValueError):
        return default, [f"参数 {v!r} 非法，已回退为 {default}"]
    if not (lo <= x <= hi):
        return (lo if x < lo else hi), [f"参数 {v!r} 超界，已裁剪到 [{lo},{hi}]"]
    return x, []


def validate(payload: dict) -> dict:
    """Pure, model-free validation. Returns normalized values + warnings/errors."""
    warnings: list[str] = []
    errors: list[str] = []

    raw = payload.get("image") or ""
    img = None
    if not raw:
        errors.append("未提供图像：请先上传或选择一张老照片")
    else:
        try:
            b64 = raw.split(",", 1)[1] if raw.startswith("data:") else raw
            data = base64.b64decode(b64, validate=True)
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception as e:                                   # noqa: BLE001
            errors.append(f"无法解析图像（{type(e).__name__}）：请上传有效 PNG/JPEG")
            img = None

    if img is not None:
        if img.mode != "RGB":
            warnings.append(f"输入为 {img.mode} 模式，已自动转换为 RGB")
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > 4096:
            warnings.append(f"输入尺寸 {w}×{h} 过大，已按比例缩小到最长边 2048")
            img.thumbnail((2048, 2048), Image.LANCZOS)
        elif min(w, h) < 32:
            warnings.append(f"输入尺寸 {w}×{h} 过小，已放大到 256×256")
            img = img.resize((256, 256), Image.LANCZOS)

    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        warnings.append("文本为空，已回退到中性修复指令")
        prompt = NEUTRAL_RESTORE_PROMPT
    if len(prompt) > 1000:
        warnings.append(f"文本长度 {len(prompt)} 超过 1000，已截断")
        prompt = prompt[:1000]

    steps, w1 = _clamp(payload.get("steps", 20), 1, 50, 20)
    size, w2 = _clamp(payload.get("size", 512), 256, 1024, 512)
    size = int(size // 16 * 16)
    guidance, w3 = _clamp(payload.get("guidance", 4.0), 1.0, 8.0, 4.0)
    seed, w4 = _clamp(payload.get("seed", 20261113), 0, 2**31 - 1, 20261113)
    warnings += w1 + w2 + w3 + w4

    return {
        "ok": not errors, "errors": errors, "warnings": warnings,
        "image": img, "prompt": prompt, "steps": int(steps), "size": int(size),
        "guidance": float(guidance), "seed": int(seed),
        "normalized": {"prompt_chars": len(prompt), "steps": int(steps),
                       "size": int(size), "guidance": float(guidance),
                       "seed": int(seed),
                       "image_mode": img.mode if img else None,
                       "image_size": list(img.size) if img else None},
    }


def _status() -> dict:
    st = {"weights_dir": str(WEIGHTS.relative_to(ROOT)),
          "weights_present": WEIGHTS.is_dir(),
          "pipeline_state": _PIPE_STATE, "pipeline_error": _PIPE_ERROR,
          "last": _LAST, "presets": PRESETS}
    try:
        import torch
        st["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            st["gpu_free_mib"] = round(free / 1024**2)
            st["gpu_total_mib"] = round(total / 1024**2)
    except Exception as e:                                        # noqa: BLE001
        st["torch_error"] = f"{type(e).__name__}: {e}"
    return st


def _load_pipeline():
    global _PIPE, _PIPE_STATE, _PIPE_ERROR
    with _LOCK:
        if _PIPE_STATE == "READY":
            return _PIPE
        if not WEIGHTS.is_dir():
            _PIPE_STATE = "FAILED"
            _PIPE_ERROR = f"缺少 BF16 完整 pipeline: {WEIGHTS}"
            raise FileNotFoundError(_PIPE_ERROR)
        _PIPE_STATE = "LOADING"
        try:
            import torch
            from diffusers import Flux2KleinPipeline
            pipe = Flux2KleinPipeline.from_pretrained(
                str(WEIGHTS), torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True)
            pipe.enable_model_cpu_offload(gpu_id=0)
            _PIPE = pipe
            _PIPE_STATE = "READY"
            _PIPE_ERROR = ""
            return _PIPE
        except Exception as e:                                    # noqa: BLE001
            _PIPE_STATE = "FAILED"
            _PIPE_ERROR = f"{type(e).__name__}: {e}"
            raise


def generate(v: dict) -> dict:
    global _LAST
    pipe = _load_pipeline()
    import torch
    with _LOCK:
        t0 = time.time()
        img = v["image"].resize((v["size"], v["size"]), Image.LANCZOS)
        gen = torch.Generator(device="cuda").manual_seed(v["seed"])
        out = None
        used_guidance = v["guidance"]
        try:
            out = pipe(prompt=v["prompt"], image=[img],
                       height=v["size"], width=v["size"],
                       num_inference_steps=v["steps"],
                       guidance_scale=used_guidance,
                       generator=gen).images[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            used_guidance = 1.0
            v["warnings"].append("CUDA OOM，已自动以 guidance=1.0 重试")
            gen = torch.Generator(device="cuda").manual_seed(v["seed"])
            out = pipe(prompt=v["prompt"], image=[img],
                       height=v["size"], width=v["size"],
                       num_inference_steps=v["steps"],
                       guidance_scale=1.0, generator=gen).images[0]
        buf = io.BytesIO()
        out.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        dt = round(time.time() - t0, 1)
        _LAST = {"seconds": dt, "prompt": v["prompt"][:120],
                 "warnings": v["warnings"],
                 "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        torch.cuda.empty_cache()
        return {"ok": True, "image": "data:image/png;base64," + b64,
                "seconds": dt, "warnings": v["warnings"],
                "normalized": v["normalized"],
                "used_guidance": used_guidance}


# ---------------------------------------------------------------------------- UI
HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>多模态参考人脸图像恢复 · 交互 Demo</title>
<style>
:root{--bg:#f6f8fb;--card:#fff;--bd:#d9e2ee;--ink:#1e2a3a;--mut:#5b6b7c;
--blue:#3c6fd6;--orange:#de8a20;--green:#2e8b57;--red:#c0392b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:"Microsoft YaHei","Segoe UI",sans-serif;font-size:14px}
header{background:#fff;border-bottom:1px solid var(--bd);padding:16px 24px}
h1{margin:0;font-size:19px}
.sub{color:var(--mut);font-size:12.5px;margin-top:5px;line-height:1.6}
.warn{background:#fff4e2;border:1px solid #f0c987;color:#8a5a18;
padding:9px 12px;border-radius:8px;font-size:12.3px;margin-top:9px}
.wrap{display:grid;grid-template-columns:minmax(340px,1fr) minmax(340px,1fr);
gap:18px;padding:18px 24px;max-width:1500px;margin:auto}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;
padding:16px;box-shadow:0 1px 3px rgba(30,42,58,.05)}
.card h2{font-size:14.5px;margin:0 0 12px;padding-bottom:8px;
border-bottom:1px solid var(--bd)}
label{display:block;font-size:12.5px;color:var(--mut);margin:10px 0 5px}
textarea,input[type=text]{width:100%;border:1px solid var(--bd);border-radius:8px;
padding:9px;font-family:inherit;font-size:13px;resize:vertical}
textarea{height:92px}
input[type=range]{width:100%}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.pill{display:inline-block;border:1px solid var(--bd);background:#f8fafc;
border-radius:20px;padding:5px 11px;margin:3px 4px 3px 0;cursor:pointer;
font-size:12px;color:var(--ink)}
.pill:hover{border-color:var(--blue);color:var(--blue);background:#eef4ff}
.pill.edit{border-color:#e7d3b2;background:#fffaf1}
.group{font-size:11.5px;color:var(--mut);margin-top:9px}
button.gen{width:100%;margin-top:14px;background:var(--blue);color:#fff;
border:0;border-radius:9px;padding:12px;font-size:14.5px;font-weight:700;
cursor:pointer}
button.gen:disabled{background:#9db0cc;cursor:wait}
button.small{border:1px solid var(--bd);background:#fff;border-radius:7px;
padding:5px 9px;font-size:11.8px;cursor:pointer;margin:3px 4px 3px 0}
button.small:hover{border-color:var(--orange);color:var(--orange)}
.imgbox{background:#eef2f7;border:1px dashed var(--bd);border-radius:9px;
min-height:270px;display:flex;align-items:center;justify-content:center;
overflow:hidden;position:relative}
.imgbox img{max-width:100%;max-height:460px;display:block}
.thumbs{display:flex;gap:7px;flex-wrap:wrap;margin-top:8px}
.thumbs img{width:62px;height:62px;object-fit:cover;border:2px solid transparent;
border-radius:7px;cursor:pointer}
.thumbs img.sel{border-color:var(--blue)}
#status{font-size:12.3px;color:var(--mut);line-height:1.7;margin-top:10px;
white-space:pre-wrap}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11.5px;
background:#eef4ff;color:var(--blue)}
pre{background:#f7f9fc;border:1px solid var(--bd);border-radius:8px;padding:10px;
font-size:11.5px;max-height:220px;overflow:auto;white-space:pre-wrap}
.full{grid-column:1/-1}
.err{color:var(--red)} .ok{color:var(--green)} .wr{color:var(--orange)}
</style></head><body>
<header><h1>多模态参考人脸图像恢复 · 交互 Demo</h1>
<div class="sub">上传老照片 + 输入文本 → 真实 FLUX.2 [klein] 4B Base 出图 ·
原生 Qwen3 文本编码 + 原生 <code>image=[ref]</code> · RTX 4070 Ti 12 GB ·
模型 CPU offload · 首次生成需加载 17 GB 权重（约 30–60 s）</div>
<div class="warn"><b>边界声明：</b>本 Demo 展示的是<b>冻结的原生 FLUX.2 能力（零训练）</b>，
不是本项目自训文本旁路的结果；一图多目标严格 A/B 的最新 verdict 仍为
<b>NOT-YET-SUPPORTED（7.28% &lt; 10%）</b>。两者不可混用。</div></header>

<div class="wrap">
<section class="card"><h2>① 输入</h2>
<label>老照片（点击上传，或选下方内置样例）</label>
<input type="file" id="file" accept="image/*">
<div class="thumbs" id="thumbs"></div>
<div class="imgbox" id="inbox"><span style="color:#8a99ab">尚未选择图像</span></div>
<label>文本指令（可点预设快速填入）</label>
<div id="presets"></div>
<textarea id="prompt"></textarea>
<div class="row">
 <div><label>steps <span id="vsteps">20</span></label>
  <input type="range" id="steps" min="4" max="40" value="20"></div>
 <div><label>size <span id="vsize">512</span></label>
  <input type="range" id="size" min="256" max="768" step="16" value="512"></div>
 <div><label>guidance <span id="vguid">4.0</span></label>
  <input type="range" id="guidance" min="1" max="8" step="0.5" value="4"></div>
 <div><label>seed</label>
  <input type="text" id="seed" value="20261113"></div>
</div>
<button class="gen" id="go">生成修复 / 编辑图像</button>
<div id="status">状态：加载中…</div>
</section>

<section class="card"><h2>② 输出</h2>
<div class="imgbox" id="outbox"><span style="color:#8a99ab">等待生成</span></div>
<div id="outmeta" style="font-size:12.3px;color:var(--mut);margin-top:10px"></div>
<label>系统状态 / 最近一次生成</label>
<pre id="sysinfo">加载中…</pre>
</section>

<section class="card full"><h2>③ 鲁棒性 / 边界情况（不加载模型，秒级验证输入校验与自动归一化）</h2>
<div id="edge"></div>
<pre id="edgeres">点击上方任一按钮，查看系统如何优雅处理边界输入。</pre>
<div class="sub" style="margin-top:8px">这些用例对应课程要求「可展示鲁棒性 / 边界情况」：
空图、空文本、超长文本、损坏文件、越界参数、灰度输入。校验逻辑在
<code>app/demo_server.py::validate()</code>，与真实生成共用同一入口。</div>
</section>
</div>

<script>
let INPUT = "";
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function showBox(id, dataUrl){
  const el = $(id);
  if (!dataUrl) { el.innerHTML = '<span style="color:#8a99ab">无图像</span>'; return; }
  el.innerHTML = '<img src="' + dataUrl + '">';
}
function setInput(u){
  INPUT = u; showBox('inbox', u);
  document.querySelectorAll('#thumbs img').forEach(i =>
    i.classList.toggle('sel', i.dataset.u === u));
}
function buildPresets(list){
  let html = '', group = '';
  list.forEach(p => {
    if (p.group !== group) { group = p.group; html += '<div class="group">' + esc(group) + '</div>'; }
    html += '<span class="pill ' + (p.group === '内容编辑' ? 'edit' : '') +
            '" data-p="' + encodeURIComponent(p.prompt) + '">' + esc(p.label) + '</span>';
  });
  $('presets').innerHTML = html;
  document.querySelectorAll('#presets .pill').forEach(el =>
    el.onclick = () => { $('prompt').value = decodeURIComponent(el.dataset.p); });
}
async function refresh(){
  try {
    const s = await (await fetch('/api/status')).json();
    $('status').textContent = '状态：权重 ' + (s.weights_present ? '已就位' : '缺失') +
      ' · pipeline ' + s.pipeline_state +
      (s.gpu_free_mib ? ' · GPU 空闲 ' + s.gpu_free_mib + ' MiB' : '') +
      (s.pipeline_error ? '\n错误：' + s.pipeline_error : '');
    $('sysinfo').textContent = JSON.stringify(s, null, 2);
    return s;
  } catch(e) { $('status').textContent = '状态接口失败：' + e; }
}
async function loadExamples(s){
  buildPresets(s.presets);
  const r = await (await fetch('/api/examples')).json();
  $('thumbs').innerHTML = r.items.map(i =>
    '<img src="' + i.url + '" data-u="' + i.url + '" title="' + esc(i.name) + '">').join('');
  document.querySelectorAll('#thumbs img').forEach(i => i.onclick = () => setInput(i.dataset.u));
  if (r.items.length) setInput(r.items[0].url);
}
$('file').onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  const rd = new FileReader(); rd.onload = () => setInput(rd.result); rd.readAsDataURL(f);
};
['steps','size','guidance'].forEach(k => $(k).oninput = () => {
  $('v' + (k === 'guidance' ? 'guid' : k)).textContent = $(k).value;
});
$('go').onclick = async () => {
  const btn = $('go'); btn.disabled = true; btn.textContent = '生成中（首次需加载 17 GB 权重）…';
  $('outmeta').textContent = '';
  try {
    const payload = {image: INPUT, prompt: $('prompt').value,
      steps: +$('steps').value, size: +$('size').value,
      guidance: +$('guidance').value, seed: +$('seed').value};
    const res = await fetch('/api/generate', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    const j = await res.json();
    if (!res.ok || !j.ok) throw new Error((j.errors || []).join('；') || j.error || '生成失败');
    showBox('outbox', j.image);
    $('outmeta').innerHTML = '<span class="ok">生成成功</span> · ' + j.seconds + ' s · ' +
      'guidance=' + j.used_guidance +
      (j.warnings.length ? '<br><span class="wr">' + esc(j.warnings.join('；')) + '</span>' : '');
  } catch(e) { $('outmeta').innerHTML = '<span class="err">' + esc(e.message) + '</span>'; }
  btn.disabled = false; btn.textContent = '生成修复 / 编辑图像'; refresh();
};

function grayDataUrl(){
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const x = c.getContext('2d'); const g = x.createLinearGradient(0,0,128,128);
  g.addColorStop(0,'#111'); g.addColorStop(1,'#eee'); x.fillStyle = g; x.fillRect(0,0,128,128);
  return c.toDataURL('image/png');
}
const EDGE = [
  ['空图', () => ({image:'', prompt:'restore this photo'})],
  ['空文本', () => ({image:INPUT, prompt:''})],
  ['超长文本 2000 字', () => ({image:INPUT, prompt:'repair the photo '.repeat(130)})],
  ['损坏图像文件', () => ({image:'data:image/png;base64,bm90X2FuX2ltYWdl', prompt:'restore'})],
  ['越界参数 steps=999 size=4096', () => ({image:INPUT, prompt:'restore', steps:999, size:4096, guidance:99, seed:-1})],
  ['灰度 / 非 RGB 输入', () => ({image:grayDataUrl(), prompt:'restore'})],
];
$('edge').innerHTML = EDGE.map((e,i) => '<button class="small" data-i="'+i+'">'+esc(e[0])+'</button>').join('');
document.querySelectorAll('#edge button').forEach(b => b.onclick = async () => {
  const payload = EDGE[+b.dataset.i][1]();
  $('edgeres').textContent = '请求中…';
  try {
    const r = await fetch('/api/validate', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    const j = await r.json();
    $('edgeres').textContent = '用例：' + EDGE[+b.dataset.i][0] + '\n' +
      'HTTP ' + r.status + '  ok=' + j.ok + '\n' +
      'errors  : ' + JSON.stringify(j.errors) + '\n' +
      'warnings: ' + JSON.stringify(j.warnings) + '\n' +
      '归一化后: ' + JSON.stringify(j.normalized);
  } catch(e) { $('edgeres').textContent = String(e); }
});
(async () => { const s = await refresh(); if (s) await loadExamples(s); })();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "CourseDemo/1.0"

    def log_message(self, fmt, *args):
        print(f"[demo] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        p = self.path.split("?", 1)[0]
        if p in ("/", "/index.html"):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif p == "/api/status":
            self._json(200, _status())
        elif p == "/api/examples":
            items = []
            if EXAMPLE_DIR.is_dir():
                for f in sorted(EXAMPLE_DIR.glob("*.jpg"))[:6]:
                    im = Image.open(f).convert("RGB")
                    # keep the native 512x512: the UI shows a CSS thumbnail but
                    # sends FULL resolution to the model (better restoration)
                    im.thumbnail((512, 512), Image.LANCZOS)
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=88)
                    items.append({"name": f.name,
                                  "url": "data:image/jpeg;base64,"
                                         + base64.b64encode(buf.getvalue()).decode("ascii")})
            self._json(200, {"items": items})
        else:
            self._json(404, {"error": "not found", "path": p})

    def do_POST(self):
        p = self.path.split("?", 1)[0]
        if p not in ("/api/validate", "/api/generate"):
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:                                     # noqa: BLE001
            self._json(400, {"ok": False,
                             "errors": [f"请求体不是合法 JSON: {type(e).__name__}"]})
            return
        v = validate(payload)
        if p == "/api/validate":
            self._json(200 if v["ok"] else 400, {
                "ok": v["ok"], "errors": v["errors"],
                "warnings": v["warnings"], "normalized": v["normalized"]})
            return
        if not v["ok"]:
            self._json(400, {"ok": False, "errors": v["errors"],
                             "warnings": v["warnings"]})
            return
        try:
            self._json(200, generate(v))
        except Exception as e:                                     # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"ok": False,
                             "error": f"{type(e).__name__}: {e}",
                             "errors": [str(e)],
                             "warnings": v["warnings"]})


def selftest() -> int:
    """Run every boundary case WITHOUT loading the 17 GB model."""
    cases = [
        ("空图", {"image": "", "prompt": "restore"}),
        ("空文本", {"image": "", "prompt": ""}),
        ("超长文本", {"image": "", "prompt": "x" * 2000}),
        ("损坏图像", {"image": "data:image/png;base64,bm90X2FuX2ltYWdl",
                      "prompt": "restore"}),
        ("越界参数", {"image": "", "prompt": "restore", "steps": 999,
                      "size": 4096, "guidance": 99, "seed": -1}),
    ]
    ok = True
    for name, p in cases:
        v = validate(p)
        print(f"[selftest] {name:10s} ok={v['ok']!s:5s} "
              f"errors={v['errors']} warnings={v['warnings']} "
              f"norm={v['normalized']}")
        ok &= isinstance(v["normalized"], dict)
    # a real RGB image should pass
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 90, 60)).save(buf, format="PNG")
    v = validate({"image": "data:image/png;base64,"
                           + base64.b64encode(buf.getvalue()).decode("ascii"),
                  "prompt": "restore this photo"})
    print(f"[selftest] 合法输入   ok={v['ok']} errors={v['errors']}")
    ok &= v["ok"]
    print("[selftest] PASS" if ok else "[selftest] FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[demo] http://{a.host}:{a.port}/  weights={WEIGHTS.is_dir()}", flush=True)
    print("[demo] Ctrl+C to stop", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
