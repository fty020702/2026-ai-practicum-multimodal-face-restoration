from __future__ import annotations

"""Show what the training data actually looks like.

Reproducible visual + structured example of RestorationDataset:

    D:/Anaconda/envs/ChatAI/python.exe scripts/show_data_example.py

Writes outputs/samples/data_example.png and outputs/logs/data_example.json.
Read-only with respect to the dataset: it instantiates the dataset but does
not modify data/ or any config.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont

from src.data.dataset import RestorationDataset
from src.utils import ROOT, load_config, save_json

CELL = 192
LABEL_W = 430
PAD = 12
ROW_H = CELL + 46


def _font(sz):
    for p in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def _upscale(arr):
    return Image.fromarray(arr).resize((CELL, CELL), Image.NEAREST)


def _wrap(text, max_chars):
    lines, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def main():
    cfg = load_config()
    ds = RestorationDataset(cfg, root=ROOT)
    tasks = list(cfg["data"]["degradations"])

    item = ds[0]
    rep = {
        "source": ds.source,
        "num_pairs": len(ds),
        "image_size": cfg["data"]["image_size"],
        "seed": cfg["project"]["seed"],
        "degradation_types": tasks,
        "note": ("data/clean is empty -> fully synthetic and deterministic; "
                 "drop real .png/.jpg into data/clean to switch to real images"),
        "sample_0": {
            "task": item["task"],
            "prompt": item["prompt"],
            "clean": {"shape": list(item["clean"].shape),
                      "dtype": str(item["clean"].dtype),
                      "range": [round(float(item["clean"].min()), 4),
                                round(float(item["clean"].max()), 4)]},
            "degraded": {"shape": list(item["degraded"].shape),
                         "dtype": str(item["degraded"].dtype),
                         "range": [round(float(item["degraded"].min()), 4),
                                   round(float(item["degraded"].max()), 4)]},
        },
        "unique_prompts": sorted({s[3] for s in ds.samples}),
    }
    out_json = ROOT / "outputs" / "logs" / "data_example.json"
    save_json(rep, out_json)
    print(json.dumps(rep, ensure_ascii=False, indent=2))

    rows = [next(s for s in ds.samples if s[2] == t) for t in tasks]

    W = LABEL_W + 2 * CELL + 3 * PAD
    H = 40 + len(rows) * ROW_H + PAD
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(canvas)
    f_title, f_big, f_small = _font(18), _font(15), _font(13)

    d.text((PAD, 10),
           f"RestorationDataset - one example per degradation type  "
           f"(source={ds.source}, {len(ds)} pairs, "
           f"{cfg['data']['image_size']}x{cfg['data']['image_size']})",
           fill=(20, 20, 20), font=f_title)

    for i, (clean, deg, task, prompt) in enumerate(rows):
        y = 40 + i * ROW_H
        canvas.paste(_upscale(clean), (PAD, y))
        canvas.paste(_upscale(deg), (PAD * 2 + CELL, y))
        lx = PAD * 3 + 2 * CELL
        d.text((lx, y + 6), f"task   : {task}", fill=(20, 20, 20), font=f_big)
        for k, line in enumerate(_wrap(prompt, 50)):
            prefix = "prompt : " if k == 0 else "         "
            d.text((lx, y + 30 + k * 17), prefix + line,
                   fill=(60, 60, 60), font=f_small)
        d.text((PAD, y + CELL + 4), "GT clean", fill=(90, 90, 90), font=f_small)
        d.text((PAD * 2 + CELL, y + CELL + 4), "INPUT degraded",
               fill=(90, 90, 90), font=f_small)
        d.line([(PAD, y + CELL + 24), (W - PAD, y + CELL + 24)],
               fill=(220, 220, 220), width=1)

    out_png = ROOT / "outputs" / "samples" / "data_example.png"
    canvas.save(out_png)
    print(f"[data-example] -> {out_png}")
    print(f"[data-example] -> {out_json}")


if __name__ == "__main__":
    main()
