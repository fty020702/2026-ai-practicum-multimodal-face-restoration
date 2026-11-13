from __future__ import annotations

"""Inspect the downloaded InstructPix2Pix shard BEFORE building anything.

Answers, from the actual bytes (no guessing):
  * schema, row count, image encoding / resolution
  * how many DISTINCT original_image bytes there are
  * whether the SAME source image appears with MULTIPLE different edit_prompt
    -> if yes, this official shard is ALREADY an A-class one-input-many-targets
       dataset and we can use it directly
  * how many distinct original_prompt values have >=2 edits

    D:/Anaconda/envs/ChatAI/python.exe scripts/inspect_ip2p_shard.py

Writes outputs/logs/ip2p_shard_inspect.json
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq
from PIL import Image
import io

from src.utils import ROOT


def main() -> int:
    raw = ROOT / "data" / "ip2p_500m" / "raw"
    files = sorted(raw.glob("*.parquet"))
    if not files:
        print("[inspect] no parquet yet; wait for fetch_ip2p_shard.py")
        return 1
    p = files[0]
    pf = pq.ParquetFile(p)
    rep = {
        "file": str(p.relative_to(ROOT)),
        "file_bytes": p.stat().st_size,
        "n_rows": pf.metadata.num_rows,
        "n_row_groups": pf.metadata.num_row_groups,
        "schema": [(f.name, str(f.type)) for f in pf.schema_arrow],
    }

    # --- text columns are small; read them fully ---
    cols = [c for c in ("original_prompt", "edit_prompt", "edited_prompt")
            if c in pf.schema_arrow.names]
    t = pf.read(columns=cols)
    texts = {c: t.column(c).to_pylist() for c in cols}

    by_prompt = defaultdict(list)
    for i in range(rep["n_rows"]):
        key = (texts.get("original_prompt") or [""] * rep["n_rows"])[i] or ""
        by_prompt[key.strip().lower()].append(i)
    multi = {k: v for k, v in by_prompt.items() if len(v) >= 2}
    rep["distinct_original_prompts"] = len(by_prompt)
    rep["prompts_with_multiple_rows"] = len(multi)
    rep["max_rows_for_one_prompt"] = max((len(v) for v in by_prompt.values()),
                                         default=0)

    # --- image columns: stream in batches, hash bytes, sample one ---
    imgcols = [c for c in ("original_image", "edited_image")
               if c in pf.schema_arrow.names]
    hashes = {c: defaultdict(int) for c in imgcols}
    samples = {}
    for batch in pf.iter_batches(batch_size=64, columns=imgcols):
        for c in imgcols:
            for item in batch.column(c).to_pylist():
                if not item:
                    continue
                b = item.get("bytes") if isinstance(item, dict) else None
                if not b:
                    continue
                hashes[c][hashlib.sha1(b).hexdigest()] += 1
                if c not in samples:
                    try:
                        im = Image.open(io.BytesIO(b))
                        samples[c] = {
                            "format": im.format, "size": list(im.size),
                            "mode": im.mode, "bytes": len(b),
                            "path_field": item.get("path") if isinstance(item, dict) else None,
                        }
                    except Exception as e:                       # noqa: BLE001
                        samples[c] = {"error": str(e)[:120]}

    for c in imgcols:
        rep[f"{c}_distinct_bytes"] = len(hashes[c])
        rep[f"{c}_max_identical_rows"] = max(hashes[c].values(), default=0)
        rep[f"{c}_rows_with_duplicate_bytes"] = sum(
            n for n in hashes[c].values() if n > 1
        )
    rep["image_samples"] = samples

    # Does one identical SOURCE image have several different instructions?
    src_col = "original_image"
    if src_col in pf.schema_arrow.names:
        idx_by_hash = defaultdict(list)
        row = 0
        for batch in pf.iter_batches(batch_size=64, columns=[src_col]):
            for item in batch.column(src_col).to_pylist():
                if isinstance(item, dict) and item.get("bytes"):
                    idx_by_hash[hashlib.sha1(item["bytes"]).hexdigest()].append(row)
                row += 1
        groups = [v for v in idx_by_hash.values() if len(v) >= 2]
        edits = texts.get("edit_prompt") or [""] * rep["n_rows"]
        distinct_edit_groups = sum(
            1 for g in groups if len({edits[i] for i in g}) >= 2
        )
        rep["identical_source_image_groups"] = len(groups)
        rep["identical_source_groups_with_different_edits"] = distinct_edit_groups
        rep["a_class_native"] = distinct_edit_groups > 0

    out = ROOT / "outputs" / "logs" / "ip2p_shard_inspect.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("schema", "image_samples")},
                     ensure_ascii=False, indent=2))
    print("schema:", rep["schema"])
    print("image_samples:", json.dumps(samples, ensure_ascii=False))
    print(f"[inspect] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
