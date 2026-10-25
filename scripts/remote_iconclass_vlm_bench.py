#!/usr/bin/env python3
"""Speed and fidelity of the published 4B Iconclass VLM on 50 test images (results/iconclass-tete/PREINSCRIPTION.md, amendment 6).

Usage: remote_iconclass_vlm_bench.py <preds_dir> <out>
Runs davanstrien/qwen35-4b-iconclass-sft-brillfull at its pinned revision, fp16 on the GPU, with the settings of the published file
passk/sft4b_1ep.json (training prompt, greedy, 768 new tokens, repetition penalty 1.1), batch 1, after 3 warm-up images, on the same
50 test images as the head's latency (seed 0). Writes <out>/vlm_bench.json: end-to-end seconds per image (image -> parsed codes) and
fidelity = share of images whose code set and JSON status equal the published greedy output.
"""

import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PREDS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
REPO, REV = "davanstrien/iconclass-vlm-brillfull", "8045366ff092a06e8e1c4e47f1eb43cb2a54d732"
MODEL, MODEL_REV = "davanstrien/qwen35-4b-iconclass-sft-brillfull", "597a5f5c756c21f372bf73d4fa5abcfb745daecb"

import pyarrow.parquet as pq  # noqa: E402
import transformers  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

table = pq.read_table(hf_hub_download(REPO, "data/test-00000-of-00001.parquet", repo_type="dataset", revision=REV),
                      columns=["image", "label", "messages"])
labels, images, messages = (table.column(c).to_pylist() for c in ("label", "image", "messages"))
pub = sorted(json.load(open(PREDS / "passk/sft4b_1ep.json"))["records"], key=lambda r: r["index"])
assert len(pub) == len(labels) == 788 and all(sorted(r["gt_codes"]) == sorted(y) for r, y in zip(pub, labels, strict=True))

# The training prompt: the user turn's text of the dataset's own conversation for that image.
def prompt_of(i):
    user = next(m for m in messages[i] if m["role"] == "user")
    content = user["content"]
    return content if isinstance(content, str) else next(c["text"] for c in content if c.get("type") == "text" and c.get("text"))


processor = AutoProcessor.from_pretrained(MODEL, revision=MODEL_REV)
model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=MODEL_REV, dtype=torch.float16).to("cuda").eval()


def parse(text):
    start = text.find("{")
    try:
        d, _ = json.JSONDecoder().raw_decode(text[start:]) if start >= 0 else ({}, 0)
        codes = d.get("iconclass-codes") if isinstance(d, dict) else None
        ok = isinstance(codes, list) and all(isinstance(c, str) for c in codes)
        return (list(codes), True) if ok else ([], False)  # off-schema output counts as a failure
    except json.JSONDecodeError:
        return [], False


@torch.no_grad()
def run(i):
    im = Image.open(io.BytesIO(images[i]["bytes"])).convert("RGB")
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_of(i)}]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[im], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=768, do_sample=False, repetition_penalty=1.1)
    raw = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return (*parse(raw), raw)


pick = np.random.default_rng(0).choice(788, 50, replace=False)
for i in pick[:3]:
    run(int(i))
rows = []
partial = OUT / "vlm_bench.partial.jsonl"
for i in pick:
    torch.cuda.synchronize()
    t0 = time.time()
    codes, ok, raw = run(int(i))
    torch.cuda.synchronize()
    g = pub[int(i)]["greedy"]
    rows.append({"index": int(i), "seconds": time.time() - t0, "codes": codes, "valid_json": ok,
                 "same_as_published": ok == bool(g.get("valid_json")) and sorted(set(codes)) == sorted(set(g["codes"])),
                 "raw": raw})
    with open(partial, "a") as f:
        f.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
json.dump({"model": [MODEL, MODEL_REV], "transformers": transformers.__version__, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "seconds_median": float(np.median([r["seconds"] for r in rows])),
           "seconds_p90": float(np.quantile([r["seconds"] for r in rows], 0.9)),
           "fidelity": float(np.mean([r["same_as_published"] for r in rows])), "rows": rows},
          open(OUT / "vlm_bench.json", "w"), indent=1)
print("done", flush=True)
