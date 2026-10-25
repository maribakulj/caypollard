#!/usr/bin/env python3
"""The published Iconclass VLMs on the frozen external test (results/iconclass-externe/PREINSCRIPTION.md).

Usage: remote_external_vlm.py 4b|9b <ext_out> <ext_dir> <out>
<ext_out> is the folder of scripts/remote_external_iconclass.py after `prepare` (test.json, frozen.json, vu.json); <ext_dir> holds
images/. Settings are those of the author's published file passk/sft4b_1ep.json (training prompt, greedy, 768 new tokens,
repetition penalty 1.1, default template), fp16, batch 1. Writes <out>/vlm_<size>.jsonl as it goes (resumable: done images are
skipped) and <out>/vlm_<size>_run.json. Scoring is done afterwards, with the same projection and measures as the heads.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

SIZE, EXTOUT, EXT, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"4b": ("davanstrien/qwen35-4b-iconclass-sft-brillfull", "597a5f5c756c21f372bf73d4fa5abcfb745daecb"),
          "9b": ("davanstrien/qwen35-9b-iconclass-sft-brillfull", "4d83102aec672c09eacb07568c8f81367616a1d7")}
MODEL, MODEL_REV = MODELS[SIZE]
REPO, REV = "davanstrien/iconclass-vlm-brillfull", "8045366ff092a06e8e1c4e47f1eb43cb2a54d732"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


assert json.load(open(EXTOUT / "vu.json"))["frozen_sha256"] == sha(EXTOUT / "frozen.json"), "draw not frozen and looked at"
assert sha(EXTOUT / "test.json") == json.load(open(EXTOUT / "frozen.json"))["test.json"]
test = json.load(open(EXTOUT / "test.json"))
assert test
# The pixels: each test image must have the hash recorded in the signed feature cache of the draw.
import numpy as np  # noqa: E402

frozen = json.load(open(EXTOUT / "frozen.json"))
assert sha(EXTOUT / "candidats_feats.npz") == frozen["candidats_feats.npz"]
cache = np.load(EXTOUT / "candidats_feats.npz")
img_sha = dict(zip(cache["files"].tolist(), cache["img_sha"].tolist(), strict=True))
for item in test:
    assert sha(EXT / "images" / item["filename"]) == img_sha[item["filename"]], item["filename"]

import pyarrow.parquet as pq  # noqa: E402
import transformers  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

# The training prompt: the user text of the dataset's conversations, which must be the same for every test row.
msgs = pq.read_table(hf_hub_download(REPO, "data/test-00000-of-00001.parquet", repo_type="dataset", revision=REV),
                     columns=["messages"]).column("messages").to_pylist()


def user_text(conv):
    content = next(m for m in conv if m["role"] == "user")["content"]
    return content if isinstance(content, str) else next(c["text"] for c in content if c.get("type") == "text" and c.get("text"))


prompts = {user_text(c) for c in msgs}
assert len(prompts) == 1, prompts
PROMPT = prompts.pop()

processor = AutoProcessor.from_pretrained(MODEL, revision=MODEL_REV)
model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=MODEL_REV, dtype=torch.float16, device_map="cuda").eval()


def parse(text):
    start = text.find("{")
    try:
        d, _ = json.JSONDecoder().raw_decode(text[start:]) if start >= 0 else ({}, 0)
        codes = d.get("iconclass-codes") if isinstance(d, dict) else None
        ok = isinstance(codes, list) and all(isinstance(c, str) for c in codes)
        return (list(codes), True) if ok else ([], False)
    except json.JSONDecodeError:
        return [], False


@torch.no_grad()
def run(path):
    im = Image.open(path).convert("RGB")
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[im], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=768, do_sample=False, repetition_penalty=1.1)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


log = OUT / f"vlm_{SIZE}.jsonl"
RUN = {"model": [MODEL, MODEL_REV], "settings": "greedy|768|rep1.1|default-template|fp16|batch1|warmup3",
       "prompt": PROMPT, "frozen_sha256": sha(EXTOUT / "frozen.json"), "script_sha256": sha(__file__),
       "transformers": transformers.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
run_file = OUT / f"vlm_{SIZE}_run.json"
if run_file.exists():
    assert json.load(open(run_file))["key"] == RUN, "an earlier run with other settings, versions or GPU: use another folder"
else:
    assert not log.exists(), "a log without its run manifest: refused"
    with open(OUT / f"vlm_{SIZE}_run.tmp", "w") as f:
        json.dump({"key": RUN}, f, indent=1)
    os.replace(OUT / f"vlm_{SIZE}_run.tmp", run_file)
# Resume: only a truncated LAST line is repaired (atomically); a bad line in the middle stops the run.
rows = []
if log.exists():
    lines = open(log).read().split("\n")
    complete, tail = lines[:-1], lines[-1]  # tail is "" when the file ends with a newline
    for line in complete:
        rows.append(json.loads(line))  # raises on corruption in the middle
    if tail:
        try:
            rows.append(json.loads(tail))
            with open(log, "a") as f:  # complete but unterminated: end it before appending
                f.write("\n")
        except json.JSONDecodeError:
            with open(OUT / f"vlm_{SIZE}.tmp", "w") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            os.replace(OUT / f"vlm_{SIZE}.tmp", log)
done = {r["filename"] for r in rows}
assert len(done) == len(rows) and done <= {t["filename"] for t in test}
for item in test[:3]:  # warm-up, not logged
    run(EXT / "images" / item["filename"])
for k, item in enumerate(test):
    if item["filename"] in done:
        continue
    torch.cuda.synchronize()
    t0 = time.time()
    raw = run(EXT / "images" / item["filename"])
    torch.cuda.synchronize()
    codes, ok = parse(raw)
    with open(log, "a") as f:
        f.write(json.dumps({"filename": item["filename"], "codes": codes, "valid_json": ok, "seconds": time.time() - t0,
                            "raw": raw}, ensure_ascii=False) + "\n")
    if k % 50 == 0:
        print(k, len(test), flush=True)
final = [json.loads(line) for line in open(log)]
assert sorted(r["filename"] for r in final) == sorted(t["filename"] for t in test), "incomplete or duplicated"
print("done", flush=True)
