#!/usr/bin/env python3
"""Run the ``qwen25vl3b-scene`` brick on a rented GPU (Colab or Kaggle), zero-shot.

Qwen2.5-VL-3B-Instruct reads every picture of a bench with the same prompt as
``opus-scene`` (without the line asking to open a file) and writes one JSON per picture,
in the same shape as the ``scene`` cache. Greedy decoding, pictures between 256 and 640
x 28 x 28 pixels. Too heavy for the Mac (7.5 GB of weights): it runs remotely, e.g.

    colab new -s cayqwen --gpu T4
    colab upload -s cayqwen code.tar /content/code.tar
    colab upload -s cayqwen relations.tar.gz /content/relations.tar.gz
    colab exec -s cayqwen -f scripts/remote_scene_qwen.py
    colab download -s cayqwen /content/scene-qwen.tar.gz scene-qwen.tar.gz
    colab stop -s cayqwen

where ``code.tar`` is ``git archive HEAD src scripts`` and ``relations.tar.gz`` the bench
folder (``tar -chzf relations.tar.gz -C data/derived relations``). The archive unpacks
into ``data/derived/atelier/relations/scene--qwen25vl3b-scene``.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = os.environ.get("CAY_REMOTE_ROOT", "/content")
WORK = ROOT + "/cay"
OUT = ROOT + "/scene--qwen25vl3b-scene"

os.makedirs(WORK + "/data/derived", exist_ok=True)
with tarfile.open(ROOT + "/code.tar") as archive:
    archive.extractall(WORK)
with tarfile.open(ROOT + "/relations.tar.gz") as archive:
    archive.extractall(WORK + "/data/derived")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.17.0"], check=True)
sys.path.insert(0, WORK + "/src")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

from caypollard.atelier.bricks import (  # noqa: E402
    SCENE_POSTURES,
    SCENE_PROMPT,
    SCENE_RELATIONS,
    SCENE_ROLES,
)

prompt = SCENE_PROMPT.split("\n", 2)[2].format(
    path="",
    roles=", ".join(SCENE_ROLES),
    postures=", ".join(SCENE_POSTURES),
    relations=", ".join(SCENE_RELATIONS),
)
print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
repo = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    repo, torch_dtype=torch.float16, device_map="cuda"
).eval()
processor = AutoProcessor.from_pretrained(repo, min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28)
os.makedirs(OUT, exist_ok=True)
for path in sorted(glob.glob(WORK + "/data/derived/relations/images/*.jpg")):
    item = os.path.basename(path)[:-4]
    if os.path.exists(f"{OUT}/{item}.json"):
        continue
    start = time.time()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image = Image.open(path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=700, do_sample=False)
    new = ids[:, inputs.input_ids.shape[1] :]
    answer = processor.batch_decode(new, skip_special_tokens=True)[0]
    s = answer.find("{")
    try:
        data, _ = json.JSONDecoder().raw_decode(answer[s:]) if s >= 0 else ({}, 0)
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data = {
        "figures": [],
        "relations": [],
        "subject": "",
        **data,
        "_raw": answer,
        "seconds": round(time.time() - start, 1),
    }
    with open(f"{OUT}/{item}.json", "w") as f:
        json.dump(data, f, ensure_ascii=False)
    print(item, len(data.get("relations", [])), data["seconds"], flush=True)
shutil.make_archive(ROOT + "/scene-qwen", "gztar", OUT)
print("done", len(os.listdir(OUT)), flush=True)
