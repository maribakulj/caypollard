#!/usr/bin/env python3
"""Train a scene-graph student on a rented GPU: Qwen2.5-VL-7B with LoRA, Opus as teacher.

Runs on Colab (L4, 24 GB) or Kaggle; everything sits in ``ROOT`` (default ``/content``):

- ``code.tar`` (``git archive HEAD src scripts``);
- ``train.tar.gz`` (``data/derived/scene-train``, dereferenced), ``teacher.tar.gz`` (the
  Opus graphs, ``data/derived/atelier/entrainement/scene--opus-scene``);
- ``relations.tar.gz`` and ``subjects.tar.gz`` (the two benches it is scored on).

Steps, all fixed before any result:

1. Near-copies of bench pictures leave the training set: DINOv2-base CLS cosine above 0.9
   to any picture of either bench (the threshold used to clean the ``relations`` bench).
   Graphs without figures are dropped too.
2. 5 % of the rest (seed 0) is held out to watch the loss on unseen pictures.
3. LoRA on the language model only (rank 16, alpha 32, dropout 0.05, on the attention and
   MLP projections), the vision tower frozen; bf16 weights, gradient checkpointing, one
   picture per step, 8 steps accumulated, learning rate 1e-4 with linear decay, 2 epochs.
   The loss counts the answer only: the teacher's JSON (figures, relations, subject).
4. The trained model reads both benches with the zero-shot prompt and settings (greedy,
   256-640 x 28 x 28 pixels); one JSON per picture, as the ``scene`` cache expects.

Output: ``ROOT/student.tar.gz`` holding ``adapter/``, ``log.json`` (losses, kept and
dropped pictures, timings) and ``relations/``, ``sujets/`` (the graphs). ``CAY_SMOKE=1``
runs 4 training steps and reads 3 pictures, to check the chain before paying for it.
"""

import glob
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = os.environ.get("CAY_REMOTE_ROOT", "/content")
SMOKE = os.environ.get("CAY_SMOKE") == "1"
REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
WORK = ROOT + "/cay"
OUT = ROOT + "/student"

os.makedirs(WORK + "/data/derived", exist_ok=True)
for name, where in (
    ("code.tar", WORK),
    ("train.tar.gz", WORK + "/data/derived"),
    ("teacher.tar.gz", WORK + "/teacher"),
    ("relations.tar.gz", WORK + "/data/derived"),
    ("subjects.tar.gz", WORK + "/data/derived"),
):
    with tarfile.open(f"{ROOT}/{name}") as archive:
        archive.extractall(where)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "transformers==5.17.0", "peft"], check=True
)
# Colab ships an old torchao that peft refuses to import alongside; it is not used here.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=False)
sys.path.insert(0, WORK + "/src")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import (  # noqa: E402
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from caypollard.atelier.bricks import (  # noqa: E402
    SCENE_POSTURES,
    SCENE_PROMPT,
    SCENE_RELATIONS,
    SCENE_ROLES,
)

PROMPT = SCENE_PROMPT.split("\n", 2)[2].format(
    path="",
    roles=", ".join(SCENE_ROLES),
    postures=", ".join(SCENE_POSTURES),
    relations=", ".join(SCENE_RELATIONS),
)
log = {"model": REPO, "smoke": SMOKE}
start = time.time()


def pictures(folder):
    return sorted(glob.glob(f"{WORK}/data/derived/{folder}/images/*.jpg"))


# 1. Near-copies of bench pictures leave the training set.
train_paths = pictures("scene-train")
bench_paths = pictures("relations") + pictures("subjects")
dino = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda").eval()
dino_in = AutoImageProcessor.from_pretrained("facebook/dinov2-base")


def embed(paths, encoder):
    out = []
    for i in range(0, len(paths), 32):
        batch = [Image.open(p).convert("RGB") for p in paths[i : i + 32]]
        with torch.no_grad():
            cls = encoder(**dino_in(images=batch, return_tensors="pt").to("cuda")).pooler_output
        out.append(torch.nn.functional.normalize(cls.float(), dim=-1))
    return torch.cat(out)


closest = (embed(train_paths, dino) @ embed(bench_paths, dino).T).max(1).values
del dino
torch.cuda.empty_cache()
teacher = {}
for p in glob.glob(WORK + "/teacher/*.json"):
    with open(p) as f:
        teacher[os.path.basename(p)[:-5]] = json.load(f)
examples, dropped = [], {"near_copy": [], "no_graph": []}
for path, c in zip(train_paths, closest.tolist(), strict=True):
    item = os.path.basename(path)[:-4]
    graph = teacher.get(item, {})
    if c > 0.9:
        dropped["near_copy"].append(item)
    elif not graph.get("figures"):
        dropped["no_graph"].append(item)
    else:
        answer = {
            k: graph.get(k, [] if k != "subject" else "")
            for k in ("figures", "relations", "subject")
        }
        examples.append((path, json.dumps(answer, ensure_ascii=False)))
random.Random(0).shuffle(examples)
held = max(1, len(examples) // 20)
val, train = examples[:held], examples[held:]
log.update(kept=len(examples), train=len(train), val=len(val), dropped=dropped)
print(f"train {len(train)}, val {len(val)}, dropped {sum(map(len, dropped.values()))}", flush=True)

# 2-3. LoRA on the language model, the answer only in the loss.
from peft import LoraConfig, get_peft_model  # noqa: E402

processor = AutoProcessor.from_pretrained(REPO, min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    REPO, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
log["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)


def batch_for(path, answer=None):
    user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(user, add_generation_prompt=True)
    image = Image.open(path).convert("RGB")
    if answer is None:
        return processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
    full = processor.apply_chat_template(
        [*user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    )
    inputs = processor(text=[full], images=[image], return_tensors="pt").to("cuda")
    n_prompt = processor(text=[prompt], images=[image], return_tensors="pt").input_ids.shape[1]
    labels = inputs.input_ids.clone()
    labels[:, :n_prompt] = -100
    inputs["labels"] = labels
    return inputs


def val_loss():
    model.eval()
    losses = []
    with torch.no_grad():
        for path, answer in val[: 3 if SMOKE else len(val)]:
            losses.append(model(**batch_for(path, answer)).loss.item())
    model.train()
    return sum(losses) / len(losses)


epochs, accumulate = 2, 8
steps = 4 if SMOKE else epochs * len(train)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda s: max(0.0, 1 - s / max(1, steps // accumulate))
)
log["losses"], log["val_losses"] = [], [val_loss()]
model.train()
running = 0.0
for step in range(steps):
    path, answer = train[step % len(train)]
    loss = model(**batch_for(path, answer)).loss / accumulate
    loss.backward()
    running += loss.item()
    if (step + 1) % accumulate == 0 or step + 1 == steps:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        log["losses"].append(round(running, 4))
        running = 0.0
    if (step + 1) % len(train) == 0 or step + 1 == steps:
        log["val_losses"].append(val_loss())
        print(f"step {step + 1}/{steps} val {log['val_losses'][-1]:.4f}", flush=True)
    if (step + 1) % 100 == 0:
        print(f"step {step + 1}/{steps} loss {log['losses'][-1]}", flush=True)
log["train_seconds"] = round(time.time() - start)
model.save_pretrained(OUT + "/adapter")

# 4. The student reads both benches.
model.eval()
for bench, folder in (("relations", "relations"), ("sujets", "subjects")):
    os.makedirs(f"{OUT}/{bench}", exist_ok=True)
    for path in pictures(folder)[: 3 if SMOKE else None]:
        item = os.path.basename(path)[:-4]
        t0 = time.time()
        inputs = batch_for(path)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=700, do_sample=False)
        text = processor.batch_decode(
            ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0]
        s = text.find("{")
        try:
            data, _ = json.JSONDecoder().raw_decode(text[s:]) if s >= 0 else ({}, 0)
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data = {
            "figures": [],
            "relations": [],
            "subject": "",
            **data,
            "_raw": text,
            "seconds": round(time.time() - t0, 1),
        }
        with open(f"{OUT}/{bench}/{item}.json", "w") as f:
            json.dump(data, f, ensure_ascii=False)
    print(f"{bench} read", flush=True)
log["total_seconds"] = round(time.time() - start)
with open(OUT + "/log.json", "w") as f:
    json.dump(log, f, indent=1)
shutil.make_archive(ROOT + "/student", "gztar", OUT)
print("done", flush=True)
