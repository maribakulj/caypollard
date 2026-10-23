#!/usr/bin/env python3
"""Train a scene-graph student on a rented GPU: Qwen2.5-VL-7B with LoRA, Opus as teacher.

Runs on Colab (L4, 24 GB) or Kaggle; everything sits in ``ROOT`` (default ``/content``):

- ``code.tar`` (``git archive HEAD src scripts``);
- ``train.tar.gz`` (``data/derived/scene-train``, dereferenced), ``teacher.tar.gz`` (the
  Opus graphs, ``data/derived/atelier/entrainement/scene--opus-scene``);
- ``relations.tar.gz`` and ``subjects.tar.gz`` (the two benches it is scored on).

Steps, all fixed before any result (revised after the Codex review of 23/10/2026):

1. The manifests are authoritative. Every training picture must have its Opus graph: an
   absent graph stops the run (outside the smoke check); graphs without figures are left
   out and counted apart. Near-copies of bench pictures leave the training set: DINOv2-base
   CLS cosine above 0.9 to any picture of either bench, logged with the bench neighbour.
2. 5 % of the rest (seed 0) is held out to watch the loss; it separates files, not works
   (copies inside the training set are not grouped). The split ids are logged.
3. The same model WITHOUT adapter, in the same precision (bf16), first reads both benches:
   the zero-shot reference the trained model is compared with.
4. LoRA on the language model only (rank 16, alpha 32, dropout 0.05, on the attention and
   MLP projections; the trainable names are checked), the vision tower frozen; gradient
   checkpointing, use_cache off while training, one picture per step, 8 steps accumulated
   (the last group weighted by its real size), AdamW at 1e-4 decaying linearly over
   ceil(steps / 8) updates, 2 epochs in the same order, seeds 0. The loss counts the
   answer only: the teacher's JSON (figures, relations, subject).
5. The trained model reads both benches with the zero-shot prompt and settings (greedy,
   700 new tokens, 256-640 x 28 x 28 pixels); one JSON per picture.

Output: ``ROOT/student.tar.gz`` holding ``adapter/``, ``log.json`` (written as the run goes:
versions, archive hashes, model revision, exclusions, split ids, losses, GPU memory peaks)
and ``zeroshot-bf16/`` and ``lora/``, each with ``relations/`` and ``sujets/``. peft is not
pinned; its version is logged. ``CAY_SMOKE=1`` runs 4 training steps and reads 3 pictures
per bench, to check the chain before paying for it.
"""

import glob
import hashlib
import json
import math
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
log = {"model": REPO, "smoke": SMOKE, "versions": {}}
start = time.time()
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def write_log():
    os.makedirs(OUT, exist_ok=True)
    log["seconds"] = round(time.time() - start)
    log["gpu_max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    log["gpu_max_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    with open(OUT + "/log.json", "w") as f:
        json.dump(log, f, indent=1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_paths(folder):
    """The pictures a manifest lists (the manifests are authoritative, not the folders)."""
    with open(f"{WORK}/data/derived/{folder}/manifest.json") as f:
        ids = [row["id"] for row in json.load(f)]
    paths = [f"{WORK}/data/derived/{folder}/images/{i}.jpg" for i in ids]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{folder}: {len(missing)} pictures listed but absent, e.g. {missing[:3]}")
    return paths


import peft  # noqa: E402
import transformers  # noqa: E402

log["versions"] = {
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
}
log["archives_sha256"] = {
    n: sha256(f"{ROOT}/{n}")
    for n in ("code.tar", "train.tar.gz", "teacher.tar.gz", "relations.tar.gz", "subjects.tar.gz")
}

# 1. Every training picture must have its teacher graph; near-copies of bench pictures leave.
train_paths = manifest_paths("scene-train")
bench_paths = manifest_paths("relations") + manifest_paths("subjects")
teacher = {}
for p in glob.glob(WORK + "/teacher/*.json"):
    with open(p) as f:
        teacher[os.path.basename(p)[:-5]] = json.load(f)
absent = [os.path.basename(p)[:-4] for p in train_paths if os.path.basename(p)[:-4] not in teacher]
log["teacher_absent"] = absent
if absent and not SMOKE:
    write_log()
    raise SystemExit(f"{len(absent)} training pictures have no teacher graph: freeze the archive")
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


best = (embed(train_paths, dino) @ embed(bench_paths, dino).T).max(1)
del dino
torch.cuda.empty_cache()
examples = []
dropped = {"near_copy": [], "empty_graph": [], "absent": absent}
for path, c, j in zip(train_paths, best.values.tolist(), best.indices.tolist(), strict=True):
    item = os.path.basename(path)[:-4]
    graph = teacher.get(item)
    if graph is None:
        continue
    if c > 0.9:
        dropped["near_copy"].append(
            {
                "id": item,
                "bench_neighbour": os.path.basename(bench_paths[j])[:-4],
                "cos": round(c, 4),
            }
        )
    elif not graph.get("figures"):
        dropped["empty_graph"].append(item)
    else:
        answer = {
            k: graph.get(k, [] if k != "subject" else "")
            for k in ("figures", "relations", "subject")
        }
        examples.append((path, json.dumps(answer, ensure_ascii=False)))
random.Random(0).shuffle(examples)
held = max(1, len(examples) // 20)
val, train = examples[:held], examples[held:]
if not train or not val:
    write_log()
    raise SystemExit(f"nothing to train on: train {len(train)}, val {len(val)}")
log.update(
    kept=len(examples),
    train=len(train),
    val=len(val),
    dropped=dropped,
    val_ids=[os.path.basename(p)[:-4] for p, _ in val],
    train_ids=[os.path.basename(p)[:-4] for p, _ in train],
    validation_note="files, not works: copies inside the training set are not grouped",
)
print(
    f"train {len(train)}, val {len(val)}, near copies {len(dropped['near_copy'])}, "
    f"empty {len(dropped['empty_graph'])}, absent {len(absent)}",
    flush=True,
)
write_log()

processor = AutoProcessor.from_pretrained(REPO, min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    REPO, torch_dtype=torch.bfloat16, device_map="cuda"
)
log["model_revision"] = getattr(model.config, "_commit_hash", None)


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


def read_benches(tag):
    """The model as it stands reads both benches with the zero-shot prompt and settings."""
    model.eval()
    for bench, folder in (("relations", "relations"), ("sujets", "subjects")):
        os.makedirs(f"{OUT}/{tag}/{bench}", exist_ok=True)
        for path in manifest_paths(folder)[: 3 if SMOKE else None]:
            item = os.path.basename(path)[:-4]
            t0 = time.time()
            inputs = batch_for(path)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=700, do_sample=False, use_cache=True)
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
            with open(f"{OUT}/{tag}/{bench}/{item}.json", "w") as f:
                json.dump(data, f, ensure_ascii=False)
        print(f"{tag} {bench} read", flush=True)
    write_log()


# 2. The same model without any adapter, in the same precision: the zero-shot reference.
read_benches("zeroshot-bf16")

# 3. LoRA on the language model, the answer only in the loss.
from peft import LoraConfig, get_peft_model  # noqa: E402

model.config.use_cache = False
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
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
if not trainable or any("language_model" not in n or "lora_" not in n for n in trainable):
    raise SystemExit(f"unexpected trainable parameters: {trainable[:5]}")
log["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)


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
updates = math.ceil(steps / accumulate)
log.update(
    epochs=epochs, accumulate=accumulate, steps=steps, updates=updates, order="same each epoch"
)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda u: max(0.0, 1 - u / updates))
log["losses"], log["val_losses"] = [], [val_loss()]
model.train()
running = 0.0
for step in range(steps):
    group_start = step - step % accumulate
    group = min(accumulate, steps - group_start)  # the last group may be shorter
    path, answer = train[step % len(train)]
    loss = model(**batch_for(path, answer)).loss / group
    loss.backward()
    running += loss.item()
    if step + 1 == group_start + group:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        log["losses"].append(round(running, 4))
        running = 0.0
    if (step + 1) % len(train) == 0 or step + 1 == steps:
        log["val_losses"].append(val_loss())
        print(f"step {step + 1}/{steps} val {log['val_losses'][-1]:.4f}", flush=True)
        write_log()
    if (step + 1) % 100 == 0:
        print(f"step {step + 1}/{steps} loss {log['losses'][-1]}", flush=True)
log["train_seconds"] = round(time.time() - start)
model.save_pretrained(OUT + "/adapter")
write_log()

# 4. The student reads both benches.
model.config.use_cache = True
read_benches("lora")
shutil.make_archive(ROOT + "/student", "gztar", OUT)
print("done", flush=True)
