#!/usr/bin/env python3
"""A home-made scene-graph head on a frozen vision encoder, trained from scratch on Opus.

The "family 1" arm of the training trial: no language model. A frozen encoder gives patch
tokens; a small head, trained from random weights on the Opus graphs of the training
pictures, predicts up to 8 figures (a role or "none", and a posture, in the closed
vocabulary of ``opus-scene``) and the relations between every ordered pair of predicted
figures (18 verbs, several allowed). It cannot leave the vocabulary and knows no story.
Its output is the same JSON as the ``scene`` bricks, scored the same way.

Runs on Colab or Kaggle; in ``ROOT`` (default ``/content``): ``code.tar``,
``train.tar.gz`` (``data/derived/scene-train``), ``teacher.tar.gz`` (the frozen Opus
graphs), ``relations.tar.gz`` and ``subjects.tar.gz``. ``CAY_ENCODERS`` lists encoders
among ``pe-spatial``, ``dinov2``, ``siglip2`` (default: all three, fixed together).

Settings, fixed before any result (24/10/2026):

- the training pictures, exclusions (near-copies of bench pictures, DINOv2 cosine above
  0.9; graphs without figures) and the 5 % validation split (seed 0) are those of
  ``scripts/remote_train_scene_lora.py``, recomputed the same way and logged with their
  ids; since the head also sees training pictures mirrored, a picture whose MIRROR is a
  near-copy of a bench picture is also left out (counted apart: the LoRA run filtered
  originals only);
- known limit, accepted before any result (Codex review): the Hungarian matching uses
  roles and postures only, so two identical figures may be swapped; the graph is the
  same but the relation loss is not;
- encoders, frozen, eval mode: PE-Spatial large 448 px (timm
  ``vit_pe_spatial_large_patch14_448.fb``), DINOv2-base 448 px, SigLIP 2 base 384 px
  (``google/siglip2-base-patch16-384``); patch tokens pooled to a 16 x 16 grid; the
  training pictures are also encoded mirrored (relations and roles do not change in a
  mirror), the bench pictures only as they are;
- head: the 256 grid tokens projected to 256 dimensions plus a learned 2-D position
  embedding; a 3-layer transformer decoder (8 heads, width 256, dropout 0.1) with 8
  learned queries; per query a role classifier (10 roles + "none") and a posture
  classifier (7); per ordered pair of queries an MLP on the two query states giving 18
  independent relation scores;
- training: each picture's teacher figures (at most 8, in order) matched to the queries
  by the Hungarian algorithm on the role and posture log-probabilities; losses = role
  cross-entropy (unmatched queries -> "none", weight 0.2 for "none") + posture
  cross-entropy on matched queries + relation binary cross-entropy on all ordered pairs of
  matched queries (positives from the teacher, the rest negative); AdamW 3e-4, weight
  decay 0.01, batch 32, 60 epochs, cosine decay; the epoch with the lowest validation loss
  is kept; seed 0;
- reading: a query is a figure when "none" is not its most likely class; a relation is
  kept when its score is above 0.5; roles, postures and verbs are written as the
  vocabulary words.

Output: ``ROOT/head.tar.gz`` with, per encoder, ``<encoder>/relations/*.json`` and
``<encoder>/sujets/*.json`` (bricks ``head-<encoder>-scene``), and ``log.json``.
"""

import glob
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = os.environ.get("CAY_REMOTE_ROOT", "/content")
ENCODERS = os.environ.get("CAY_ENCODERS", "pe-spatial,dinov2,siglip2").split(",")
WORK = ROOT + "/cay"
OUT = ROOT + "/head"

shutil.rmtree(WORK, ignore_errors=True)
shutil.rmtree(OUT, ignore_errors=True)
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
    [sys.executable, "-m", "pip", "install", "-q", "transformers==5.17.0", "timm"], check=True
)
sys.path.insert(0, WORK + "/src")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from caypollard.atelier.bricks import SCENE_POSTURES, SCENE_RELATIONS, SCENE_ROLES  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
DEV = "cuda"
ROLES = list(SCENE_ROLES)  # index len(ROLES) = "none"
POSTURES = list(SCENE_POSTURES)
VERBS = list(SCENE_RELATIONS)
Q = 8
log = {"encoders": ENCODERS, "start": time.time()}


def write_log():
    os.makedirs(OUT, exist_ok=True)
    with open(OUT + "/log.json", "w") as f:
        json.dump(log, f, indent=1)


def manifest_paths(folder):
    with open(f"{WORK}/data/derived/{folder}/manifest.json") as f:
        ids = [row["id"] for row in json.load(f)]
    return [f"{WORK}/data/derived/{folder}/images/{i}.jpg" for i in ids]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


log["archives_sha256"] = {
    n: sha256(f"{ROOT}/{n}")
    for n in ("code.tar", "train.tar.gz", "teacher.tar.gz", "relations.tar.gz", "subjects.tar.gz")
}

# --- the training examples, as in remote_train_scene_lora.py ---------------------------
train_paths = manifest_paths("scene-train")
benches = {"relations": manifest_paths("relations"), "sujets": manifest_paths("subjects")}
teacher = {}
for p in glob.glob(WORK + "/teacher/*.json"):
    with open(p) as f:
        teacher[os.path.basename(p)[:-5]] = json.load(f)
absent = [p for p in train_paths if os.path.basename(p)[:-4] not in teacher]
if absent:
    raise SystemExit(f"{len(absent)} training pictures without teacher graph")


def target(graph):
    """Teacher figures (role index, posture index) and relation triples between them."""
    figs, index = [], {}
    for f in graph.get("figures") or []:
        if not isinstance(f, dict) or len(figs) == Q:
            continue
        if f.get("role") in ROLES and f.get("posture") in POSTURES:
            index[f.get("id")] = len(figs)
            figs.append((ROLES.index(f["role"]), POSTURES.index(f["posture"])))
    rels = set()
    for r in graph.get("relations") or []:
        if isinstance(r, dict) and r.get("rel") in VERBS:
            a, b = index.get(r.get("a")), index.get(r.get("b"))
            if a is not None and b is not None and a != b:
                rels.add((a, b, VERBS.index(r["rel"])))
    return figs, sorted(rels)


from transformers import AutoImageProcessor, AutoModel  # noqa: E402

dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval()
dino_in = AutoImageProcessor.from_pretrained("facebook/dinov2-base")


def cls_embed(paths, model):
    out = []
    for i in range(0, len(paths), 32):
        batch = [Image.open(p).convert("RGB") for p in paths[i : i + 32]]
        with torch.no_grad():
            cls = model(**dino_in(images=batch, return_tensors="pt").to(DEV)).pooler_output
        out.append(torch.nn.functional.normalize(cls.float(), dim=-1))
    return torch.cat(out)


def cls_embed_mirrored(paths, model):
    out = []
    for i in range(0, len(paths), 32):
        batch = [ImageOps.mirror(Image.open(p).convert("RGB")) for p in paths[i : i + 32]]
        with torch.no_grad():
            cls = model(**dino_in(images=batch, return_tensors="pt").to(DEV)).pooler_output
        out.append(torch.nn.functional.normalize(cls.float(), dim=-1))
    return torch.cat(out)


bench_all = [p for paths in benches.values() for p in paths]
bench_v = cls_embed(bench_all, dino)
closest = (cls_embed(train_paths, dino) @ bench_v.T).max(1).values
# The training pictures are also seen mirrored: a mirrored near-copy of a bench picture
# would leak. Counted apart, so the LoRA run (which filtered originals only) can be judged.
closest_mirror = (cls_embed_mirrored(train_paths, dino) @ bench_v.T).max(1).values
del dino
torch.cuda.empty_cache()
examples = []
excluded = {"near_copy": [], "mirror_near_copy_only": [], "empty_graph": []}
lora_examples = []  # the split exactly as the LoRA run computed it (originals only)
for path, c, cm in zip(train_paths, closest.tolist(), closest_mirror.tolist(), strict=True):
    item = os.path.basename(path)[:-4]
    graph = teacher[item]
    if c > 0.9:
        excluded["near_copy"].append(item)
        continue
    if not graph.get("figures"):
        excluded["empty_graph"].append(item)
        continue
    lora_examples.append(item)
    if cm > 0.9:
        excluded["mirror_near_copy_only"].append(item)
        continue
    examples.append((path, target(graph)))
lora_order = list(lora_examples)
random.Random(0).shuffle(lora_order)
lora_held = max(1, len(lora_order) // 20)
log["lora_split"] = {"val": lora_order[:lora_held], "train": lora_order[lora_held:]}
# The head keeps the LoRA split, minus the mirrored near-copies (if any).
dropped = set(excluded["mirror_near_copy_only"])
by_id = {os.path.basename(p)[:-4]: (p, t) for p, t in examples}
val = [by_id[i] for i in lora_order[:lora_held] if i not in dropped]
train = [by_id[i] for i in lora_order[lora_held:] if i not in dropped]
log.update(
    train=len(train),
    val=len(val),
    excluded=excluded,
    train_ids=[os.path.basename(p)[:-4] for p, _ in train],
    val_ids=[os.path.basename(p)[:-4] for p, _ in val],
    lora_counts_expected={"train": 1110, "val": 58},
    lora_counts_recomputed={"train": len(lora_order) - lora_held, "val": lora_held},
    matching_limit=(
        "the Hungarian matching uses roles and postures only: two identical figures may be "
        "swapped, which leaves the graph unchanged but changes the relation loss (Codex)"
    ),
)
print(f"train {len(train)} val {len(val)}", flush=True)
write_log()


# --- encoders --------------------------------------------------------------------------
def grid16(tokens):
    b, n, d = tokens.shape
    g = round(n**0.5)
    if g * g != n:
        raise ValueError(f"{n} tokens do not make a square grid")
    x = tokens.reshape(b, g, g, d).permute(0, 3, 1, 2).float()
    return torch.nn.functional.adaptive_avg_pool2d(x, 16).flatten(2).transpose(1, 2)


def encoder(name):
    if name == "pe-spatial":
        import timm

        m = timm.create_model("vit_pe_spatial_large_patch14_448.fb", pretrained=True).to(DEV)
        m.eval()
        tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=m))
        skip = getattr(m, "num_prefix_tokens", 0)

        def run(images):
            x = torch.stack([tf(im) for im in images]).to(DEV)
            return grid16(m.forward_features(x)[:, skip:])

        return run
    if name == "dinov2":
        m = AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval()
        proc = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base", size={"height": 448, "width": 448}, do_center_crop=False
        )

        def run(images):
            return grid16(
                m(**proc(images=images, return_tensors="pt").to(DEV)).last_hidden_state[:, 1:]
            )

        return run
    if name == "siglip2":
        from transformers import SiglipVisionModel

        repo = "google/siglip2-base-patch16-384"
        m = SiglipVisionModel.from_pretrained(repo).to(DEV).eval()
        proc = AutoImageProcessor.from_pretrained(repo)

        def run(images):
            return grid16(m(**proc(images=images, return_tensors="pt").to(DEV)).last_hidden_state)

        return run
    raise SystemExit(f"unknown encoder {name}")


def features(run, paths, mirror=False):
    out = []
    for i in range(0, len(paths), 16):
        images = [Image.open(p).convert("RGB") for p in paths[i : i + 16]]
        if mirror:
            images = [ImageOps.mirror(im) for im in images]
        with torch.no_grad():
            out.append(run(images).half().cpu())
    return torch.cat(out)


# --- the head --------------------------------------------------------------------------
class Head(torch.nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        d = 256
        self.proj = torch.nn.Linear(dim_in, d)
        self.pos = torch.nn.Parameter(torch.randn(1, 256, d) * 0.02)
        self.queries = torch.nn.Parameter(torch.randn(1, Q, d))  # spread, or the queries stay alike
        layer = torch.nn.TransformerDecoderLayer(d, 8, 4 * d, dropout=0.1, batch_first=True)
        self.decoder = torch.nn.TransformerDecoder(layer, 3)
        self.role = torch.nn.Linear(d, len(ROLES) + 1)
        self.posture = torch.nn.Linear(d, len(POSTURES))
        self.pair = torch.nn.Sequential(
            torch.nn.Linear(2 * d, d), torch.nn.ReLU(), torch.nn.Linear(d, len(VERBS))
        )

    def forward(self, tokens):
        memory = self.proj(tokens) + self.pos
        h = self.decoder(self.queries.expand(len(tokens), -1, -1), memory)
        pairs = torch.cat(
            [h.unsqueeze(2).expand(-1, -1, Q, -1), h.unsqueeze(1).expand(-1, Q, -1, -1)], -1
        )
        return self.role(h), self.posture(h), self.pair(pairs)


def loss_for(head, feats, targets):
    role, posture, rel = head(feats.to(DEV).float())
    lr, lp = role.log_softmax(-1), posture.log_softmax(-1)
    none = len(ROLES)
    weight = torch.ones(none + 1, device=DEV)
    weight[none] = 0.2
    role_target = torch.full((len(feats), Q), none, device=DEV)
    rel_target = torch.zeros_like(rel)
    rel_mask = torch.zeros(rel.shape[:3], device=DEV)
    posture_terms = []
    for i, (figs, rels) in enumerate(targets):
        if not figs:
            continue
        cost = (
            -(lr[i][:, [f[0] for f in figs]] + lp[i][:, [f[1] for f in figs]])
            .detach()
            .cpu()
            .numpy()
        )
        qs, ts = linear_sum_assignment(cost)
        slot = dict(zip(ts.tolist(), qs.tolist(), strict=True))
        for q, t in zip(qs.tolist(), ts.tolist(), strict=True):
            role_target[i, q] = figs[t][0]
            posture_terms.append(lp[i, q, figs[t][1]])
        for a in slot.values():
            for b in slot.values():
                if a != b:
                    rel_mask[i, a, b] = 1
        for a, b, v in rels:
            rel_target[i, slot[a], slot[b], v] = 1
    loss = torch.nn.functional.cross_entropy(
        role.reshape(-1, none + 1), role_target.reshape(-1), weight=weight
    )
    if posture_terms:
        loss = loss - torch.stack(posture_terms).mean()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(rel, rel_target, reduction="none")
    if rel_mask.sum() > 0:
        loss = loss + (bce.mean(-1) * rel_mask).sum() / rel_mask.sum()
    return loss


def read(head, feats):
    head.eval()
    with torch.no_grad():
        role, posture, rel = head(feats.to(DEV).float())
    out = []
    for i in range(len(feats)):
        figures, ids = [], {}
        for q in range(Q):
            r = int(role[i, q].argmax())
            if r == len(ROLES):
                continue
            ids[q] = f"f{len(figures) + 1}"
            figures.append(
                {"id": ids[q], "role": ROLES[r], "posture": POSTURES[int(posture[i, q].argmax())]}
            )
        relations = [
            {"a": ids[a], "rel": VERBS[v], "b": ids[b]}
            for a in ids
            for b in ids
            if a != b
            for v in range(len(VERBS))
            if rel[i, a, b, v] > 0
        ]
        out.append({"figures": figures, "relations": relations, "subject": ""})
    return out


for name in ENCODERS:
    t0 = time.time()
    run = encoder(name)
    train_x = torch.cat(
        [features(run, [p for p, _ in train]), features(run, [p for p, _ in train], mirror=True)]
    )
    train_y = [t for _, t in train] * 2
    val_x = features(run, [p for p, _ in val])
    val_y = [t for _, t in val]
    bench_x = {b: features(run, paths) for b, paths in benches.items()}
    del run
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    head = Head(train_x.shape[-1]).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    epochs, batch = 60, 32
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, best_epoch, history = float("inf"), None, 0, []
    order = list(range(len(train_x)))
    for epoch in range(epochs):
        head.train()
        random.Random(epoch).shuffle(order)
        for s in range(0, len(order), batch):
            idx = order[s : s + batch]
            loss = loss_for(head, train_x[idx], [train_y[i] for i in idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            v = loss_for(head, val_x, val_y).item()
        history.append(v)
        if v < best:
            best, best_epoch = v, epoch + 1
            best_state = {k: t.detach().clone() for k, t in head.state_dict().items()}
            os.makedirs(f"{OUT}/{name}", exist_ok=True)
            torch.save(best_state, f"{OUT}/{name}/head.pt")
        log[name] = {"val_losses": [round(h, 5) for h in history], "best_epoch": best_epoch}
        write_log()
    head.load_state_dict(best_state)
    for b, x in bench_x.items():
        os.makedirs(f"{OUT}/{name}/{b}", exist_ok=True)
        for path, graph in zip(benches[b], read(head, x), strict=True):
            with open(f"{OUT}/{name}/{b}/{os.path.basename(path)[:-4]}.json", "w") as f:
                json.dump(graph, f)
    log[name] = {
        "val_losses": [round(h, 5) for h in history],
        "best_epoch": best_epoch,
        "parameters": sum(p.numel() for p in head.parameters()),
        "seconds": round(time.time() - t0),
    }
    print(f"{name}: best epoch {log[name]['best_epoch']}, val {best:.4f}", flush=True)
    write_log()
shutil.make_archive(ROOT + "/head", "gztar", OUT)
print("done", flush=True)
