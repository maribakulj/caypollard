#!/usr/bin/env python3
"""Iconclass codes from a frozen encoder + a small head, against the published VLM predictions
(results/iconclass-tete/PREINSCRIPTION.md, amended after Codex's review of 25/10/2026).

Usage: remote_iconclass_head.py <preds_dir> <out>
<preds_dir> holds the author's published per-image predictions (passk/sft4b_1ep.json and passk/sft9b_1ep.json, required;
passk/sft4b_orig.json, descriptive only). The dataset is read from the Hugging Face Hub at its pinned revision, test shard first,
then train shards; only features are kept, each shard with a manifest that a resumed run must match. Writes <out>/run.json,
<out>/preflight.json, <out>/journal.jsonl, <out>/head.pt, <out>/resultats.json, <out>/predictions.json.
"""

import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PREDS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
(OUT / "feats").mkdir(parents=True, exist_ok=True)
REPO, REV = "davanstrien/iconclass-vlm-brillfull", "8045366ff092a06e8e1c4e47f1eb43cb2a54d732"
RADIO, RADIO_REV = "nvidia/C-RADIOv3-L", "9d0413465e8a91e67bbf2c1ad342815478d1b906"
TEST = "data/test-00000-of-00001.parquet"
TRAIN = [f"data/train-{i:05d}-of-00011.parquet" for i in range(11)]
SMOKE = int(os.environ.get("CAY_SMOKE", "0"))  # rows per shard in a dry run; its results are technical only
FEATURE_CONFIG = f"{RADIO}@{RADIO_REV}|square-grey128|448|bicubic|autocast-fp16|summary+meanspatial|l2-each-then-l2"
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
started = time.time()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


import transformers  # noqa: E402

json.dump({"smoke": SMOKE, "device": dev, "torch": torch.__version__, "transformers": transformers.__version__,
           "dataset": [REPO, REV], "feature_config": FEATURE_CONFIG, "script_sha256": sha(__file__),
           "preds_sha256": {f: sha(PREDS / f) for f in ("passk/sft4b_1ep.json", "passk/sft9b_1ep.json", "passk/sft4b_orig.json")
                            if (PREDS / f).exists()}}, open(OUT / "run.json", "w"), indent=1)


# ---- Iconclass hierarchy read from the notation (amendment 1) ----
# A notation is one or more parts joined by ':'. A part is read as tokens: each of the first two digits, then a letter or a doubled
# letter, then each digit, then any bracket (a name '(NAME)' or a key '(+..)') as one token, and the digits after it one by one.
# Its ancestors are the prefixes ending on a token. A compound keeps its own string and the ancestors of each part.
PART = re.compile(r"\d\d?[A-Z]{0,2}\d*(?:\([^)]*\)\d*)*")
TOKEN = re.compile(r"\([^)]*\)|[A-Z]{1,2}|\d")
_anc_cache: dict = {}


def ancestors(code):
    """The code and all its ancestors; an unreadable code is itself only, and counted."""
    code = code.strip()
    if code in _anc_cache:
        return _anc_cache[code]
    parts = code.split(":")
    out = {code}
    if all(PART.fullmatch(p) for p in parts):
        for p in parts:
            cur = ""
            for t in TOKEN.findall(p):
                cur += t
                out.add(cur)
    _anc_cache[code] = frozenset(out)
    return _anc_cache[code]


def readable(code):
    return all(PART.fullmatch(p) for p in code.strip().split(":"))


def hier_f1(gt, pred):
    if not pred or not gt:
        return 0.0
    G = set().union(*map(ancestors, gt))
    P = set().union(*map(ancestors, pred))
    i = len(G & P)
    return 0.0 if i == 0 else 2 * i / (len(G) + len(P))


def most_specific(codes):
    codes = list(dict.fromkeys(codes))
    implied = set()
    for c in codes:
        implied |= ancestors(c) - {c}
    return [c for c in codes if c not in implied]


# ---- Phase 0: test labels and the published references, checked before any encoding (amendment 2) ----
from huggingface_hub import hf_hub_download  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def read_shard(shard):
    path = hf_hub_download(REPO, shard, repo_type="dataset", revision=REV)
    t = pq.read_table(path, columns=["image", "label"])
    return path, t


test_path, test_table = read_shard(TEST)
YT_full = test_table.column("label").to_pylist()
assert SMOKE or len(YT_full) == 788, len(YT_full)
refs, descriptive = {}, {}
for name, f, required in (("sft4b_1ep", "passk/sft4b_1ep.json", True), ("sft9b_1ep", "passk/sft9b_1ep.json", True),
                          ("sft4b_orig", "passk/sft4b_orig.json", False)):
    p = PREDS / f
    if not p.exists():
        assert not required, f"required reference missing: {f}"
        continue
    d = json.load(open(p))
    recs = d["records"]
    idx = [r["index"] for r in recs]
    assert sorted(idx) == list(range(len(YT_full))) and len(set(idx)) == len(idx), f"{f}: indices are not 0..{len(YT_full) - 1}"
    recs = sorted(recs, key=lambda r: r["index"])
    bad = [r["index"] for r, y in zip(recs, YT_full, strict=True) if sorted(r["gt_codes"]) != sorted(y)]
    assert not bad, f"{f}: labels differ from the pinned test at {bad[:5]}"
    preds = [r["greedy"]["codes"] if r["greedy"].get("valid_json") else [] for r in recs]
    entry = {"preds": preds, "declared_hf1": float(np.mean([r["greedy"]["hf1"] for r in recs])),
             "invalid_json": sum(not r["greedy"].get("valid_json") for r in recs), "model": str(d.get("summary", ""))[:200]}
    (refs if required else descriptive)[name] = entry
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "preds"} for k, v in {**refs, **descriptive}.items()},
          open(OUT / "preflight.json", "w"), indent=1)
print("preflight ok:", {k: v["invalid_json"] for k, v in refs.items()}, flush=True)

# ---- Phase 1: C-RADIOv3-L features, test shard first, resumable under a manifest (amendment 3) ----
from transformers import AutoModel, CLIPImageProcessor  # noqa: E402

model = AutoModel.from_pretrained(RADIO, revision=RADIO_REV, trust_remote_code=True).to(dev).eval()
processor = CLIPImageProcessor.from_pretrained(RADIO, revision=RADIO_REV)


def square(im):
    im = im.convert("RGB")
    s = max(im.size)
    c = Image.new("RGB", (s, s), (128, 128, 128))
    c.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
    return c.resize((448, 448), Image.BICUBIC)


@torch.no_grad()
def features(images):
    px = processor(images=[square(i) for i in images], return_tensors="pt", do_resize=False).pixel_values.to(dev)
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):  # RADIO normalises its input in float32 itself
        summary, spatial = model(px)
    v = torch.cat([torch.nn.functional.normalize(summary.float(), dim=-1),
                   torch.nn.functional.normalize(spatial.float().mean(1), dim=-1)], 1)
    return torch.nn.functional.normalize(v, dim=-1).cpu().numpy().astype(np.float16)


def encode_shard(shard, preloaded=None):
    tag = Path(shard).stem
    npz, man = OUT / "feats" / f"{tag}.npz", OUT / "feats" / f"{tag}.json"
    path, table = preloaded or read_shard(shard)
    n = table.num_rows if not SMOKE else min(SMOKE, table.num_rows)
    labels = table.column("label").to_pylist()[:n]
    manifest = {"dataset": [REPO, REV], "shard": shard, "rows": n, "smoke": SMOKE, "feature_config": FEATURE_CONFIG,
                "labels_sha256": hashlib.sha256(json.dumps(labels).encode()).hexdigest()}
    if npz.exists() and man.exists() and json.load(open(man)) == manifest:
        z = np.load(npz)
        assert z["x"].shape[0] == n and np.isfinite(z["x"]).all()
        return z["x"], labels, list(z["md5"])
    imgs = table.column("image").to_pylist()[:n]
    out, md5 = [], []
    for i in range(0, n, 32):
        chunk = imgs[i:i + 32]
        md5 += [hashlib.md5(x["bytes"]).hexdigest() for x in chunk]
        out.append(features([Image.open(io.BytesIO(x["bytes"])) for x in chunk]))
    x = np.concatenate(out)
    assert x.shape[0] == n and np.isfinite(x).all()
    np.savez(OUT / "feats" / f"{tag}.tmp.npz", x=x, md5=np.array(md5))
    os.replace(OUT / "feats" / f"{tag}.tmp.npz", npz)
    with open(OUT / "feats" / f"{tag}.tmp.json", "w") as f:
        json.dump(manifest, f)
    os.replace(OUT / "feats" / f"{tag}.tmp.json", man)
    if path != test_path:
        os.remove(path)  # keep only the features
    print(f"{tag}: {n} rows, {time.time() - started:.0f} s", flush=True)
    return x, labels, md5


XT, YT, MT = encode_shard(TEST, (test_path, test_table))
parts = [encode_shard(s) for s in TRAIN]
X = np.concatenate([p[0] for p in parts]).astype(np.float32)
Y = [y for p in parts for y in p[1]]
M = [m for p in parts for m in p[2]]
XT = XT.astype(np.float32)
if not SMOKE:
    assert len(X) == 86216 and len(XT) == 788, (len(X), len(XT))
refs = {k: {**v, "preds": v["preds"][:len(YT)]} for k, v in refs.items()}

# Validation: 5 % of the train, drawn by groups of byte-identical images (seed 0); exact train/test copies are reported.
groups = sorted(set(M))
gval = set(np.array(groups)[np.random.default_rng(0).random(len(groups)) < 0.05])
val = np.array([m in gval for m in M])
copies_train_test = int(sum(m in set(M) for m in MT))
print(f"train {len(X)} (val {val.sum()}), test {len(XT)}, exact test copies in train {copies_train_test}", flush=True)


def per_image(gts, preds):
    return np.array([hier_f1(g, p) for g, p in zip(gts, preds, strict=True)])


def exact(gts, preds):
    tp = sum(len(set(g) & set(p)) for g, p in zip(gts, preds, strict=True))
    npred, ngt = sum(len(set(p)) for p in preds), sum(len(set(g)) for g in gts)
    pr, rc = tp / max(npred, 1), tp / max(ngt, 1)
    return {"precision": pr, "recall": rc, "f1": 0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)}


def boot_diff(a, b):
    idx = np.random.default_rng(0).integers(0, len(a), (2000, len(a)))
    d = (a[idx] - b[idx]).mean(1)
    return {"diff": float(a.mean() - b.mean()), "low95": float(np.quantile(d, 0.025)), "high95": float(np.quantile(d, 0.975))}


grid = [round(float(x), 2) for x in np.arange(0.05, 0.61, 0.05)]

# ---- K: nearest neighbours ----
Xtr, Ytr, Xva, Yva = X[~val], [y for y, v in zip(Y, val) if not v], X[val], [y for y, v in zip(Y, val) if v]
Xtr_dev = torch.from_numpy(Xtr).to(dev)


def knn_votes(Q, k=20):
    out = []
    for i in range(0, len(Q), 1024):
        w, j = (torch.from_numpy(Q[i:i + 1024]).to(dev) @ Xtr_dev.T).topk(min(k, len(Xtr)), dim=1)
        for wi, ji in zip(w.cpu().numpy(), j.cpu().numpy(), strict=True):
            votes = {}
            for wt, jj in zip(wi, ji, strict=True):
                for c in set(Ytr[jj]):
                    votes[c] = votes.get(c, 0) + max(float(wt), 0)
            tot = max(float(wi.clip(0).sum()), 1e-9)
            out.append({c: v / tot for c, v in votes.items()})
    return out


def knn_pred(votes, t):
    return [most_specific([c for c, v in d.items() if v >= t]) for d in votes]


vv = knn_votes(Xva)
k_t = max(grid, key=lambda t: per_image(Yva, knn_pred(vv, t)).mean())

# ---- T: MLP head on codes seen at least 5 times, with their ancestors ----
counts = {}
for y in Ytr:
    for c in set(y):
        counts[c] = counts.get(c, 0) + 1
vocab = sorted({a for c, n in counts.items() if n >= 5 for a in ancestors(c)})
vi = {c: i for i, c in enumerate(vocab)}
print(f"vocab {len(vocab)}, dense targets {4 * len(Ytr) * len(vocab) / 1e9:.2f} GB", flush=True)


def targets(ys):
    t = np.zeros((len(ys), len(vocab)), dtype=np.float32)
    for r, y in enumerate(ys):
        for c in y:
            for a in ancestors(c):
                if a in vi:
                    t[r, vi[a]] = 1
    return t


Ttr = torch.from_numpy(targets(Ytr))
head = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 2048), torch.nn.GELU(), torch.nn.Dropout(0.2),
                           torch.nn.Linear(2048, len(vocab))).to(dev)
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)


def asym_loss(logits, y, gn=4.0, gp=0.0, clip=0.05):
    """Asymmetric loss; every absent label is a negative (catalogue incompleteness is not masked: declared)."""
    p = torch.sigmoid(logits)
    pn = (p - clip).clamp(min=0)
    lp = y * (1 - p) ** gp * torch.log(p.clamp(min=1e-8))
    ln = (1 - y) * pn ** gn * torch.log((1 - pn).clamp(min=1e-8))
    return -(lp + ln).sum(1).mean()


@torch.no_grad()
def probs(Q):
    head.eval()
    return torch.cat([torch.sigmoid(head(torch.from_numpy(Q[i:i + 4096]).to(dev))) for i in range(0, len(Q), 4096)]).cpu().numpy()


def head_pred(P, t):
    return [most_specific([vocab[j] for j in np.nonzero(row >= t)[0]]) for row in P]


Xtr_t = torch.from_numpy(Xtr)
best_score, best_epoch, best_t = -1.0, None, None
gen = torch.Generator().manual_seed(0)
for epoch in range(15):
    head.train()
    perm = torch.randperm(len(Xtr_t), generator=gen)
    for i in range(0, len(perm), 512):
        b = perm[i:i + 512]
        loss = asym_loss(head(Xtr_t[b].to(dev)), Ttr[b].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
    Pva = probs(Xva)
    t = max(grid, key=lambda t: per_image(Yva, head_pred(Pva, t)).mean())
    score = float(per_image(Yva, head_pred(Pva, t)).mean())
    with open(OUT / "journal.jsonl", "a") as f:
        f.write(json.dumps({"epoch": epoch, "val_hier_f1": score, "threshold": t, "loss": float(loss)}) + "\n")
    print(f"epoch {epoch}: val hier-F1 {score:.4f} at t={t}", flush=True)
    if score > best_score:
        best_score, best_epoch, best_t = score, epoch, t
        torch.save({"state": head.state_dict(), "vocab": vocab, "threshold": t, "epoch": epoch,
                    "feature_config": FEATURE_CONFIG}, OUT / "head.pt")
head.load_state_dict(torch.load(OUT / "head.pt")["state"])
val_scores = {"K": float(per_image(Yva, knn_pred(vv, k_t)).mean()), "T": best_score}
chosen = max(val_scores, key=lambda k: val_scores[k])  # chosen on validation, never on test

K_test = knn_pred(knn_votes(XT), k_t)
T_test = head_pred(probs(XT), best_t)
freq = [c for c, _ in sorted(counts.items(), key=lambda x: -x[1])[:4]]
systems = {"K_knn": K_test, "T_head": T_test, "frequent": [freq for _ in YT],
           **{k: v["preds"] for k, v in refs.items()}, **{k: v["preds"][:len(YT)] for k, v in descriptive.items()}}


# ---- End-to-end latency of the chosen system on 50 test images, batch 1, after warm-up (amendment 4) ----
@torch.no_grad()
def run_chosen(im):
    f = features([im]).astype(np.float32)
    return knn_pred(knn_votes(f), k_t)[0] if chosen == "K" else head_pred(probs(f), best_t)[0]


test_imgs = test_table.column("image").to_pylist()
pick = np.random.default_rng(0).choice(len(YT), min(50, len(YT)), replace=False)
for j in pick[:3]:
    run_chosen(Image.open(io.BytesIO(test_imgs[j]["bytes"])))
lat = []
for j in pick:
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    run_chosen(Image.open(io.BytesIO(test_imgs[j]["bytes"])))
    if dev == "cuda":
        torch.cuda.synchronize()
    lat.append(time.time() - t0)

per = {k: per_image(YT, v) for k, v in systems.items()}
unread = [c for y in YT for c in y if not readable(c)]
res = {"technical_only": bool(SMOKE), "val_scores": val_scores, "chosen_on_validation": chosen, "knn_threshold": k_t,
       "head": {"vocab": len(vocab), "best_epoch": best_epoch, "threshold": best_t},
       "test_codes_out_of_vocab": {"share": float(np.mean([c not in vi for y in YT for c in y])),
                                   "count": int(sum(c not in vi for y in YT for c in y)),
                                   "images": int(sum(any(c not in vi for c in y) for y in YT))},
       "unreadable_test_codes": len(unread), "exact_test_copies_in_train": copies_train_test,
       "latency_chosen_s_batch1": {"median": float(np.median(lat)), "p90": float(np.quantile(lat, 0.9))},
       "author_declared_hf1": {k: v["declared_hf1"] for k, v in {**refs, **descriptive}.items()},
       "excluded_from_verdict": {"sft4b_orig": "contaminated per the author", "frequent": "baseline"}, "systems": {}}
for k, v in systems.items():
    res["systems"][k] = {"hier_f1": float(per[k].mean()), "exact": exact(YT, v),
                         "codes_per_image": float(np.mean([len(p) for p in v]))}
ours = "K_knn" if chosen == "K" else "T_head"
for ref in refs:
    res[f"paired_{ours}_vs_{ref}"] = boot_diff(per[ours], per[ref])
json.dump(res, open(OUT / "resultats.json", "w"), indent=1)
json.dump({"K_knn": K_test, "T_head": T_test}, open(OUT / "predictions.json", "w"))
print("done", round(time.time() - started), flush=True)
