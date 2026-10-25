#!/usr/bin/env python3
"""External Iconclass bench from Emblematica (results/iconclass-externe/PREINSCRIPTION.md): second copy filter, frozen draw, heads.

Usage: remote_external_iconclass.py prepare|evaluate <head_out> <ext_dir> <out>
Two separate phases (amendment of 25/10/2026, Codex): `prepare` filters, draws and freezes the lists, then stops; the sheet of the
closest remaining pairs is looked at where the ICAI images are, and <out>/vu.json is written there with the sha256 of frozen.json;
only then does `evaluate` score, and only on lists whose checksums match.
<head_out> is the output folder of scripts/remote_iconclass_head.py (feats/*.npz with their manifests, head.pt, run.json).
<ext_dir> holds candidates.jsonl, near256.json and images/<emblem>.jpg. Writes <out>/filtre.json, <out>/nearest.json (for the
sheet, built where the ICAI images are), <out>/dev.json, <out>/test.json (frozen lists), <out>/resultats.json, <out>/predictions.json.
The encoder, the hierarchy, the measures, the kNN and the head are those of the head script, copied here unchanged.
"""

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PHASE, HEAD, EXT, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
assert PHASE in ("prepare", "evaluate", "evaluate-on")
# evaluate-on <reference draw>: score another head (e.g. the PE-Spatial brick) on the lists frozen and looked at for an earlier
# head, without filtering or drawing again, so that both heads face the very same images.
REF = Path(sys.argv[5]) if PHASE == "evaluate-on" else None
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
started = time.time()
head_run = json.load(open(HEAD / "run.json"))
assert head_run["smoke"] == 0, "the heads must come from the full run"
FEATURE_CONFIG = head_run["feature_config"]

# ---- Copied from scripts/remote_iconclass_head.py: hierarchy and measures ----
PART = re.compile(r"\d\d?[A-Z]{0,2}\d*(?:\([^)]*\)\d*)*")
TOKEN = re.compile(r"\([^)]*\)|[A-Z]{1,2}|\d")
_anc_cache: dict = {}


def ancestors(code):
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


def exact(gts, preds):
    tp = sum(len(set(g) & set(p)) for g, p in zip(gts, preds, strict=True))
    npred, ngt = sum(len(set(p)) for p in preds), sum(len(set(g)) for g in gts)
    pr, rc = tp / max(npred, 1), tp / max(ngt, 1)
    return {"precision": pr, "recall": rc, "f1": 0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)}


# ---- Copied from scripts/remote_iconclass_head.py: the encoder ----
ENCODERS = {  # name -> (repository, pinned revision, feature configuration written into every manifest)
    "c-radio": ("nvidia/C-RADIOv3-L", "9d0413465e8a91e67bbf2c1ad342815478d1b906",
                "|square-grey128|448|bicubic|autocast-fp16|summary+meanspatial|l2-each-then-l2"),
    "pe-spatial": ("timm/vit_pe_spatial_large_patch14_448.fb", "ac6f424680a305470340dac75a0a9b4c47ee030f",
                   "|square-grey128|448|bicubic|autocast-fp16|meanpatch|l2"),
}


def load_encoder(name):
    """The frozen encoder and its feature function: image list -> float16 L2-normalised vectors."""
    repo, rev, _ = ENCODERS[name]
    if name == "c-radio":
        from transformers import AutoModel, CLIPImageProcessor

        model = AutoModel.from_pretrained(repo, revision=rev, trust_remote_code=True).to(dev).eval()
        processor = CLIPImageProcessor.from_pretrained(repo, revision=rev)

        @torch.no_grad()
        def features(images):
            px = processor(images=[square(i) for i in images], return_tensors="pt", do_resize=False).pixel_values.to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):  # RADIO normalises its input itself
                summary, spatial = model(px)
            v = torch.cat([torch.nn.functional.normalize(summary.float(), dim=-1),
                           torch.nn.functional.normalize(spatial.float().mean(1), dim=-1)], 1)
            return torch.nn.functional.normalize(v, dim=-1).cpu().numpy().astype(np.float16)

        return features
    import timm
    import torchvision.transforms as T
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(repo, "model.safetensors", revision=rev)
    model = timm.create_model("vit_pe_spatial_large_patch14_448", pretrained=True,
                              pretrained_cfg_overlay={"file": weights}).to(dev).eval()
    to_t = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])  # the model's own mean and std

    @torch.no_grad()
    def features(images):
        x = torch.stack([to_t(square(i)) for i in images]).to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            tokens = model.forward_features(x)
        v = tokens[:, getattr(model, "num_prefix_tokens", 0):].float().mean(1)  # mean of the patch tokens (timm's own pooling)
        return torch.nn.functional.normalize(v, dim=-1).cpu().numpy().astype(np.float16)

    return features


def square(im):
    im = im.convert("RGB")
    s = max(im.size)
    c = Image.new("RGB", (s, s), (128, 128, 128))
    c.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
    return c.resize((448, 448), Image.BICUBIC)


ENCODER = next(k for k, (repo, rev, cfg) in ENCODERS.items() if FEATURE_CONFIG == f"{repo}@{rev}{cfg}")
features = load_encoder(ENCODER)


# ---- ICAI features of the full run, checked against their manifests ----
train_tags = [f"train-{i:05d}-of-00011" for i in range(11)]
icai_x, icai_md5, Xtr_all, Ytr_all = [], [], [], []
for tag in train_tags + ["test-00000-of-00001"]:
    man = json.load(open(HEAD / "feats" / f"{tag}.json"))
    assert man["smoke"] == 0 and man["feature_config"] == FEATURE_CONFIG, tag
    z = np.load(HEAD / "feats" / f"{tag}.npz")
    icai_x.append(z["x"])
    icai_md5 += list(z["md5"])
icai = torch.from_numpy(np.concatenate(icai_x).astype(np.float32)).to(dev)
assert icai.shape[0] == 86216 + 788

# ---- Candidates after filter 1 (books with any picture under 50 bits of an ICAI image are out) ----
near = json.load(open(EXT / "near256.json"))
bad_books = {b for _, d, _, b in near if 0 <= d < 50}
rows = [json.loads(line) for line in open(EXT / "candidates.jsonl")]
rows = [r for r in rows if r["book_id"] is not None and r["book_id"] not in bad_books]
print(f"filter 1: {len(rows)} picturae, {len({r['book_id'] for r in rows})} books", flush=True)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


IMG_SHA = [hashlib.sha256((EXT / "images" / r["filename"]).read_bytes()).hexdigest() for r in rows]
CACHE = OUT / "candidats_feats.npz"
if (CACHE.exists() and list(np.load(CACHE)["files"]) == [r["filename"] for r in rows]
        and list(np.load(CACHE)["img_sha"]) == IMG_SHA and str(np.load(CACHE)["config"]) == FEATURE_CONFIG):
    X = np.load(CACHE)["x"].astype(np.float32)
else:
    X = []
    for i in range(0, len(rows), 32):
        X.append(features([Image.open(EXT / "images" / r["filename"]) for r in rows[i:i + 32]]))
    X = np.concatenate(X)
    np.savez(OUT / "candidats_feats.tmp.npz", x=X, files=np.array([r["filename"] for r in rows]), img_sha=np.array(IMG_SHA),
             config=np.array(FEATURE_CONFIG))
    import os
    os.replace(OUT / "candidats_feats.tmp.npz", CACHE)
    X = X.astype(np.float32)

def signature():
    """Everything the evaluation depends on: lists, filter, nearest pairs, cache, candidate truths and books, images, head artefacts,
    feature config and this script."""
    return ({n: sha(OUT / n) for n in ("dev.json", "test.json", "filtre.json", "nearest.json", "candidats_feats.npz")}
            | {"candidates.jsonl": sha(EXT / "candidates.jsonl"), "near256.json": sha(EXT / "near256.json"),
               "images_sha256": hashlib.sha256("".join(IMG_SHA).encode()).hexdigest(),
               "head_pt": sha(HEAD / "head.pt"), "head_resultats": sha(HEAD / "resultats.json"), "head_run": sha(HEAD / "run.json"),
               "head_feats": {f.name: sha(f) for f in sorted((HEAD / "feats").glob("*")) if f.suffix in (".npz", ".json")},
               "feature_config": FEATURE_CONFIG, "script": sha(__file__)})


if PHASE == "prepare":
    best_cos, best_j = [], []
    for i in range(0, len(X), 1024):
        s_, j = (torch.from_numpy(X[i:i + 1024]).to(dev) @ icai.T).max(1)
        best_cos += s_.cpu().tolist()
        best_j += j.cpu().tolist()
    books_out = sorted({r["book_id"] for r, c in zip(rows, best_cos, strict=True) if c >= 0.95})
    keep = [i for i, r in enumerate(rows) if r["book_id"] not in books_out]
    json.dump({"filter1_picturae": len(rows), "filter1_books": len({r["book_id"] for r in rows}),
               "books_out_cos_095": books_out, "kept_picturae": len(keep),
               "kept_books": len({rows[i]["book_id"] for i in keep}),
               "cos_histogram": np.histogram(best_cos, bins=[0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01])[0].tolist()},
              open(OUT / "filtre.json", "w"), indent=1)
    order = sorted(keep, key=lambda i: -best_cos[i])[:12]
    json.dump([{"filename": rows[i]["filename"], "cos": best_cos[i], "icai_md5": str(icai_md5[best_j[i]])} for i in order],
              open(OUT / "nearest.json", "w"), indent=1)
    # Frozen draw: whole books to dev or test, at most 15 picturae per book, seed 0
    rng = np.random.default_rng(0)
    by_book: dict = {}
    for i in keep:
        by_book.setdefault(rows[i]["book_id"], []).append(i)
    books = sorted(by_book)
    rng.shuffle(books)
    dev_idx, test_idx = [], []
    for b in books:
        chosen_i = sorted(rng.choice(by_book[b], min(15, len(by_book[b])), replace=False).tolist())
        (dev_idx if len(dev_idx) < 200 else test_idx).extend(chosen_i)
        if len(test_idx) >= 800:
            break
    json.dump([{"filename": rows[i]["filename"], "book_id": rows[i]["book_id"]} for i in dev_idx], open(OUT / "dev.json", "w"))
    json.dump([{"filename": rows[i]["filename"], "book_id": rows[i]["book_id"]} for i in test_idx], open(OUT / "test.json", "w"))
    assert dev_idx and test_idx, "empty draw"
    json.dump(signature(), open(OUT / "frozen.json", "w"), indent=1)
    print(f"prepared: dev {len(dev_idx)}, test {len(test_idx)}; look at the sheet, then write vu.json", flush=True)
    sys.exit(0)

if PHASE == "evaluate":
    assert json.load(open(OUT / "vu.json"))["frozen_sha256"] == sha(OUT / "frozen.json"), "the sheet has not been looked at"
    assert json.load(open(OUT / "frozen.json")) == signature(), "something changed since the draw was frozen"
    LISTS = OUT
else:
    ref = json.load(open(REF / "frozen.json"))
    assert json.load(open(REF / "vu.json"))["frozen_sha256"] == sha(REF / "frozen.json"), "reference draw not looked at"
    for n in ("dev.json", "test.json"):
        assert sha(REF / n) == ref[n], n
    assert sha(EXT / "candidates.jsonl") == ref["candidates.jsonl"] and sha(EXT / "near256.json") == ref["near256.json"]
    assert hashlib.sha256("".join(IMG_SHA).encode()).hexdigest() == ref["images_sha256"], "other images than the reference draw"
    json.dump({"reference_frozen_sha256": sha(REF / "frozen.json"), "candidats_feats.npz": sha(OUT / "candidats_feats.npz"),
               "head_pt": sha(HEAD / "head.pt"), "head_resultats": sha(HEAD / "resultats.json"), "head_run": sha(HEAD / "run.json"),
               "head_feats": {f.name: sha(f) for f in sorted((HEAD / "feats").glob("*")) if f.suffix in (".npz", ".json")},
               "feature_config": FEATURE_CONFIG, "script": sha(__file__)}, open(OUT / "reference.json", "w"), indent=1)
    LISTS = REF
pos = {r["filename"]: i for i, r in enumerate(rows)}
dev_idx = [pos[x["filename"]] for x in json.load(open(LISTS / "dev.json"))]
test_idx = [pos[x["filename"]] for x in json.load(open(LISTS / "test.json"))]
assert dev_idx and test_idx, "empty lists"

# ---- The system chosen on validation in the head run, as learnt (no tuning here) ----
res_head = json.load(open(HEAD / "resultats.json"))
chosen = res_head["chosen_on_validation"]
ckpt = torch.load(HEAD / "head.pt", weights_only=False)
vocab, t_head = ckpt["vocab"], ckpt["threshold"]
head = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 2048), torch.nn.GELU(), torch.nn.Dropout(0.2),
                           torch.nn.Linear(2048, len(vocab))).to(dev)
head.load_state_dict(ckpt["state"])
head.eval()
if True:  # both T and K are evaluated as frozen; the kNN needs the train labels, read again from the pinned dataset
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    REPO, REV = "davanstrien/iconclass-vlm-brillfull", "8045366ff092a06e8e1c4e47f1eb43cb2a54d732"
    Y_tr = [y for tag in train_tags for y in pq.read_table(hf_hub_download(REPO, f"data/{tag}.parquet", repo_type="dataset",
                                                                          revision=REV), columns=["label"]).column("label").to_pylist()]
    M = list(np.concatenate([np.load(HEAD / "feats" / f"{tag}.npz")["md5"] for tag in train_tags]))
    groups = sorted(set(M))
    gval = set(np.array(groups)[np.random.default_rng(0).random(len(groups)) < 0.05])
    trmask = np.array([m not in gval for m in M])
    Xtr_dev = icai[:86216][torch.from_numpy(trmask).to(dev)]
    Ytr = [y for y, k in zip(Y_tr, trmask, strict=True) if k]
    k_t = res_head["knn_threshold"]


@torch.no_grad()
def head_pred(Q):
    P = torch.sigmoid(head(torch.from_numpy(Q).to(dev))).cpu().numpy()
    return [most_specific([vocab[j] for j in np.nonzero(row >= t_head)[0]]) for row in P]


@torch.no_grad()
def knn_pred(Q, k=20):
    out = []
    w, j = (torch.from_numpy(Q).to(dev) @ Xtr_dev.T).topk(k, dim=1)
    for wi, ji in zip(w.cpu().numpy(), j.cpu().numpy(), strict=True):
        votes = {}
        for wt, jj in zip(wi, ji, strict=True):
            for c in set(Ytr[jj]):
                votes[c] = votes.get(c, 0) + max(float(wt), 0)
        tot = max(float(wi.clip(0).sum()), 1e-9)
        out.append(most_specific([c for c, v in votes.items() if v / tot >= k_t]))
    return out


def project(codes, drop86):
    """The principal measure leaves chapter 86 (written mottoes and sayings) out of BOTH truth and prediction."""
    return [c for c in codes if not (drop86 and c.strip().startswith("86"))]


out = {"head_run_sha": head_run["script_sha256"], "chosen_on_validation": chosen, "systems": {}}
preds = {}
systems = {"T_head": head_pred, "K_knn": knn_pred}
for split, idx in (("dev", dev_idx), ("test", test_idx)):
    for name, fn in systems.items():
        preds[f"{name}_{split}"] = fn(X[idx])
        for drop86, label in ((True, "principal_sans_86"), (False, "toutes_notations")):
            pairs = [(project(rows[i]["iconclass"], drop86), project(p, drop86), rows[i]["book_id"])
                     for i, p in zip(idx, preds[f"{name}_{split}"], strict=True)]
            kept = [(g, p, b) for g, p, b in pairs if g]  # images with no truth left are out of the denominator, counted
            f = [hier_f1(g, p) for g, p, _ in kept]
            out["systems"][f"{name}/{split}/{label}"] = {"hier_f1": float(np.mean(f)), "per_image": f,
                                                         "books": [b for _, _, b in kept],
                                                         "exact": exact([g for g, _, _ in kept], [p for _, p, _ in kept]),
                                                         "images_scored": len(kept), "images_without_truth": len(pairs) - len(kept)}
out["unreadable_truth_codes_test"] = sum(not readable(c) for i in test_idx for c in rows[i]["iconclass"])
json.dump(out, open(OUT / "resultats.json", "w"), indent=1)
json.dump(preds, open(OUT / "predictions.json", "w"))
print("done", round(time.time() - started), flush=True)
