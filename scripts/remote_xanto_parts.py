#!/usr/bin/env python3
"""Pose and figure parts against the global vector, on the Xanto oracle (results/xanto-parties/PREINSCRIPTION.md).

Usage: remote_xanto_parts.py <data> <out>
<data> holds the Xanto oracle folder as it is on the Mac (``recherche-sources/`` with a_/b_/c_oracle.csv, oracle_temoins.csv,
requetes_boites.csv, resultats_oracle_dinov3.csv, cache_oracle_dinov3l_384.npy, connues/, met/) next to
``corpus-images/rovigo/``, and ``src/`` (the workshop code, for the ViTPose Human-Art loader). Writes <out>/rangs.csv,
<out>/resume.json, <out>/planche.png and <out>/run.json. The crops, the rank and the B0 vectors are those of Xanto's
oracle_dinov3.py, copied as they are; B0 must give back its published ranks or the run stops.
"""

import csv
import glob
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageOps

DATA, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
D = DATA / "recherche-sources"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(DATA / "src"))
from caypollard.atelier.bricks import COCO_JOINTS, _load_vitpose  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
Image.MAX_IMAGE_PIXELS = None
started = time.time()

# ---- Copied from xanto/recherche-sources/oracle_dinov3.py (same crops, same vectors, same rank) ----
from transformers import AutoModel  # noqa: E402

M = os.environ.get("CAY_DINOV3", "facebook/dinov3-vitl16-pretrain-lvd1689m")  # a local folder on Kaggle
model = AutoModel.from_pretrained(M).to(dev).eval()
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TO_T = T.Compose([T.ToTensor(), NORM])


def carre(im, sz):
    w, h = im.size
    s = max(w, h)
    g = int(np.asarray(ImageOps.grayscale(im)).mean())
    c = Image.new("RGB", (s, s), (g, g, g))
    c.paste(im, ((s - w) // 2, (s - h) // 2))
    return c.resize((sz, sz), Image.BICUBIC)


def clamp(boxes, W, H):
    return [(max(0, min(x0, W - 8)), max(0, min(y0, H - 8)), max(min(x1, W), min(x0, W - 8) + 8),
             max(min(y1, H), min(y0, H - 8) + 8)) for (x0, y0, x1, y1) in boxes]


@torch.no_grad()
def embed(im, boxes, mirror=False, bs=8):
    im = im.convert("RGB")
    W, H = im.size
    if mirror:
        im = ImageOps.mirror(im)
    ims = [carre(im.crop(b), 384) for b in clamp(boxes, W, H)]
    out = []
    for i in range(0, len(ims), bs):
        x = torch.stack([TO_T(c) for c in ims[i:i + bs]]).to(dev)
        out.append(torch.nn.functional.normalize(model(pixel_values=x).last_hidden_state[:, 0], dim=-1).cpu().numpy())
    return np.concatenate(out)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


orc, vus = {}, set()
for f in sorted(glob.glob(str(D / "?_oracle.csv"))):
    for r in csv.DictReader(open(f)):
        vus.add(r["requete"])
        if r["trouve"] != "non" and r["x0"]:
            orc[r["requete"]] = (r["estampe"], tuple(int(float(r[k])) for k in ("x0", "y0", "x1", "y1")), r["trouve"])
crops = []
for q, (img, b, _) in orc.items():
    crops.append((img, b, "oracle:" + q))
for r in csv.DictReader(open(D / "oracle_temoins.csv")):
    if not r["x0"]:
        continue
    b = tuple(int(r[k]) for k in ("x0", "y0", "x1", "y1"))
    if b[2] - b[0] < 8 or b[3] - b[1] < 8:
        continue
    img = r["image"]
    if any(o[0].split("/")[-1] == img.split("/")[-1] and iou(o[1], b) > 0.5 for o in orc.values()):
        continue
    crops.append((img, b, "temoin"))
by = {}
for i, (img, b, _) in enumerate(crops):
    by.setdefault(img, []).append(i)
V = np.load(D / "cache_oracle_dinov3l_384.npy")
assert V.shape[0] == len(crops), (V.shape, len(crops))
labels = [c[2] for c in crops]
# ---- end of the copy ----


def gallery_path(img):
    return D / img if (D / img).exists() else D / "connues" / img.split("/")[-1]


# Pose: ViTPose-B Human-Art, top-down in each box, joints kept above 0.3 (the workshop's vitpose-humanart brick).
pose_model, processor = _load_vitpose(
    "vitpose-b-humanart.pth",
    "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/"
    "td-hm_ViTPose-base_8xb64-210e_humanart-256x192-b417f546_20230611.pth",
)


@torch.no_grad()
def poses(im, boxes, bs=32):
    im = im.convert("RGB")
    out = []
    for i in range(0, len(boxes), bs):
        coco = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes[i:i + bs]]
        inputs = processor(im, boxes=[coco], return_tensors="pt").to(dev)
        res = processor.post_process_pose_estimation(pose_model(**inputs), boxes=[coco])[0]
        for r in res:
            out.append({n: (float(p[0]), float(p[1])) for n, p, s in
                        zip(COCO_JOINTS, r["keypoints"].tolist(), r["scores"].tolist(), strict=True) if n and s >= 0.3})
    return out


PARTS = {
    "tete": ("nose", "left_shoulder", "right_shoulder"),
    "buste": ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
    "bras_g": ("left_shoulder", "left_elbow", "left_wrist"),
    "bras_d": ("right_shoulder", "right_elbow", "right_wrist"),
    "jambe_g": ("left_hip", "left_knee", "left_ankle"),
    "jambe_d": ("right_hip", "right_knee", "right_ankle"),
}


def part_boxes(k, fig):
    """Six boxes from the joints: enclosing box of the part's joints, grown by 25 % and to at least a quarter of the
    figure's height on each side. A part needs all its joints."""
    h = fig[3] - fig[1]
    out = {}
    for name, joints in PARTS.items():
        if not all(j in k for j in joints):
            continue
        xs, ys = [k[j][0] for j in joints], [k[j][1] for j in joints]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        w, hh = max(1.25 * (max(xs) - min(xs)), h / 4), max(1.25 * (max(ys) - min(ys)), h / 4)
        out[name] = (int(cx - w / 2), int(cy - hh / 2), int(cx + w / 2), int(cy + hh / 2))
    return out


# CAY_SMOKE=n: a dry run on the gallery images of the first n queries only; ranks are meaningless and B0 is not checked.
SMOKE = int(os.environ.get("CAY_SMOKE", "0"))
N = len(crops)
TYPES = {"composizione": "composition"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


import transformers  # noqa: E402
import torchvision  # noqa: E402

json.dump({"smoke": SMOKE, "device": dev, "torch": torch.__version__, "transformers": transformers.__version__,
           "torchvision": torchvision.__version__, "script_sha256": sha(__file__),
           "inputs_sha256": {n: sha(D / n) for n in ("a_oracle.csv", "b_oracle.csv", "c_oracle.csv", "oracle_temoins.csv",
                                                     "requetes_boites.csv", "resultats_oracle_dinov3.csv",
                                                     "cache_oracle_dinov3l_384.npy")},
           "dinov3": M, "dinov3_weights_sha256": sha(Path(M) / "model.safetensors") if Path(M).is_dir() else None,
           "vitpose_sha256": sha(Path("data/models/vitpose/vitpose-b-humanart.pth"))}, open(OUT / "run.json", "w"), indent=1)

# ---- Phase 1: the query lot and B0, checked in full before any variant (amendment 1 of the pre-registration) ----
queries = [q for q in csv.DictReader(open(D / "requetes_boites.csv")) if q["requete"] in vus]
published = {r["requete"]: r for r in csv.DictReader(open(D / "resultats_oracle_dinov3.csv"))}
ids = [q["requete"] for q in queries]
assert len(ids) == len(set(ids)), "duplicate query ids"
assert set(ids) == set(published), (sorted(set(ids) ^ set(published)))
assert vus - set(ids) == {"Q49"}, sorted(vus - set(ids))  # Q49: oracle box but no plate box, never evaluated
if SMOKE:
    queries = [q for q in queries if q["requete"] in sorted(orc)[:SMOKE]]
assert all(("oracle:" + q["requete"] in labels) == (published[q["requete"]]["rang"] != "") for q in queries)

Q = []  # one dict per query: photo, box, B0 scores per sense, target index, excluded crops
for q in queries:
    t = "oracle:" + q["requete"]
    item = {"q": q, "type": TYPES.get(q["type"], q["type"])}
    if t in labels:
        photo = Image.open(DATA / "corpus-images" / "rovigo" / q["photo"]).convert("RGB")
        box = tuple(int(float(q[k])) for k in ("x0", "y0", "x1", "y1"))
        crop = photo.crop(box)
        full = [(0, 0, crop.width, crop.height)]
        b0s = V @ np.concatenate([embed(crop, full), embed(crop, full, mirror=True)]).T  # (N, 2): endroit, miroir
        ti = labels.index(t)
        same = [i for i, (img, b, l) in enumerate(crops) if l.startswith("oracle:") and i != ti
                and img == crops[ti][0] and iou(b, crops[ti][1]) > 0.5]
        s = b0s.max(1).copy()
        s[same] = -np.inf
        historic = int((s > s[ti]).sum()) + 1
        if not SMOKE and str(historic) != published[q["requete"]]["rang"]:
            sys.exit(f"B0 does not reproduce Xanto's oracle for {q['requete']}: {historic} vs {published[q['requete']]['rang']}")
        item.update(photo=photo, box=box, crop=crop, b0s=b0s, ti=ti, same=same, b0_historic=historic)
    Q.append(item)
json.dump({"queries": len(Q), "with_oracle": sum("ti" in x for x in Q), "b0_reproduced": not SMOKE,
           "ranks": {x["q"]["requete"]: x.get("b0_historic") for x in Q}}, open(OUT / "b0_controle.json", "w"), indent=1)
print(f"B0 checked on {len(Q)} queries, {time.time() - started:.0f} s", flush=True)

# ---- Phase 2: pose and part vectors of every gallery crop, saved as it goes and resumable ----
if SMOKE:
    keep = {crops[x["ti"]][0] for x in Q if "ti" in x}
    by = {img: idx for img, idx in by.items() if img in keep}
CACHE = OUT / "galerie.pkl"
_run = json.load(open(OUT / "run.json"))
RESUME_KEY = [N, _run["script_sha256"], _run["inputs_sha256"], _run["dinov3_weights_sha256"], _run["vitpose_sha256"]]
assert SMOKE or _run["dinov3_weights_sha256"], "set CAY_DINOV3 to the frozen local weights folder"
import pickle  # noqa: E402

state = {"done": [], "pose": {}, "parts": {}}
if CACHE.exists():
    state = pickle.loads(CACHE.read_bytes())
    assert state.get("key") == RESUME_KEY, "cache from another script, input or model"
state.update(key=RESUME_KEY)
done = set(state["done"])
for n, (img, idx) in enumerate(by.items()):
    if img in done:
        continue
    im = Image.open(gallery_path(img))
    ks = poses(im, [crops[i][1] for i in idx])
    jobs = []
    for i, k in zip(idx, ks, strict=True):
        state["pose"][i] = k
        for name, b in part_boxes(k, crops[i][1]).items():
            jobs.append((i, name, b))
    if jobs:
        for (i, name, _), v in zip(jobs, embed(im, [b for _, _, b in jobs]), strict=True):
            state["parts"].setdefault(i, {})[name] = v
    state["done"].append(img)
    if len(state["done"]) % 50 == 0 or n == len(by) - 1:
        CACHE.write_bytes(pickle.dumps(state))
        print(f"gallery images {len(state['done'])}/{len(by)}, {time.time() - started:.0f} s", flush=True)
G_POSE = [state["pose"].get(i) for i in range(N)]
G_PARTS = [state["parts"].get(i, {}) for i in range(N)]


def pose_sim(a, b):
    common = [j for j in a if j in b]
    if len(common) < 6:
        return -np.inf
    pa, pb = np.array([a[j] for j in common]), np.array([b[j] for j in common])
    pa, pb = pa - pa.mean(0), pb - pb.mean(0)
    pa /= np.sqrt((pa ** 2).sum(1).mean()) + 1e-9
    pb /= np.sqrt((pb ** 2).sum(1).mean()) + 1e-9
    return -float(np.linalg.norm(pa - pb, axis=1).mean())


def fusion_ranks(score):
    """Ranks for the fusion: 1 + number strictly above; every invalid score (-inf) gets the same worst rank N, so a
    modality missing for everyone leaves the other one's order unchanged."""
    order = np.argsort(-score, kind="stable")
    sorted_s = score[order]
    r = np.empty(N, dtype=np.int64)
    r[order] = np.searchsorted(-sorted_s, -sorted_s, side="left") + 1
    r[~np.isfinite(score)] = N
    return r


def fuse(a, b):
    out = 1 / (60 + fusion_ranks(a)) + 1 / (60 + fusion_ranks(b))
    out[~np.isfinite(a) & ~np.isfinite(b)] = -np.inf
    return out


def verdict_rank(s, ti):
    """Rank used by the verdict, the same for every variant: ties counted against the target (1 + number of crops at or
    above its score, itself excepted); an invalid target score is the last rank, a failure."""
    if not np.isfinite(s[ti]):
        return N
    return int((s >= s[ti]).sum())


# ---- Phase 3: variants, query by query ----
rows, sheet = [], []
for x in Q:
    q = x["q"]
    row = {"requete": q["requete"], "type": x["type"], "trouve_plat": q["trouve"]}
    if "ti" not in x:
        rows.append({**row, "oracle": "introuvable", **{f"rang_{v}": N for v in ("B0", "P", "F", "K", "KP")}})
        continue
    photo, box, ti, same, b0s = x["photo"], x["box"], x["ti"], x["same"], x["b0s"]
    P, K = np.full(N, -np.inf), np.full(N, -np.inf)
    seen = []
    for sense, mirror in enumerate((False, True)):
        ph = ImageOps.mirror(photo) if mirror else photo
        bx = (photo.width - box[2], box[1], photo.width - box[0], box[3]) if mirror else box
        k = poses(ph, [bx])[0]
        pb = part_boxes(k, bx)
        pv = dict(zip(pb, embed(ph, list(pb.values())), strict=True)) if pb else {}
        seen.append((k, pb))
        P = np.maximum(P, [pose_sim(k, g) if g is not None else -np.inf for g in G_POSE])
        # K, per sense: mean cosine over the parts present on both sides; fewer than two: that sense's B0 cosine
        k_sense = np.array([np.mean([float(pv[n] @ G_PARTS[i][n]) for n in pv if n in G_PARTS[i]])
                            if sum(n in G_PARTS[i] for n in pv) >= 2 else b0s[i, sense] for i in range(N)])
        K = np.maximum(K, k_sense)
    scores = {"B0": b0s.max(1).copy(), "P": P, "K": K}
    for s in scores.values():
        s[same] = -np.inf
    scores["F"] = fuse(scores["B0"], scores["P"])
    scores["KP"] = fuse(scores["K"], scores["P"])
    common_parts = max(sum(n in G_PARTS[ti] for n in pb) for _, pb in seen)
    row.update(oracle=orc[q["requete"]][2], b0_historique=x["b0_historic"],
               joints_plat=max(len(k) for k, _ in seen), joints_oracle=len(G_POSE[ti] or {}),
               parties_plat=max(len(pb) for _, pb in seen), parties_oracle=len(G_PARTS[ti]), parties_communes=common_parts)
    for name, s in scores.items():
        s = s.copy()
        s[same] = -np.inf
        row["rang_" + name] = verdict_rank(s, ti)
    rows.append(row)
    sheet.append((q["requete"], x["crop"], seen[0], ti, box))

# The sheet first, to be looked at before the numbers (queries with an oracle crop only; seed 0).
random.seed(0)
picked = random.sample(sheet, min(6, len(sheet)))
tiles = []
for qid, crop, (k, pb), ti, qbox in picked:
    img, b, _ = crops[ti]
    gal = Image.open(gallery_path(img)).convert("RGB")
    pairs = [(crop, k, pb, (qbox[0], qbox[1])), (gal.crop(b), G_POSE[ti] or {}, part_boxes(G_POSE[ti] or {}, b), (b[0], b[1]))]
    row_t = []
    for im, kk, parts, bx0 in pairs:
        im = im.copy()
        dr = ImageDraw.Draw(im)
        rad, lw = max(3, im.width // 60), max(2, im.width // 150)
        for _, (x0, y0, x1, y1) in parts.items():
            dr.rectangle((x0 - bx0[0], y0 - bx0[1], x1 - bx0[0], y1 - bx0[1]), outline=(0, 160, 255), width=lw)
        for px, py in kk.values():
            dr.ellipse((px - bx0[0] - rad, py - bx0[1] - rad, px - bx0[0] + rad, py - bx0[1] + rad), fill=(255, 40, 40))
        im.thumbnail((360, 360))
        row_t.append(im)
    tiles.append((qid, row_t))
sheet_img = Image.new("RGB", (740, 380 * max(1, len(tiles))), "white")
for n, (qid, (a, b)) in enumerate(tiles):
    sheet_img.paste(a, (0, 380 * n + 20))
    sheet_img.paste(b, (380, 380 * n + 20))
    ImageDraw.Draw(sheet_img).text((4, 380 * n + 2), qid, fill="black")
sheet_img.save(OUT / "planche.png")

fields = ["requete", "type", "trouve_plat", "oracle", "b0_historique", "joints_plat", "joints_oracle", "parties_plat",
          "parties_oracle", "parties_communes", "rang_B0", "rang_P", "rang_F", "rang_K", "rang_KP"]
with open(OUT / "rangs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, restval="")
    w.writeheader()
    w.writerows(rows)


def sign_p(up, down):
    from math import comb

    n = up + down
    if n == 0:
        return 1.0
    k = min(up, down)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


summary: dict = {"queries": len(rows), "with_oracle": sum(r.get("oracle") != "introuvable" for r in rows), "crops": N,
                 "gallery_crops_without_six_joints": int(sum(len(k or {}) < 6 for k in G_POSE)),
                 "prints_behind_the_oracle_crops": len({crops[x["ti"]][0] for x in Q if "ti" in x})}
for name in ("B0", "P", "F", "K", "KP"):
    r = [x["rang_" + name] for x in rows]
    v: dict = {f"top{k}": sum(x <= k for x in r) for k in (1, 10, 50)}
    v["denominator"] = len(r)
    v["median_all_queries"] = sorted(r)[len(r) // 2]
    if name != "B0":
        up = sum(x["rang_" + name] < x["rang_B0"] for x in rows)
        down = sum(x["rang_" + name] > x["rang_B0"] for x in rows)
        v.update(up=up, down=down, p_sign_two_sided=sign_p(up, down),
                 top10_gained=sum(x["rang_" + name] <= 10 < x["rang_B0"] for x in rows),
                 top10_lost=sum(x["rang_B0"] <= 10 < x["rang_" + name] for x in rows),
                 wins=v["top10"] >= 18 and up > down and sign_p(up, down) < 0.0125)
    v["top10_by_type"] = {t: sum(x["rang_" + name] <= 10 for x in rows if x["type"] == t) for t in sorted({x["type"] for x in rows})}
    summary[name] = v
json.dump(summary, open(OUT / "resume.json", "w"), indent=1, ensure_ascii=False)
print("done: planche.png first, then resume.json", flush=True)
