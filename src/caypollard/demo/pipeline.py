"""Recompute every representation for one picture, keeping each intermediate step visible.

Each step calls the experiment's own function through `bridge.script`, on the fitted
models `scripts/fit_demo_models.py` recovered, and records what it saw: the image it
produced, the numbers it computed, the parameters it ran with, and whether those are
the parameters the frozen pool was built with. A picture from the pool can be analysed
too, and each live vector is then compared with its frozen row: that is how a reader
can check that what the demo shows is what the benchmarks measured.
"""

from __future__ import annotations

import base64
import collections
import colorsys
import io
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from caypollard import sketch as sketching

from .bridge import script
from .explain import DEFAULTS
from .models import atoms, composites, l2, relation_keys, stroke_atoms
from .namer import name_nodes, vocabulary

_encoder = None
_encoder_lock = threading.Lock()


def encoder():
    """The DINOv2 encoder the pool was built with, loaded once."""
    global _encoder
    with _encoder_lock:
        if _encoder is None:
            from caypollard.vision.encoders import DEFAULT_MODELS, HuggingFaceVisionEncoder

            _encoder = HuggingFaceVisionEncoder(DEFAULT_MODELS["dinov2-base"], device="auto")
        return _encoder


_pose_model = None


def pose_model():
    """torchvision's Keypoint R-CNN, COCO weights, loaded once on the same device as DINOv2."""
    global _pose_model
    with _encoder_lock:
        if _pose_model is None:
            import torch
            from torchvision.models.detection import (
                KeypointRCNN_ResNet50_FPN_Weights,
                keypointrcnn_resnet50_fpn,
            )

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
            _pose_model = (model.to(device).eval(), device)
        return _pose_model


def detect_poses(
    image: Image.Image, *, max_side: int, score: float
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """People the detector accepts: (score, keypoints xy in the resized frame, keypoint scores)."""
    import torch
    from torchvision.transforms import functional

    model, device = pose_model()
    if max(image.size) > max_side:
        factor = max_side / max(image.size)
        image = image.resize(
            (max(int(image.width * factor), 1), max(int(image.height * factor), 1)), Image.LANCZOS
        )
    tensor = functional.to_tensor(image).to(device)
    with torch.inference_mode():
        output = model([tensor])[0]
    found = []
    for i in range(len(output["scores"])):
        s = float(output["scores"][i])
        if s >= score:
            found.append(
                (
                    s,
                    output["keypoints"][i].detach().cpu().numpy()[:, :2],
                    output["keypoints_scores"][i].detach().cpu().numpy(),
                )
            )
    return found, image.size


def skeleton(
    image: Image.Image, figures: list, size: tuple[int, int], *, threshold: float
) -> Image.Image:
    pose = script("build_pose_channel")
    canvas = image.convert("RGB").resize(size)
    draw = ImageDraw.Draw(canvas)
    links = [(a, b) for _, a, b in pose.SEGMENTS] + [
        (pose.L_SHO, pose.R_SHO),
        (pose.L_HIP, pose.R_HIP),
        (pose.L_SHO, pose.L_HIP),
        (pose.R_SHO, pose.R_HIP),
    ]
    for n, (_, keypoints, scores) in enumerate(figures):
        colour = sign_colour(n * 37 + 5, 256)
        for a, b in links:
            if scores[a] >= threshold and scores[b] >= threshold:
                draw.line([tuple(keypoints[a]), tuple(keypoints[b])], fill=colour, width=3)
        for k, (x, y) in enumerate(keypoints):
            if scores[k] >= threshold:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=colour, outline=(0, 0, 0))
    return canvas


def data_url(image: Image.Image, *, quality: int = 85, kind: str = "JPEG") -> str:
    buffer = io.BytesIO()
    if kind == "JPEG":
        image.convert("RGB").save(buffer, "JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        image.save(buffer, "PNG")
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def sign_colour(sign: int, total: int) -> tuple[int, int, int]:
    hue = (sign * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def region_masks(
    path: Path, *, side: int, min_area: float, max_regions: int, smoothing: float, threshold: float
) -> tuple[list[dict], np.ndarray]:
    """The same cut as `segment_shapes.segment`, kept as label images for drawing.

    Mirrors the script line for line so that the k-th mask is the k-th region the script
    describes; the caller checks the pairing on areas and says so if it ever differs.
    """
    image = Image.open(path).convert("L").resize((side, side), Image.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    smoothed = ndimage.gaussian_filter(array, sigma=side / smoothing)
    low, high = np.percentile(smoothed, 2.0), np.percentile(smoothed, 98.0)
    smoothed = np.clip((smoothed - low) / (high - low), 0.0, 1.0) if high > low else smoothed
    page_tone = float(np.median(smoothed))
    found: list[dict] = []
    for polarity, name in ((-1.0, "sombre"), (1.0, "clair")):
        binary = (polarity * (smoothed - page_tone)) > threshold
        binary = ndimage.binary_opening(binary, np.ones((3, 3)))
        labelled, count = ndimage.label(binary)
        for index in range(1, count + 1):
            mask = labelled == index
            area = float(mask.sum()) / mask.size
            if area < min_area:
                continue
            found.append({"area": round(area, 5), "tone": name, "mask": mask})
    found.sort(key=lambda r: -r["area"])
    found = found[:max_regions]
    return found, smoothed


def overlay(
    path: Path,
    regions: list[dict],
    masks: list[dict],
    signs: np.ndarray | None,
    *,
    side: int,
    total_signs: int,
) -> Image.Image:
    base = Image.open(path).convert("RGB").resize((side, side), Image.LANCZOS)
    faded = Image.blend(base, Image.new("RGB", base.size, (255, 255, 255)), 0.55)
    layer = np.asarray(faded, dtype=np.float32)
    for k, m in enumerate(masks):
        colour = sign_colour(int(signs[k]) if signs is not None else k, total_signs)
        layer[m["mask"]] = layer[m["mask"]] * 0.35 + np.asarray(colour, dtype=np.float32) * 0.65
    out = Image.fromarray(layer.astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for k, region in enumerate(regions):
        x, y = region["centroid_x"] * side, region["centroid_y"] * side
        label = f"{k + 1}"
        draw.rectangle([x - 7, y - 7, x + 8, y + 8], fill=(20, 20, 20))
        draw.text((x - 3, y - 6), label, fill=(255, 255, 255))
    return out


def heatmap(block: np.ndarray, *, side: int = 256) -> Image.Image:
    scaled = block / block.max() if block.max() > 0 else block
    grey = (255 - scaled * 255).astype(np.uint8)
    return Image.fromarray(grey).resize((side, side), Image.NEAREST)


def balanced_image(image: Image.Image, side: int) -> Image.Image:
    small = np.asarray(image.convert("RGB").resize((side, side), Image.BILINEAR), dtype=np.float32)
    means = small.reshape(-1, 3).mean(axis=0)
    means[means == 0] = 1.0
    gains = means.mean() / means
    return Image.fromarray(np.clip(small * gains, 0, 255).astype(np.uint8))


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.dot(l2(a), l2(b)))


def top_entries(vector: np.ndarray, names: list[str] | None, n: int = 8) -> list[dict]:
    order = np.argsort(-vector)[:n]
    return [
        {"index": int(i), "name": names[i] if names else str(int(i)), "value": float(vector[i])}
        for i in order
        if vector[i] > 0
    ]


class Analysis:
    def __init__(self, path: Path, params: dict[str, Any], *, namer: str = "sonnet") -> None:
        self.path = path
        self.params = {**DEFAULTS, **{k: v for k, v in params.items() if k in DEFAULTS}}
        self.namer = namer
        self.vectors: dict[str, np.ndarray] = {}
        self.steps: list[dict] = []
        self.timings: dict[str, float] = {}

    # ---- helpers ----
    def default(self, *keys: str) -> bool:
        return all(math.isclose(float(self.params[k]), float(DEFAULTS[k])) for k in keys)

    def step(self, key: str, **payload: Any) -> None:
        payload["key"] = key
        payload.setdefault("faithful", True)
        self.steps.append(payload)

    def timed(self, key: str, start: float) -> None:
        self.timings[key] = round(time.time() - start, 3)

    # ---- the pipeline ----
    def run(self, pool=None, item_id: str | None = None) -> dict:
        p = self.params
        seg_keys = ("seg_side", "seg_min_area", "seg_max_regions", "seg_smoothing", "seg_threshold")
        image = Image.open(self.path).convert("RGB")
        self.step("source", image=data_url(image.copy().convert("RGB")), size=list(image.size))

        # renders
        t = time.time()
        render = script("render_medium_invariant").render
        renders = {}
        for mode in ("gray", "edges", "shape", "silhouette", "mass"):
            renders[mode] = render(
                self.path,
                mode=mode,
                side=int(p["render_side"]),
                sigma=float(p["render_sigma"]),
                coarse=int(p["render_coarse"]),
            )
        self.timed("renders", t)
        self.step(
            "renders",
            images={m: data_url(im.resize((256, 256))) for m, im in renders.items()},
            faithful=self.default("render_side", "render_sigma", "render_coarse"),
        )

        # strokes: contours in two colours, strokes, sketch, pictogram
        t = time.time()
        sk_params = {k[3:]: v for k, v in p.items() if k.startswith("sk_")}
        sk = sketching.sketch(image, sk_params)
        self.timed("traits", t)
        described = [sketching.stroke_descriptor(line, sk.size) for line in sk.sketch]
        self.sketch = sk
        self.step(
            "traits",
            faithful=self.default(*[k for k in p if k.startswith("sk_")]),
            images={
                "contours": data_url(sketching.edges_image(sk), kind="PNG"),
                "traits": data_url(sketching.render(sk.strokes, sk.size, width=1), kind="PNG"),
                "croquis": data_url(sketching.render(sk.sketch, sk.size, width=2), kind="PNG"),
                "pictogramme": data_url(
                    sketching.render(sk.pictogram, sk.size, width=3), kind="PNG"
                ),
            },
            counts={
                "traits": len(sk.strokes),
                "croquis": len(sk.sketch),
                "pictogramme": len(sk.pictogram),
            },
            ink=float(sk.edges.mean()),
            features=list(sketching.STROKE_FEATURES),
            strokes=[
                {
                    "n": i + 1,
                    "points": len(line),
                    "length": round(sketching.length(line), 1),
                    "descriptor": [round(float(v), 3) for v in d],
                }
                for i, (line, d) in enumerate(zip(sk.sketch[:12], described[:12], strict=True))
            ],
        )

        # pixels and silhouette embeddings, and the contour drawing through the same encoder
        t = time.time()
        enc = encoder()
        self.vectors["pixels"] = enc.encode([image], normalize=True)[0]
        self.vectors["croquis"] = enc.encode(
            [sketching.edges_image(sk).convert("RGB")], normalize=True
        )[0]
        try:
            stroke_vocab = stroke_atoms()
        except FileNotFoundError:
            stroke_vocab = None
        if stroke_vocab is not None and described:
            signs = stroke_vocab.assign(np.stack(described))
            h = np.zeros(stroke_vocab.size, dtype=np.float32)
            for sign, line in zip(signs, sk.sketch, strict=True):
                h[int(sign)] += sketching.length(line)
            if h.any():
                self.vectors["traits"] = l2(h)
            self.steps[-1]["signs"] = [int(x) for x in signs]
        self.steps[-1]["available"] = "traits" in self.vectors
        buffer = io.BytesIO()
        renders["silhouette"].save(buffer, "JPEG", quality=92)
        buffer.seek(0)
        self.vectors["silhouette"] = enc.encode(
            [Image.open(buffer).convert("RGB")], normalize=True
        )[0]
        self.timed("pixels", t)
        self.step(
            "pixels",
            model=str(enc.spec.model_id),
            device=str(enc.device),
            dimension=768,
            norm=float(np.linalg.norm(self.vectors["pixels"])),
            preview=[float(v) for v in self.vectors["pixels"][:32]],
        )
        self.step(
            "silhouette",
            dimension=768,
            faithful=self.default("render_side", "render_coarse"),
            preview=[float(v) for v in self.vectors["silhouette"][:32]],
        )

        # segmentation
        t = time.time()
        seg = script("segment_shapes")
        regions = seg.segment(
            self.path,
            side=int(p["seg_side"]),
            min_area=float(p["seg_min_area"]),
            max_regions=int(p["seg_max_regions"]),
            smoothing=float(p["seg_smoothing"]),
            threshold=float(p["seg_threshold"]),
        )
        masks, smoothed = region_masks(
            self.path,
            side=int(p["seg_side"]),
            min_area=float(p["seg_min_area"]),
            max_regions=int(p["seg_max_regions"]),
            smoothing=float(p["seg_smoothing"]),
            threshold=float(p["seg_threshold"]),
        )
        paired = len(masks) == len(regions) and all(
            m["tone"] == r["tone"] and abs(m["area"] - r["area"]) < 1e-6
            for m, r in zip(masks, regions, strict=False)
        )
        self.timed("segmentation", t)
        descriptor = script("build_shape_vocabulary").descriptor
        described = (
            np.stack([descriptor(r) for r in regions]) if regions else np.zeros((0, 34), np.float32)
        )

        # formes muettes (v2), and the pool atoms for repetition and groups, and v4 for the mix
        signs_v2 = atoms("atoms-v2-256").assign(described) if regions else np.zeros(0, int)
        signs_pool = atoms("atoms-pool").assign(described) if regions else np.zeros(0, int)
        signs_v4 = atoms("atoms-v4-256").assign(described) if regions else np.zeros(0, int)

        smooth_img = Image.fromarray((smoothed * 255).astype(np.uint8)).resize((256, 256))
        self.step(
            "segmentation",
            faithful=self.default(*seg_keys),
            count=len(regions),
            paired=paired,
            smoothed=data_url(smooth_img),
            overlay=data_url(
                overlay(
                    self.path,
                    regions,
                    masks,
                    signs_v2 if paired else None,
                    side=int(p["seg_side"]),
                    total_signs=256,
                ),
                kind="PNG",
            ),
            regions=[
                {
                    "n": k + 1,
                    "area": r["area"],
                    "tone": r["tone"],
                    "elongation": r["elongation"],
                    "solidity": r["solidity"],
                    "holes": r["holes"],
                    "scale_vs_median": r["scale_vs_median"],
                    "centroid": [r["centroid_x"], r["centroid_y"]],
                    "extent": [r["extent_x"], r["extent_y"]],
                    "hu": r["hu"],
                    "radial": r["radial"],
                    "sign_v2": int(signs_v2[k]),
                    "sign_pool": int(signs_pool[k]),
                    "sign_v4": int(signs_v4[k]),
                    "descriptor": [float(v) for v in described[k]],
                }
                for k, r in enumerate(regions)
            ],
        )
        if regions:
            recon = script("render_region_reconstruction").reconstruct(regions, 448, oriented=True)
            self.steps[-1]["reconstruction"] = data_url(recon.resize((256, 256)))
        self.step(
            "descriptors",
            count=len(regions),
            names=[
                "log1p(aire×100)",  # noqa: RUF001
                "log1p(élongation)",
                "remplissage",
                "sombre",
                "log1p(trous)",
                "log1p(échelle/médiane)",
                "hu1",
                "hu2",
                "hu3",
                "hu4",
                *[f"radial {i * 15}°" for i in range(24)],
            ],
        )

        def histogram(signs: np.ndarray, size: int) -> np.ndarray:
            h = np.zeros(size, dtype=np.float32)
            for s, r in zip(signs, regions, strict=True):
                h[int(s)] += float(r["area"])
            return l2(h)

        if regions:
            self.vectors["formes"] = histogram(signs_v2, 256)
            self.vectors["formes_v4"] = histogram(signs_v4, 256)
            dist = atoms("atoms-v2-256").distances(described)
            nearest = [float(dist[k, int(signs_v2[k])]) for k in range(len(regions))]
        else:
            nearest = []
        self.step(
            "formes",
            faithful=self.default(*seg_keys),
            dimension=256,
            available="formes" in self.vectors,
            entries=top_entries(self.vectors["formes"], None) if regions else [],
            distance_to_sign=nearest,
        )

        # relations between signs: the grammar, on the 64-sign vocabulary
        rel = script("build_shape_relations")
        keys, key_index = relation_keys()
        top = regions[: int(p["rel_max_regions"])]
        triples = []
        if len(top) >= 2:
            signs_64 = atoms("atoms-64").assign(np.stack([descriptor(r) for r in top]))
            vec = np.zeros(len(keys), dtype=np.float32)
            for i, a in enumerate(top):
                for j, b in enumerate(top):
                    if i == j:
                        continue
                    name = rel.relation(a, b)
                    key = (int(signs_64[i]), name, int(signs_64[j]))
                    kept = key in key_index
                    if kept:
                        vec[key_index[key]] += 1.0
                    triples.append(
                        {
                            "a": i + 1,
                            "sign_a": key[0],
                            "relation": name,
                            "b": j + 1,
                            "sign_b": key[2],
                            "kept": kept,
                        }
                    )
            if vec.any():
                self.vectors["relations"] = l2(vec)
        self.step(
            "relations",
            faithful=self.default("rel_max_regions") and self.default(*seg_keys),
            dimension=len(keys),
            available="relations" in self.vectors,
            triples=triples,
            kept=sum(1 for t_ in triples if t_["kept"]),
            signs=64,
        )

        # composition
        t = time.time()
        comp = script("build_composition_channel")
        grid = int(p["comp_grid"])
        block = comp.mass_map(self.path, grid=grid)
        self.vectors["composition"] = l2(comp.composition(self.path, regions, grid=grid))
        self.timed("composition", t)
        v = self.vectors["composition"]
        self.step(
            "composition",
            faithful=self.default("comp_grid") and self.default(*seg_keys),
            comparable=grid == 8,
            dimension=int(v.shape[0]),
            heatmap=data_url(heatmap(block), kind="PNG"),
            rows=[float(x) for x in block.sum(axis=1)],
            columns=[float(x) for x in block.sum(axis=0)],
            scalars=dict(
                zip(
                    [
                        "symétrie g/d",
                        "symétrie h/b",
                        "masse au centre",
                        "entropie",
                        "dispersion des centroïdes",
                        "log élongation du nuage",
                        "log nb régions",
                    ],
                    [float(x) for x in comp.composition(self.path, regions, grid=grid)[-7:]],
                    strict=True,
                )
            ),
        )

        # repetition
        rep = script("build_repetition_channel")
        if regions:
            counts = collections.Counter(int(s) for s in signs_pool)
            profile = np.zeros(len(rep.BUCKETS), dtype=np.float32)
            typed = np.zeros(256 * len(rep.BUCKETS), dtype=np.float32)
            for s, c in counts.items():
                b = rep.bucket_of(c)
                profile[b] += 1.0
                typed[s * len(rep.BUCKETS) + b] = 1.0
            if profile.sum() > 0:
                profile /= profile.sum()
            self.vectors["repetition"] = l2(np.concatenate([profile, typed]))
            buckets = {
                name: int(sum(1 for c in counts.values() if rep.bucket_of(c) == i))
                for i, (_, _, name) in enumerate(rep.BUCKETS)
            }
            repeated = sorted(((s, c) for s, c in counts.items() if c > 1), key=lambda x: -x[1])[:8]
        else:
            buckets, repeated = {}, []
        self.step(
            "repetition",
            faithful=self.default(*seg_keys),
            dimension=1028,
            available="repetition" in self.vectors,
            buckets=buckets,
            repeated=[{"sign": s, "count": c} for s, c in repeated],
        )

        # groups and composites, then the record
        grp = script("build_shape_groups")
        group_rows = []
        if regions:
            margin, ratio, min_parts = (
                float(p["grp_margin"]),
                float(p["grp_ratio"]),
                int(p["grp_min_parts"]),
            )
            described_groups, members_list = [], []
            for members in grp.groups_of(regions, margin=margin, ratio=ratio):
                if len(members) < min_parts:
                    continue
                xs0, ys0, xs1, ys1 = zip(*(grp.box(regions[i]) for i in members), strict=True)
                width, height = max(xs1) - min(xs0), max(ys1) - min(ys0)
                area = float(sum(regions[i]["area"] for i in members))
                h = np.zeros(256, dtype=np.float32)
                for i in members:
                    h[int(signs_pool[i])] += float(regions[i]["area"])
                norm = np.linalg.norm(h)
                if norm > 0:
                    h /= norm
                outline = np.asarray(
                    [
                        np.log1p(area * 100.0),
                        np.log1p(max(width, height) / max(min(width, height), 1e-6)),
                        area / max(width * height, 1e-6),
                        np.log1p(len(members)),
                        float(np.median([regions[i]["centroid_y"] for i in members])),
                    ],
                    dtype=np.float32,
                )
                described_groups.append(np.concatenate([h, outline]))
                members_list.append(members)
            if described_groups:
                matrix = np.stack(described_groups)
                matrix[:, 256:] *= np.sqrt(256 / 5.0)
                centres = composites()
                d = ((matrix[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
                assigned = d.argmin(axis=1)
                vec = np.zeros(centres.shape[0], dtype=np.float32)
                for c in assigned:
                    vec[int(c)] += 1.0
                self.vectors["composites"] = l2(vec)
                group_rows = [
                    {"members": [i + 1 for i in m], "composite": int(c)}
                    for m, c in zip(members_list, assigned, strict=True)
                ]
        self.step(
            "groups",
            faithful=self.default("grp_margin", "grp_ratio", "grp_min_parts")
            and self.default(*seg_keys),
            groups=group_rows,
            available="composites" in self.vectors,
        )
        if {"composites", "composition", "repetition"} <= set(self.vectors):
            self.vectors["record"] = l2(
                np.concatenate(
                    [
                        l2(self.vectors["composites"]) * 1.0,
                        l2(self.vectors["composition"]) * 2.0,
                        l2(self.vectors["repetition"]) * 0.5,
                    ]
                )
            )
        self.step(
            "record",
            available="record" in self.vectors,
            dimension=1243,
            faithful=all(
                s.get("faithful", True)
                for s in self.steps
                if s["key"] in ("groups", "composition", "repetition")
            ),
            comparable="record" in self.vectors and grid == 8,
        )

        # surface: palette and signal
        t = time.time()
        surf = script("build_surface_channels")
        side = int(p["surface_side"])
        self.vectors["palette"] = l2(np.asarray(surf.palette_of(image, side), dtype=np.float32))
        self.vectors["signal"] = l2(np.asarray(surf.signal_of(image, side), dtype=np.float32))
        self.timed("surface", t)
        terms = list(surf.TERMS)
        term_names = [t[0] if isinstance(t, (tuple, list)) else str(t) for t in terms]
        self.step(
            "palette",
            faithful=self.default("surface_side"),
            dimension=11,
            balanced=data_url(balanced_image(image, side).resize((256, 256))),
            terms=[
                {"name": n, "value": float(v)}
                for n, v in zip(term_names, self.vectors["palette"], strict=True)
            ],
        )
        sig = self.vectors["signal"]
        self.step(
            "signal",
            faithful=self.default("surface_side"),
            dimension=16,
            histogram=[float(x) for x in sig[:12]],
            scalars={
                "moyenne": float(sig[12]),
                "écart-type": float(sig[13]),
                "rugosité": float(sig[14]),
                "étendue p95−p5": float(sig[15]),  # noqa: RUF001
            },
        )

        # named nodes
        t = time.time()
        nodes_info: dict = {"nodes": [], "raw": "", "model": None, "error": "nommeur désactivé"}
        if self.namer == "sonnet":
            nodes_info = name_nodes(self.path, model="sonnet", max_nodes=int(p["nodes_max"]))
        self.timed("nommes", t)
        words = vocabulary()
        if nodes_info["nodes"]:
            vec = np.zeros(len(words), dtype=np.float32)
            for node in nodes_info["nodes"][: int(p["nodes_kept"])]:
                vec[words.index(node["name"])] += 1.0
            self.vectors["nommes"] = l2(vec)
        self.step(
            "nommes",
            faithful=self.default("nodes_max", "nodes_kept"),
            dimension=len(words),
            available="nommes" in self.vectors,
            nodes=nodes_info["nodes"],
            model=nodes_info["model"],
            raw=nodes_info["raw"][:600],
            error=nodes_info["error"],
            cells=script("name_nodes_vlm").CELLS,
            counted=int(p["nodes_kept"]),
        )

        # pose
        t = time.time()
        pose = script("build_pose_channel")
        figures, frame = detect_poses(
            image, max_side=int(p["pose_max_side"]), score=float(p["pose_score"])
        )
        kept_figures, angles = [], []
        for score_, keypoints, kp_scores in figures:
            feats = pose.pose_features(keypoints, kp_scores, threshold=float(p["pose_keypoint"]))
            if feats is None:
                continue
            kept_figures.append((score_, keypoints, kp_scores))
            row = {"score": round(score_, 3), "segments": []}
            for n, (name, _, _) in enumerate(pose.SEGMENTS):
                c, s_, present = feats[3 * n], feats[3 * n + 1], feats[3 * n + 2]
                row["segments"].append(
                    {
                        "name": name,
                        "present": bool(present),
                        "angle": round(math.degrees(math.atan2(float(s_), float(c))), 1)
                        if present
                        else None,
                    }
                )
            row["shoulders_over_torso"] = round(float(feats[-1]), 3)
            angles.append((feats, row))
            if len(kept_figures) >= int(p["pose_max_figures"]):
                break
        if kept_figures:
            stack = np.stack([f for f, _ in angles])
            self.vectors["pose"] = l2(
                np.concatenate(
                    [stack.mean(axis=0), stack.std(axis=0), [math.log1p(len(kept_figures))]]
                ).astype(np.float32)
            )
        self.timed("pose", t)
        self.step(
            "pose",
            faithful=self.default(
                "pose_score", "pose_keypoint", "pose_max_figures", "pose_max_side"
            ),
            dimension=57,
            available="pose" in self.vectors,
            detected=len(figures),
            usable=len(kept_figures),
            skeleton=data_url(
                skeleton(
                    image, kept_figures or figures, frame, threshold=float(p["pose_keypoint"])
                ).resize((320, int(320 * frame[1] / frame[0])))
            ),
            figures=[row for _, row in angles],
            device=pose_model()[1],
        )

        # the mix
        if {"nommes", "pixels", "palette", "signal", "formes_v4"} <= set(self.vectors):
            self.vectors["mix"] = l2(
                np.concatenate(
                    [
                        l2(self.vectors["nommes"]),
                        l2(self.vectors["pixels"]),
                        l2(self.vectors["palette"]),
                        l2(self.vectors["signal"]),
                        l2(self.vectors["formes_v4"]),
                    ]
                )
            )
        self.step("mix", available="mix" in self.vectors, dimension=1167)

        # reproduction against the frozen pool, when the picture is one of its rows
        reproduction = {}
        if pool is not None and item_id:
            from .pool import CHANNELS

            for key in CHANNELS:
                if key in self.vectors:
                    frozen = pool.frozen_vector(key, item_id)
                    reproduction[key] = (
                        cosine(self.vectors[key], frozen) if frozen is not None else None
                    )
        return {
            "params": self.params,
            "steps": self.steps,
            "timings": self.timings,
            "channels": {k: int(v.shape[0]) for k, v in self.vectors.items()},
            "reproduction": reproduction,
        }


def analyse(
    path: Path,
    params: dict[str, Any] | None = None,
    *,
    namer: str = "sonnet",
    pool=None,
    item_id: str | None = None,
) -> tuple[Analysis, dict]:
    analysis = Analysis(path, params or {}, namer=namer)
    return analysis, analysis.run(pool=pool, item_id=item_id)
