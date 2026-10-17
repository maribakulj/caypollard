"""The bricks: tools that can be plugged into the workshop, one kind each.

- ``figures``: picture -> one box per figure (x0, y0, x1, y1).
- ``pose``: picture + boxes -> body points per box, in the pilot's joint names.
- ``vector``: pictures -> one L2-normalised vector each (the whole picture).
- ``regions``: pictures -> one L2-normalised vector per patch (a grid over the picture),
  compared patch by patch rather than as wholes.
- ``scene``: picture -> its figures (a generic role and a posture each) and the relations
  between them, in a closed vocabulary: a scene graph without names.

A brick states where it comes from, how heavy its weights are, and whether it counts as
a heavy job on this machine (more than 1 GB in memory: run it through
``~/outils-seg/lourd.sh``, at most two at a time). Models load on first use only.
Settings (prompts, thresholds) are fixed in the brick, before results are seen; a
variant with other settings is another brick with another name.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PILOT_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"

# The pilot's joint names, in COCO's 17-point order (eyes and ears are not kept).
COCO_JOINTS = (
    "nose",
    None,
    None,
    None,
    None,
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
COCO_KEPT = frozenset(n for n in COCO_JOINTS if n)
FIGURE_PROMPT = ("a person", "a skeleton", "an angel", "a child")


def device() -> str:
    import torch

    if torch.cuda.is_available():  # on Kaggle or Colab
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class Brick:
    name: str
    kind: str  # "figures", "pose" or "vector"
    summary: str
    source: str
    weights_mb: int
    heavy: bool = False
    _model: Any = field(default=None, repr=False)

    def meta(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "summary": self.summary,
            "source": self.source,
            "weights_mb": self.weights_mb,
            "heavy": self.heavy,
        }


# --- figures ------------------------------------------------------------------------


class OpusFigures(Brick):
    """Opus's own boxes, read from the pilot description (``describe/<id>.json``)."""

    def boxes(self, image, item_id: str, bench) -> list[list[float]]:
        return [f["box"] for f in _opus(bench, item_id).get("figures", []) if f.get("box")]


class GroundingDinoFigures(Brick):
    def boxes(self, image, item_id: str, bench) -> list[list[float]]:
        import torch
        from torchvision.ops import nms
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if self._model is None:
            repo = "IDEA-Research/grounding-dino-base"
            self._model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(repo).to(device()).eval(),
                AutoProcessor.from_pretrained(repo),
            )
        model, processor = self._model
        text = ". ".join(FIGURE_PROMPT) + "."
        inputs = processor(images=image, text=text, return_tensors="pt").to(device())
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )[0]
        return _nms(result["boxes"].cpu(), result["scores"].cpu(), nms)


class Owlv2Figures(Brick):
    def boxes(self, image, item_id: str, bench) -> list[list[float]]:
        import torch
        from torchvision.ops import nms
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        if self._model is None:
            repo = "google/owlv2-base-patch16-ensemble"
            self._model = (
                Owlv2ForObjectDetection.from_pretrained(repo).to(device()).eval(),
                Owlv2Processor.from_pretrained(repo),
            )
        model, processor = self._model
        inputs = processor(text=[list(FIGURE_PROMPT)], images=image, return_tensors="pt").to(
            device()
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # OWLv2 pads the picture to a square; boxes are given for the padded square.
        side = max(image.size)
        result = processor.post_process_grounded_object_detection(
            outputs, threshold=0.1, target_sizes=[(side, side)]
        )[0]
        boxes = result["boxes"].cpu()
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image.size[0])
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image.size[1])
        return _nms(boxes, result["scores"].cpu(), nms)


def _nms(boxes, scores, nms) -> list[list[float]]:
    if not len(boxes):
        return []
    keep = nms(boxes, scores, 0.5)
    return [[round(v, 1) for v in boxes[i].tolist()] for i in keep]


# --- pose ---------------------------------------------------------------------------


class OpusPose(Brick):
    """Opus's points, read from the pilot description; only with Opus's own boxes."""

    def points(self, image, boxes, item_id: str, bench) -> list[dict]:
        figures = _opus(bench, item_id).get("figures", [])
        by_box = {tuple(f["box"]): f.get("keypoints") or {} for f in figures if f.get("box")}
        try:
            return [{n: p for n, p in by_box[tuple(b)].items() if p} for b in boxes]
        except KeyError as exc:
            raise ValueError("the opus pose brick only works with the opus figures brick") from exc


class VitPose(Brick):
    """ViTPose-B, top-down in each box; mmpose checkpoints converted to transformers."""

    checkpoint: str = ""
    url: str = ""

    def points(self, image, boxes, item_id: str, bench) -> list[dict]:
        import torch

        if not boxes:
            return []
        if self._model is None:
            self._model = _load_vitpose(self.checkpoint, self.url)
        model, processor = self._model
        coco = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
        inputs = processor(image, boxes=[coco], return_tensors="pt").to(device())
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_pose_estimation(outputs, boxes=[coco])[0]
        out = []
        for res in results:
            k = {}
            for name, p, s in zip(
                COCO_JOINTS, res["keypoints"].tolist(), res["scores"].tolist(), strict=True
            ):
                if name and s >= 0.3:
                    k[name] = [round(p[0], 1), round(p[1], 1)]
            out.append(k)
        return out


def _load_vitpose(checkpoint: str, url: str):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import VitPoseConfig, VitPoseForPoseEstimation, VitPoseImageProcessor

    repo = "usyd-community/vitpose-base-simple"
    path = Path("data/models/vitpose") / checkpoint
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    config = VitPoseConfig.from_json_file(hf_hub_download(repo, "config.json"))
    config.use_simple_decoder = False
    model = VitPoseForPoseEstimation(config)
    mm = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(convert_vitpose(mm), strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint does not fit: {missing[:3]} {unexpected[:3]}")
    return model.to(device()).eval(), VitPoseImageProcessor.from_pretrained(repo)


def convert_vitpose(mm: dict) -> dict:
    """mmpose ViTPose-B (classic decoder) names to transformers' VitPose names."""
    out = {
        "backbone.embeddings.position_embeddings": mm["backbone.pos_embed"],
        "backbone.embeddings.patch_embeddings.projection.weight": mm[
            "backbone.patch_embed.projection.weight"
        ],
        "backbone.embeddings.patch_embeddings.projection.bias": mm[
            "backbone.patch_embed.projection.bias"
        ],
        "backbone.layernorm.weight": mm["backbone.ln1.weight"],
        "backbone.layernorm.bias": mm["backbone.ln1.bias"],
    }
    layers = 1 + max(int(k.split(".")[2]) for k in mm if k.startswith("backbone.layers."))
    for i in range(layers):
        src, dst = f"backbone.layers.{i}.", f"backbone.encoder.layer.{i}."
        for part in ("weight", "bias"):
            q, k, v = mm[src + f"attn.qkv.{part}"].chunk(3, dim=0)
            out[dst + f"attention.attention.query.{part}"] = q
            out[dst + f"attention.attention.key.{part}"] = k
            out[dst + f"attention.attention.value.{part}"] = v
            out[dst + f"attention.output.dense.{part}"] = mm[src + f"attn.proj.{part}"]
            out[dst + f"layernorm_before.{part}"] = mm[src + f"ln1.{part}"]
            out[dst + f"layernorm_after.{part}"] = mm[src + f"ln2.{part}"]
            out[dst + f"mlp.fc1.{part}"] = mm[src + f"ffn.layers.0.0.{part}"]
            out[dst + f"mlp.fc2.{part}"] = mm[src + f"ffn.layers.1.{part}"]
    out["head.deconv1.weight"] = mm["head.deconv_layers.0.weight"]
    out["head.deconv2.weight"] = mm["head.deconv_layers.3.weight"]
    for src, dst in (("1", "batchnorm1"), ("4", "batchnorm2")):
        for part in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            out[f"head.{dst}.{part}"] = mm[f"head.deconv_layers.{src}.{part}"]
    out["head.conv.weight"] = mm["head.final_layer.weight"]
    out["head.conv.bias"] = mm["head.final_layer.bias"]
    return out


# --- vectors ------------------------------------------------------------------------


class HuggingFaceVector(Brick):
    """A whole-picture encoder from ``caypollard.vision`` (CLS or projected features)."""

    spec: str = ""

    def encode(self, images: list) -> np.ndarray:
        from caypollard.vision.encoders import DEFAULT_MODELS, HuggingFaceVisionEncoder

        if self._model is None:
            self._model = HuggingFaceVisionEncoder(DEFAULT_MODELS[self.spec], device="auto")
        return self._model.encode(images)


class DinoPatches(Brick):
    """DINOv2's patch tokens at 224 px: a 16 x 16 grid, one 768-d vector per square."""

    symmetric: bool = True

    def patches(self, images: list) -> np.ndarray:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if self._model is None:
            repo = "facebook/dinov2-base"
            processor = AutoImageProcessor.from_pretrained(
                repo, size={"height": 224, "width": 224}, do_center_crop=False
            )
            self._model = (AutoModel.from_pretrained(repo).to(device()).eval(), processor)
        model, processor = self._model
        inputs = processor(images=images, return_tensors="pt").to(device())
        with torch.no_grad():
            tokens = model(**inputs).last_hidden_state[:, 1:]  # drop the CLS token
        tokens = torch.nn.functional.normalize(tokens, dim=-1)
        return tokens.cpu().numpy().astype(np.float16)


SCENE_ROLES = (
    "man",
    "woman",
    "old man",
    "old woman",
    "child",
    "infant",
    "angel",
    "soldier",
    "crowd",
    "animal",
)
SCENE_POSTURES = ("standing", "seated", "kneeling", "lying", "bending", "flying", "riding")
SCENE_RELATIONS = (
    "holds",
    "carries",
    "embraces",
    "touches",
    "reaches toward",
    "points at",
    "looks at",
    "kneels before",
    "bows to",
    "gives to",
    "blesses",
    "supports",
    "mourns over",
    "strikes",
    "leads",
    "lies before",
    "is above",
    "stands beside",
)
SCENE_PROMPT = """Read the image file {path} with the Read tool.

Describe what is visible in it as a scene graph, without naming anyone and without saying \
what story or subject it shows: no proper names, no titles, no religious identities. Give \
only the figures that take part in the main scene (at most 8; a group can be one figure \
with role "crowd").

Answer with JSON only:
{{"figures": [{{"id": "f1", "role": ..., "posture": ...}}, ...],
 "relations": [{{"a": "f1", "rel": ..., "b": "f2"}}, ...],
 "subject": "..."}}

role is one of: {roles}.
posture is one of: {postures}.
rel is one of: {relations} (a and b are figure ids).
Give every relation you can see between the figures, and only those.
Last, and only last, "subject": the subject you think the picture shows, in a few words."""


class ClaudeScene(Brick):
    """A vision-language model reads the scene graph, headless (``claude -p``), once per
    picture. The subject it names is kept apart and never used by the distance."""

    model: str = "opus"

    def scene(self, path: Path) -> dict:
        import subprocess

        prompt = SCENE_PROMPT.format(
            path=path.resolve(),
            roles=", ".join(SCENE_ROLES),
            postures=", ".join(SCENE_POSTURES),
            relations=", ".join(SCENE_RELATIONS),
        )
        run = subprocess.run(
            [
                "claude",
                "-p",
                "--model",
                self.model,
                "--allowedTools",
                "Read",
                "--output-format",
                "json",
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            stdin=subprocess.DEVNULL,
        )
        text = str(json.loads(run.stdout).get("result", ""))
        start = text.find("{")
        # No JSON at all is a failed call (API error, usage limit...), not an empty scene:
        # raising keeps it out of the cache so the next run asks again.
        if text.startswith("API Error") or start < 0:
            raise RuntimeError(f"{path.name}: {text[:80]}")
        try:
            data, _ = json.JSONDecoder().raw_decode(text[start:]) if start >= 0 else ({}, 0)
        except json.JSONDecodeError:
            data = {}
        return {"figures": [], "relations": [], "subject": "", **data, "_raw": text}


class ElsewhereScene(Brick):
    """Scene graphs computed elsewhere (a Kaggle notebook) with the same prompt, read from
    the cache; nothing runs here."""

    def scene(self, path: Path) -> dict:
        raise RuntimeError(f"{self.name} runs on Kaggle ({self.source}); fetch its cache first")


def _opus(bench, item_id: str) -> dict:
    path = bench.root / "describe" / f"{item_id}.json"
    if not path.exists():
        sys.path.insert(0, str(PILOT_SCRIPTS))
        from pilot_describe import describe  # one call to Claude Opus, about a minute

        data = describe(bench.image(item_id), "opus", 600)
        if str(data.get("_raw", "")).startswith("API Error"):
            raise RuntimeError(f"{item_id}: Opus unreachable — {data['_raw'][:80]}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return json.loads(path.read_text())


def _vitpose(name: str, summary: str, checkpoint: str, url: str) -> VitPose:
    brick = VitPose(name, "pose", summary, url, 360)
    brick.checkpoint, brick.url = checkpoint, url
    return brick


def _vector(name: str, spec: str, summary: str, source: str, mb: int) -> HuggingFaceVector:
    brick = HuggingFaceVector(name, "vector", summary, source, mb)
    brick.spec = spec
    return brick


def _patches(name: str, symmetric: bool, summary: str) -> DinoPatches:
    brick = DinoPatches(
        name, "regions", summary, "https://huggingface.co/facebook/dinov2-base", 350
    )
    brick.symmetric = symmetric
    return brick


def _scene(name: str, model: str, summary: str) -> ClaudeScene:
    brick = ClaudeScene(name, "scene", summary, "claude -p (Claude Code, headless)", 0)
    brick.model = model
    return brick


MMPOSE = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap"
BRICKS: dict[tuple[str, str], Brick] = {
    (b.kind, b.name): b
    for b in (
        OpusFigures(
            "opus",
            "figures",
            "Claude Opus names each figure and boxes it (one pass, ~1 min a picture, no review)",
            "scripts/pilot_describe.py",
            0,
        ),
        GroundingDinoFigures(
            "grounding-dino",
            "figures",
            "Grounding DINO base, zero-shot; prompt 'a person. a skeleton. an angel. a child.', "
            "boxes above 0.3, NMS 0.5",
            "https://huggingface.co/IDEA-Research/grounding-dino-base",
            900,
        ),
        Owlv2Figures(
            "owlv2",
            "figures",
            "OWLv2 base (ensemble), zero-shot; the same four queries, boxes above 0.1, NMS 0.5",
            "https://huggingface.co/google/owlv2-base-patch16-ensemble",
            620,
        ),
        OpusPose(
            "opus",
            "pose",
            "Claude Opus's body points (15 joints, crown and chin included)",
            "scripts/pilot_describe.py",
            0,
        ),
        _vitpose(
            "vitpose-humanart",
            "ViTPose-B trained on COCO + Human-Art (paintings, sculpture, drawings): 0.759 AP "
            "on Human-Art with given boxes; points above 0.3",
            "vitpose-b-humanart.pth",
            f"{MMPOSE}/human_art/td-hm_ViTPose-base_8xb64-210e_humanart-256x192-b417f546_20230611.pth",
        ),
        _vitpose(
            "vitpose-coco",
            "ViTPose-B trained on photographs only (COCO): 0.555 AP on Human-Art with given "
            "boxes; points above 0.3. The control for what Human-Art adds",
            "vitpose-b-coco.pth",
            f"{MMPOSE}/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth",
        ),
        _vector(
            "dinov2",
            "dinov2-base",
            "DINOv2 base, CLS token (self-supervised, 2023)",
            "https://huggingface.co/facebook/dinov2-base",
            350,
        ),
        _vector(
            "clip",
            "clip-vit-b32",
            "CLIP ViT-B/32 image features (image-text, 2021)",
            "https://huggingface.co/openai/clip-vit-base-patch32",
            600,
        ),
        _vector(
            "siglip",
            "siglip-base-224",
            "SigLIP base 224 image features (image-text, 2023)",
            "https://huggingface.co/google/siglip-base-patch16-224",
            800,
        ),
        _scene(
            "opus-scene",
            "opus",
            "Claude Opus reads a scene graph without names: figures with a generic role and "
            "a posture, relations in a closed vocabulary of 18 verbs; one pass, no review",
        ),
        ElsewhereScene(
            "qwen25vl3b-scene",
            "scene",
            "Qwen2.5-VL-3B-Instruct, zero-shot, the same prompt as opus-scene; on Kaggle (T4), "
            "greedy decoding, pictures at 256-640 x 28 x 28 pixels",
            "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct",
            7500,
            heavy=True,
        ),
        # Both regions bricks were fixed together, before either was run.
        _patches(
            "dinov2-patchs",
            True,
            "DINOv2 base patch tokens (16 x 16 at 224 px, whole picture, no crop); each patch "
            "takes its best match in the other picture, mean of both directions",
        ),
        _patches(
            "dinov2-patchs-requete",
            False,
            "The same patches, matched from the query only: a detail taken from a larger "
            "picture can still find it, but a busy picture can match everything",
        ),
    )
}


def brick(kind: str, name: str) -> Brick:
    try:
        return BRICKS[(kind, name)]
    except KeyError:
        known = ", ".join(n for k, n in BRICKS if k == kind)
        raise SystemExit(f"no {kind} brick named {name!r}; known: {known}") from None
