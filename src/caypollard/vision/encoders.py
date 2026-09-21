"""Lazy Hugging Face image encoders for Phase-2 visual baselines.

Heavy libraries are imported only when an encoder is instantiated. This keeps the
core benchmark/test environment small while allowing the same package to run real
DINOv2, CLIP, and SigLIP extraction when the ``vision`` extra is installed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..embeddings.store import l2_normalize


@dataclass(frozen=True)
class VisionModelSpec:
    family: str
    model_id: str
    revision: str | None = None


DEFAULT_MODELS: dict[str, VisionModelSpec] = {
    "dinov2-base": VisionModelSpec("dinov2", "facebook/dinov2-base"),
    "clip-vit-b32": VisionModelSpec("clip", "openai/clip-vit-base-patch32"),
    "siglip-base-224": VisionModelSpec("siglip", "google/siglip-base-patch16-224"),
}


def _projected_image_features(result: Any) -> Any:
    """Return the projected image-feature tensor from ``get_image_features``.

    Transformers 4.x returned the projected tensor directly. Transformers 5.x
    returns the full vision output object with ``pooler_output`` replaced by the
    projection, so the tensor has to be unwrapped. Both shapes are accepted
    rather than pinning one major version, since the recorded library version in
    the embedding metadata is what makes a run reproducible.
    """
    if hasattr(result, "detach"):
        return result
    pooled = getattr(result, "pooler_output", None)
    if pooled is None:
        raise TypeError(
            f"get_image_features returned {type(result).__name__} without a pooler_output tensor"
        )
    return pooled


class HuggingFaceVisionEncoder:
    """Thin adapter exposing a uniform ``encode`` method for baseline models."""

    def __init__(self, spec: VisionModelSpec, *, device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Install the vision dependencies with `uv sync --extra vision`"
            ) from exc

        self._torch = torch
        self.spec = spec
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        kwargs: dict[str, Any] = {}
        if spec.revision:
            kwargs["revision"] = spec.revision

        if spec.family not in {"dinov2", "clip", "siglip"}:
            raise ValueError(f"Unsupported vision family: {spec.family}")
        # Only images are ever encoded here, so the image processor is loaded
        # directly rather than the multimodal wrapper. For CLIP and SigLIP the
        # wrapper additionally pulls in a text tokenizer, which for SigLIP means
        # a hard SentencePiece dependency for a branch that is never executed.
        self.processor = AutoImageProcessor.from_pretrained(spec.model_id, **kwargs)

        self.model = AutoModel.from_pretrained(spec.model_id, **kwargs).to(device)
        self.model.eval()

    @property
    def resolved_revision(self) -> str | None:
        """Best-effort immutable Hub commit recorded by Transformers."""
        return getattr(self.model.config, "_commit_hash", None)

    @property
    def metadata(self) -> dict[str, Any]:
        from importlib.metadata import version

        processor_config = (
            self.processor.to_dict() if hasattr(self.processor, "to_dict") else None
        )
        return {
            "family": self.spec.family,
            "model_id": self.spec.model_id,
            "requested_revision": self.spec.revision,
            "resolved_revision": self.resolved_revision,
            "device": self.device,
            "processor_class": type(self.processor).__name__,
            "processor_config": processor_config,
            "torch_version": version("torch"),
            "transformers_version": version("transformers"),
        }

    def release_cache(self) -> None:
        """Hand freed device memory back to the system.

        On MPS the allocator keeps freed blocks in a cache that grows across a
        long extraction, and it is that cache, not the model, that the machine
        runs out of: a 9 756-image run was killed at 2 000. Callers that loop
        over thousands of images should call this periodically. A no-op
        anywhere else.
        """
        if str(self.device) == "mps":
            self._torch.mps.empty_cache()

    def encode(self, images: Iterable[Any], *, normalize: bool = True) -> np.ndarray:
        """Encode PIL-compatible images into a float32 matrix."""
        batch = list(images)
        if not batch:
            raise ValueError("encode requires at least one image")
        torch = self._torch
        inputs = self.processor(images=batch, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            if self.spec.family in {"clip", "siglip"}:
                features = _projected_image_features(self.model.get_image_features(**inputs))
            else:
                output = self.model(**inputs)
                features = output.last_hidden_state[:, 0]
        matrix = features.detach().cpu().float().numpy()
        return l2_normalize(matrix) if normalize else matrix.astype(np.float32, copy=False)
