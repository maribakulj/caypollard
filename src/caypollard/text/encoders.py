"""Lazy Hugging Face sentence encoders for the T modality.

The model matrix declares seven conditions -- V, G, T, V+T, G+T, V+G, V+G+T --
and three of them plus half of a fourth had no implementation, because nothing
in the project encoded text. Emblematica supplies 30 402 transcribed mottoes, in
Latin, German, French, Dutch and English, often in the same volume, so the
encoder has to be multilingual and must not assume modern orthography.

Pooling is mean over the attention mask rather than the CLS token: the models
below are trained as sentence encoders with mean pooling, and taking CLS instead
silently produces a worse space that still looks like a valid embedding. The
choice is recorded in the metadata alongside the resolved revision, because it
changes the vectors and therefore every number computed from them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..embeddings.store import l2_normalize


@dataclass(frozen=True)
class TextModelSpec:
    family: str
    model_id: str
    revision: str | None = None
    query_prefix: str = ""
    max_length: int = 128


DEFAULT_MODELS: dict[str, TextModelSpec] = {
    # LaBSE carries a WordPiece vocabulary, so it needs no SentencePiece build,
    # and is trained on 109 languages including Latin-script early modern spelling.
    "labse": TextModelSpec("bert", "sentence-transformers/LaBSE"),
    # E5 expects its inputs marked; the prefix is part of the model contract.
    "e5-multilingual": TextModelSpec(
        "xlm-roberta", "intfloat/multilingual-e5-base", query_prefix="query: "
    ),
    "minilm-multilingual": TextModelSpec(
        "xlm-roberta", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
}


def mean_pool(hidden_states: Any, attention_mask: Any) -> Any:
    """Average token vectors over real tokens only, never over padding."""
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class HuggingFaceTextEncoder:
    """Thin adapter exposing the same ``encode`` contract as the vision encoders."""

    def __init__(self, spec: TextModelSpec, *, device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Install the text dependencies with `uv sync --extra text`"
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
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, **kwargs)
        self.model = AutoModel.from_pretrained(spec.model_id, **kwargs).to(device)
        self.model.eval()

    @property
    def resolved_revision(self) -> str | None:
        return getattr(self.model.config, "_commit_hash", None)

    @property
    def metadata(self) -> dict[str, Any]:
        from importlib.metadata import version

        return {
            "family": self.spec.family,
            "model_id": self.spec.model_id,
            "requested_revision": self.spec.revision,
            "resolved_revision": self.resolved_revision,
            "device": self.device,
            "tokenizer_class": type(self.tokenizer).__name__,
            "pooling": "mean_over_attention_mask",
            "query_prefix": self.spec.query_prefix,
            "max_length": self.spec.max_length,
            "torch_version": version("torch"),
            "transformers_version": version("transformers"),
        }

    def encode(self, texts: Iterable[str], *, normalize: bool = True) -> np.ndarray:
        batch: Sequence[str] = [self.spec.query_prefix + text for text in texts]
        if not batch:
            raise ValueError("encode requires at least one text")
        torch = self._torch
        inputs = self.tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=self.spec.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model(**inputs)
            pooled = mean_pool(output.last_hidden_state, inputs["attention_mask"])
        matrix = pooled.detach().cpu().float().numpy()
        return l2_normalize(matrix) if normalize else matrix.astype(np.float32, copy=False)


def resolve_text_spec(preset: str | None, model_id: str | None) -> TextModelSpec:
    if model_id:
        return TextModelSpec("custom", model_id)
    if preset is None:
        raise ValueError("either a preset or an explicit model id is required")
    if preset not in DEFAULT_MODELS:
        known = sorted(DEFAULT_MODELS)
        raise ValueError(f"unknown text preset {preset!r}; expected one of {known}")
    return DEFAULT_MODELS[preset]
