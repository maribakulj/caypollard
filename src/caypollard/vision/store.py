"""Portable storage for image embeddings and their identifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EmbeddingTable:
    """Aligned identifiers, embedding matrix, and run metadata."""

    ids: tuple[str, ...]
    vectors: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a two-dimensional matrix")
        if len(self.ids) != self.vectors.shape[0]:
            raise ValueError("ids and vectors must contain the same number of rows")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("embedding ids must be unique")


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized float32 vectors."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero vectors cannot be L2-normalized")
    return matrix / norms


def save_embedding_table(
    path: str | Path,
    *,
    ids: list[str] | tuple[str, ...],
    vectors: np.ndarray,
    metadata: dict[str, Any],
    normalize: bool = True,
) -> tuple[Path, Path]:
    """Save embeddings as NPZ plus a human-readable JSON metadata sidecar."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    matrix = l2_normalize(vectors) if normalize else np.asarray(vectors, dtype=np.float32)
    clean_ids = tuple(str(value) for value in ids)
    EmbeddingTable(clean_ids, matrix, dict(metadata))

    np.savez_compressed(target, ids=np.asarray(clean_ids), vectors=matrix)
    metadata_path = target.with_suffix(target.suffix + ".metadata.json")
    payload = dict(metadata)
    payload.update(
        {
            "n_items": len(clean_ids),
            "dimension": int(matrix.shape[1]) if matrix.size else 0,
            "l2_normalized": bool(normalize),
            "embedding_file": target.name,
        }
    )
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target, metadata_path


def load_embedding_table(path: str | Path) -> EmbeddingTable:
    """Load an embedding table and its required metadata sidecar."""
    target = Path(path)
    with np.load(target, allow_pickle=False) as payload:
        ids = tuple(str(value) for value in payload["ids"].tolist())
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
    metadata_path = target.with_suffix(target.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return EmbeddingTable(ids=ids, vectors=vectors, metadata=metadata)
