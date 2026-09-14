"""Small learned alignment models for frozen visual and graph embeddings.

The module intentionally keeps the learned component narrow: precomputed encoders stay
frozen and only projection heads are trained. Test identifiers must never be supplied to
``train_joint_alignment``; the function accepts explicit train and validation partitions.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from caypollard.embeddings.store import EmbeddingTable, l2_normalize


@dataclass(frozen=True)
class AlignmentConfig:
    """Hyperparameters for the small shared-space projection model."""

    projection_dim: int = 128
    hidden_dim: int | None = None
    temperature: float = 0.07
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 100
    patience: int = 15
    seed: int = 42

    def __post_init__(self) -> None:
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive when provided")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")


@dataclass(frozen=True)
class AlignmentRun:
    """Result of one projection-head training run."""

    ids: tuple[str, ...]
    visual_projected: np.ndarray
    graph_projected: np.ndarray
    joint_projected: np.ndarray
    metadata: dict[str, Any]
    history: tuple[dict[str, float | int], ...]

    def as_tables(self) -> tuple[EmbeddingTable, EmbeddingTable, EmbeddingTable]:
        common = dict(self.metadata)
        visual = EmbeddingTable(
            self.ids,
            self.visual_projected,
            {**common, "representation": "projected_visual"},
        )
        graph = EmbeddingTable(
            self.ids,
            self.graph_projected,
            {**common, "representation": "projected_graph"},
        )
        joint = EmbeddingTable(
            self.ids,
            self.joint_projected,
            {**common, "representation": "joint_mean"},
        )
        return visual, graph, joint


def ids_digest(ids: Iterable[str]) -> str:
    """Return a stable digest for an identifier set without bloating run metadata."""
    payload = "\n".join(sorted({str(item_id) for item_id in ids})).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def aligned_embedding_matrices(
    visual: EmbeddingTable,
    graph: EmbeddingTable,
    ids: Iterable[str],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Return aligned, normalized matrices in deterministic identifier order."""
    visual_row = {item_id: i for i, item_id in enumerate(visual.ids)}
    graph_row = {item_id: i for i, item_id in enumerate(graph.ids)}
    requested = tuple(sorted({str(item_id) for item_id in ids}))
    missing_visual = [item_id for item_id in requested if item_id not in visual_row]
    missing_graph = [item_id for item_id in requested if item_id not in graph_row]
    if missing_visual:
        raise ValueError(f"ids missing from visual embeddings: {', '.join(missing_visual[:5])}")
    if missing_graph:
        raise ValueError(f"ids missing from graph embeddings: {', '.join(missing_graph[:5])}")
    if len(requested) < 2:
        raise ValueError("at least two aligned ids are required")
    visual_matrix = l2_normalize(
        np.stack([visual.vectors[visual_row[item_id]] for item_id in requested])
    )
    graph_matrix = l2_normalize(
        np.stack([graph.vectors[graph_row[item_id]] for item_id in requested])
    )
    return requested, visual_matrix, graph_matrix


def _torch_imports():
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "learned alignment requires the optional alignment dependencies; "
            "install caypollard[alignment]"
        ) from exc
    return torch, nn, F


def _build_projection_head(input_dim: int, config: AlignmentConfig):
    torch, nn, _ = _torch_imports()
    if config.hidden_dim is None:
        return nn.Linear(input_dim, config.projection_dim)
    return nn.Sequential(
        nn.Linear(input_dim, config.hidden_dim),
        nn.GELU(),
        nn.Linear(config.hidden_dim, config.projection_dim),
    )


def symmetric_info_nce_loss(
    visual_projection,
    graph_projection,
    *,
    temperature: float,
):
    """Symmetric cross-modal InfoNCE loss with diagonal positive pairs."""
    torch, _, F = _torch_imports()
    if visual_projection.ndim != 2 or graph_projection.ndim != 2:
        raise ValueError("projection tensors must be two-dimensional")
    if visual_projection.shape != graph_projection.shape:
        raise ValueError("visual and graph projection tensors must have identical shapes")
    if visual_projection.shape[0] < 2:
        raise ValueError("InfoNCE requires at least two aligned pairs")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    visual_projection = F.normalize(visual_projection, dim=1)
    graph_projection = F.normalize(graph_projection, dim=1)
    logits = visual_projection @ graph_projection.T / temperature
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def _set_deterministic_seed(seed: int) -> None:
    torch, _, _ = _torch_imports()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA in CI
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:  # pragma: no cover - backend-dependent
        pass


def _project_numpy(head, matrix: np.ndarray, device: str) -> np.ndarray:
    torch, _, F = _torch_imports()
    head.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        projected = F.normalize(head(tensor), dim=1)
    return projected.detach().cpu().numpy().astype(np.float32, copy=False)


def train_joint_alignment(
    visual: EmbeddingTable,
    graph: EmbeddingTable,
    *,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    output_ids: Sequence[str] | None = None,
    config: AlignmentConfig | None = None,
    device: str = "cpu",
) -> AlignmentRun:
    """Train two small projection heads using train/validation partitions only.

    ``output_ids`` controls which aligned rows are projected after training. It may include
    test identifiers because no optimisation is performed on them. The returned metadata
    records train/validation identifiers so leakage can be audited downstream.
    """
    torch, _, _ = _torch_imports()
    config = config or AlignmentConfig()
    _set_deterministic_seed(config.seed)

    train_ids_tuple, train_visual, train_graph = aligned_embedding_matrices(
        visual, graph, train_ids
    )
    val_ids_tuple, val_visual, val_graph = aligned_embedding_matrices(
        visual, graph, validation_ids
    )
    overlap = set(train_ids_tuple).intersection(val_ids_tuple)
    if overlap:
        raise ValueError(f"train and validation ids overlap: {sorted(overlap)[:5]}")

    selected_output_ids = (
        tuple(sorted(set(visual.ids).intersection(graph.ids)))
        if output_ids is None
        else tuple(str(item_id) for item_id in output_ids)
    )
    out_ids, out_visual, out_graph = aligned_embedding_matrices(
        visual, graph, selected_output_ids
    )

    visual_head = _build_projection_head(train_visual.shape[1], config).to(device)
    graph_head = _build_projection_head(train_graph.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        list(visual_head.parameters()) + list(graph_head.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_visual_tensor = torch.as_tensor(train_visual, dtype=torch.float32)
    train_graph_tensor = torch.as_tensor(train_graph, dtype=torch.float32)
    val_visual_tensor = torch.as_tensor(val_visual, dtype=torch.float32, device=device)
    val_graph_tensor = torch.as_tensor(val_graph, dtype=torch.float32, device=device)

    rng = np.random.default_rng(config.seed)
    best_loss = float("inf")
    best_epoch = 0
    best_visual_state = deepcopy(visual_head.state_dict())
    best_graph_state = deepcopy(graph_head.state_dict())
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        visual_head.train()
        graph_head.train()
        permutation = rng.permutation(len(train_ids_tuple))
        train_losses: list[float] = []
        for start in range(0, len(permutation), config.batch_size):
            batch_indices = permutation[start : start + config.batch_size]
            if len(batch_indices) < 2:
                continue
            batch_visual = train_visual_tensor[batch_indices].to(device)
            batch_graph = train_graph_tensor[batch_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = symmetric_info_nce_loss(
                visual_head(batch_visual),
                graph_head(batch_graph),
                temperature=config.temperature,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        if not train_losses:
            raise ValueError("training partition is too small for the configured batch size")

        visual_head.eval()
        graph_head.eval()
        with torch.no_grad():
            val_loss_tensor = symmetric_info_nce_loss(
                visual_head(val_visual_tensor),
                graph_head(val_graph_tensor),
                temperature=config.temperature,
            )
        val_loss = float(val_loss_tensor.detach().cpu())
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})

        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch
            best_visual_state = deepcopy(visual_head.state_dict())
            best_graph_state = deepcopy(graph_head.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    visual_head.load_state_dict(best_visual_state)
    graph_head.load_state_dict(best_graph_state)
    projected_visual = _project_numpy(visual_head, out_visual, device)
    projected_graph = _project_numpy(graph_head, out_graph, device)
    joint = l2_normalize(projected_visual + projected_graph)

    metadata: dict[str, Any] = {
        "family": "multimodal",
        "method": "frozen-encoder-symmetric-infonce-projection",
        "protocol": "protocol-v0.4",
        "config": asdict(config),
        "device": device,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_ran": len(history),
        "train_id_count": len(train_ids_tuple),
        "validation_id_count": len(val_ids_tuple),
        "train_ids_sha256": ids_digest(train_ids_tuple),
        "validation_ids_sha256": ids_digest(val_ids_tuple),
        "output_id_count": len(out_ids),
        "visual_source": visual.metadata.get("method") or visual.metadata.get("model_id"),
        "graph_source": graph.metadata.get("method") or graph.metadata.get("model_id"),
        "positive_pairing": "same object id across modalities",
        "negative_sampling": "in-batch negatives",
        "test_optimisation": False,
    }
    return AlignmentRun(
        ids=out_ids,
        visual_projected=projected_visual,
        graph_projected=projected_graph,
        joint_projected=joint,
        metadata=metadata,
        history=tuple(history),
    )


def embedding_collapse_diagnostics(table: EmbeddingTable) -> dict[str, float | int]:
    """Return simple collapse/hubness-adjacent diagnostics for one embedding table."""
    vectors = l2_normalize(table.vectors)
    n_items, dimension = vectors.shape
    if n_items < 2:
        raise ValueError("at least two embeddings are required")
    similarities = vectors @ vectors.T
    upper = similarities[np.triu_indices(n_items, k=1)]
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    squared = singular_values**2
    effective_rank = 0.0
    if float(squared.sum()) > 0:
        effective_rank = float((squared.sum() ** 2) / np.square(squared).sum())
    return {
        "n_items": n_items,
        "dimension": dimension,
        "mean_off_diagonal_cosine": float(upper.mean()),
        "std_off_diagonal_cosine": float(upper.std()),
        "max_off_diagonal_cosine": float(upper.max()),
        "effective_rank": effective_rank,
    }
