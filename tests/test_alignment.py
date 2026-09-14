import numpy as np
import pytest

from caypollard.alignment import (
    AlignmentConfig,
    aligned_embedding_matrices,
    embedding_collapse_diagnostics,
    symmetric_info_nce_loss,
    train_joint_alignment,
)
from caypollard.embeddings.store import EmbeddingTable, l2_normalize


def _tables(seed=0):
    rng = np.random.default_rng(seed)
    latent = l2_normalize(rng.normal(size=(12, 4)).astype(np.float32))
    visual_map = rng.normal(size=(4, 7)).astype(np.float32)
    graph_map = rng.normal(size=(4, 6)).astype(np.float32)
    visual = l2_normalize(latent @ visual_map + 0.02 * rng.normal(size=(12, 7)))
    graph = l2_normalize(latent @ graph_map + 0.02 * rng.normal(size=(12, 6)))
    ids = tuple(f"item-{i:02d}" for i in range(12))
    return (
        EmbeddingTable(ids, visual, {"method": "fixture-visual"}),
        EmbeddingTable(ids, graph, {"method": "fixture-graph"}),
    )


def test_aligned_embedding_matrices_is_deterministic():
    visual, graph = _tables()
    ids, v, g = aligned_embedding_matrices(visual, graph, ["item-03", "item-01"])
    assert ids == ("item-01", "item-03")
    assert v.shape == (2, 7)
    assert g.shape == (2, 6)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_symmetric_info_nce_prefers_correct_pairing():
    torch = pytest.importorskip("torch")
    identity = torch.eye(4)
    correct = symmetric_info_nce_loss(identity, identity, temperature=0.1)
    wrong = symmetric_info_nce_loss(identity, identity.flip(0), temperature=0.1)
    assert float(correct) < float(wrong)


def test_training_is_reproducible_and_never_optimises_on_test_ids():
    pytest.importorskip("torch")
    visual, graph = _tables()
    config = AlignmentConfig(
        projection_dim=5,
        hidden_dim=None,
        epochs=30,
        patience=8,
        batch_size=4,
        learning_rate=0.02,
        seed=7,
    )
    kwargs = dict(
        train_ids=visual.ids[:8],
        validation_ids=visual.ids[8:10],
        output_ids=visual.ids,
        config=config,
    )
    first = train_joint_alignment(visual, graph, **kwargs)
    second = train_joint_alignment(visual, graph, **kwargs)
    assert np.allclose(first.joint_projected, second.joint_projected, atol=1e-6)
    assert first.metadata["test_optimisation"] is False
    assert first.metadata["train_id_count"] == 8
    assert first.metadata["validation_id_count"] == 2
    assert len(first.metadata["train_ids_sha256"]) == 64
    assert len(first.metadata["validation_ids_sha256"]) == 64
    assert first.metadata["best_epoch"] >= 1
    assert len(first.history) <= config.epochs
    assert np.allclose(np.linalg.norm(first.joint_projected, axis=1), 1.0, atol=1e-5)


def test_training_rejects_train_validation_overlap():
    pytest.importorskip("torch")
    visual, graph = _tables()
    with pytest.raises(ValueError, match="overlap"):
        train_joint_alignment(
            visual,
            graph,
            train_ids=visual.ids[:6],
            validation_ids=visual.ids[5:8],
            config=AlignmentConfig(epochs=1, batch_size=3),
        )


def test_collapse_diagnostics_distinguish_spread_from_near_collapse():
    spread = EmbeddingTable(
        ("a", "b", "c", "d"),
        np.eye(4, dtype=np.float32),
        {},
    )
    nearly_same = l2_normalize(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.01, 0.0],
                [1.0, 0.0, 0.01],
                [1.0, 0.01, 0.01],
            ],
            dtype=np.float32,
        )
    )
    collapsed = EmbeddingTable(("a", "b", "c", "d"), nearly_same, {})
    spread_stats = embedding_collapse_diagnostics(spread)
    collapsed_stats = embedding_collapse_diagnostics(collapsed)
    assert collapsed_stats["mean_off_diagonal_cosine"] > spread_stats["mean_off_diagonal_cosine"]
    assert spread_stats["effective_rank"] > 1.0
