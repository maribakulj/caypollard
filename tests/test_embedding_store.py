import json

import numpy as np
import pytest

from caypollard.embeddings.store import (
    EmbeddingTable,
    l2_normalize,
    load_embedding_table,
    save_embedding_table,
)


def test_l2_normalize():
    result = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
    assert np.linalg.norm(result, axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_zero_vector_rejected():
    with pytest.raises(ValueError):
        l2_normalize(np.array([[0.0, 0.0]], dtype=np.float32))


def test_embedding_table_round_trip(tmp_path):
    path = tmp_path / "embeddings.npz"
    save_embedding_table(
        path,
        ids=["a", "b"],
        vectors=np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        metadata={"model_id": "fixture/model", "resolved_revision": "abc123"},
    )
    table = load_embedding_table(path)
    assert table.ids == ("a", "b")
    assert table.vectors.shape == (2, 2)
    assert table.metadata["model_id"] == "fixture/model"
    metadata = json.loads((tmp_path / "embeddings.npz.metadata.json").read_text())
    assert metadata["dimension"] == 2
    assert metadata["l2_normalized"] is True


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        EmbeddingTable(("x", "x"), np.ones((2, 2), dtype=np.float32), {})
