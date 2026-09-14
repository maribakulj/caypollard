import numpy as np

from caypollard.embeddings.store import EmbeddingTable
from caypollard.fusion import (
    SimilarityBounds,
    fit_similarity_bounds,
    late_fusion,
    late_fusion_embedding_table,
    minmax_scale,
    rerank_visual_candidates,
)


def _tables():
    ids = ("a", "b", "c", "d")
    visual = EmbeddingTable(
        ids=ids,
        vectors=np.asarray([[1, 0], [0.9, 0.1], [0.1, 0.9], [0, 1]], dtype=np.float32),
        metadata={"method": "visual-fixture"},
    )
    graph = EmbeddingTable(
        ids=ids,
        vectors=np.asarray([[1, 0], [0.1, 0.9], [0.95, 0.05], [0, 1]], dtype=np.float32),
        metadata={"method": "graph-fixture"},
    )
    return visual, graph


def test_late_fusion_endpoints_match_modalities():
    visual = np.asarray([0.2, 0.9])
    graph = np.asarray([0.8, 0.1])
    assert np.allclose(late_fusion(visual, graph, 1.0), visual)
    assert np.allclose(late_fusion(visual, graph, 0.0), graph)


def test_minmax_scale_uses_external_bounds_without_clipping():
    scores = np.asarray([-1.0, 0.0, 2.0])
    scaled = minmax_scale(scores, low=0.0, high=1.0)
    assert np.allclose(scaled, [-1.0, 0.0, 2.0])


def test_similarity_bounds_are_deterministic():
    visual, _ = _tables()
    first, meta1 = fit_similarity_bounds(visual, list(visual.ids), sample_pairs=100, seed=4)
    second, meta2 = fit_similarity_bounds(visual, list(visual.ids), sample_pairs=100, seed=4)
    assert first == second
    assert meta1 == meta2
    assert first.high > first.low


def test_weighted_concatenation_matches_minmax_late_fusion_ranking():
    visual, graph = _tables()
    vb = SimilarityBounds(low=-1.0, high=1.0)
    gb = SimilarityBounds(low=-0.5, high=1.0)
    alpha = 0.6
    fused = late_fusion_embedding_table(
        visual, graph, alpha=alpha, visual_bounds=vb, graph_bounds=gb
    )
    q = 0
    fused_scores = fused.vectors @ fused.vectors[q]
    v = visual.vectors / np.linalg.norm(visual.vectors, axis=1, keepdims=True)
    g = graph.vectors / np.linalg.norm(graph.vectors, axis=1, keepdims=True)
    direct = late_fusion(
        minmax_scale(v @ v[q], low=vb.low, high=vb.high),
        minmax_scale(g @ g[q], low=gb.low, high=gb.high),
        alpha=alpha,
    )
    assert list(np.argsort(-fused_scores)) == list(np.argsort(-direct))


def test_reranking_can_change_visual_order():
    visual, graph = _tables()
    vb = SimilarityBounds(low=-1.0, high=1.0)
    gb = SimilarityBounds(low=-1.0, high=1.0)
    visual_only = rerank_visual_candidates(
        visual,
        graph,
        query_id="a",
        candidate_ids=visual.ids,
        candidate_k=3,
        alpha=1.0,
        visual_bounds=vb,
        graph_bounds=gb,
    )
    graph_heavy = rerank_visual_candidates(
        visual,
        graph,
        query_id="a",
        candidate_ids=visual.ids,
        candidate_k=3,
        alpha=0.1,
        visual_bounds=vb,
        graph_bounds=gb,
    )
    assert visual_only[0][0] == "b"
    assert graph_heavy[0][0] == "c"
