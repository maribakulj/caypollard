"""Transparent multimodal fusion baselines."""

from __future__ import annotations

import numpy as np


def late_fusion(
    visual_scores: np.ndarray,
    graph_scores: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Combine aligned visual and graph similarity scores.

    Parameters
    ----------
    visual_scores:
        Array of visual similarity scores.
    graph_scores:
        Array of graph similarity scores with the same shape.
    alpha:
        Weight assigned to visual similarity. Must lie in [0, 1].
    """
    if visual_scores.shape != graph_scores.shape:
        raise ValueError("visual_scores and graph_scores must have identical shapes")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return alpha * visual_scores + (1.0 - alpha) * graph_scores


def minmax_scale(scores: np.ndarray, *, low: float, high: float) -> np.ndarray:
    """Scale scores using externally fitted bounds.

    Bounds should normally be estimated on validation data, not test data.
    """
    if high <= low:
        raise ValueError("high must be greater than low")
    return (scores - low) / (high - low)
