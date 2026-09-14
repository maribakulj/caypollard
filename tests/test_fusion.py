import numpy as np
import pytest

from caypollard.fusion import late_fusion, minmax_scale


def test_late_fusion_midpoint():
    visual = np.array([1.0, 0.0])
    graph = np.array([0.0, 1.0])
    result = late_fusion(visual, graph, alpha=0.5)
    np.testing.assert_allclose(result, np.array([0.5, 0.5]))


def test_late_fusion_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        late_fusion(np.array([1.0]), np.array([1.0]), alpha=1.1)


def test_minmax_uses_external_bounds():
    result = minmax_scale(np.array([2.0, 4.0]), low=0.0, high=4.0)
    np.testing.assert_allclose(result, np.array([0.5, 1.0]))
