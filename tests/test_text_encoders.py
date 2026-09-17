"""Tests for the text encoder that do not require the optional extra."""

from __future__ import annotations

import numpy as np
import pytest

from caypollard.text.encoders import (
    DEFAULT_MODELS,
    TextModelSpec,
    mean_pool,
    resolve_text_spec,
)


class FakeTensor:
    """The three tensor operations mean_pool uses, on numpy arrays."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array, dtype=float)

    @property
    def dtype(self):
        return self.array.dtype

    def unsqueeze(self, axis: int) -> FakeTensor:
        return FakeTensor(np.expand_dims(self.array, axis))

    def to(self, dtype) -> FakeTensor:
        return FakeTensor(self.array.astype(dtype))

    def sum(self, dim: int) -> FakeTensor:
        return FakeTensor(self.array.sum(axis=dim))

    def clamp(self, min: float) -> FakeTensor:
        return FakeTensor(np.clip(self.array, min, None))

    def __mul__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(self.array * other.array)

    def __truediv__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(self.array / other.array)


def test_mean_pool_ignores_padding_tokens() -> None:
    # Two tokens of content then one of padding: the padded vector must not
    # drag the mean, which is exactly what CLS-or-naive-mean pooling gets wrong.
    hidden = FakeTensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = FakeTensor([[1, 1, 0]])
    pooled = mean_pool(hidden, mask)
    np.testing.assert_allclose(pooled.array, [[2.0, 2.0]])


def test_mean_pool_survives_an_all_padding_row_without_dividing_by_zero() -> None:
    hidden = FakeTensor([[[5.0, 5.0]]])
    mask = FakeTensor([[0]])
    pooled = mean_pool(hidden, mask)
    assert np.isfinite(pooled.array).all()


def test_every_preset_declares_a_model_id() -> None:
    assert DEFAULT_MODELS
    for name, spec in DEFAULT_MODELS.items():
        assert isinstance(spec, TextModelSpec)
        assert "/" in spec.model_id, name


def test_e5_carries_the_prefix_its_contract_requires() -> None:
    assert DEFAULT_MODELS["e5-multilingual"].query_prefix == "query: "
    assert DEFAULT_MODELS["labse"].query_prefix == ""


def test_resolve_prefers_an_explicit_model_id_over_a_preset() -> None:
    spec = resolve_text_spec("labse", "some-org/some-model")
    assert spec.model_id == "some-org/some-model"
    assert spec.family == "custom"


def test_resolve_rejects_an_unknown_preset_and_an_empty_request() -> None:
    with pytest.raises(ValueError, match="unknown text preset"):
        resolve_text_spec("not-a-model", None)
    with pytest.raises(ValueError, match="either a preset"):
        resolve_text_spec(None, None)
